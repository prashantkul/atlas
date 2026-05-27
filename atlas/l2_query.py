"""L2 Trust-Conditioned Query Classifier for ATLAS.

Trains two per-query classifiers — a baseline (without trust score) and an
ATLAS variant (with the L1 trust score) — then compares their AUC to
quantify the lift from account-level trust conditioning.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from atlas.utils import (
  DATA_DIR,
  MODELS_DIR,
  QUERIES_PARQUET,
  SEED,
  ensure_dirs,
  seed_everything,
  setup_logging,
)

log = logging.getLogger(__name__)

BASELINE_FEATURES = [
  "query_risk_raw",
  "topic_is_ccl",
  "jailbreak_technique_used",
  "turn_position",
  "session_refusal_count",
]

ATLAS_FEATURES = BASELINE_FEATURES + ["trust_score"]

LGB_PARAMS: dict = dict(
  num_leaves=31,
  learning_rate=0.05,
  n_estimators=200,
  min_child_samples=20,
  random_state=SEED,
  verbose=-1,
  is_unbalance=True,
)


def _load_and_merge(
  queries_path: Path,
  l1_predictions_path: Path,
) -> pd.DataFrame:
  log.info("Loading queries from %s", queries_path)
  queries = pd.read_parquet(queries_path)
  log.info("Loaded %d queries", len(queries))

  log.info("Loading L1 predictions from %s", l1_predictions_path)
  l1 = pd.read_parquet(l1_predictions_path)
  log.info("Loaded %d L1 predictions", len(l1))

  queries = queries.drop(columns=["archetype"], errors="ignore")
  df = queries.merge(l1[["account_id", "archetype", "trust_score"]], on="account_id", how="left")
  log.info("Merged dataset: %d rows, trust_score NaN count: %d",
           len(df), df["trust_score"].isna().sum())
  return df


def _create_labels(df: pd.DataFrame) -> pd.Series:
  is_malicious_account = df["archetype"].isin(["persistent_adversary", "sleeper"])
  risky_content = (df["query_risk_raw"] >= 0.5) | (df["jailbreak_technique_used"] == 1)
  should_block = (is_malicious_account & risky_content).astype(int)

  log.info("Label distribution: block=%d, allow=%d",
           (should_block == 1).sum(), (should_block == 0).sum())
  return should_block


def _split(
  df: pd.DataFrame,
  label: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
  df_train, df_temp, y_train, y_temp = train_test_split(
    df, label, test_size=0.30, stratify=label, random_state=SEED,
  )
  df_val, df_test, y_val, y_test = train_test_split(
    df_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED,
  )
  log.info("Split sizes — train: %d, val: %d, test: %d",
           len(df_train), len(df_val), len(df_test))
  return df_train, df_val, df_test, y_train, y_val, y_test


def _train_model(
  X_train: pd.DataFrame,
  y_train: pd.Series,
  X_val: pd.DataFrame,
  y_val: pd.Series,
  features: list[str],
  name: str,
) -> lgb.LGBMClassifier:
  log.info("Training %s with features: %s", name, features)
  model = lgb.LGBMClassifier(**LGB_PARAMS)
  model.fit(
    X_train[features],
    y_train,
    eval_set=[(X_val[features], y_val)],
    callbacks=[lgb.early_stopping(20, verbose=False)],
  )
  log.info("%s best iteration: %d", name, model.best_iteration_)
  return model


def _evaluate(
  model: lgb.LGBMClassifier,
  X: pd.DataFrame,
  y: pd.Series,
  features: list[str],
  name: str,
) -> float:
  probs = model.predict_proba(X[features])[:, 1]
  auc = roc_auc_score(y, probs)
  log.info("%s test AUC: %.4f", name, auc)
  return auc


def train_and_evaluate(
  queries_path: Path = QUERIES_PARQUET,
  l1_predictions_path: Path = DATA_DIR / "l1_predictions.parquet",
  baseline_model_path: Path = MODELS_DIR / "l2_baseline_model.pkl",
  atlas_model_path: Path = MODELS_DIR / "l2_atlas_model.pkl",
  predictions_path: Path = DATA_DIR / "l2_predictions.parquet",
) -> None:
  df = _load_and_merge(queries_path, l1_predictions_path)
  label = _create_labels(df)

  df_train, df_val, df_test, y_train, y_val, y_test = _split(df, label)

  baseline_model = _train_model(
    df_train, y_train, df_val, y_val, BASELINE_FEATURES, "L2_baseline",
  )
  atlas_model = _train_model(
    df_train, y_train, df_val, y_val, ATLAS_FEATURES, "L2_atlas",
  )

  auc_baseline = _evaluate(baseline_model, df_test, y_test, BASELINE_FEATURES, "L2_baseline")
  auc_atlas = _evaluate(atlas_model, df_test, y_test, ATLAS_FEATURES, "L2_atlas")

  delta = auc_atlas - auc_baseline
  log.info("AUC lift from trust conditioning: %.4f (%.2f%% relative)",
           delta, 100.0 * delta / max(auc_baseline, 1e-9))

  joblib.dump(baseline_model, baseline_model_path)
  log.info("Saved baseline model -> %s", baseline_model_path)
  joblib.dump(atlas_model, atlas_model_path)
  log.info("Saved ATLAS model -> %s", atlas_model_path)

  baseline_prob = baseline_model.predict_proba(df_test[BASELINE_FEATURES])[:, 1]
  atlas_prob = atlas_model.predict_proba(df_test[ATLAS_FEATURES])[:, 1]

  predictions = pd.DataFrame({
    "account_id": df_test["account_id"].values,
    "archetype": df_test["archetype"].values,
    "label": y_test.values,
    "baseline_pred": (baseline_prob >= 0.5).astype(int),
    "baseline_prob": baseline_prob,
    "atlas_pred": (atlas_prob >= 0.5).astype(int),
    "atlas_prob": atlas_prob,
    "trust_score": df_test["trust_score"].values,
    "query_risk_raw": df_test["query_risk_raw"].values,
    "topic_is_ccl": df_test["topic_is_ccl"].values,
  })

  predictions.to_parquet(predictions_path, index=False)
  log.info("Saved predictions (%d rows) -> %s", len(predictions), predictions_path)


def main() -> None:
  parser = argparse.ArgumentParser(description="ATLAS L2 Trust-Conditioned Query Classifier")
  parser.add_argument("--queries", type=str, default=str(QUERIES_PARQUET),
                      help="Path to queries parquet")
  parser.add_argument("--l1-predictions", type=str,
                      default=str(DATA_DIR / "l1_predictions.parquet"),
                      help="Path to L1 predictions parquet")
  parser.add_argument("--baseline-model", type=str,
                      default=str(MODELS_DIR / "l2_baseline_model.pkl"),
                      help="Output path for baseline model")
  parser.add_argument("--atlas-model", type=str,
                      default=str(MODELS_DIR / "l2_atlas_model.pkl"),
                      help="Output path for ATLAS model")
  parser.add_argument("--predictions", type=str,
                      default=str(DATA_DIR / "l2_predictions.parquet"),
                      help="Output path for predictions parquet")
  args = parser.parse_args()

  setup_logging()
  seed_everything()
  ensure_dirs()

  train_and_evaluate(
    queries_path=Path(args.queries),
    l1_predictions_path=Path(args.l1_predictions),
    baseline_model_path=Path(args.baseline_model),
    atlas_model_path=Path(args.atlas_model),
    predictions_path=Path(args.predictions),
  )


if __name__ == "__main__":
  main()
