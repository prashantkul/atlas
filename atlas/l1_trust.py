"""L1 Account Trust Scorer — LightGBM binary classifier producing trust scores [0,1]."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from atlas.utils import (
  ACCOUNTS_PARQUET,
  ALL_FEATURES,
  DATA_DIR,
  MODELS_DIR,
  SEED,
  ensure_dirs,
  seed_everything,
  setup_logging,
)

log = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
  df = pd.read_parquet(path)
  required = {"account_id", "archetype", "label"} | set(ALL_FEATURES)
  missing = required - set(df.columns)
  if missing:
    raise ValueError(f"Missing columns: {missing}")
  return df


def split_data(
  df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
  splitter_tv = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
  train_idx, temp_idx = next(splitter_tv.split(df, df["archetype"]))

  temp_df = df.iloc[temp_idx]
  splitter_vt = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
  val_rel_idx, test_rel_idx = next(splitter_vt.split(temp_df, temp_df["archetype"]))

  val_idx = temp_idx[val_rel_idx]
  test_idx = temp_idx[test_rel_idx]

  split_indices = {
    "train": np.array(train_idx),
    "val": np.array(val_idx),
    "test": np.array(test_idx),
  }

  return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx], split_indices


def train_model(
  train_df: pd.DataFrame,
  val_df: pd.DataFrame,
) -> lgb.LGBMClassifier:
  X_train, y_train = train_df[ALL_FEATURES], train_df["label"]
  X_val, y_val = val_df[ALL_FEATURES], val_df["label"]

  model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=300,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    is_unbalance=True,
    random_state=SEED,
    verbose=-1,
  )

  model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
      lgb.early_stopping(stopping_rounds=20, verbose=False),
      lgb.log_evaluation(period=0),
    ],
  )

  return model


def evaluate(
  model: lgb.LGBMClassifier,
  df: pd.DataFrame,
  split_name: str,
) -> float:
  X = df[ALL_FEATURES]
  y = df["label"]
  proba = model.predict_proba(X)[:, 1]
  auc = roc_auc_score(y, proba)
  log.info("%s AUC: %.4f", split_name, auc)
  return auc


def build_predictions(
  model: lgb.LGBMClassifier,
  df: pd.DataFrame,
) -> pd.DataFrame:
  trust_scores = model.predict_proba(df[ALL_FEATURES])[:, 1]
  return pd.DataFrame({
    "account_id": df["account_id"].values,
    "archetype": df["archetype"].values,
    "label": df["label"].values,
    "trust_score": trust_scores,
  })


def main() -> None:
  parser = argparse.ArgumentParser(description="L1 Account Trust Scorer")
  parser.add_argument("--data-path", type=Path, default=ACCOUNTS_PARQUET)
  parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
  args = parser.parse_args()

  setup_logging()
  seed_everything()
  ensure_dirs()

  data_path: Path = args.data_path
  model_dir: Path = args.model_dir
  model_dir.mkdir(parents=True, exist_ok=True)

  log.info("Loading data from %s", data_path)
  df = load_data(data_path)
  log.info("Loaded %d accounts with %d features", len(df), len(ALL_FEATURES))

  train_df, val_df, test_df, split_indices = split_data(df)
  log.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

  log.info("Training LightGBM model")
  model = train_model(train_df, val_df)

  evaluate(model, train_df, "Train")
  evaluate(model, val_df, "Val")
  evaluate(model, test_df, "Test")

  model_path = model_dir / "l1_trust_model.pkl"
  joblib.dump(model, model_path)
  log.info("Model saved to %s", model_path)

  split_path = model_dir / "l1_split.pkl"
  joblib.dump(split_indices, split_path)
  log.info("Split indices saved to %s", split_path)

  preds_df = build_predictions(model, df)
  preds_path = DATA_DIR / "l1_predictions.parquet"
  preds_df.to_parquet(preds_path, index=False)
  log.info("Predictions saved to %s", preds_path)


if __name__ == "__main__":
  main()
