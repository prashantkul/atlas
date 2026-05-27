"""Full evaluation suite for ATLAS — L1 trust scorer and L2 query classifier."""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
  accuracy_score,
  auc,
  confusion_matrix,
  f1_score,
  precision_recall_curve,
  roc_auc_score,
  roc_curve,
)

from atlas.utils import (
  ACCOUNTS_PARQUET,
  ALL_FEATURES,
  DATA_DIR,
  MODELS_DIR,
  OUTPUTS_DIR,
  PLOTS_DIR,
  ensure_dirs,
  seed_everything,
  setup_logging,
)

log = logging.getLogger(__name__)

ARCHETYPE_COLORS: dict[str, str] = {
  "clean_enterprise": "#2196F3",
  "clean_consumer": "#4CAF50",
  "persistent_adversary": "#F44336",
  "sleeper": "#FF9800",
}

FIGSIZE = (10, 6)
DPI = 150


def _style() -> None:
  sns.set_style("whitegrid")
  plt.rcParams.update({"figure.figsize": FIGSIZE, "savefig.dpi": DPI})


# ---------------------------------------------------------------------------
# L1 evaluation
# ---------------------------------------------------------------------------

def _l1_metrics(
  model: object,
  df: pd.DataFrame,
) -> dict[str, float]:
  X = df[ALL_FEATURES]
  y = df["label"].values
  proba = model.predict_proba(X)[:, 1]  # type: ignore[union-attr]
  preds = (proba >= 0.5).astype(int)

  fpr_curve, tpr_curve, _ = roc_curve(y, proba)
  prec_curve, rec_curve, _ = precision_recall_curve(y, proba)

  return {
    "auc_roc": roc_auc_score(y, proba),
    "auc_pr": auc(rec_curve, prec_curve),
    "accuracy": accuracy_score(y, preds),
    "f1": f1_score(y, preds),
  }


def _l1_trust_distributions(df: pd.DataFrame) -> Path:
  fig, ax = plt.subplots(figsize=FIGSIZE)
  archetypes = sorted(ARCHETYPE_COLORS.keys())
  data = [df.loc[df["archetype"] == a, "trust_score"].values for a in archetypes]
  colors = [ARCHETYPE_COLORS[a] for a in archetypes]

  bp = ax.boxplot(data, labels=archetypes, patch_artist=True, showfliers=True)
  for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

  ax.set_ylabel("Trust Score")
  ax.set_title("L1 Trust Score Distribution by Archetype")
  ax.set_ylim(-0.05, 1.05)
  fig.tight_layout()
  path = PLOTS_DIR / "l1_trust_distributions.png"
  fig.savefig(path, dpi=DPI)
  plt.close(fig)
  log.info("Saved %s", path)
  return path


def _l1_false_positive_rate(df: pd.DataFrame) -> dict[str, float]:
  clean = df[df["archetype"].isin(["clean_enterprise", "clean_consumer"])]
  flagged = (clean["trust_score"] < 0.5).sum()
  total = len(clean)
  rate = flagged / max(total, 1)
  log.info("L1 FP rate (clean accounts with trust < 0.5): %d / %d = %.4f", flagged, total, rate)
  return {"fp_count": int(flagged), "fp_total": int(total), "fp_rate": float(rate)}


def _l1_sleeper_detection(model: object) -> dict[str, float]:
  traj_path = DATA_DIR / "sleeper_trajectories.parquet"
  if not traj_path.exists():
    log.warning("Sleeper trajectories not found at %s — skipping", traj_path)
    return {}

  traj = pd.read_parquet(traj_path)
  feature_cols = [c for c in traj.columns if c not in ("account_id", "day")]

  traj["trust_score"] = model.predict_proba(traj[feature_cols])[:, 1]  # type: ignore[union-attr]

  latencies: list[float] = []
  for aid, grp in traj.groupby("account_id"):
    grp = grp.sort_values("day")
    shift_mask = grp["day"] >= 60
    detected = grp.loc[shift_mask & (grp["trust_score"] < 0.5)]
    if not detected.empty:
      first_detect_day = detected["day"].iloc[0]
      shift_start = grp.loc[shift_mask, "day"].iloc[0]
      latencies.append(float(first_detect_day - shift_start))

  if not latencies:
    log.warning("No sleeper detections found")
    return {}

  latencies_arr = np.array(latencies)
  median_lat = float(np.median(latencies_arr))
  p90_lat = float(np.percentile(latencies_arr, 90))
  log.info("Sleeper detection latency — median: %.1f days, p90: %.1f days", median_lat, p90_lat)

  fig, ax = plt.subplots(figsize=FIGSIZE)
  ax.hist(latencies_arr, bins=20, color=ARCHETYPE_COLORS["sleeper"], alpha=0.7, edgecolor="black")
  ax.axvline(median_lat, color="red", linestyle="--", label=f"Median = {median_lat:.1f}d")
  ax.axvline(p90_lat, color="darkred", linestyle=":", label=f"p90 = {p90_lat:.1f}d")
  ax.set_xlabel("Detection Latency (days after behavioral shift)")
  ax.set_ylabel("Count")
  ax.set_title("L1 Sleeper Detection Latency")
  ax.legend()
  fig.tight_layout()
  path = PLOTS_DIR / "l1_sleeper_detection.png"
  fig.savefig(path, dpi=DPI)
  plt.close(fig)
  log.info("Saved %s", path)
  return {"median_latency_days": median_lat, "p90_latency_days": p90_lat}


def _l1_shap(model: object, df: pd.DataFrame) -> None:
  X = df[ALL_FEATURES]
  explainer = shap.TreeExplainer(model)
  shap_values = explainer(X)

  fig = plt.figure(figsize=FIGSIZE)
  shap.plots.beeswarm(shap_values, show=False)
  fig = plt.gcf()
  fig.tight_layout()
  path = PLOTS_DIR / "l1_shap_summary.png"
  fig.savefig(path, dpi=DPI, bbox_inches="tight")
  plt.close("all")
  log.info("Saved %s", path)

  for archetype in ARCHETYPE_COLORS:
    arch_mask = df["archetype"] == archetype
    if arch_mask.sum() == 0:
      continue
    idx = df.loc[arch_mask].index[0]
    row_pos = list(X.index).index(idx)

    fig = plt.figure(figsize=FIGSIZE)
    shap.plots.waterfall(shap_values[row_pos], show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Waterfall — {archetype}", fontsize=12)
    fig.tight_layout()
    wf_path = PLOTS_DIR / f"l1_shap_waterfall_{archetype}.png"
    fig.savefig(wf_path, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    log.info("Saved %s", wf_path)


def _l1_calibration(model: object, df: pd.DataFrame) -> None:
  X = df[ALL_FEATURES]
  y = df["label"].values
  proba = model.predict_proba(X)[:, 1]  # type: ignore[union-attr]

  fraction_pos, mean_predicted = calibration_curve(y, proba, n_bins=10, strategy="uniform")

  fig, ax = plt.subplots(figsize=FIGSIZE)
  ax.plot(mean_predicted, fraction_pos, "s-", color="#2196F3", label="L1 Model")
  ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
  ax.set_xlabel("Mean Predicted Probability")
  ax.set_ylabel("Fraction of Positives")
  ax.set_title("L1 Calibration (Reliability Diagram)")
  ax.legend()
  fig.tight_layout()
  path = PLOTS_DIR / "l1_calibration.png"
  fig.savefig(path, dpi=DPI)
  plt.close(fig)
  log.info("Saved %s", path)


def _l1_feature_importance(model: object) -> None:
  importances = model.feature_importances_  # type: ignore[union-attr]
  sorted_idx = np.argsort(importances)
  features_sorted = [ALL_FEATURES[i] for i in sorted_idx]
  values_sorted = importances[sorted_idx]

  fig, ax = plt.subplots(figsize=FIGSIZE)
  ax.barh(features_sorted, values_sorted, color="#2196F3", alpha=0.8)
  ax.set_xlabel("Feature Importance (split count)")
  ax.set_title("L1 Feature Importance")
  fig.tight_layout()
  path = PLOTS_DIR / "l1_feature_importance.png"
  fig.savefig(path, dpi=DPI)
  plt.close(fig)
  log.info("Saved %s", path)


def evaluate_l1(
  model_path: Path,
  split_path: Path,
  accounts_path: Path,
) -> dict:
  log.info("=== L1 Evaluation ===")
  model = joblib.load(model_path)
  split = joblib.load(split_path)
  df = pd.read_parquet(accounts_path)

  test_df = df.iloc[split["test"]].copy()
  log.info("Test set: %d accounts", len(test_df))

  test_df["trust_score"] = model.predict_proba(test_df[ALL_FEATURES])[:, 1]

  metrics = _l1_metrics(model, test_df)
  log.info("L1 test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

  all_df = df.copy()
  all_df["trust_score"] = model.predict_proba(all_df[ALL_FEATURES])[:, 1]
  _l1_trust_distributions(all_df)

  fp_info = _l1_false_positive_rate(all_df)

  sleeper_info = _l1_sleeper_detection(model)

  _l1_shap(model, test_df)

  _l1_calibration(model, test_df)

  _l1_feature_importance(model)

  return {
    "metrics": metrics,
    "false_positive": fp_info,
    "sleeper_detection": sleeper_info,
  }


# ---------------------------------------------------------------------------
# L2 evaluation
# ---------------------------------------------------------------------------

def _l2_overall_metrics(preds: pd.DataFrame) -> dict[str, dict[str, float]]:
  results: dict[str, dict[str, float]] = {}
  for variant, pred_col, prob_col in [
    ("baseline", "baseline_pred", "baseline_prob"),
    ("atlas", "atlas_pred", "atlas_prob"),
  ]:
    y_true = preds["label"].values
    y_pred = preds[pred_col].values
    y_prob = preds[prob_col].values
    results[variant] = {
      "accuracy": accuracy_score(y_true, y_pred),
      "f1": f1_score(y_true, y_pred),
      "auc": roc_auc_score(y_true, y_prob),
    }
  return results


def _l2_enterprise_fp_rate(preds: pd.DataFrame) -> dict[str, float]:
  mask = (preds["archetype"] == "clean_enterprise") & (preds["topic_is_ccl"] == 1)
  subset = preds[mask]
  if len(subset) == 0:
    log.warning("No clean enterprise CCL queries found")
    return {"baseline_fp": 0.0, "atlas_fp": 0.0}

  baseline_fp = (subset["baseline_pred"] == 1).sum() / len(subset)
  atlas_fp = (subset["atlas_pred"] == 1).sum() / len(subset)
  log.info("Enterprise pharma FP — baseline: %.4f, atlas: %.4f (n=%d)",
           baseline_fp, atlas_fp, len(subset))
  return {
    "baseline_fp": float(baseline_fp),
    "atlas_fp": float(atlas_fp),
    "n_queries": int(len(subset)),
  }


def _l2_adversary_fn_rate(preds: pd.DataFrame) -> dict[str, float]:
  mask = (preds["archetype"] == "persistent_adversary") & (preds["label"] == 1)
  subset = preds[mask]
  if len(subset) == 0:
    log.warning("No adversary should_block queries found")
    return {"baseline_fn": 0.0, "atlas_fn": 0.0}

  baseline_fn = (subset["baseline_pred"] == 0).sum() / len(subset)
  atlas_fn = (subset["atlas_pred"] == 0).sum() / len(subset)
  log.info("Adversary FN — baseline: %.4f, atlas: %.4f (n=%d)",
           baseline_fn, atlas_fn, len(subset))
  return {
    "baseline_fn": float(baseline_fn),
    "atlas_fn": float(atlas_fn),
    "n_queries": int(len(subset)),
  }


def _l2_confusion_matrices(preds: pd.DataFrame) -> None:
  archetypes = sorted(preds["archetype"].unique())
  n_arch = len(archetypes)
  fig, axes = plt.subplots(n_arch, 2, figsize=(12, 4 * n_arch))
  if n_arch == 1:
    axes = axes.reshape(1, -1)

  for row_idx, archetype in enumerate(archetypes):
    subset = preds[preds["archetype"] == archetype]
    y_true = subset["label"].values

    for col_idx, (variant, pred_col) in enumerate([
      ("Baseline", "baseline_pred"),
      ("ATLAS", "atlas_pred"),
    ]):
      ax = axes[row_idx, col_idx]
      y_pred = subset[pred_col].values
      cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                  xticklabels=["Allow", "Block"], yticklabels=["Allow", "Block"])
      ax.set_title(f"{variant} — {archetype}")
      ax.set_ylabel("Actual")
      ax.set_xlabel("Predicted")

  fig.suptitle("L2 Confusion Matrices by Archetype", fontsize=14, y=1.02)
  fig.tight_layout()
  path = PLOTS_DIR / "l2_confusion_matrices.png"
  fig.savefig(path, dpi=DPI, bbox_inches="tight")
  plt.close(fig)
  log.info("Saved %s", path)


def _l2_threshold_modulation(preds: pd.DataFrame) -> None:
  mask = (
    (preds["query_risk_raw"] >= 0.4)
    & (preds["query_risk_raw"] <= 0.6)
    & (preds["topic_is_ccl"] == 1)
  )
  subset = preds[mask].copy()
  if len(subset) == 0:
    log.warning("No medium-risk CCL queries found for threshold modulation plot")
    return

  bins = np.arange(0, 1.1, 0.1)
  labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
  subset["trust_bin"] = pd.cut(subset["trust_score"], bins=bins, labels=labels, include_lowest=True)

  grouped = subset.groupby("trust_bin", observed=False)["atlas_prob"].mean()

  fig, ax = plt.subplots(figsize=FIGSIZE)
  x_positions = np.arange(len(labels))
  values = [grouped.get(label, np.nan) for label in labels]
  ax.bar(x_positions, values, color="#2196F3", alpha=0.7, edgecolor="black")
  ax.plot(x_positions, values, "o-", color="#F44336", linewidth=2)
  ax.set_xticks(x_positions)
  ax.set_xticklabels(labels, rotation=45, ha="right")
  ax.set_xlabel("Trust Score Bin")
  ax.set_ylabel("Mean P(block)")
  ax.set_title("L2 ATLAS: Threshold Modulation on Medium-Risk CCL Queries")
  fig.tight_layout()
  path = PLOTS_DIR / "l2_threshold_modulation.png"
  fig.savefig(path, dpi=DPI)
  plt.close(fig)
  log.info("Saved %s", path)


def evaluate_l2(
  baseline_model_path: Path,
  atlas_model_path: Path,
  predictions_path: Path,
) -> dict:
  log.info("=== L2 Evaluation ===")
  joblib.load(baseline_model_path)
  joblib.load(atlas_model_path)
  preds = pd.read_parquet(predictions_path)
  log.info("Loaded %d L2 predictions", len(preds))

  overall = _l2_overall_metrics(preds)
  log.info("L2 overall — baseline: %s", {k: f"{v:.4f}" for k, v in overall["baseline"].items()})
  log.info("L2 overall — atlas:    %s", {k: f"{v:.4f}" for k, v in overall["atlas"].items()})

  enterprise_fp = _l2_enterprise_fp_rate(preds)
  adversary_fn = _l2_adversary_fn_rate(preds)

  _l2_confusion_matrices(preds)
  _l2_threshold_modulation(preds)

  return {
    "overall": overall,
    "enterprise_fp": enterprise_fp,
    "adversary_fn": adversary_fn,
  }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(l1_results: dict, l2_results: dict) -> Path:
  l1m = l1_results["metrics"]
  fp = l1_results["false_positive"]
  sl = l1_results.get("sleeper_detection", {})
  l2o = l2_results["overall"]
  efp = l2_results["enterprise_fp"]
  afn = l2_results["adversary_fn"]

  acc_lift = l2o["atlas"]["accuracy"] - l2o["baseline"]["accuracy"]
  f1_lift = l2o["atlas"]["f1"] - l2o["baseline"]["f1"]
  auc_lift = l2o["atlas"]["auc"] - l2o["baseline"]["auc"]
  fp_reduction = efp["baseline_fp"] - efp["atlas_fp"]

  report = textwrap.dedent(f"""\
    # ATLAS Evaluation Report

    ## Abstract

    ATLAS (Account Trust-Layered Adaptive Safety) conditions LLM query-level safety decisions
    on persistent account trust scores. The L1 trust scorer achieves {l1m['auc_roc']:.3f} AUC-ROC
    on held-out accounts. The L2 trust-conditioned query classifier improves AUC by
    {auc_lift:+.4f} over the baseline, while reducing false positive rate on legitimate
    enterprise CCL queries from {efp['baseline_fp']:.1%} to {efp['atlas_fp']:.1%}
    (a {fp_reduction:.1%} absolute reduction).

    ## Method

    **L1 Account Trust Scorer.** A LightGBM binary classifier trained on 18 account-level
    features (identity, behavioral, session) to predict whether an account is adversarial
    (label=1) or clean (label=0). Outputs a continuous trust score in [0, 1] via
    `predict_proba`.

    **L2 Query Classifier.** Two LightGBM classifiers predict whether a query should be
    blocked. The *baseline* uses 5 query-level features; the *ATLAS* variant adds the L1
    trust score as a 6th feature, enabling trust-conditioned decisions.

    ## Results

    ### L1 Trust Scorer

    | Metric   | Value  |
    |----------|--------|
    | AUC-ROC  | {l1m['auc_roc']:.4f} |
    | AUC-PR   | {l1m['auc_pr']:.4f} |
    | Accuracy | {l1m['accuracy']:.4f} |
    | F1       | {l1m['f1']:.4f} |

    **False positive rate** (clean accounts with trust < 0.5): {fp['fp_count']} / {fp['fp_total']} = {fp['fp_rate']:.4f}

    **Trust score distributions** by archetype:

    ![L1 Trust Distributions](plots/l1_trust_distributions.png)

    **Feature importance:**

    ![L1 Feature Importance](plots/l1_feature_importance.png)

    **Calibration:**

    ![L1 Calibration](plots/l1_calibration.png)

    **SHAP summary:**

    ![L1 SHAP Summary](plots/l1_shap_summary.png)
  """)

  for archetype in ARCHETYPE_COLORS:
    report += f"\n![SHAP Waterfall — {archetype}](plots/l1_shap_waterfall_{archetype}.png)\n"

  if sl:
    report += textwrap.dedent(f"""
    **Sleeper detection latency:**
    - Median: {sl['median_latency_days']:.1f} days
    - 90th percentile: {sl['p90_latency_days']:.1f} days

    ![L1 Sleeper Detection](plots/l1_sleeper_detection.png)
    """)

  report += textwrap.dedent(f"""
    ### L2 Query Classifier

    | Metric   | Baseline | ATLAS  | Delta  |
    |----------|----------|--------|--------|
    | Accuracy | {l2o['baseline']['accuracy']:.4f}   | {l2o['atlas']['accuracy']:.4f} | {acc_lift:+.4f} |
    | F1       | {l2o['baseline']['f1']:.4f}   | {l2o['atlas']['f1']:.4f} | {f1_lift:+.4f} |
    | AUC      | {l2o['baseline']['auc']:.4f}   | {l2o['atlas']['auc']:.4f} | {auc_lift:+.4f} |

    **False positive rate on clean enterprise CCL queries** (the headline number):
    - Baseline: {efp['baseline_fp']:.4f} ({efp['baseline_fp']:.1%})
    - ATLAS: {efp['atlas_fp']:.4f} ({efp['atlas_fp']:.1%})
    - Reduction: {fp_reduction:.4f} ({fp_reduction:.1%} absolute)
    - n = {efp.get('n_queries', 'N/A')} queries

    **False negative rate on adversary queries:**
    - Baseline: {afn['baseline_fn']:.4f}
    - ATLAS: {afn['atlas_fn']:.4f}
    - n = {afn.get('n_queries', 'N/A')} queries

    **Confusion matrices by archetype:**

    ![L2 Confusion Matrices](plots/l2_confusion_matrices.png)

    **Threshold modulation** (medium-risk CCL queries, P(block) vs trust score):

    ![L2 Threshold Modulation](plots/l2_threshold_modulation.png)

    ## Discussion

    ### Limitations

    1. **Synthetic data.** All accounts and queries are generated from hand-crafted archetypes.
       Real adversary behavior is more diverse and adversarial drift patterns are harder to
       detect than our sleeper simulation suggests.
    2. **Feature set.** The 18 L1 features are a subset of what a production system would use.
       Features like IP geolocation, device fingerprinting, and payment history are absent.
    3. **Scale.** This evaluation uses {fp['fp_total']} accounts. At Google scale (billions of
       accounts), even a 0.1% false positive rate would affect millions of users. The
       calibration and threshold choices would need extensive tuning.
    4. **Temporal dynamics.** The sleeper detection test uses a simplified day-by-day trajectory.
       Real behavioral shifts are more gradual and may be masked by seasonal usage patterns.
    5. **Label definition.** The `should_block` label is rule-based (high risk + jailbreak).
       In production, labels come from human review, which introduces noise and subjectivity.

    ### What Would Change at Google Scale

    - Trust scores would need to be computed incrementally (streaming features) rather than
      batch-recomputed, with staleness guarantees.
    - The L2 classifier would likely be replaced by a neural model or incorporated into the
      LLM's own safety head, with the trust score as an embedding input.
    - Fairness constraints would be critical: trust scores must not proxy for demographic
      attributes. Regular bias audits and counterfactual fairness checks would be mandatory.
    - The threshold modulation curve would be tuned per-product and per-region, with
      different risk appetites for different use cases.
    - A/B testing with online metrics (user satisfaction, appeal rate, adversary success rate)
      would replace offline AUC comparisons.

    ### Client-Side Signals: Browser Fingerprint & Device Identity

    The current ATLAS feature set is entirely server-side. A production deployment would
    benefit from client-side signals including:

    - **Browser/device fingerprint:** Canvas hash, WebGL renderer, installed fonts, screen
      resolution, timezone, and navigator properties compose a semi-unique device identifier.
      This enables detection of **account farming** (many accounts operated from the same
      device) and **credential sharing** (one account used across unusual device diversity).
      Adversaries who burn through throwaway accounts often reuse the same browser profile,
      creating a cross-account linkage invisible to server-side features alone.
    - **IP geolocation and ASN:** Data-center IPs, VPN/proxy exit nodes, and Tor relays are
      disproportionately used by adversaries. Geolocation shifts (e.g., an enterprise account
      normally in Boston suddenly querying from a residential IP in a different country) could
      feed into `channel_migration` or a new `geo_anomaly` feature.
    - **Device consistency score:** How stable the device fingerprint is over time. Legitimate
      users show a small set of consistent devices. Adversaries using anti-fingerprinting tools
      or VM rotation produce high device entropy.

    **Privacy and fairness considerations:** Client-side signals carry significant risks.
    Browser fingerprinting can proxy for socioeconomic status (older devices, less common
    browsers). IP-based signals can discriminate against users in regions with limited ISP
    diversity or where VPN usage is common for legitimate privacy reasons. Any production
    deployment would require: (1) privacy review and user consent mechanisms, (2) differential
    privacy or k-anonymity guarantees on fingerprint storage, (3) fairness audits to ensure
    these signals do not disproportionately penalize protected demographic groups, and
    (4) regulatory compliance (GDPR, CCPA) for fingerprint data retention.
  """)

  report_path = OUTPUTS_DIR / "evaluation_report.md"
  report_path.write_text(report)
  log.info("Saved evaluation report to %s", report_path)
  return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  parser = argparse.ArgumentParser(description="ATLAS Evaluation Suite")
  parser.add_argument("--l1-model", type=Path, default=MODELS_DIR / "l1_trust_model.pkl")
  parser.add_argument("--l1-split", type=Path, default=MODELS_DIR / "l1_split.pkl")
  parser.add_argument("--accounts", type=Path, default=ACCOUNTS_PARQUET)
  parser.add_argument("--l2-baseline", type=Path, default=MODELS_DIR / "l2_baseline_model.pkl")
  parser.add_argument("--l2-atlas", type=Path, default=MODELS_DIR / "l2_atlas_model.pkl")
  parser.add_argument("--l2-predictions", type=Path, default=DATA_DIR / "l2_predictions.parquet")
  args = parser.parse_args()

  setup_logging()
  seed_everything()
  ensure_dirs()
  _style()

  l1_results = evaluate_l1(args.l1_model, args.l1_split, args.accounts)

  l2_results = evaluate_l2(args.l2_baseline, args.l2_atlas, args.l2_predictions)

  _generate_report(l1_results, l2_results)

  log.info("Evaluation complete.")


if __name__ == "__main__":
  main()
