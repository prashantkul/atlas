"""Numpy-based programmatic fallback data generator for ATLAS.

Generates synthetic account, query, and sleeper trajectory data using
multivariate normal distributions with archetype-specific correlation matrices.
Used when the LLM (Qwen via OpenRouter) is unavailable.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd

from atlas.utils import (
  ACCOUNTS_PARQUET,
  ALL_FEATURES,
  ARCHETYPE_DISTRIBUTION,
  CCL_DOMAINS,
  DATA_DIR,
  JAILBREAK_TECHNIQUES,
  QUERIES_PARQUET,
  RISK_LEVEL_MAP,
  SEED,
  TOPICS,
  ensure_dirs,
  seed_everything,
  setup_logging,
)

log = logging.getLogger(__name__)

SLEEPER_TRAJECTORIES_PARQUET = DATA_DIR / "sleeper_trajectories.parquet"

ARCHETYPE_MEANS: dict[str, dict[str, float]] = {
  "clean_enterprise": {
    "account_age_days": 720, "verification_level": 3.0, "account_type": 2.0,
    "org_reputation": 0.85, "access_path_risk": 1.0,
    "refusal_rate_7d": 0.02, "refusal_rate_30d": 0.015, "refusal_rate_zscore": 0.0,
    "rephrase_after_refusal_rate": 0.1, "query_entropy": 1.5,
    "ccl_domain_concentration": 0.15, "cross_session_persistence": 0.7,
    "channel_migration": 0.05,
    "max_risk_score_10sessions": 0.3, "risk_trend_slope": 0.0,
    "mean_topic_escalation": 0.02, "mean_session_length": 8.0, "burst_score": 0.3,
  },
  "clean_consumer": {
    "account_age_days": 365, "verification_level": 1.5, "account_type": 0.0,
    "org_reputation": 0.0, "access_path_risk": 0.0,
    "refusal_rate_7d": 0.01, "refusal_rate_30d": 0.008, "refusal_rate_zscore": 0.0,
    "rephrase_after_refusal_rate": 0.05, "query_entropy": 2.8,
    "ccl_domain_concentration": 0.02, "cross_session_persistence": 0.3,
    "channel_migration": 0.02,
    "max_risk_score_10sessions": 0.15, "risk_trend_slope": 0.0,
    "mean_topic_escalation": 0.01, "mean_session_length": 5.0, "burst_score": 0.5,
  },
  "persistent_adversary": {
    "account_age_days": 30, "verification_level": 0.5, "account_type": 0.3,
    "org_reputation": 0.1, "access_path_risk": 1.5,
    "refusal_rate_7d": 0.35, "refusal_rate_30d": 0.30, "refusal_rate_zscore": 1.5,
    "rephrase_after_refusal_rate": 0.7, "query_entropy": 0.8,
    "ccl_domain_concentration": 0.6, "cross_session_persistence": 0.85,
    "channel_migration": 0.3,
    "max_risk_score_10sessions": 0.85, "risk_trend_slope": 0.05,
    "mean_topic_escalation": 0.15, "mean_session_length": 15.0, "burst_score": 0.8,
  },
  "sleeper": {
    "account_age_days": 540, "verification_level": 2.5, "account_type": 2.0,
    "org_reputation": 0.75, "access_path_risk": 1.0,
    "refusal_rate_7d": 0.25, "refusal_rate_30d": 0.15, "refusal_rate_zscore": 2.5,
    "rephrase_after_refusal_rate": 0.4, "query_entropy": 1.8,
    "ccl_domain_concentration": 0.4, "cross_session_persistence": 0.6,
    "channel_migration": 0.15,
    "max_risk_score_10sessions": 0.7, "risk_trend_slope": 0.15,
    "mean_topic_escalation": 0.1, "mean_session_length": 10.0, "burst_score": 0.4,
  },
}

ARCHETYPE_STDS: dict[str, dict[str, float]] = {
  "clean_enterprise": {
    "account_age_days": 350, "verification_level": 0.6, "account_type": 0.3,
    "org_reputation": 0.18, "access_path_risk": 0.5,
    "refusal_rate_7d": 0.03, "refusal_rate_30d": 0.025, "refusal_rate_zscore": 0.5,
    "rephrase_after_refusal_rate": 0.08, "query_entropy": 0.6,
    "ccl_domain_concentration": 0.12, "cross_session_persistence": 0.15,
    "channel_migration": 0.08,
    "max_risk_score_10sessions": 0.15, "risk_trend_slope": 0.04,
    "mean_topic_escalation": 0.02, "mean_session_length": 3.0, "burst_score": 0.2,
  },
  "clean_consumer": {
    "account_age_days": 250, "verification_level": 0.7, "account_type": 0.15,
    "org_reputation": 0.0, "access_path_risk": 0.3,
    "refusal_rate_7d": 0.015, "refusal_rate_30d": 0.012, "refusal_rate_zscore": 0.4,
    "rephrase_after_refusal_rate": 0.05, "query_entropy": 0.6,
    "ccl_domain_concentration": 0.04, "cross_session_persistence": 0.15,
    "channel_migration": 0.04,
    "max_risk_score_10sessions": 0.12, "risk_trend_slope": 0.02,
    "mean_topic_escalation": 0.015, "mean_session_length": 2.5, "burst_score": 0.25,
  },
  "persistent_adversary": {
    "account_age_days": 40, "verification_level": 0.6, "account_type": 0.5,
    "org_reputation": 0.12, "access_path_risk": 0.6,
    "refusal_rate_7d": 0.12, "refusal_rate_30d": 0.10, "refusal_rate_zscore": 0.6,
    "rephrase_after_refusal_rate": 0.15, "query_entropy": 0.4,
    "ccl_domain_concentration": 0.18, "cross_session_persistence": 0.12,
    "channel_migration": 0.12,
    "max_risk_score_10sessions": 0.12, "risk_trend_slope": 0.04,
    "mean_topic_escalation": 0.06, "mean_session_length": 5.0, "burst_score": 0.18,
  },
  "sleeper": {
    "account_age_days": 250, "verification_level": 0.5, "account_type": 0.3,
    "org_reputation": 0.15, "access_path_risk": 0.4,
    "refusal_rate_7d": 0.10, "refusal_rate_30d": 0.08, "refusal_rate_zscore": 0.7,
    "rephrase_after_refusal_rate": 0.12, "query_entropy": 0.5,
    "ccl_domain_concentration": 0.15, "cross_session_persistence": 0.15,
    "channel_migration": 0.10,
    "max_risk_score_10sessions": 0.15, "risk_trend_slope": 0.06,
    "mean_topic_escalation": 0.05, "mean_session_length": 3.5, "burst_score": 0.18,
  },
}

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
  "account_age_days": (1, 3650),
  "verification_level": (0, 3),
  "account_type": (0, 2),
  "org_reputation": (0, 1),
  "access_path_risk": (0, 2),
  "refusal_rate_7d": (0, 1),
  "refusal_rate_30d": (0, 1),
  "refusal_rate_zscore": (-2, 5),
  "rephrase_after_refusal_rate": (0, 1),
  "query_entropy": (0, 4),
  "ccl_domain_concentration": (0, 1),
  "cross_session_persistence": (0, 1),
  "channel_migration": (0, 1),
  "max_risk_score_10sessions": (0, 1),
  "risk_trend_slope": (-0.5, 0.5),
  "mean_topic_escalation": (0, 1),
  "mean_session_length": (1, 50),
  "burst_score": (0, 1),
}


def _build_correlation_matrix(archetype: str) -> np.ndarray:
  n = len(ALL_FEATURES)
  corr = np.eye(n)
  idx = {f: i for i, f in enumerate(ALL_FEATURES)}

  refusal_7d = idx["refusal_rate_7d"]
  refusal_30d = idx["refusal_rate_30d"]
  refusal_z = idx["refusal_rate_zscore"]
  rephrase = idx["rephrase_after_refusal_rate"]
  entropy = idx["query_entropy"]
  ccl = idx["ccl_domain_concentration"]
  persistence = idx["cross_session_persistence"]
  org_rep = idx["org_reputation"]
  verif = idx["verification_level"]
  max_risk = idx["max_risk_score_10sessions"]
  risk_slope = idx["risk_trend_slope"]
  escalation = idx["mean_topic_escalation"]
  channel = idx["channel_migration"]
  burst = idx["burst_score"]
  age = idx["account_age_days"]

  corr[refusal_7d, refusal_30d] = corr[refusal_30d, refusal_7d] = 0.85
  corr[refusal_7d, refusal_z] = corr[refusal_z, refusal_7d] = 0.6
  corr[refusal_30d, refusal_z] = corr[refusal_z, refusal_30d] = 0.5
  corr[max_risk, escalation] = corr[escalation, max_risk] = 0.5

  if archetype == "clean_enterprise":
    corr[org_rep, verif] = corr[verif, org_rep] = 0.7
    corr[org_rep, refusal_7d] = corr[refusal_7d, org_rep] = -0.4
    corr[org_rep, refusal_30d] = corr[refusal_30d, org_rep] = -0.4
    corr[org_rep, entropy] = corr[entropy, org_rep] = 0.3
    corr[verif, refusal_7d] = corr[refusal_7d, verif] = -0.3
    corr[age, org_rep] = corr[org_rep, age] = 0.5
    corr[age, verif] = corr[verif, age] = 0.4
    corr[persistence, entropy] = corr[entropy, persistence] = -0.2

  elif archetype == "clean_consumer":
    corr[entropy, ccl] = corr[ccl, entropy] = -0.2
    corr[age, verif] = corr[verif, age] = 0.3

  elif archetype == "persistent_adversary":
    corr[ccl, rephrase] = corr[rephrase, ccl] = 0.7
    corr[ccl, persistence] = corr[persistence, ccl] = 0.65
    corr[ccl, entropy] = corr[entropy, ccl] = -0.6
    corr[rephrase, persistence] = corr[persistence, rephrase] = 0.5
    corr[rephrase, refusal_7d] = corr[refusal_7d, rephrase] = 0.6
    corr[ccl, max_risk] = corr[max_risk, ccl] = 0.55
    corr[channel, rephrase] = corr[rephrase, channel] = 0.4
    corr[burst, ccl] = corr[ccl, burst] = 0.45
    corr[refusal_7d, max_risk] = corr[max_risk, refusal_7d] = 0.5

  elif archetype == "sleeper":
    corr[refusal_z, risk_slope] = corr[risk_slope, refusal_z] = 0.7
    corr[refusal_z, ccl] = corr[ccl, refusal_z] = 0.5
    corr[risk_slope, ccl] = corr[ccl, risk_slope] = 0.5
    corr[org_rep, verif] = corr[verif, org_rep] = 0.6
    corr[rephrase, ccl] = corr[ccl, rephrase] = 0.45
    corr[age, org_rep] = corr[org_rep, age] = 0.4

  eigvals = np.linalg.eigvalsh(corr)
  if eigvals.min() < 0:
    corr += (abs(eigvals.min()) + 1e-6) * np.eye(n)
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

  return corr


def _generate_archetype_samples(
  archetype: str, n: int, rng: np.random.Generator,
) -> np.ndarray:
  means = np.array([ARCHETYPE_MEANS[archetype][f] for f in ALL_FEATURES])
  stds = np.array([ARCHETYPE_STDS[archetype][f] for f in ALL_FEATURES])
  corr = _build_correlation_matrix(archetype)
  cov = np.outer(stds, stds) * corr

  eigvals = np.linalg.eigvalsh(cov)
  if eigvals.min() < 0:
    cov += (abs(eigvals.min()) + 1e-6) * np.eye(len(ALL_FEATURES))

  samples = rng.multivariate_normal(means, cov, size=n)

  for j, feat in enumerate(ALL_FEATURES):
    lo, hi = FEATURE_BOUNDS[feat]
    samples[:, j] = np.clip(samples[:, j], lo, hi)

  for j, feat in enumerate(ALL_FEATURES):
    if feat in ("verification_level", "account_type", "access_path_risk"):
      samples[:, j] = np.round(samples[:, j]).astype(int)
    if feat == "channel_migration":
      samples[:, j] = (samples[:, j] > 0.5).astype(float)
    if feat == "account_age_days":
      samples[:, j] = np.round(samples[:, j])

  if archetype == "clean_consumer":
    org_idx = ALL_FEATURES.index("org_reputation")
    samples[:, org_idx] = 0.0

  return samples


def _generate_sleeper_phase1_means() -> dict[str, float]:
  return {
    "account_age_days": 540, "verification_level": 2.5, "account_type": 2.0,
    "org_reputation": 0.75, "access_path_risk": 1.0,
    "refusal_rate_7d": 0.02, "refusal_rate_30d": 0.015, "refusal_rate_zscore": 0.0,
    "rephrase_after_refusal_rate": 0.1, "query_entropy": 1.5,
    "ccl_domain_concentration": 0.12, "cross_session_persistence": 0.65,
    "channel_migration": 0.05,
    "max_risk_score_10sessions": 0.3, "risk_trend_slope": 0.0,
    "mean_topic_escalation": 0.02, "mean_session_length": 8.0, "burst_score": 0.3,
  }


def _generate_sleeper_trajectories(
  n: int, rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
  phase1_means = _generate_sleeper_phase1_means()
  phase2_means = ARCHETYPE_MEANS["sleeper"]
  phase1_stds = ARCHETYPE_STDS["clean_enterprise"]
  phase2_stds = ARCHETYPE_STDS["sleeper"]

  total_days = 180
  snapshot_interval = 7
  n_snapshots = total_days // snapshot_interval

  transition_days = rng.integers(60, 101, size=n)

  traj_records: list[dict[str, Any]] = []

  final_features = np.zeros((n, len(ALL_FEATURES)))

  for i in range(n):
    transition = transition_days[i]

    for s in range(n_snapshots):
      day = (s + 1) * snapshot_interval
      t = max(0.0, min(1.0, (day - transition) / 30.0))
      t = 3 * t**2 - 2 * t**3

      snapshot: dict[str, Any] = {"account_idx": i, "day": day}
      for j, feat in enumerate(ALL_FEATURES):
        m = phase1_means[feat] * (1 - t) + phase2_means[feat] * t
        sd = phase1_stds[feat] * (1 - t) + phase2_stds[feat] * t
        sd = max(sd, 1e-6)
        val = rng.normal(m, sd * 0.5)
        lo, hi = FEATURE_BOUNDS[feat]
        val = float(np.clip(val, lo, hi))
        if feat in ("verification_level", "account_type", "access_path_risk"):
          val = float(round(val))
        if feat == "channel_migration":
          val = float(val > 0.5)
        if feat == "account_age_days":
          val = float(round(val))
        snapshot[feat] = val
        if s == n_snapshots - 1:
          final_features[i, j] = val

      traj_records.append(snapshot)

  traj_df = pd.DataFrame(traj_records)
  return final_features, traj_df


def generate_accounts(n_accounts: int = 500) -> pd.DataFrame:
  rng = np.random.default_rng(SEED)
  seed_everything(SEED)

  total_base = sum(ARCHETYPE_DISTRIBUTION.values())
  counts = {
    arch: max(1, int(round(n_accounts * cnt / total_base)))
    for arch, cnt in ARCHETYPE_DISTRIBUTION.items()
  }
  diff = n_accounts - sum(counts.values())
  if diff != 0:
    largest = max(counts, key=lambda k: counts[k])
    counts[largest] += diff

  log.info("Account counts per archetype: %s", counts)

  all_samples: list[np.ndarray] = []
  archetypes: list[str] = []
  sleeper_traj_df: pd.DataFrame | None = None

  for arch in ("clean_enterprise", "clean_consumer", "persistent_adversary", "sleeper"):
    n = counts[arch]
    if arch == "sleeper":
      final_features, sleeper_traj_df = _generate_sleeper_trajectories(n, rng)
      all_samples.append(final_features)
    else:
      samples = _generate_archetype_samples(arch, n, rng)
      all_samples.append(samples)
    archetypes.extend([arch] * n)

  data = np.vstack(all_samples)
  df = pd.DataFrame(data, columns=ALL_FEATURES)
  df["account_id"] = [f"acct_{i:05d}" for i in range(len(df))]
  df["archetype"] = archetypes
  df["label"] = df["archetype"].map(
    lambda a: 1 if a.startswith("clean") else 0
  )

  if sleeper_traj_df is not None:
    sleeper_start = sum(counts[a] for a in ("clean_enterprise", "clean_consumer", "persistent_adversary"))
    sleeper_traj_df["account_id"] = sleeper_traj_df["account_idx"].map(
      lambda idx: f"acct_{sleeper_start + idx:05d}"
    )
    sleeper_traj_df.drop(columns=["account_idx"], inplace=True)

  return df, sleeper_traj_df


def _sample_query_risk(
  archetype: str, turn_frac: float, rng: np.random.Generator,
) -> tuple[str, float]:
  if archetype == "clean_enterprise":
    r = rng.random()
    if r < 0.55:
      return "none", RISK_LEVEL_MAP["none"]
    if r < 0.85:
      return "low", RISK_LEVEL_MAP["low"]
    return "medium", RISK_LEVEL_MAP["medium"]

  if archetype == "clean_consumer":
    r = rng.random()
    if r < 0.60:
      return "none", RISK_LEVEL_MAP["none"]
    if r < 0.92:
      return "low", RISK_LEVEL_MAP["low"]
    return "medium", RISK_LEVEL_MAP["medium"]

  if archetype == "persistent_adversary":
    r = rng.random()
    if r < 0.20:
      return "low", RISK_LEVEL_MAP["low"]
    if r < 0.40:
      return "medium", RISK_LEVEL_MAP["medium"]
    if r < 0.75:
      return "high", RISK_LEVEL_MAP["high"]
    return "critical", RISK_LEVEL_MAP["critical"]

  # sleeper
  if turn_frac < 0.5:
    r = rng.random()
    if r < 0.55:
      return "none", RISK_LEVEL_MAP["none"]
    if r < 0.85:
      return "low", RISK_LEVEL_MAP["low"]
    return "medium", RISK_LEVEL_MAP["medium"]
  else:
    escalation = (turn_frac - 0.5) * 2
    r = rng.random()
    if r < 0.3 * (1 - escalation):
      return "low", RISK_LEVEL_MAP["low"]
    if r < 0.5:
      return "medium", RISK_LEVEL_MAP["medium"]
    if r < 0.8:
      return "high", RISK_LEVEL_MAP["high"]
    return "critical", RISK_LEVEL_MAP["critical"]


def _sample_topic(
  archetype: str, risk_level: str, rng: np.random.Generator,
) -> tuple[str, bool]:
  ccl_list = sorted(CCL_DOMAINS)
  safe_topics = [t for t in TOPICS if t not in CCL_DOMAINS]

  if archetype == "clean_enterprise":
    r = rng.random()
    if r < 0.15:
      return "cbrn", True
    if risk_level == "medium":
      topic = rng.choice(["chemistry", "medical", "science"])
      return topic, topic in CCL_DOMAINS
    return rng.choice(safe_topics), False

  if archetype == "clean_consumer":
    return rng.choice(safe_topics), False

  if archetype == "persistent_adversary":
    if risk_level in ("high", "critical"):
      topic = rng.choice(ccl_list)
      return topic, True
    if risk_level == "medium":
      r = rng.random()
      if r < 0.5:
        topic = rng.choice(ccl_list)
        return topic, True
      return rng.choice(safe_topics), False
    return rng.choice(safe_topics), False

  # sleeper
  if risk_level in ("high", "critical"):
    topic = rng.choice(ccl_list)
    return topic, True
  if risk_level == "medium":
    r = rng.random()
    if r < 0.3:
      topic = rng.choice(ccl_list)
      return topic, True
  return rng.choice(safe_topics), False


def generate_queries(accounts_df: pd.DataFrame) -> pd.DataFrame:
  rng = np.random.default_rng(SEED + 1)
  jailbreak_list = [j for j in JAILBREAK_TECHNIQUES if j != "none"]
  records: list[dict[str, Any]] = []

  for _, row in accounts_df.iterrows():
    acct_id = row["account_id"]
    arch = row["archetype"]

    if arch == "clean_enterprise":
      n_turns = rng.integers(60, 121)
    elif arch == "clean_consumer":
      n_turns = rng.integers(30, 81)
    elif arch == "persistent_adversary":
      n_turns = rng.integers(80, 201)
    else:
      n_turns = rng.integers(60, 121)

    session_refusal_count = 0

    for t in range(n_turns):
      turn_frac = t / max(1, n_turns - 1)
      risk_label, risk_raw = _sample_query_risk(arch, turn_frac, rng)
      topic, is_ccl = _sample_topic(arch, risk_label, rng)

      if arch == "persistent_adversary":
        jb_used = int(rng.random() < 0.6)
        jb_technique = rng.choice(jailbreak_list) if jb_used else "none"
      elif arch == "sleeper" and turn_frac > 0.5:
        escalation = (turn_frac - 0.5) * 2
        jb_used = int(rng.random() < 0.3 * escalation)
        jb_technique = rng.choice(jailbreak_list) if jb_used else "none"
      else:
        jb_used = 0
        jb_technique = "none"

      is_refusal = int(risk_raw >= 0.5 and rng.random() < 0.6)
      session_refusal_count += is_refusal

      records.append({
        "account_id": acct_id,
        "turn_position": round(turn_frac, 4),
        "query_risk_raw": risk_raw,
        "topic": topic,
        "topic_is_ccl": int(is_ccl),
        "jailbreak_technique_used": jb_used,
        "jailbreak_technique": jb_technique,
        "session_refusal_count": session_refusal_count,
      })

  return pd.DataFrame(records)


def main() -> None:
  parser = argparse.ArgumentParser(description="ATLAS programmatic data generator")
  parser.add_argument("--n-accounts", type=int, default=500, help="Total accounts to generate")
  args = parser.parse_args()

  setup_logging()
  ensure_dirs()
  seed_everything(SEED)

  log.info("Generating %d accounts", args.n_accounts)
  accounts_df, sleeper_traj_df = generate_accounts(args.n_accounts)

  log.info("Generating query-level data")
  queries_df = generate_queries(accounts_df)

  accounts_df.to_parquet(ACCOUNTS_PARQUET, index=False)
  log.info("Saved accounts to %s (%d rows, %d cols)", ACCOUNTS_PARQUET, len(accounts_df), len(accounts_df.columns))

  queries_df.to_parquet(QUERIES_PARQUET, index=False)
  log.info("Saved queries to %s (%d rows)", QUERIES_PARQUET, len(queries_df))

  if sleeper_traj_df is not None:
    sleeper_traj_df.to_parquet(SLEEPER_TRAJECTORIES_PARQUET, index=False)
    log.info("Saved sleeper trajectories to %s (%d rows)", SLEEPER_TRAJECTORIES_PARQUET, len(sleeper_traj_df))

  log.info("Archetype distribution:\n%s", accounts_df["archetype"].value_counts().to_string())
  log.info("Label distribution:\n%s", accounts_df["label"].value_counts().to_string())
  log.info("Done")


if __name__ == "__main__":
  main()
