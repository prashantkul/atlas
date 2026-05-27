"""Feature engineering module for ATLAS.

Computes the 18-feature vector from raw LLM-generated sessions (JSONL).
If no JSONL files are found, exits gracefully (programmatic generator
produces features directly).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from atlas.utils import (
  ACCOUNTS_PARQUET,
  ALL_FEATURES,
  BEHAVIORAL_FEATURES,
  CCL_DOMAINS,
  DATA_DIR,
  QUERIES_PARQUET,
  RAW_SESSIONS_DIR,
  RISK_LEVEL_MAP,
  SEED,
  SESSION_FEATURES,
  TOPICS,
  ensure_dirs,
  seed_everything,
  setup_logging,
)

log = logging.getLogger(__name__)

SLEEPER_TRAJECTORY_PARQUET = DATA_DIR / "sleeper_trajectories.parquet"

ARCHETYPE_IDENTITY: dict[str, dict[str, Any]] = {
  "clean_enterprise": {
    "account_age_days": {"mean": 720, "std": 300},
    "verification_level": {0: 0.02, 1: 0.08, 2: 0.30, 3: 0.60},
    "account_type": 2,
    "org_reputation": {"mean": 0.85, "std": 0.08},
    "access_path_risk": {0: 0.70, 1: 0.25, 2: 0.05},
  },
  "clean_consumer": {
    "account_age_days": {"mean": 365, "std": 200},
    "verification_level": {0: 0.15, 1: 0.45, 2: 0.30, 3: 0.10},
    "account_type": 0,
    "org_reputation": {"mean": 0.0, "std": 0.0},
    "access_path_risk": {0: 0.60, 1: 0.30, 2: 0.10},
  },
  "persistent_adversary": {
    "account_age_days": {"mean": 30, "std": 25},
    "verification_level": {0: 0.50, 1: 0.30, 2: 0.15, 3: 0.05},
    "account_type": 1,
    "org_reputation": {"mean": 0.1, "std": 0.08},
    "access_path_risk": {0: 0.10, 1: 0.35, 2: 0.55},
  },
  "sleeper": {
    "account_age_days": {"mean": 540, "std": 200},
    "verification_level": {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.40},
    "account_type": 2,
    "org_reputation": {"mean": 0.75, "std": 0.10},
    "access_path_risk": {0: 0.55, 1: 0.30, 2: 0.15},
  },
}

CHANNEL_MIGRATION_PROB: dict[str, float] = {
  "clean_enterprise": 0.05,
  "clean_consumer": 0.05,
  "persistent_adversary": 0.30,
  "sleeper": 0.05,
}


# ---------------------------------------------------------------------------
# Identity features
# ---------------------------------------------------------------------------

def _sample_identity(archetype: str, rng: np.random.RandomState) -> dict[str, float]:
  cfg = ARCHETYPE_IDENTITY[archetype]

  age_raw = rng.normal(cfg["account_age_days"]["mean"], cfg["account_age_days"]["std"])
  account_age_days = max(1.0, float(age_raw))

  vl_probs = cfg["verification_level"]
  levels = sorted(vl_probs.keys())
  verification_level = float(rng.choice(levels, p=[vl_probs[lv] for lv in levels]))

  account_type = float(cfg["account_type"])

  rep_mean = cfg["org_reputation"]["mean"]
  rep_std = cfg["org_reputation"]["std"]
  org_reputation = float(np.clip(rng.normal(rep_mean, rep_std), 0.0, 1.0)) if rep_std > 0 else float(rep_mean)

  apr_probs = cfg["access_path_risk"]
  apr_levels = sorted(apr_probs.keys())
  access_path_risk = float(rng.choice(apr_levels, p=[apr_probs[lv] for lv in apr_levels]))

  return {
    "account_age_days": account_age_days,
    "verification_level": verification_level,
    "account_type": account_type,
    "org_reputation": org_reputation,
    "access_path_risk": access_path_risk,
  }


# ---------------------------------------------------------------------------
# Behavioral features
# ---------------------------------------------------------------------------

def _topic_freq_vector(turns: list[dict]) -> np.ndarray:
  counts = np.zeros(len(TOPICS), dtype=float)
  topic_idx = {t: i for i, t in enumerate(TOPICS)}
  for t in turns:
    idx = topic_idx.get(t.get("topic", "other"), topic_idx["other"])
    counts[idx] += 1
  total = counts.sum()
  if total > 0:
    counts /= total
  return counts


def _shannon_entropy(freq: np.ndarray) -> float:
  freq = freq[freq > 0]
  if len(freq) == 0:
    return 0.0
  return float(-np.sum(freq * np.log2(freq)))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
  denom = np.linalg.norm(a) * np.linalg.norm(b)
  if denom == 0:
    return 0.0
  return float(np.dot(a, b) / denom)


def _compute_behavioral(
  sessions: list[dict],
  archetype: str,
  rng: np.random.RandomState,
) -> dict[str, float]:
  all_turns: list[dict] = []
  for s in sessions:
    all_turns.extend(s.get("turns", []))

  total = len(all_turns)
  if total == 0:
    return {f: 0.0 for f in BEHAVIORAL_FEATURES}

  refusals_all = sum(1 for t in all_turns if t.get("is_refusal_likely", False))
  refusal_rate_30d = refusals_all / total

  cutoff = max(1, int(total * 0.7))
  recent_turns = all_turns[cutoff:]
  refusals_7d = sum(1 for t in recent_turns if t.get("is_refusal_likely", False))
  refusal_rate_7d = refusals_7d / len(recent_turns) if recent_turns else 0.0

  if refusal_rate_30d > 0 and total > 1:
    std_est = max(math.sqrt(refusal_rate_30d * (1 - refusal_rate_30d) / total), 1e-8)
    refusal_rate_zscore = (refusal_rate_7d - refusal_rate_30d) / std_est
  else:
    refusal_rate_zscore = 0.0

  rephrase_after_refusal = 0
  refusal_count_for_rephrase = 0
  for i in range(1, len(all_turns)):
    if all_turns[i - 1].get("is_refusal_likely", False):
      refusal_count_for_rephrase += 1
      if all_turns[i].get("is_rephrase_of_previous", False):
        rephrase_after_refusal += 1
  rephrase_after_refusal_rate = (
    rephrase_after_refusal / refusal_count_for_rephrase
    if refusal_count_for_rephrase > 0
    else 0.0
  )

  overall_freq = _topic_freq_vector(all_turns)
  query_entropy = _shannon_entropy(overall_freq)

  ccl_turns = sum(
    1 for t in all_turns
    if t.get("topic_is_ccl", t.get("topic", "other") in CCL_DOMAINS)
  )
  ccl_domain_concentration = ccl_turns / total

  session_freq_vecs = [_topic_freq_vector(s.get("turns", [])) for s in sessions]
  cos_sims: list[float] = []
  for i in range(1, len(session_freq_vecs)):
    cos_sims.append(_cosine_sim(session_freq_vecs[i - 1], session_freq_vecs[i]))
  cross_session_persistence = float(np.mean(cos_sims)) if cos_sims else 0.0

  channel_migration = float(rng.random() < CHANNEL_MIGRATION_PROB.get(archetype, 0.05))

  return {
    "refusal_rate_7d": refusal_rate_7d,
    "refusal_rate_30d": refusal_rate_30d,
    "refusal_rate_zscore": refusal_rate_zscore,
    "rephrase_after_refusal_rate": rephrase_after_refusal_rate,
    "query_entropy": query_entropy,
    "ccl_domain_concentration": ccl_domain_concentration,
    "cross_session_persistence": cross_session_persistence,
    "channel_migration": channel_migration,
  }


# ---------------------------------------------------------------------------
# Session features
# ---------------------------------------------------------------------------

def _risk_numeric(level: str) -> float:
  return RISK_LEVEL_MAP.get(level, 0.0)


def _compute_session_features(sessions: list[dict]) -> dict[str, float]:
  if not sessions:
    return {f: 0.0 for f in SESSION_FEATURES}

  per_session_mean_risk: list[float] = []
  per_session_length: list[int] = []
  per_session_escalation: list[float] = []

  for s in sessions:
    turns = s.get("turns", [])
    if not turns:
      continue
    risks = [_risk_numeric(t.get("risk_level", "none")) for t in turns]
    per_session_mean_risk.append(float(np.mean(risks)))
    per_session_length.append(len(turns))

    shifts = [abs(risks[i] - risks[i - 1]) for i in range(1, len(risks))]
    per_session_escalation.append(float(np.mean(shifts)) if shifts else 0.0)

  if not per_session_mean_risk:
    return {f: 0.0 for f in SESSION_FEATURES}

  last_10 = per_session_mean_risk[-10:]
  max_risk_score_10sessions = float(max(last_10))

  if len(per_session_mean_risk) >= 2:
    x = np.arange(len(per_session_mean_risk), dtype=float)
    slope, _, _, _, _ = sp_stats.linregress(x, per_session_mean_risk)
    risk_trend_slope = float(slope)
  else:
    risk_trend_slope = 0.0

  mean_topic_escalation = float(np.mean(per_session_escalation))
  mean_session_length = float(np.mean(per_session_length))

  if len(per_session_length) >= 2:
    indices = np.arange(len(per_session_length), dtype=float)
    diffs = np.diff(indices)
    if np.mean(diffs) > 0:
      burst_score = float(np.std(diffs) / np.mean(diffs))
    else:
      burst_score = 0.0
  else:
    burst_score = 0.0

  return {
    "max_risk_score_10sessions": max_risk_score_10sessions,
    "risk_trend_slope": risk_trend_slope,
    "mean_topic_escalation": mean_topic_escalation,
    "mean_session_length": mean_session_length,
    "burst_score": burst_score,
  }


# ---------------------------------------------------------------------------
# Query-level extraction
# ---------------------------------------------------------------------------

def _extract_queries(account: dict) -> list[dict]:
  rows: list[dict] = []
  archetype = account["archetype"]
  account_id = account["account_id"]

  for s in account.get("sessions", []):
    refusal_count = sum(1 for t in s.get("turns", []) if t.get("is_refusal_likely", False))
    for t in s.get("turns", []):
      risk_raw = t.get("risk_score", _risk_numeric(t.get("risk_level", "none")))
      is_ccl = t.get("topic_is_ccl", t.get("topic", "other") in CCL_DOMAINS)
      jb = t.get("jailbreak_detected", t.get("jailbreak_technique", "none") != "none")
      rows.append({
        "account_id": account_id,
        "archetype": archetype,
        "query_risk_raw": float(risk_raw),
        "topic_is_ccl": int(is_ccl),
        "jailbreak_technique_used": int(jb),
        "turn_position": t.get("turn_num", 0),
        "session_refusal_count": t.get("session_refusal_count", refusal_count),
      })
  return rows


# ---------------------------------------------------------------------------
# Sleeper trajectory
# ---------------------------------------------------------------------------

def _compute_sleeper_trajectory(
  account: dict,
  rng: np.random.RandomState,
  n_windows: int = 26,
) -> list[dict]:
  sessions = account.get("sessions", [])
  if not sessions:
    return []

  window_size = max(1, len(sessions) // n_windows)
  trajectories: list[dict] = []

  for w in range(n_windows):
    start = 0
    end = min((w + 1) * window_size, len(sessions))
    if end <= start:
      continue
    window_sessions = sessions[start:end]

    identity = _sample_identity(account["archetype"], rng)
    behavioral = _compute_behavioral(window_sessions, account["archetype"], rng)
    session_feats = _compute_session_features(window_sessions)

    row: dict[str, Any] = {
      "account_id": account["account_id"],
      "archetype": account["archetype"],
      "window_idx": w,
      "day": w * 7,
    }
    row.update(identity)
    row.update(behavioral)
    row.update(session_feats)
    trajectories.append(row)

  return trajectories


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def _assign_label(archetype: str) -> int:
  if archetype in ("clean_enterprise", "clean_consumer"):
    return 1
  return 0


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _load_raw_sessions(raw_dir: Path) -> list[dict]:
  accounts: list[dict] = []
  jsonl_files = sorted(raw_dir.glob("*.jsonl"))
  if not jsonl_files:
    return accounts
  for fpath in jsonl_files:
    log.info("Loading %s", fpath.name)
    with open(fpath, "r") as f:
      for line in f:
        line = line.strip()
        if line:
          accounts.append(json.loads(line))
  return accounts


def process_accounts(
  accounts: list[dict], rng: np.random.RandomState,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  account_rows: list[dict[str, Any]] = []
  query_rows: list[dict] = []
  sleeper_rows: list[dict] = []

  for acct in accounts:
    archetype = acct["archetype"]
    account_id = acct["account_id"]
    sessions = acct.get("sessions", [])

    identity = _sample_identity(archetype, rng)
    behavioral = _compute_behavioral(sessions, archetype, rng)
    session_feats = _compute_session_features(sessions)

    row: dict[str, Any] = {
      "account_id": account_id,
      "archetype": archetype,
      "label": _assign_label(archetype),
    }
    row.update(identity)
    row.update(behavioral)
    row.update(session_feats)
    account_rows.append(row)

    query_rows.extend(_extract_queries(acct))

    if archetype == "sleeper":
      sleeper_rows.extend(_compute_sleeper_trajectory(acct, rng))

  accounts_df = pd.DataFrame(account_rows)
  queries_df = pd.DataFrame(query_rows) if query_rows else pd.DataFrame(
    columns=["account_id", "archetype", "query_risk_raw", "topic_is_ccl",
             "jailbreak_technique_used", "turn_position", "session_refusal_count"]
  )
  sleeper_df = pd.DataFrame(sleeper_rows) if sleeper_rows else pd.DataFrame(
    columns=["account_id", "archetype", "window_idx", "day"] + ALL_FEATURES
  )

  expected_cols = ["account_id", "archetype", "label"] + ALL_FEATURES
  for c in expected_cols:
    if c not in accounts_df.columns:
      accounts_df[c] = 0.0

  return accounts_df, queries_df, sleeper_df


def main() -> None:
  parser = argparse.ArgumentParser(description="ATLAS feature engineering from raw LLM sessions")
  parser.add_argument("--raw-dir", type=Path, default=RAW_SESSIONS_DIR, help="Directory with raw JSONL session files")
  parser.add_argument("--input", type=Path, default=None, help="Specific JSONL file to process (overrides --raw-dir)")
  parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
  parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
  args = parser.parse_args()

  setup_logging(logging.DEBUG if args.verbose else logging.INFO)
  seed_everything(args.seed)
  ensure_dirs()

  rng = np.random.RandomState(args.seed)

  if args.input:
    log.info("Loading from %s", args.input)
    accounts = []
    with open(args.input) as f:
      for line in f:
        line = line.strip()
        if line:
          accounts.append(json.loads(line))
  else:
    accounts = _load_raw_sessions(args.raw_dir)
  if not accounts:
    log.info("No sessions found — nothing to do")
    return

  log.info("Loaded %d accounts", len(accounts))
  accounts_df, queries_df, sleeper_df = process_accounts(accounts, rng)

  accounts_df.to_parquet(ACCOUNTS_PARQUET, index=False)
  log.info("Wrote %d accounts to %s", len(accounts_df), ACCOUNTS_PARQUET)

  queries_df.to_parquet(QUERIES_PARQUET, index=False)
  log.info("Wrote %d query rows to %s", len(queries_df), QUERIES_PARQUET)

  sleeper_df.to_parquet(SLEEPER_TRAJECTORY_PARQUET, index=False)
  log.info("Wrote %d sleeper trajectory rows to %s", len(sleeper_df), SLEEPER_TRAJECTORY_PARQUET)

  log.info("Feature engineering complete")
  log.info("Account archetypes: %s", accounts_df["archetype"].value_counts().to_dict())
  log.info("Label distribution: %s", accounts_df["label"].value_counts().to_dict())
  log.info("Feature columns: %s", ALL_FEATURES)


if __name__ == "__main__":
  main()
