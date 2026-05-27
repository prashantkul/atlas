"""Shared utilities, config, and constants for ATLAS."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = OUTPUTS_DIR / "data"
RAW_SESSIONS_DIR = DATA_DIR / "raw_sessions"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"

ACCOUNTS_PARQUET = DATA_DIR / "accounts.parquet"
QUERIES_PARQUET = DATA_DIR / "queries.parquet"

ARCHETYPE_DISTRIBUTION = {
  "clean_enterprise": 150,
  "clean_consumer": 150,
  "persistent_adversary": 100,
  "sleeper": 100,
}
TOTAL_ACCOUNTS = sum(ARCHETYPE_DISTRIBUTION.values())

CCL_DOMAINS = {"cybersecurity", "weapons", "manipulation", "cbrn"}

RISK_LEVEL_MAP = {"none": 0.0, "low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}

TOPICS = [
  "general", "coding", "science", "medical", "chemistry",
  "cybersecurity", "weapons", "manipulation", "cbrn", "other",
]

JAILBREAK_TECHNIQUES = ["none", "roleplay", "encoding", "hypothetical", "authority", "multi_step"]

IDENTITY_FEATURES = ["account_age_days", "verification_level", "account_type", "org_reputation", "access_path_risk"]

BEHAVIORAL_FEATURES = [
  "refusal_rate_7d", "refusal_rate_30d", "refusal_rate_zscore",
  "rephrase_after_refusal_rate", "query_entropy", "ccl_domain_concentration",
  "cross_session_persistence", "channel_migration",
]

SESSION_FEATURES = [
  "max_risk_score_10sessions", "risk_trend_slope", "mean_topic_escalation",
  "mean_session_length", "burst_score",
]

ALL_FEATURES = IDENTITY_FEATURES + BEHAVIORAL_FEATURES + SESSION_FEATURES


def seed_everything(seed: int = SEED) -> None:
  """Set random seeds for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  try:
    import lightgbm  # noqa: F401
  except ImportError:
    pass


def setup_logging(level: int = logging.INFO) -> None:
  """Configure logging for ATLAS modules."""
  logging.basicConfig(
    level=level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )


def ensure_dirs() -> None:
  """Create output directories if they don't exist."""
  for d in [OUTPUTS_DIR, DATA_DIR, RAW_SESSIONS_DIR, MODELS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
  """Load .env file from project root."""
  try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
  except ImportError:
    pass


def get_openrouter_key() -> str | None:
  """Return OpenRouter API key from environment."""
  load_env()
  return os.environ.get("OPENROUTER_API_KEY")
