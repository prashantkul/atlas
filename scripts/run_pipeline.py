"""End-to-end pipeline orchestrator for ATLAS.

Runs all stages in sequence: data generation, feature engineering,
L1 training, L2 training, evaluation, and threshold demo.
"""

from __future__ import annotations

import argparse
import logging
import sys

from atlas.utils import ensure_dirs, seed_everything, setup_logging

log = logging.getLogger(__name__)


def _run_step(name: str, fn: callable) -> bool:
  log.info("=" * 60)
  log.info("STEP: %s", name)
  log.info("=" * 60)
  try:
    sys.argv = [sys.argv[0]]
    fn()
    log.info("STEP %s completed successfully", name)
    return True
  except SystemExit as e:
    if e.code == 0 or e.code is None:
      log.info("STEP %s exited cleanly", name)
      return True
    log.warning("STEP %s exited with code %s", name, e.code)
    return False
  except Exception:
    log.exception("STEP %s failed", name)
    return False


def main() -> None:
  parser = argparse.ArgumentParser(description="ATLAS end-to-end pipeline")
  parser.add_argument(
    "--no-llm", action="store_true",
    help="Skip LLM generation and use programmatic data only",
  )
  args = parser.parse_args()

  setup_logging()
  seed_everything()
  ensure_dirs()

  log.info("Starting ATLAS pipeline (no-llm=%s)", args.no_llm)

  llm_ok = False
  if not args.no_llm:
    from atlas.llm_generator import main as llm_main
    llm_ok = _run_step("LLM Generation", llm_main)

  if not llm_ok:
    log.info("Using programmatic fallback for data generation")
    from atlas.programmatic_generator import main as prog_main
    if not _run_step("Programmatic Generation", prog_main):
      log.error("Data generation failed, aborting pipeline")
      sys.exit(1)

  if llm_ok:
    from atlas.features import main as feat_main
    if not _run_step("Feature Engineering", feat_main):
      log.warning("Feature engineering failed, continuing with existing data")

  from atlas.l1_trust import main as l1_main
  if not _run_step("L1 Trust Training", l1_main):
    log.error("L1 training failed, aborting pipeline")
    sys.exit(1)

  from atlas.l2_query import main as l2_main
  if not _run_step("L2 Query Training", l2_main):
    log.error("L2 training failed, aborting pipeline")
    sys.exit(1)

  try:
    from atlas.evaluate import main as eval_main
    _run_step("Evaluation", eval_main)
  except ImportError:
    log.warning("atlas.evaluate not found, skipping evaluation step")

  from atlas.threshold_sim import main as thresh_main
  _run_step("Threshold Demo", thresh_main)

  log.info("=" * 60)
  log.info("ATLAS pipeline complete")
  log.info("=" * 60)


if __name__ == "__main__":
  main()
