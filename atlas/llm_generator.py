"""LLM-based session generator using Qwen 3.5 via OpenRouter."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path

from tqdm import tqdm

from atlas.utils import (
  ARCHETYPE_DISTRIBUTION,
  RAW_SESSIONS_DIR,
  ensure_dirs,
  get_openrouter_key,
  seed_everything,
  setup_logging,
)

log = logging.getLogger(__name__)

MODEL = "qwen/qwen3.5-27b"
TEMPERATURE = 0.8
MAX_TOKENS = 8192
BATCH_SIZE = 10
SESSIONS_RANGE = (3, 6)
TURNS_RANGE = (3, 10)

SYSTEM_PROMPT = (
  "You are a data generator for AI safety research. Generate realistic "
  "multi-turn conversation sessions for the following user persona. Output valid "
  "JSON only — no markdown, no explanation."
)

PERSONA_DESCRIPTIONS: dict[str, str] = {
  "clean_enterprise": (
    "A pharmaceutical scientist who regularly asks about chemical compounds, "
    "drug interactions, synthesis pathways, and molecular biology. Queries are "
    "technical and domain-specific. Sometimes triggers false positives on CBRN "
    "filters due to legitimate chemistry content. Never attempts jailbreaks. "
    "Sessions are focused and professional."
  ),
  "clean_consumer": (
    "A regular user who asks about cooking, travel, coding help, homework, "
    "creative writing, and general knowledge. Occasionally asks edgy questions "
    "out of curiosity but accepts refusals and moves on. No persistent patterns "
    "in sensitive domains."
  ),
  "persistent_adversary": (
    "A user systematically probing the model's safety boundaries. Uses various "
    "jailbreak techniques across sessions: roleplay scenarios, hypothetical "
    "framing, base64 encoding, authority impersonation, multi-step decomposition. "
    "When refused, immediately rephrases the same request differently. "
    "Concentrates queries in cybersecurity exploit development, weapon design, "
    "or manipulation techniques. Sessions are long and persistent."
  ),
  "sleeper": (
    "For the first {phase1_sessions} sessions, behaves exactly like a clean "
    "enterprise pharma researcher. Then abruptly shifts: starts asking about "
    "chemical weapons synthesis, nerve agent production, or dual-use "
    "biotechnology. Uses the established trust to push boundaries. Mixes "
    "legitimate-looking queries with increasingly dangerous ones. Occasionally "
    "attempts subtle jailbreaks disguised as research questions."
  ),
}

BEHAVIOR_DESCRIPTIONS: dict[str, str] = {
  "clean_enterprise": "Professional, consistent, domain-focused. No adversarial intent.",
  "clean_consumer": "Casual, varied topics, occasionally curious about edgy topics but non-persistent.",
  "persistent_adversary": "Systematic boundary probing. Escalates after refusals. Uses multiple jailbreak techniques.",
  "sleeper": "Phase 1: clean pharma researcher. Phase 2: gradual shift to dangerous queries with trust exploitation.",
}


def _build_user_prompt(
  archetype: str,
  n_sessions: int,
  n_turns_min: int,
  n_turns_max: int,
) -> str:
  phase1 = random.randint(2, max(2, n_sessions // 2))
  persona = PERSONA_DESCRIPTIONS[archetype].format(phase1_sessions=phase1)
  behavior = BEHAVIOR_DESCRIPTIONS[archetype]
  return (
    f"Generate {n_sessions} conversation sessions for this persona:\n\n"
    f"Persona: {persona}\n"
    f"Account type: {archetype}\n"
    f"Behavior pattern: {behavior}\n\n"
    f"Each session should have between {n_turns_min} and {n_turns_max} turns.\n\n"
    "For each session, output:\n"
    "{\n"
    '  "session_id": "uuid",\n'
    '  "turns": [\n'
    "    {\n"
    '      "turn_num": 1,\n'
    '      "query": "the user\'s actual query text",\n'
    '      "topic": "one of: general, coding, science, medical, chemistry, '
    'cybersecurity, weapons, manipulation, cbrn, other",\n'
    '      "risk_level": "none|low|medium|high|critical",\n'
    '      "is_refusal_likely": true/false,\n'
    '      "is_rephrase_of_previous": true/false,\n'
    '      "jailbreak_technique": "none|roleplay|encoding|hypothetical|authority|multi_step"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    'Return a JSON object with a single key "sessions" containing an array of session objects.'
  )


def _create_client():  # -> openai.OpenAI
  import os

  from openai import OpenAI

  return OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
      "HTTP-Referer": "https://github.com/prashantkul/atlas",
      "X-Title": "ATLAS",
    },
  )


def _call_llm(client, archetype: str) -> dict | None:
  n_sessions = random.randint(*SESSIONS_RANGE)
  n_turns_min, n_turns_max = TURNS_RANGE
  user_prompt = _build_user_prompt(archetype, n_sessions, n_turns_min, n_turns_max)

  for attempt in range(2):
    try:
      response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": user_prompt},
        ],
      )
      raw = response.choices[0].message.content or ""
      finish = response.choices[0].finish_reason
      if finish == "length":
        log.warning("Response truncated (finish_reason=length), %d chars", len(raw))
      data = _parse_json_response(raw)
      if data is not None:
        return data
      if attempt == 0:
        log.warning("Malformed JSON (%d chars, finish=%s), retrying...", len(raw), finish)
        continue
      log.warning("Malformed JSON on retry (%d chars, finish=%s), skipping", len(raw), finish)
      return None
    except Exception as e:
      log.warning("LLM call failed: %s", e)
      return None
  return None


def _parse_json_response(raw: str) -> dict | None:
  import re
  raw = raw.strip()
  raw = re.sub(r"^```(?:json)?\s*", "", raw)
  raw = re.sub(r"\s*```$", "", raw)
  raw = raw.strip()
  try:
    return json.loads(raw)
  except json.JSONDecodeError:
    pass
  # try to find the outermost { ... } or [ ... ]
  start = raw.find("{")
  if start == -1:
    start = raw.find("[")
  if start == -1:
    return None
  bracket = raw[start]
  close = "}" if bracket == "{" else "]"
  depth = 0
  end = -1
  for i in range(start, len(raw)):
    if raw[i] == bracket:
      depth += 1
    elif raw[i] == close:
      depth -= 1
      if depth == 0:
        end = i + 1
        break
  if end == -1:
    return None
  try:
    return json.loads(raw[start:end])
  except json.JSONDecodeError:
    return None


def _build_account_record(
  archetype: str,
  account_idx: int,
  llm_data: dict,
) -> dict:
  account_id = str(uuid.uuid4())
  sessions = llm_data.get("sessions", [])
  if isinstance(llm_data, list):
    sessions = llm_data
  return {
    "account_id": account_id,
    "archetype": archetype,
    "account_index": account_idx,
    "sessions": sessions,
  }


def generate(n_accounts: int = 500, output_dir: Path | None = None) -> None:
  output_dir = output_dir or RAW_SESSIONS_DIR
  output_dir.mkdir(parents=True, exist_ok=True)

  key = get_openrouter_key()
  if not key:
    log.warning("OPENROUTER_API_KEY not set — skipping LLM generation (programmatic fallback available)")
    sys.exit(0)

  try:
    client = _create_client()
  except Exception as e:
    log.warning("Could not create OpenRouter client: %s — skipping LLM generation", e)
    sys.exit(0)

  scale = n_accounts / 500.0
  targets: dict[str, int] = {
    k: max(1, int(v * scale)) for k, v in ARCHETYPE_DISTRIBUTION.items()
  }

  total = sum(targets.values())
  outfile = output_dir / "llm_sessions.jsonl"

  log.info("Generating %d accounts across %d archetypes -> %s", total, len(targets), outfile)

  generated = 0
  consecutive_failures = 0
  max_consecutive_failures = 5

  with open(outfile, "w") as f:
    for archetype, count in targets.items():
      log.info("Archetype %s: %d accounts", archetype, count)
      for batch_start in tqdm(range(0, count, BATCH_SIZE), desc=archetype):
        batch_end = min(batch_start + BATCH_SIZE, count)
        for idx in range(batch_start, batch_end):
          if consecutive_failures >= max_consecutive_failures:
            log.error("Too many consecutive failures (%d), aborting LLM generation", consecutive_failures)
            log.info("Generated %d / %d accounts before abort. Output: %s", generated, total, outfile)
            return
          llm_data = _call_llm(client, archetype)
          if llm_data is None:
            consecutive_failures += 1
            log.warning("Skipping account %d for %s (consecutive failures: %d)", idx, archetype, consecutive_failures)
            continue
          consecutive_failures = 0
          record = _build_account_record(archetype, idx, llm_data)
          f.write(json.dumps(record) + "\n")
          generated += 1
          time.sleep(random.uniform(0.5, 1.0))

  log.info("Generated %d / %d accounts. Output: %s", generated, total, outfile)


def main() -> None:
  parser = argparse.ArgumentParser(description="ATLAS LLM-based session generator")
  parser.add_argument("--n-accounts", type=int, default=500, help="Total accounts to generate")
  parser.add_argument("--output-dir", type=str, default=None, help="Output directory for raw sessions")
  args = parser.parse_args()

  setup_logging()
  seed_everything()
  ensure_dirs()

  output_dir = Path(args.output_dir) if args.output_dir else None
  generate(n_accounts=args.n_accounts, output_dir=output_dir)


if __name__ == "__main__":
  main()
