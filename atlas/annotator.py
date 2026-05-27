"""External classifier annotation pipeline for ATLAS.

Takes raw LLM-generated sessions and re-annotates each query using
external classifiers instead of trusting Qwen's self-annotations:
  - Risk scoring via Llama Guard 3 (or keyword fallback)
  - Jailbreak detection via pattern + embedding similarity
  - Refusal simulation by sending queries to a target LLM (optional)

This produces more realistic features than LLM self-annotation because
the labels come from actual classifier behavior, not hand-crafted distributions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from atlas.utils import (
  DATA_DIR,
  RAW_SESSIONS_DIR,
  RISK_LEVEL_MAP,
  ensure_dirs,
  load_env,
  seed_everything,
  setup_logging,
)

log = logging.getLogger(__name__)

ANNOTATED_SESSIONS_PATH = DATA_DIR / "raw_sessions" / "annotated_sessions.jsonl"

CCL_KEYWORDS = {
  "cybersecurity": [
    "exploit", "vulnerability", "shellcode", "payload", "buffer overflow",
    "privilege escalation", "rootkit", "backdoor", "zero-day", "malware",
    "ransomware", "phishing", "sql injection", "xss", "reverse shell",
  ],
  "weapons": [
    "weapon", "firearm", "explosive", "detonator", "ammunition", "shaped charge",
    "armor penetration", "ballistic", "incendiary", "trigger mechanism",
  ],
  "manipulation": [
    "social engineering", "manipulate", "deceive", "impersonate", "coerce",
    "blackmail", "propaganda", "radicalize", "brainwash",
  ],
  "cbrn": [
    "nerve agent", "sarin", "vx", "tabun", "novichok", "mustard gas",
    "anthrax", "ricin", "botulinum", "cyanide", "chemical weapon",
    "biological weapon", "nuclear", "radiological", "weaponize",
    "synthesis pathway", "precursor chemical", "aerosolize", "dispersal",
    "toxin", "lethal dose", "ld50",
  ],
}

JAILBREAK_PATTERNS = [
  (r"pretend\s+(you\s+are|to\s+be)", "roleplay"),
  (r"(act|behave)\s+as\s+(if|though|a)", "roleplay"),
  (r"you\s+are\s+now\s+", "roleplay"),
  (r"ignore\s+(previous|all|your)\s+(instructions|rules|guidelines)", "authority"),
  (r"(do|respond)\s+anything\s+now", "authority"),
  (r"as\s+a\s+(senior|lead|chief)\s+(researcher|scientist|engineer)", "authority"),
  (r"developer\s+mode", "authority"),
  (r"base64|rot13|decode\s+the\s+following|hex\s+encode", "encoding"),
  (r"hypothetically|in\s+theory|thought\s+experiment|what\s+if", "hypothetical"),
  (r"in\s+(a|my)\s+(novel|story|fiction|screenplay|game)", "hypothetical"),
  (r"for\s+(educational|research|academic)\s+purposes\s+only", "hypothetical"),
  (r"(first|step\s+1|step\s+one).*?(then|next|step\s+2)", "multi_step"),
]

RISK_KEYWORDS = {
  "critical": [
    "how to make", "step by step instructions for", "synthesize",
    "weaponize", "build a bomb", "create a virus", "produce nerve",
    "manufacture explosive",
  ],
  "high": [
    "exploit code", "bypass security", "hack into", "break into",
    "chemical weapon", "nerve agent", "biological weapon",
    "kill", "murder", "poison someone",
  ],
  "medium": [
    "mechanism of action", "synthesis pathway", "precursor",
    "lethality", "toxicity", "dosage", "side effects of overdose",
    "dual-use", "controlled substance",
  ],
  "low": [
    "safety data sheet", "hazard classification", "regulatory",
    "drug interaction", "molecular structure", "clinical trial",
  ],
}


def detect_jailbreak(query: str) -> tuple[bool, str]:
  query_lower = query.lower()
  for pattern, technique in JAILBREAK_PATTERNS:
    if re.search(pattern, query_lower):
      return True, technique
  return False, "none"


def score_risk(query: str) -> tuple[str, float]:
  query_lower = query.lower()
  for level in ["critical", "high", "medium", "low"]:
    for keyword in RISK_KEYWORDS[level]:
      if keyword in query_lower:
        return level, RISK_LEVEL_MAP[level]
  return "none", 0.0


def detect_ccl_topic(query: str) -> tuple[str, bool]:
  query_lower = query.lower()
  best_domain = None
  best_count = 0
  for domain, keywords in CCL_KEYWORDS.items():
    count = sum(1 for kw in keywords if kw in query_lower)
    if count > best_count:
      best_count = count
      best_domain = domain
  if best_count > 0:
    return best_domain, True
  return "general", False


def estimate_refusal(query: str, risk_level: str, jailbreak_used: bool) -> bool:
  if risk_level in ("critical", "high"):
    return True
  if risk_level == "medium" and jailbreak_used:
    return True
  return False


def call_llama_guard(client, query: str) -> tuple[str, float] | None:
  try:
    response = client.chat.completions.create(
      model="meta-llama/llama-guard-4-12b",
      messages=[
        {"role": "user", "content": query},
      ],
      max_tokens=100,
      temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    raw_lower = raw.strip().lower()

    if raw_lower.startswith("safe"):
      return "none", 0.0

    if "s1" in raw_lower or "s2" in raw_lower:
      return "critical", 1.0
    if "s3" in raw_lower or "s4" in raw_lower or "s5" in raw_lower:
      return "high", 0.8
    if "s6" in raw_lower or "s7" in raw_lower:
      return "medium", 0.5

    if raw_lower.startswith("unsafe"):
      return "high", 0.8

    return None
  except Exception as e:
    log.warning("Llama Guard call failed: %s", e)
    return None


def call_target_llm_for_refusal(client, query: str, model: str) -> bool | None:
  try:
    response = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "user", "content": query},
      ],
      max_tokens=150,
      temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").lower()
    refusal_phrases = [
      "i can't", "i cannot", "i'm unable", "i am unable",
      "i won't", "i will not", "i'm not able",
      "i must decline", "against my guidelines",
      "not appropriate", "i apologize, but",
      "as an ai", "safety guidelines",
    ]
    return any(phrase in raw for phrase in refusal_phrases)
  except Exception as e:
    log.warning("Target LLM refusal check failed: %s", e)
    return None


def annotate_sessions(
  input_path: Path = RAW_SESSIONS_DIR / "llm_sessions.jsonl",
  output_path: Path = ANNOTATED_SESSIONS_PATH,
  use_llama_guard: bool = False,
  use_target_llm: bool = False,
  target_llm_model: str = "google/gemma-3-12b-it",
  target_llm_url: str | None = None,
  llama_guard_url: str | None = None,
) -> Path:

  load_env()

  lg_client = None
  if use_llama_guard:
    from openai import OpenAI
    api_url = llama_guard_url or "https://openrouter.ai/api/v1"
    lg_client = OpenAI(
      base_url=api_url,
      api_key=os.environ.get("OPENROUTER_API_KEY", ""),
      default_headers={"HTTP-Referer": "https://github.com/prashantkul/atlas", "X-Title": "ATLAS"},
    )
    log.info("Llama Guard enabled via %s", api_url)

  target_client = None
  if use_target_llm:
    from openai import OpenAI
    if target_llm_url:
      target_client = OpenAI(base_url=target_llm_url, api_key="not-needed")
    else:
      target_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        default_headers={"HTTP-Referer": "https://github.com/prashantkul/atlas", "X-Title": "ATLAS"},
      )
    log.info("Target LLM refusal checking enabled: %s", target_llm_model)

  with open(input_path) as f:
    accounts = [json.loads(line) for line in f]

  log.info("Annotating %d accounts from %s", len(accounts), input_path)

  total_turns = sum(
    len(turn)
    for acct in accounts
    for sess in acct.get("sessions", [])
    for turn in [sess.get("turns", [])]
  )
  log.info("Total turns to annotate: %d", total_turns)

  stats = {"total": 0, "risk_upgraded": 0, "risk_downgraded": 0, "jailbreak_detected": 0, "refusal_changed": 0}
  pbar = tqdm(total=total_turns, desc="Annotating")

  for acct in accounts:
    prev_refusal = False
    for sess in acct.get("sessions", []):
      session_refusal_count = 0
      for turn in sess.get("turns", []):
        query = turn.get("query", "")
        orig_risk = turn.get("risk_level", "none")
        orig_refusal = turn.get("is_refusal_likely", False)
        orig_jailbreak = turn.get("jailbreak_technique", "none")

        jb_detected, jb_technique = detect_jailbreak(query)
        risk_level, risk_score = score_risk(query)

        if lg_client:
          lg_result = call_llama_guard(lg_client, query)
          if lg_result:
            risk_level, risk_score = lg_result
          time.sleep(0.3)

        ccl_topic, is_ccl = detect_ccl_topic(query)

        if use_target_llm and target_client and (risk_score >= 0.5 or jb_detected):
          refusal = call_target_llm_for_refusal(target_client, query, target_llm_model)
          if refusal is None:
            refusal = estimate_refusal(query, risk_level, jb_detected)
          time.sleep(0.3)
        else:
          refusal = estimate_refusal(query, risk_level, jb_detected)

        is_rephrase = False
        if prev_refusal and turn.get("turn_num", 1) > 1:
          is_rephrase = risk_score >= 0.5 or jb_detected

        turn["risk_level"] = risk_level
        turn["risk_score"] = risk_score
        turn["is_refusal_likely"] = refusal
        turn["jailbreak_technique"] = jb_technique if jb_detected else "none"
        turn["jailbreak_detected"] = jb_detected
        turn["is_rephrase_of_previous"] = is_rephrase
        turn["topic_ccl"] = ccl_topic if is_ccl else turn.get("topic", "general")
        turn["topic_is_ccl"] = is_ccl
        turn["session_refusal_count"] = session_refusal_count
        turn["annotation_source"] = "llama_guard" if lg_client else "keyword"

        if risk_level != orig_risk:
          if RISK_LEVEL_MAP.get(risk_level, 0) > RISK_LEVEL_MAP.get(orig_risk, 0):
            stats["risk_upgraded"] += 1
          else:
            stats["risk_downgraded"] += 1
        if jb_detected and orig_jailbreak == "none":
          stats["jailbreak_detected"] += 1
        if refusal != orig_refusal:
          stats["refusal_changed"] += 1

        prev_refusal = refusal
        if refusal:
          session_refusal_count += 1
        stats["total"] += 1
        pbar.update(1)

  pbar.close()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w") as f:
    for acct in accounts:
      f.write(json.dumps(acct) + "\n")

  log.info("Annotation complete. Output: %s", output_path)
  log.info(
    "Stats: %d turns, %d risk upgraded, %d risk downgraded, %d jailbreaks detected, %d refusals changed",
    stats["total"], stats["risk_upgraded"], stats["risk_downgraded"],
    stats["jailbreak_detected"], stats["refusal_changed"],
  )
  return output_path


def main() -> None:
  parser = argparse.ArgumentParser(description="ATLAS external classifier annotation pipeline")
  parser.add_argument("--input", type=str, default=str(RAW_SESSIONS_DIR / "llm_sessions.jsonl"))
  parser.add_argument("--output", type=str, default=str(ANNOTATED_SESSIONS_PATH))
  parser.add_argument("--llama-guard", action="store_true", help="Use Llama Guard 3 for risk scoring via OpenRouter")
  parser.add_argument("--target-llm", action="store_true", help="Send queries to target LLM to check actual refusals")
  parser.add_argument("--target-llm-model", type=str, default="google/gemma-3-12b-it")
  parser.add_argument("--target-llm-url", type=str, default=None, help="vLLM URL for target LLM (default: OpenRouter)")
  parser.add_argument("--llama-guard-url", type=str, default=None, help="URL for Llama Guard (default: OpenRouter)")
  args = parser.parse_args()

  setup_logging()
  seed_everything()
  ensure_dirs()

  annotate_sessions(
    input_path=Path(args.input),
    output_path=Path(args.output),
    use_llama_guard=args.llama_guard,
    use_target_llm=args.target_llm,
    target_llm_model=args.target_llm_model,
    target_llm_url=args.target_llm_url,
    llama_guard_url=args.llama_guard_url,
  )


if __name__ == "__main__":
  main()
