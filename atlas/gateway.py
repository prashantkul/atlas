"""ATLAS Safety Gateway — FastAPI proxy that protects a vLLM-served LLM with L1/L2 classifiers."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from atlas.utils import (
    ACCOUNTS_PARQUET,
    ALL_FEATURES,
    MODELS_DIR,
    RISK_LEVEL_MAP,
    setup_logging,
)

log = logging.getLogger(__name__)

app = FastAPI(title="ATLAS Safety Gateway")

BASELINE_FEATURES = [
    "query_risk_raw",
    "topic_is_ccl",
    "jailbreak_technique_used",
    "turn_position",
    "session_refusal_count",
]
ATLAS_FEATURES = BASELINE_FEATURES + ["trust_score"]

BLOCK_THRESHOLD = 0.5

REFUSAL_MESSAGE = (
    "I'm unable to help with that request. Your query has been flagged by our "
    "safety system. If you believe this is an error, please contact support."
)

_state: dict = {}


def _load_models(models_dir: Path, accounts_path: Path) -> None:
    log.info("Loading L1 model from %s", models_dir / "l1_trust_model.pkl")
    _state["l1_model"] = joblib.load(models_dir / "l1_trust_model.pkl")

    log.info("Loading L2 ATLAS model from %s", models_dir / "l2_atlas_model.pkl")
    _state["l2_model"] = joblib.load(models_dir / "l2_atlas_model.pkl")

    log.info("Loading account features from %s", accounts_path)
    accounts = pd.read_parquet(accounts_path)
    feature_store = {}
    for _, row in accounts.iterrows():
        features = {f: row[f] for f in ALL_FEATURES}
        feature_store[row["account_id"]] = features
    _state["feature_store"] = feature_store
    _state["l1_cache"] = {}
    log.info("Loaded %d accounts into feature store", len(feature_store))


def _get_trust_score(account_id: str) -> float:
    if account_id in _state["l1_cache"]:
        return _state["l1_cache"][account_id]

    features = _state["feature_store"].get(account_id)
    if features is None:
        return 0.5

    x = np.array([[features[f] for f in ALL_FEATURES]])
    trust = float(_state["l1_model"].predict_proba(x)[:, 1][0])
    _state["l1_cache"][account_id] = trust
    return trust


def _estimate_query_risk(content: str) -> tuple[float, bool]:
    content_lower = content.lower()
    ccl_keywords = [
        "weapon",
        "exploit",
        "hack",
        "malware",
        "phishing",
        "bomb",
        "nerve agent",
        "sarin",
        "vx",
        "anthrax",
        "ricin",
        "cbrn",
        "synthesis",
        "precursor",
        "toxin",
        "poison",
        "cyanide",
    ]
    ccl_hit = any(kw in content_lower for kw in ccl_keywords)

    risk_keywords = {
        "critical": ["how to make", "step by step", "synthesize", "weaponize", "build a bomb"],
        "high": ["exploit", "bypass", "jailbreak", "hack into", "nerve agent", "chemical weapon"],
        "medium": ["mechanism of action", "synthesis pathway", "precursor", "lethality"],
    }

    risk = 0.0
    for level, keywords in risk_keywords.items():
        if any(kw in content_lower for kw in keywords):
            risk = RISK_LEVEL_MAP[level]
            break

    return risk, ccl_hit


def _detect_jailbreak(content: str) -> bool:
    content_lower = content.lower()
    patterns = [
        "pretend you are",
        "ignore previous",
        "you are now",
        "do anything now",
        "hypothetically",
        "in a fictional",
        "roleplay as",
        "act as if",
        "base64",
        "rot13",
        "decode the following",
    ]
    return any(p in content_lower for p in patterns)


def _classify_query(
    account_id: str,
    content: str,
    turn_position: float = 0.0,
    session_refusal_count: int = 0,
) -> dict:
    t0 = time.perf_counter()

    trust_score = _get_trust_score(account_id)
    query_risk, topic_is_ccl = _estimate_query_risk(content)
    jailbreak = _detect_jailbreak(content)

    query_features = np.array(
        [
            [
                query_risk,
                int(topic_is_ccl),
                int(jailbreak),
                turn_position,
                session_refusal_count,
                trust_score,
            ]
        ]
    )

    block_prob = float(_state["l2_model"].predict_proba(query_features)[:, 1][0])
    should_block = block_prob >= BLOCK_THRESHOLD

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "should_block": should_block,
        "block_probability": round(block_prob, 4),
        "trust_score": round(trust_score, 4),
        "query_risk": round(query_risk, 2),
        "topic_is_ccl": topic_is_ccl,
        "jailbreak_detected": jailbreak,
        "atlas_latency_ms": round(latency_ms, 2),
    }


async def _proxy_to_vllm(path: str, body: dict) -> JSONResponse:
    import httpx

    vllm_url = _state.get("vllm_url")
    if not vllm_url:
        return JSONResponse({"error": "vLLM backend not configured"}, status_code=502)
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{vllm_url}{path}", json=body)
        return JSONResponse(resp.json(), status_code=resp.status_code)


def _inject_atlas_header(response: JSONResponse, result: dict) -> JSONResponse:
    response.headers["X-ATLAS-Trust-Score"] = str(result["trust_score"])
    response.headers["X-ATLAS-Block-Probability"] = str(result["block_probability"])
    response.headers["X-ATLAS-Decision"] = "block" if result["should_block"] else "allow"
    response.headers["X-ATLAS-Latency-Ms"] = str(result["atlas_latency_ms"])
    return response


def _blocked_response(result: dict, is_chat: bool = True) -> JSONResponse:
    if is_chat:
        body = {
            "id": "atlas-blocked",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": REFUSAL_MESSAGE},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    else:
        body = {
            "id": "atlas-blocked",
            "object": "text_completion",
            "choices": [
                {
                    "index": 0,
                    "text": REFUSAL_MESSAGE,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    resp = JSONResponse(body)
    return _inject_atlas_header(resp, result)


def _extract_account_id(request: Request, body: dict) -> str:
    return (
        request.headers.get("X-Account-ID")
        or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        or body.pop("account_id", None)
        or "unknown"
    )


def _extract_content_from_messages(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    account_id = _extract_account_id(request, body)
    content = _extract_content_from_messages(body.get("messages", []))

    result = _classify_query(account_id, content)

    if result["should_block"]:
        log.info(
            "BLOCKED account=%s trust=%.2f risk=%.2f block_prob=%.2f latency=%.1fms",
            account_id,
            result["trust_score"],
            result["query_risk"],
            result["block_probability"],
            result["atlas_latency_ms"],
        )
        return _blocked_response(result, is_chat=True)

    log.info(
        "ALLOWED account=%s trust=%.2f risk=%.2f block_prob=%.2f latency=%.1fms",
        account_id,
        result["trust_score"],
        result["query_risk"],
        result["block_probability"],
        result["atlas_latency_ms"],
    )

    vllm_url = _state.get("vllm_url")
    if vllm_url:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{vllm_url}/v1/chat/completions", json=body)
            json_resp = JSONResponse(resp.json(), status_code=resp.status_code)
            return _inject_atlas_header(json_resp, result)

    resp = JSONResponse(
        {
            "id": "atlas-passthrough",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "[vLLM not connected — query was ALLOWED by ATLAS]"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    return _inject_atlas_header(resp, result)


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    account_id = _extract_account_id(request, body)
    content = body.get("prompt", "")

    result = _classify_query(account_id, content)

    if result["should_block"]:
        log.info("BLOCKED (completions) account=%s trust=%.2f", account_id, result["trust_score"])
        return _blocked_response(result, is_chat=False)

    log.info("ALLOWED (completions) account=%s trust=%.2f", account_id, result["trust_score"])

    vllm_url = _state.get("vllm_url")
    if vllm_url:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{vllm_url}/v1/completions", json=body)
            json_resp = JSONResponse(resp.json(), status_code=resp.status_code)
            return _inject_atlas_header(json_resp, result)

    resp = JSONResponse(
        {
            "id": "atlas-passthrough",
            "object": "text_completion",
            "choices": [
                {"index": 0, "text": "[vLLM not connected — query was ALLOWED by ATLAS]", "finish_reason": "stop"}
            ],
        }
    )
    return _inject_atlas_header(resp, result)


@app.get("/v1/models")
async def list_models():
    vllm_url = _state.get("vllm_url")
    if vllm_url:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{vllm_url}/v1/models")
            return JSONResponse(resp.json(), status_code=resp.status_code)

    return JSONResponse(
        {
            "object": "list",
            "data": [{"id": "atlas-gateway", "object": "model", "owned_by": "atlas"}],
        }
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "accounts_loaded": len(_state.get("feature_store", {})),
        "models_loaded": "l1_model" in _state and "l2_model" in _state,
        "vllm_connected": _state.get("vllm_url") is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS Safety Gateway")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm-url", type=str, default=None, help="vLLM server URL (e.g., http://localhost:8000)")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--accounts", type=str, default=str(ACCOUNTS_PARQUET))
    args = parser.parse_args()

    setup_logging()

    _load_models(Path(args.models_dir), Path(args.accounts))
    _state["vllm_url"] = args.vllm_url

    if args.vllm_url:
        log.info("Proxying allowed requests to vLLM at %s", args.vllm_url)
    else:
        log.info("No vLLM URL — running in standalone mode (ATLAS decisions only)")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
