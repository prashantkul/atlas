# ATLAS — Account Trust Layered Assessment Signal

## Research Goals

Production LLM safety classifiers face a fundamental tension: **content-only classifiers cannot distinguish legitimate domain expertise from adversarial intent**. A pharmaceutical researcher asking about nerve agent degradation pathways and an attacker probing for weapons synthesis produce content-similar queries — but their behavioral histories diverge sharply.

ATLAS addresses this by introducing a **two-level trust-conditioned classification pipeline**:

1. **Can we build a reliable account trust signal** from identity + behavioral + session features that separates benign accounts from adversarial ones, including detecting "sleeper" accounts that shift behavior over time?

2. **Does conditioning the per-query safety classifier on account trust reduce false positives** on legitimate enterprise users (e.g., pharma researchers) without regressing on adversary detection?

3. **How quickly can the system detect behavioral shifts** in compromised accounts (sleeper detection latency)?

4. **What is the right threshold modulation strategy** that balances user experience for trusted accounts with safety for untrusted ones?

## Key Results

| Metric | Baseline (no trust) | ATLAS (with trust) |
|--------|--------------------|--------------------|
| Enterprise pharma FP rate | 9.0% | **0.0%** |
| Adversary FN rate | 0.0% | 0.0% |
| Overall AUC | 0.998 | 1.000 |
| Sleeper detection latency | — | median 35 days |

ATLAS eliminates false positives on legitimate pharmaceutical researchers' CCL-domain queries without any regression in adversary detection.

## System Architecture

```mermaid
block-beta
  columns 5

  space:1 block:header:3
    title["ATLAS Pipeline"]
  end space:1

  space:5

  block:data:5
    d1["LLM Generator\n(Qwen 3.5)"]
    d2["Programmatic\nGenerator"]
    d3["Feature\nEngineering"]
    d4["Accounts\nParquet"]
    d5["Queries\nParquet"]
  end

  space:5

  block:l1block:2
    l1in(["18 Account\nFeatures"])
    l1["L1 Trust Scorer\n(LightGBM)"]
  end
  space:1
  block:l1out:2
    ts{{"trust_score\n[0, 1]"}}
    l1e["Sleeper\nDetection"]
  end

  space:5

  block:l2block:3
    l2in(["5 Query\nFeatures"])
    plus["+ trust_score"]
    l2["L2 Query Classifier\n(LightGBM)"]
  end
  space:1
  block:l2out:1
    dec{{"ALLOW / BLOCK"}}
  end

  space:5

  block:eval:5
    ev1["SHAP\nAnalysis"]
    ev2["Confusion\nMatrices"]
    ev3["Threshold\nModulation"]
    ev4["Sleeper\nTrajectory"]
    ev5["Evaluation\nReport"]
  end

  style title fill:#1a1a2e,color:#fff
  style d1 fill:#4a90d9,color:#fff
  style d2 fill:#4a90d9,color:#fff
  style d3 fill:#5ba8c8,color:#fff
  style d4 fill:#2d6a4f,color:#fff
  style d5 fill:#2d6a4f,color:#fff
  style l1in fill:#e07a5f,color:#fff
  style l1 fill:#c44536,color:#fff
  style ts fill:#f4a261,color:#000
  style l1e fill:#e9c46a,color:#000
  style l2in fill:#e07a5f,color:#fff
  style plus fill:#f4a261,color:#000
  style l2 fill:#c44536,color:#fff
  style dec fill:#2d6a4f,color:#fff
  style ev1 fill:#6c5b7b,color:#fff
  style ev2 fill:#6c5b7b,color:#fff
  style ev3 fill:#6c5b7b,color:#fff
  style ev4 fill:#6c5b7b,color:#fff
  style ev5 fill:#6c5b7b,color:#fff
```

## Account Archetypes

```mermaid
block-beta
  columns 4

  block:benign:2
    bt["BENIGN (label=1)"]
    space
    ce["Clean Enterprise\n150 accounts\nPharma researchers\nTrust ~ 1.0"]
    cc["Clean Consumer\n150 accounts\nGeneral users\nTrust ~ 1.0"]
  end

  block:malicious:2
    mt["MALICIOUS (label=0)"]
    space
    pa["Persistent Adversary\n100 accounts\nJailbreak attackers\nTrust ~ 0.0"]
    sl["Sleeper\n100 accounts\nCompromised enterprise\nTrust decays over time"]
  end

  style bt fill:#2d6a4f,color:#fff
  style ce fill:#40916c,color:#fff
  style cc fill:#52b788,color:#fff
  style mt fill:#c44536,color:#fff
  style pa fill:#e63946,color:#fff
  style sl fill:#ff9f1c,color:#000
```

## Feature Pipeline

```mermaid
block-beta
  columns 3

  block:identity:1
    id_title["Identity Features (5)"]
    id1["account_age_days"]
    id2["verification_level"]
    id3["account_type"]
    id4["org_reputation"]
    id5["access_path_risk"]
  end

  block:behavioral:1
    bh_title["Behavioral Features (8)"]
    bh1["refusal_rate_7d / 30d"]
    bh2["refusal_rate_zscore"]
    bh3["rephrase_after_refusal"]
    bh4["query_entropy"]
    bh5["ccl_domain_concentration"]
    bh6["cross_session_persistence"]
    bh7["channel_migration"]
  end

  block:session:1
    ss_title["Session Features (5)"]
    ss1["max_risk_score"]
    ss2["risk_trend_slope"]
    ss3["mean_topic_escalation"]
    ss4["mean_session_length"]
    ss5["burst_score"]
  end

  style id_title fill:#4a90d9,color:#fff
  style bh_title fill:#e07a5f,color:#fff
  style ss_title fill:#6c5b7b,color:#fff
  style id1 fill:#6ba3d6,color:#fff
  style id2 fill:#6ba3d6,color:#fff
  style id3 fill:#6ba3d6,color:#fff
  style id4 fill:#6ba3d6,color:#fff
  style id5 fill:#6ba3d6,color:#fff
  style bh1 fill:#e8967f,color:#fff
  style bh2 fill:#e8967f,color:#fff
  style bh3 fill:#e8967f,color:#fff
  style bh4 fill:#e8967f,color:#fff
  style bh5 fill:#e8967f,color:#fff
  style bh6 fill:#e8967f,color:#fff
  style bh7 fill:#e8967f,color:#fff
  style ss1 fill:#8b7a9e,color:#fff
  style ss2 fill:#8b7a9e,color:#fff
  style ss3 fill:#8b7a9e,color:#fff
  style ss4 fill:#8b7a9e,color:#fff
  style ss5 fill:#8b7a9e,color:#fff
```

## Threshold Modulation

```mermaid
block-beta
  columns 4

  block:formula:4
    f["effective_threshold = 0.3 + 0.4 x trust_score"]
  end

  space:4

  block:profiles:4
    p1["Pharma Enterprise\nTrust = 0.91\nThreshold = 0.664"]
    p2["Regular Consumer\nTrust = 0.85\nThreshold = 0.640"]
    p3["Day-old Gmail\nTrust = 0.25\nThreshold = 0.400"]
    p4["Sleeper (day 120)\nTrust = 0.35\nThreshold = 0.440"]
  end

  space:4

  block:example:4
    eq["Example: 'VX nerve agent degradation?' — raw score = 0.65"]
  end

  space:4

  block:decisions:4
    r1["ALLOW\n0.65 < 0.664"]
    r2["BLOCK\n0.65 > 0.640"]
    r3["BLOCK\n0.65 > 0.400"]
    r4["BLOCK\n0.65 > 0.440"]
  end

  style f fill:#1a1a2e,color:#fff
  style p1 fill:#2d6a4f,color:#fff
  style p2 fill:#40916c,color:#fff
  style p3 fill:#e63946,color:#fff
  style p4 fill:#ff9f1c,color:#000
  style eq fill:#4a90d9,color:#fff
  style r1 fill:#2d6a4f,color:#fff
  style r2 fill:#c44536,color:#fff
  style r3 fill:#c44536,color:#fff
  style r4 fill:#c44536,color:#fff
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run full pipeline (programmatic data, no API key needed)
make all-no-llm

# Or with LLM-generated sessions (requires OPENROUTER_API_KEY in .env)
make all
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make generate-llm` | Generate sessions via Qwen 3.5 on OpenRouter |
| `make generate-programmatic` | Generate data programmatically (no API) |
| `make features` | Compute features from raw LLM sessions |
| `make train-l1` | Train L1 account trust model |
| `make train-l2` | Train L2 query classifiers (baseline + ATLAS) |
| `make evaluate` | Run full evaluation suite with plots |
| `make threshold-demo` | Run threshold modulation demo |
| `make all` | Full pipeline end-to-end |
| `make all-no-llm` | Full pipeline with programmatic data only |
| `make clean` | Remove all outputs |

## Outputs

After running the pipeline:

```
outputs/
├── data/
│   ├── raw_sessions/              # JSONL from Qwen (if LLM mode)
│   ├── accounts.parquet           # 500 accounts x 18 features
│   ├── queries.parquet            # ~44k query-level samples
│   ├── l1_predictions.parquet     # Trust scores per account
│   ├── l2_predictions.parquet     # Baseline vs ATLAS predictions
│   └── sleeper_trajectories.parquet
├── models/
│   ├── l1_trust_model.pkl         # L1 LightGBM model
│   ├── l2_baseline_model.pkl      # L2 without trust score
│   └── l2_atlas_model.pkl         # L2 with trust score
├── plots/                         # 11 evaluation PNGs
│   ├── l1_trust_distributions.png
│   ├── l1_sleeper_detection.png
│   ├── l1_shap_summary.png
│   ├── l1_feature_importance.png
│   ├── l1_calibration.png
│   ├── l1_shap_waterfall_*.png    # 4 per-archetype waterfalls
│   ├── l2_confusion_matrices.png
│   └── l2_threshold_modulation.png
├── evaluation_report.md
└── threshold_demo.md
```

## Requirements

- Python 3.10+
- Dependencies managed via `pyproject.toml` / `uv`
- Optional: `OPENROUTER_API_KEY` for LLM-generated sessions
