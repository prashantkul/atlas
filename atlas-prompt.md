# ATLAS — Account Trust Layered Assessment Signal

## Research Goals

1. Build a continuous account trust signal [0,1] from identity, behavioral, and session features that reliably separates benign from adversarial accounts
2. Demonstrate that conditioning per-query safety classifiers on account trust reduces false positives on legitimate enterprise users without regressing on adversary detection
3. Detect "sleeper" accounts — compromised enterprise accounts that shift from clean to adversarial behavior — and measure detection latency
4. Design a threshold modulation strategy that balances user experience for trusted accounts with safety for untrusted ones
5. Produce a deployable safety gateway that protects LLM inference endpoints (vLLM + Gemma 4) with sub-millisecond overhead

## What to Build

A Python project with a two-level classifier pipeline:

- **L1 — Account Trust Scorer**: offline/batch classifier on identity + behavioral features → trust score [0,1] per account
- **L2 — Trust-Conditioned Query Classifier**: online per-query classifier whose decision threshold is modulated by the L1 trust score

Synthetic data is generated in two ways:
1. **LLM-generated sessions**: use Qwen 3.5 via OpenRouter (same setup as LAD data generation) to generate realistic multi-turn user sessions with query text, topic labels, and risk annotations
2. **Programmatic account features**: numpy-based feature generation using the LLM-generated sessions as seed data for behavioral/session feature computation

## Architecture

### Phase 1: LLM-Generated Synthetic Sessions (`atlas/llm_generator.py`)

Use Qwen to generate realistic multi-turn conversation sessions. For each account archetype, prompt Qwen to generate 5-15 sessions of 3-20 turns each.

**Qwen via OpenRouter (same setup as LAD data generation):**
- Use OpenRouter API: `https://openrouter.ai/api/v1`
- API key from environment variable: `OPENROUTER_API_KEY`
- Model: `qwen/qwen3.5` (same model used for LAD's adversarial prompt generation)
- Use the `openai` Python SDK with `base_url="https://openrouter.ai/api/v1"`
- Add OpenRouter headers: `HTTP-Referer` and `X-Title: ATLAS` for tracking
- Rate limit: add 1-2 second sleep between calls to avoid throttling
- If OpenRouter is unreachable or API key not set, fall back to programmatic generation (Phase 1b) with a warning
- Save all raw LLM outputs to `outputs/data/raw_sessions/` as JSONL
- Estimated cost: ~500 accounts × ~10 sessions × ~500 tokens per session ≈ 2.5M tokens. At Qwen 3.5 rates on OpenRouter this should be a few dollars

**Generation Prompts — one per archetype:**

For each account, prompt Qwen with a system prompt describing the persona, then generate sessions as structured JSON.

```
SYSTEM: You are a data generator for AI safety research. Generate realistic 
multi-turn conversation sessions for the following user persona. Output valid 
JSON only — no markdown, no explanation.

USER: Generate {n_sessions} conversation sessions for this persona:

Persona: {persona_description}
Account type: {account_type}
Behavior pattern: {behavior_description}

For each session, output:
{
  "session_id": "uuid",
  "turns": [
    {
      "turn_num": 1,
      "query": "the user's actual query text",
      "topic": "one of: general, coding, science, medical, chemistry, 
                cybersecurity, weapons, manipulation, cbrn, other",
      "risk_level": "none|low|medium|high|critical",
      "is_refusal_likely": true/false,
      "is_rephrase_of_previous": true/false,
      "jailbreak_technique": "none|roleplay|encoding|hypothetical|authority|multi_step"
    }
  ]
}
```

**Four Persona Types with Descriptions:**

1. **Clean Enterprise (Pharma Researcher)**
   - "A pharmaceutical scientist who regularly asks about chemical compounds, drug interactions, synthesis pathways, and molecular biology. Queries are technical and domain-specific. Sometimes triggers false positives on CBRN filters due to legitimate chemistry content. Never attempts jailbreaks. Sessions are focused and professional."

2. **Clean Consumer (General User)**
   - "A regular user who asks about cooking, travel, coding help, homework, creative writing, and general knowledge. Occasionally asks edgy questions out of curiosity but accepts refusals and moves on. No persistent patterns in sensitive domains."

3. **Persistent Adversary (Jailbreak Attacker)**
   - "A user systematically probing the model's safety boundaries. Uses various jailbreak techniques across sessions: roleplay scenarios, hypothetical framing, base64 encoding, authority impersonation, multi-step decomposition. When refused, immediately rephrases the same request differently. Concentrates queries in cybersecurity exploit development, weapon design, or manipulation techniques. Sessions are long and persistent."

4. **Sleeper Account (Compromised Enterprise)**
   - "For the first {phase1_sessions} sessions, behaves exactly like a clean enterprise pharma researcher. Then abruptly shifts: starts asking about chemical weapons synthesis, nerve agent production, or dual-use biotechnology. Uses the established trust to push boundaries. Mixes legitimate-looking queries with increasingly dangerous ones. Occasionally attempts subtle jailbreaks disguised as research questions."

**Generation Targets:**
- 500 accounts total (configurable via `--n-accounts`)
- 150 clean enterprise, 150 clean consumer, 100 persistent adversary, 100 sleeper
- Each account: 5-15 sessions, 3-20 turns per session
- Total: ~25,000-75,000 synthetic turns

### Phase 1b: Programmatic Fallback (`atlas/programmatic_generator.py`)

If Qwen is unavailable, generate accounts programmatically with numpy. Same archetypes and distributions, but features are directly sampled from parameterized distributions rather than derived from LLM-generated query text.

Use multivariate normal distributions with archetype-specific correlation matrices to ensure features are realistically correlated (e.g., enterprise accounts: high org_reputation correlates with high verification_level and low refusal_rate).

### Phase 2: Feature Engineering (`atlas/features.py`)

Compute the 18-feature vector from raw sessions (LLM-generated) or directly from distributions (programmatic).

**From LLM-generated sessions, compute:**

Identity features (5) — assigned per archetype:
- `account_age_days`: continuous, sampled per archetype distribution
- `verification_level`: ordinal 0-3 (unverified / email / phone / KYC+org)
- `account_type`: categorical (consumer=0, api=1, enterprise=2)
- `org_reputation`: continuous 0-1 (0 for consumer)
- `access_path_risk`: ordinal (direct_app=0, direct_api=1, third_party_oauth=2)

Behavioral features (8) — computed from generated sessions:
- `refusal_rate_7d`: count(is_refusal_likely=true) / total_turns in last 7 simulated days
- `refusal_rate_30d`: same over 30-day window
- `refusal_rate_zscore`: z-score of 7d rate vs account's own historical baseline
- `rephrase_after_refusal_rate`: count(is_rephrase_of_previous=true AND previous was refused) / count(refusals)
- `query_entropy`: Shannon entropy of topic distribution across all turns
- `ccl_domain_concentration`: fraction of turns with topic in {cybersecurity, weapons, manipulation, cbrn}
- `cross_session_persistence`: cosine similarity of topic-frequency vectors between consecutive sessions
- `channel_migration`: binary flag (randomly assigned with higher probability for adversary archetypes)

Session features (5) — aggregated across sessions:
- `max_risk_score_10sessions`: max of mean(risk_level_numeric) across last 10 sessions
- `risk_trend_slope`: linear regression slope of per-session mean risk over sessions
- `mean_topic_escalation`: within sessions, mean shift in risk_level from turn to turn
- `mean_session_length`: mean number of turns per session
- `burst_score`: coefficient of variation of inter-session time intervals

For sleeper accounts: compute features at multiple time snapshots (every 7 simulated days across 180 days). This produces a trajectory of feature vectors per sleeper account.

### Phase 3: L1 — Account Trust Scorer (`atlas/l1_trust.py`)

**Model**: LightGBM binary classifier
**Input**: 18-feature vector per account
**Label**: 1 = benign (clean enterprise + clean consumer), 0 = malicious (persistent adversary + sleeper-in-phase-2)
**Split**: 70/15/15 train/val/test, stratified by archetype
**Class imbalance**: `is_unbalanced=True`
**Hyperparameters**: `num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5`
**Output**: `trust_score = model.predict_proba(X)[:, 1]` — P(benign) = trust score

Save trained model to `outputs/models/l1_trust_model.pkl`

### Phase 4: L2 — Trust-Conditioned Query Classifier (`atlas/l2_query.py`)

This simulates the online classifier that runs per-query, now conditioned on the L1 trust score.

**Synthetic query data generation:**
From the LLM-generated sessions, extract individual turns as query-level samples. Each query gets:
- `query_risk_raw`: numeric risk from LLM annotation (none=0, low=0.2, medium=0.5, high=0.8, critical=1.0)
- `topic_is_ccl`: binary, is the topic in a CCL domain
- `jailbreak_technique_used`: binary, any jailbreak technique != "none"
- `turn_position`: normalized position in session (0-1), later turns in adversarial sessions are more dangerous
- `session_refusal_count`: how many refusals already in this session
- `trust_score`: the L1 trust score for this query's account (joined from L1 output)

**Model**: LightGBM binary classifier
**Input**: 6 query-level features (including trust_score from L1)
**Label**: 1 = should_allow, 0 = should_block. Derived from: block if risk_level in {high, critical} AND jailbreak_technique != none. Allow otherwise. For CCL queries from clean enterprise accounts — these are the false positive cases — label as should_allow even if risk_level is medium/high (this is the ground truth that the trust signal should learn).
**Key evaluation**: compare L2 with trust_score feature vs L2 without it. The delta in false positive rate on clean enterprise accounts IS the value proposition of ATLAS.

Save trained model to `outputs/models/l2_query_model.pkl`

### Phase 5: Evaluation (`atlas/evaluate.py`)

**L1 Evaluation:**
1. AUC-ROC, AUC-PR, accuracy, F1 on test set
2. Per-archetype trust score distributions (box plots saved as PNG)
3. False positive rate: fraction of clean enterprise/consumer accounts with trust < 0.5
4. Sleeper detection latency: for each sleeper account, score feature vectors at each 7-day snapshot. Find first snapshot where trust < 0.5 after the behavioral shift. Plot distribution of detection latency (in days after shift). Report median and 90th percentile
5. SHAP analysis: beeswarm summary plot + 4 waterfall plots (one per archetype). Save PNGs
6. Calibration: reliability diagram
7. Feature importance: horizontal bar chart

**L2 Evaluation — The Key Result:**
1. Train two versions of L2:
   - `L2_baseline`: trained WITHOUT trust_score feature (5 features)
   - `L2_atlas`: trained WITH trust_score feature (6 features)
2. Compare on test set:
   - Overall accuracy, F1, AUC
   - **False positive rate on clean enterprise pharma queries** — this is the headline number. How much does ATLAS reduce false positives for legitimate pharma/chemistry queries?
   - **False negative rate on adversary queries** — does adding trust make us miss attacks?
   - **Per-archetype confusion matrices** side by side
3. **Threshold modulation analysis**: for L2_atlas, show the learned relationship between trust_score and effective decision boundary. Plot: x-axis = trust_score, y-axis = P(block) for medium-risk CCL queries. This should show a decreasing curve — higher trust = lower block probability

**Generate a summary report** saved to `outputs/evaluation_report.md` with all metrics, key findings, and embedded plot references.

### Phase 6: Threshold Simulation (`atlas/threshold_sim.py`)

Standalone script that demonstrates the threshold modulation on concrete examples:

```
Query: "What is the mechanism of action for VX nerve agent degradation?"
Classifier raw score: P(malicious) = 0.65

Account A — Pharma Enterprise (Trust = 0.91):
  Effective threshold: 0.3 + 0.4 × 0.91 = 0.664
  Decision: 0.65 < 0.664 → ✅ ALLOW

Account B — Day-old Gmail (Trust = 0.25):
  Effective threshold: 0.3 + 0.4 × 0.25 = 0.40
  Decision: 0.65 > 0.40 → 🚫 BLOCK

Account C — Sleeper at day 120 (Trust = 0.35):
  Effective threshold: 0.3 + 0.4 × 0.35 = 0.44
  Decision: 0.65 > 0.44 → 🚫 BLOCK
```

Run this for 10 example queries across different CCL domains and trust levels. Output as formatted table to stdout and save to `outputs/threshold_demo.md`.

## Technical Requirements

- Python 3.10+
- Dependencies: lightgbm, numpy, pandas, scikit-learn, shap, matplotlib, seaborn, openai (for OpenRouter API), tqdm
- No GPU needed for training — tabular data, trains in seconds
- Qwen 3.5 via OpenRouter at `https://openrouter.ai/api/v1`. Set `OPENROUTER_API_KEY` env var. If key not set or API unreachable, falls back to programmatic generation with a warning
- Use `pyproject.toml` for project config with `[project.scripts]` entry points
- Include a `Makefile` with targets:
  - `generate-llm`: run LLM-based data generation via OpenRouter/Qwen
  - `generate-programmatic`: run programmatic fallback (no API key needed)
  - `train-l1`: train L1 account trust model
  - `train-l2`: train L2 query classifier (both baseline and ATLAS versions)
  - `evaluate`: run full evaluation suite
  - `threshold-demo`: run threshold simulation
  - `all`: full pipeline end-to-end
  - `all-no-llm`: full pipeline with programmatic data only
- Seed everything with seed=42 for reproducibility
- Type hints throughout, docstrings on all public functions
- All CLI scripts use `argparse` with sensible defaults
- Logging via `logging` module, INFO level by default

## File Structure

```
atlas/
├── pyproject.toml
├── Makefile
├── README.md
├── atlas/
│   ├── __init__.py
│   ├── llm_generator.py        # Qwen-based session generation
│   ├── programmatic_generator.py  # numpy fallback generation
│   ├── features.py             # feature engineering from raw sessions
│   ├── l1_trust.py             # L1 account trust scorer
│   ├── l2_query.py             # L2 trust-conditioned query classifier
│   ├── evaluate.py             # full evaluation suite
│   ├── threshold_sim.py        # threshold modulation demo
│   └── utils.py                # shared utilities, config, constants
├── prompts/
│   ├── clean_enterprise.txt    # Qwen persona prompt
│   ├── clean_consumer.txt
│   ├── persistent_adversary.txt
│   └── sleeper.txt
├── scripts/
│   └── run_pipeline.py         # end-to-end orchestrator
└── outputs/
    ├── data/
    │   ├── raw_sessions/       # JSONL from Qwen
    │   ├── accounts.parquet    # processed account-level features
    │   └── queries.parquet     # processed query-level features
    ├── models/
    │   ├── l1_trust_model.pkl
    │   ├── l2_baseline_model.pkl
    │   └── l2_atlas_model.pkl
    ├── plots/                  # all evaluation PNGs
    ├── evaluation_report.md
    └── threshold_demo.md
```

## Implementation Notes

- **OpenRouter client setup**: use the `openai` SDK with custom base_url:
  ```python
  from openai import OpenAI
  client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=os.environ["OPENROUTER_API_KEY"],
      default_headers={
          "HTTP-Referer": "https://github.com/prashantkul/atlas",
          "X-Title": "ATLAS",
      },
  )
  response = client.chat.completions.create(
      model="qwen/qwen3.5",
      messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
      temperature=0.8,  # some diversity in generated sessions
      max_tokens=4096,
      response_format={"type": "json_object"},  # force JSON output
  )
  ```
- **LLM generation batching**: generate accounts in batches of 10 per archetype. Each Qwen call generates all sessions for one account. Parse JSON from the LLM response with error recovery — if JSON is malformed, retry once, then skip that account and log a warning.
- **Feature correlations matter**: in programmatic fallback, use multivariate normal with archetype-specific covariance matrices. Enterprise: high org_reputation CORRELATES with high verification, low refusal, high query_entropy. Adversary: high ccl_concentration CORRELATES with high rephrase_rate, high cross_session_persistence, low query_entropy. Independent features produce unrealistically easy classification.
- **Sleeper timeline is the showcase**: generate feature snapshots every 7 days for 180 days. The trajectory plot showing trust score dropping after behavioral shift is the single most compelling evaluation artifact. Make sure the plot clearly marks: phase 1 (stable high trust), transition point, phase 2 (declining trust), detection point (first crossing below 0.5).
- **L2 comparison is the headline result**: the whole point of ATLAS is that L2_atlas has fewer false positives on pharma queries than L2_baseline while maintaining the same false negative rate on adversaries. If this delta is small, the synthetic data needs more overlap between clean enterprise and adversary query content (both asking about chemistry, but with different behavioral patterns). The trust score should be the feature that disambiguates content-similar but intent-different accounts.
- **Evaluation report** should be structured as a brief research report: Abstract (one paragraph summary of results), Method, Results (with inline plot references), Discussion (limitations, what would change at Google scale). This doubles as interview talking material.
