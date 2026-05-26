# Unified benchmark pipeline

End-to-end inference + LLM-as-judge evaluation + analysis pipeline for
cybersecurity benchmarks. Self-contained under this directory so it does not
conflict with the parent repository's existing scripts.

## Layout

```
unified-benchmark-pipeline/
├── evaluate.py                    # core inference helpers + API clients (Azure OpenAI, Anthropic)
├── run_inference_benchmarks.py    # collect model responses across all benchmarks
├── run_evaluate_llm_judge.py      # unified LLM-as-judge: extract + verdict in one call
├── analyze_gold_errors.py         # majority-vote analysis to flag suspect gold answers
├── calibrate_max_tokens.py        # per-task token-budget calibration via vLLM
├── fix_summaries.py               # regenerate summary.json from *_detailed.jsonl
├── download_models.sh             # helper to pre-download models
├── run_judge_all_models.sh        # batch driver: judge across all models
├── .env.example                   # template for AZURE_API_KEY, AZURE_JUDGE_ENDPOINT
├── experiment/
│   └── 01_full_scale_evaluation.py # full-scale config across all models × all tasks
└── analysis/                      # re-runnable post-judge analysis (see analysis/README.md)
    ├── lib/                       # shared utilities (loaders, plot_style, voting, …)
    ├── results_table.py
    ├── judge_agreement.py
    ├── gold_error_voting.py
    └── …
```

## Quick start

```bash
cd unified-benchmark-pipeline

# 1. Copy and fill in API credentials
cp .env.example .env
# edit .env to set AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME

# 2. Collect model responses
python run_inference_benchmarks.py \
    --model_path /path/to/model \
    --use_vllm \
    --output_dir outputs/responses_<MODEL_NAME> \
    --tasks mcq rcm vsp ate cti_taa sevenllm ...

# 3. Judge the responses
python run_evaluate_llm_judge.py \
    --response_dir outputs/responses_<MODEL_NAME> \
    --judge_use_api \
    --judge_api_endpoint "$AZURE_ENDPOINT" \
    --judge_api_model "$AZURE_DEPLOYMENT_NAME" \
    --output outputs/judge_<MODEL_NAME>/eval_results

# 4. Re-runnable analysis on the judged outputs
cd analysis
PYTHONPATH=. python -m analysis.judge_agreement
PYTHONPATH=. python -m analysis.results_table
PYTHONPATH=. python -m analysis.gold_error_voting
```

See `analysis/README.md` for the full analysis workflow (results table, judge
agreement, gold-error voting, plotting).

## Supported benchmarks

24 sub-tasks across:

- **CTI-Bench**: MCQ, RCM, VSP, ATE, TAA (TSV)
- **AthenaBench**: CKT, RMS, TAA, ATE (expanded), RCM (compiled), VSP
- **SECURE**: MAET, CWET, KCV
- **SecEval** · **CISSP** · **CyberMetric-500** · **SecBench** · **MMLU-CS**
- **RedSage-MCQ**: Frameworks, Generals, Skills, CLI, Kali
- **SEvenLLM-Bench** (English subset, structured CTI extraction)

## Supported judge API styles

- `chat_completions` — Azure OpenAI Chat Completions (default)
- `azure_responses` — Azure Responses API (GPT-5.4)
- `anthropic_messages` — Azure-hosted Anthropic /v1/messages (Claude on Azure)

Set `--judge_api_style` to choose. The judge prompt is unified across tasks:
extraction + verdict in a single call, with per-task `format_hint` and
`compare_rule` controlling canonicalization and equality semantics.

## Standardized pipeline configuration

The unified harness applies the following fixed choices across all models and
tasks. Each choice is a deliberate departure from the benchmark-specific
defaults documented in [BENCHMARKS.md](../BENCHMARKS.md) and is selected to
remove measurement artifacts rather than to change task semantics.

### Inference

| Setting | Value | Rationale |
|---------|-------|-----------|
| Temperature | 0.0 | Deterministic output; eliminates temperature-drift failure mode ($\mathcal{F}_3(\mathcal{I})$) |
| top\_p | 1.0 | No nucleus truncation at zero temperature |
| do\_sample | False | Greedy decoding for HF Transformers backend |
| min\_tokens (vLLM) | 50 | Prevents EOS-only responses on models with early-exit tendency (e.g. Gemma-4) |
| Token budget | Per-task, calibrated | Set per task via `calibrate_max_tokens.py` (see table below); resolves token-budget failure mode ($\mathcal{F}_2(\mathcal{I})$) |
| seed | 42 (where applicable) | For reproducibility on non-deterministic backends |

**Per-task calibrated token budgets** (representative values from a non-thinking model):

| Task | Budget | Task | Budget |
|------|-------:|------|-------:|
| `mcq` | 1024 | `ckt` | 1024 |
| `rcm` | 512 | `rms` | 512 |
| `vsp` | 2048 | `taa` | 512 |
| `ate` | 1024 | `athena_ate` | 256 |
| `seceval` | 256 | `athena_rcm` | 1024 |
| `cybermetric` | 256 | `athena_vsp` | 1024 |
| `secbench` | 256 | `secure_*` | 256 |
| `mmlu-cs` | 512 | `redsage_*` | 256 |

Budgets for thinking models (Qwen3, Gemma-4) are scaled up (typically 4×–8×)
to accommodate the chain-of-thought prefix. Pass your calibration file via
`--max_tokens_config slurm/calibration_<MODEL>.json`.

### Prompt construction

- **Chat template:** Always applied via `tokenizer.apply_chat_template()`,
  ensuring model-specific tokens (`<|im_start|>`, `[INST]`, etc.) are correctly
  injected. Eliminates template-incompatibility failure mode ($\mathcal{F}_3(\mathcal{P})$).
- **System prompts:** Applied where the benchmark specifies one:
  - CTI-Bench tasks: `"You are a cybersecurity expert specializing in cyberthreat intelligence."`
  - CyberMetric: `"You are a security expert who answers questions."`
  - SecEval: `"Below are multiple-choice questions concerning cybersecurity. Please select the correct answers and respond with the letters ABCD only."` (+ 1-shot example)
  - AthenaBench, SECURE, SecBench, MMLU-CS, RedSage-Bench: no system prompt
- **Prompt source:** Benchmark-provided `Prompt` column used as-is where available
  (CTI-Bench, AthenaBench). Reconstructed from question/answer fields where no
  template is provided (SecBench, MMLU-CS, CISSP).

### Thinking model handling

Reasoning models (Qwen3-series, Gemma-4) produce `<think>…</think>` blocks
before their final answer. The pipeline handles these in two ways:

1. **Token budget:** Extended budgets (set via `--max_tokens_config`) ensure the
   thinking trace does not consume the entire budget before an answer is generated.
2. **Stop sequence:** The RedSage-Bench original stop sequence `["\n"]` fires
   inside the thinking preamble and returns an empty string. The unified pipeline
   applies the stop post-generation: the thinking block is stripped first, then
   `"\n"` is applied to the answer portion only. This recovers 86 percentage
   points on RedSage-Bench for reasoning models ($\mathcal{F}_1(\mathcal{I})$).

### Extraction and scoring (LLM judge)

All tasks use a single unified judge call that performs extraction and verdict
together. No regex post-processing is applied. The judge returns:

```json
{
  "extracted_answer": "<answer in canonical form>",
  "verdict": "CORRECT" | "INCORRECT",
  "justification": "<one sentence>"
}
```

**Per-task canonicalization rules** enforced by the judge:

| Task type | Extracted form | Comparison rule |
|-----------|---------------|-----------------|
| MCQ (all single-answer tasks) | Single uppercase letter A–E | Exact match, case-insensitive |
| SecEval (multi-answer) | Sorted concatenated letters, e.g. `"ABC"` | Exact set match; partial credit not awarded |
| RCM | Comma-separated sorted CWE-IDs, e.g. `"CWE-79,CWE-89"` | Exact set match; `"79"` normalised to `"CWE-79"` |
| VSP | CVSS:3.1 vector string | Exact match after prefix normalisation (`CVSS:3.0/` → `CVSS:3.1/`) |
| ATE | Sorted parent MITRE T-IDs, e.g. `"T1027,T1059"` | Exact set match; subtechnique suffixes stripped |
| RMS | Sorted MITRE M-IDs, e.g. `"M1018,M1026"` | Exact set match |
| TAA | Canonical threat-actor name | Alias-aware match (e.g. `"APT28"` ≡ `"Fancy Bear"`) |

### Denominator policy

All attempted items are included in the denominator regardless of whether the
model produced a parseable response. Unparseable or empty responses count as
incorrect. This eliminates denominator-inflation failure mode ($\mathcal{F}_2(\mathcal{E})$),
which can inflate scores up to 500× when only valid responses are counted.
