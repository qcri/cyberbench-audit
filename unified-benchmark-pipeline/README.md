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
# edit .env to set AZURE_API_KEY, AZURE_JUDGE_ENDPOINT

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
    --judge_api_endpoint "$AZURE_JUDGE_ENDPOINT" \
    --judge_api_model gpt-5.4 \
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
