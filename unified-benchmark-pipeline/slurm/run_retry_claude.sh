#!/usr/bin/env bash
#SBATCH --job-name=retry_claude
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -uo pipefail

# Retry the 1 inference ERROR (athena_rcm) and 7 judge skips
# (athena_vsp×2, redsage_skills×2, seceval×1, secure_maet×2) for claude.

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks

set -a; [ -f "$ROOT_DIR/.env" ] && source "$ROOT_DIR/.env"; set +a

INFER_ENDPOINT="https://qcri-claude-hub-cyberxpert.services.ai.azure.com/anthropic/v1/messages"
INFER_MODEL="claude-sonnet-4-6-cyberxpert"
JUDGE_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
JUDGE_MODEL="gpt-5.4"

RESPONSE_DIR="$ROOT_DIR/outputs/responses_claude-sonnet-4-6-cyberxpert"
JUDGE_DIR="$ROOT_DIR/outputs/judge_claude-sonnet-4-6-cyberxpert/eval_results"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Start: $(date)"
echo "============================================================"

# ── Phase A: inference retry (athena_rcm only — claude uses Anthropic API) ──
# retry_failed_samples.py uses run_inference_benchmarks.chat_completion_api by
# default; for claude we need anthropic_messages so we run the full inference
# pipeline restricted to athena_rcm (skip_completed protects the existing 1999
# OK rows since the file already exists). Easier: just delete the file and
# re-run the task — but that would waste 2k API calls. Instead patch in-place
# via the same anthropic_messages_api the inference uses.
echo ""
echo "=== Phase A: inference retry — athena_rcm ERROR rows ==="
$PYTHON - <<'PYEOF'
import os, sys, json
sys.path.insert(0, '/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks')
from evaluate import anthropic_messages_api

path = "/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/outputs/responses_claude-sonnet-4-6-cyberxpert/athena_rcm_responses.jsonl"
endpoint = "https://qcri-claude-hub-cyberxpert.services.ai.azure.com/anthropic/v1/messages"
model = "claude-sonnet-4-6-cyberxpert"
key = os.environ["AZURE_CLAUDE_KEY"]

rows = []
with open(path) as f:
    for line in f:
        rows.append(json.loads(line))

n_err = sum(1 for r in rows if r.get("model_response","").startswith("ERROR:"))
print(f"  {n_err} ERROR rows; retrying...")
n_fixed = 0
for r in rows:
    if r.get("model_response","").startswith("ERROR:"):
        new = anthropic_messages_api(endpoint, model, r["prompt"], api_key=key, max_tokens=4096)
        r["model_response"] = new
        if not new.startswith("ERROR:"):
            n_fixed += 1
        print(f"    idx={r.get('index')}: {'FIXED' if not new.startswith('ERROR:') else 'still ERROR'}")
with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"  fixed {n_fixed}/{n_err}")
PYEOF

# ── Phase B: judge retry (the 7 skipped rows) ────────────────────────────────
echo ""
echo "=== Phase B: judge retry on claude skips ==="
$PYTHON "$ROOT_DIR/retry_failed_judges.py" \
    --judge_dir "$JUDGE_DIR" \
    --response_dir "$RESPONSE_DIR" \
    --judge_api_endpoint "$JUDGE_ENDPOINT" \
    --judge_api_model "$JUDGE_MODEL" \
    --judge_api_key "$AZURE_API_KEY" \
    --workers 8

# ── Phase C: re-judge athena_rcm (the inference was just fixed) ──────────────
echo ""
echo "=== Phase C: re-judge claude athena_rcm with the fixed responses ==="
rm -f "$JUDGE_DIR/athena_rcm_detailed.jsonl"
$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir       "$RESPONSE_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$JUDGE_ENDPOINT" \
    --judge_api_model    "$JUDGE_MODEL" \
    --judge_api_key      "$AZURE_API_KEY" \
    --n_workers          8 \
    --tasks              athena_rcm \
    --output             "$JUDGE_DIR"

echo ""
echo "============================================================"
echo "End: $(date)"
echo "============================================================"
