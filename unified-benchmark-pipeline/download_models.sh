#!/usr/bin/env bash
# download_models.sh — Pre-download all models to external storage
#
# All models are stored under /export/cyb-ai-research-data/aberriche/models/
# to avoid filling the home directory quota.
#
# Run this script on the LOGIN NODE (which has internet access) before
# submitting SLURM inference jobs.
#
# Usage:
#   bash download_models.sh               # Download all new models
#   bash download_models.sh qwen gemma    # Download specific model(s) by keyword
#
set -euo pipefail

MODELS_DIR="/export/cyb-ai-research-data/aberriche/models"
HF_CACHE="/export/cyb-ai-research-data/aberriche/hf_cache"

mkdir -p "$MODELS_DIR"
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"

# ── Helper ────────────────────────────────────────────────────────────────────
download_model() {
    local hf_id="$1"    # e.g. Qwen/Qwen3.6-35B-A3B
    local local_name="$2"  # directory name under $MODELS_DIR
    local local_path="$MODELS_DIR/$local_name"

    if [ -d "$local_path" ] && [ "$(ls -A "$local_path" 2>/dev/null)" ]; then
        echo "[SKIP] $hf_id already exists at $local_path"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "Downloading: $hf_id"
    echo "Destination: $local_path"
    echo "============================================================"

    huggingface-cli download "$hf_id" \
        --local-dir "$local_path" \
        --local-dir-use-symlinks False

    echo "[DONE] $hf_id → $local_path"
}

# ── Model list ────────────────────────────────────────────────────────────────
# Filter by keyword if argument(s) provided
FILTER="${*:-all}"

should_download() {
    local name="$1"
    if [ "$FILTER" = "all" ]; then return 0; fi
    for kw in $FILTER; do
        if echo "$name" | grep -qi "$kw"; then return 0; fi
    done
    return 1
}

echo "Storage root : $MODELS_DIR"
echo "HF cache     : $HF_CACHE"
echo "Filter       : $FILTER"
echo ""

# ── Open-source models (new) ──────────────────────────────────────────────────
if should_download "qwen"; then
    download_model "Qwen/Qwen3.6-35B-A3B"          "Qwen3.6-35B-A3B"
fi

if should_download "gemma"; then
    download_model "google/gemma-4-31B"             "gemma-4-31B"
fi

if should_download "fanar"; then
    download_model "QCRI/Fanar-2-27B-Instruct"      "Fanar-2-27B-Instruct"
fi

if should_download "gpt-oss" || should_download "gptoss"; then
    download_model "openai/gpt-oss-20b"             "gpt-oss-20b"
fi

# ── Security-tuned models (existing — re-download to new path if needed) ──────
if should_download "foundation"; then
    download_model "fdtn-ai/Foundation-Sec-8B-Instruct"   "Foundation-Sec-8B-Instruct"
fi

if should_download "redsage" || should_download "qwen3"; then
    download_model "RISys-Lab/RedSage-Qwen3-8B-DPO"      "RedSage-Qwen3-8B-DPO"
fi

if should_download "primus" || should_download "llama"; then
    download_model "trendmicro-ailab/Llama-Primus-Merged" "Llama-Primus-Merged"
fi

echo ""
echo "============================================================"
echo "All requested downloads complete."
echo "Models stored in: $MODELS_DIR"
echo "============================================================"
