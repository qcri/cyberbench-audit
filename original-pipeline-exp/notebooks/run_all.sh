#!/usr/bin/env bash
# Execute all original-pipeline-exp notebooks headlessly (no heavy inference steps).
# Run on a compute node — the login node blocks heavy I/O.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ORIG_NB_REGEN=0 jupyter nbconvert --to notebook --execute \
  00_overview.ipynb \
  01_regex_baseline.ipynb \
  02_judge_vs_regex.ipynb \
  03_prompt_sensitivity.ipynb \
  04_inference_config.ipynb \
  05_logprob_vs_generative.ipynb
