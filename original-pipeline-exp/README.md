# Original Pipeline Experiments

Scripts that reproduce each benchmark's original evaluation pipeline as closely as
possible. These are a core part of the audit: by running benchmarks under their own
published evaluation logic, we measure the failure modes documented in the paper: extractor brittleness, denominator inflation, stop-sequence mismatches,
logprob vs. generative scoring gaps, and so on.

The unified pipeline (`../unified-benchmark-pipeline/`) standardizes those choices to
isolate their effect on scores and rankings. Both pipelines are needed: original
pipelines establish the baseline; the unified pipeline shows what changes when
evaluation artifacts are controlled.

## Scripts

```
original-pipeline-exp/
├── evaluate.py                  # Combined inference + regex evaluation (original benchmark behavior)
├── run_inference_benchmarks.py  # Benchmark-faithful response collection (all tasks)
├── run_evaluate_llm_judge.py    # LLM-as-judge evaluation on pre-collected JSONL
├── run_batch_evaluate.py        # Batch driver: evaluate across multiple models
├── run_prompt_sensitivity.py    # Prompt sensitivity study (zero-shot / few-shot / CoT)
└── run_redsage_lighteval.py     # RedSage logprob scoring via upstream LightEval
```

## Script Descriptions

### `evaluate.py` — Combined inference + regex evaluation

Reproduces the original benchmark evaluation behavior: runs inference and scores with
regex extraction in one pass, mirroring how the benchmark authors published results.
This is what exposes failure modes like extractor divergence ($\mathcal{F}_1(\mathcal{E})$)
and denominator inflation ($\mathcal{F}_2(\mathcal{E})$).
Supports CTI-Bench (MCQ, RCM, VSP, ATE), CyberMetric-500, SecEval, and CISSP.

```bash
python evaluate.py \
  --model_path "meta-llama/Llama-3.3-70B-Instruct" \
  --tasks mcq rcm vsp ate seceval cybermetric \
  --output eval_results.json

# With CISSP
python evaluate.py \
  --model_path "meta-llama/Llama-3.3-70B-Instruct" \
  --tasks mcq cissp \
  --cissp_path ../cissp.json \
  --output eval_results.json

# API inference
python evaluate.py \
  --use_api \
  --api_endpoint "https://api.openai.com/v1/chat/completions" \
  --api_model "gpt-4" \
  --tasks mcq rcm \
  --output eval_results_gpt4.json
```

---

### `run_inference_benchmarks.py` — Benchmark-faithful response collection

Collects raw model responses for all 24 sub-tasks in a benchmark-faithful format and
writes one JSONL file per task. Separates collection from scoring so the same outputs
can be passed through the original regex evaluator, the LLM judge, or the unified
pipeline to measure evaluator-induced score differences.

Supports all benchmark families:
- **CTI-Bench** original TSVs: MCQ, RCM, RCM-2021, VSP, ATE, TAA
- **AthenaBench** original JSONLs: CKT, ATE, RCM, RMS, VSP, TAA
- **SECURE** original TSVs: MAET, CWET, KCV
- **SecEval**, **CyberMetric-500**, **SecBench**, **MMLU-CS**, **CISSP**
- **RedSageMCQ**: Frameworks, Generals, Skills, CLI, Kali
- `mmlu-cs-logprobs` — logprob scoring mode to mirror the original MMLU evaluation

```bash
python run_inference_benchmarks.py \
  --model_path "meta-llama/Llama-3.3-70B-Instruct" \
  --use_vllm \
  --tasks mcq rcm vsp ate ckt rms taa \
  --output_dir outputs/responses_llama33

# API inference
python run_inference_benchmarks.py \
  --use_api \
  --api_endpoint "https://api.openai.com/v1/chat/completions" \
  --api_model "gpt-4" \
  --tasks seceval cybermetric \
  --output_dir outputs/responses_gpt4
```

---

### `run_evaluate_llm_judge.py` — LLM-as-judge evaluation

Evaluates pre-collected JSONL responses using an LLM judge instead of regex extraction.
Running both `evaluate.py` and this script on the same outputs quantifies the extractor
divergence failure mode ($\mathcal{F}_1(\mathcal{E})$). Supports all benchmarks.

```bash
python run_evaluate_llm_judge.py \
  --response_dir outputs/responses_llama33 \
  --judge_use_api \
  --judge_api_endpoint "$AZURE_ENDPOINT" \
  --judge_api_model "$AZURE_DEPLOYMENT_NAME" \
  --output outputs/judge_llama33/eval_results
```

---

### `run_batch_evaluate.py` — Batch evaluation across models

Iterates over all models in a raw-results directory and evaluates each JSONL in-process,
writing per-model JSON results without spawning subprocesses per task.

```bash
python run_batch_evaluate.py \
  --raw_results_dir outputs/ \
  --output_dir outputs/batch_results/

# Dry run (show what would run without executing)
python run_batch_evaluate.py \
  --raw_results_dir outputs/ \
  --output_dir outputs/batch_results/ \
  --dry_run

# Specific models only
python run_batch_evaluate.py \
  --raw_results_dir outputs/ \
  --output_dir outputs/batch_results/ \
  --models llama33 gpt4
```

---

### `run_prompt_sensitivity.py` — Prompt sensitivity study

Controlled study of prompt-stage failure modes ($\mathcal{F}_1$–$\mathcal{F}_3(\mathcal{P})$).
For each task, collects 100 seeded samples under three prompt modes:

- `zero_shot` — original benchmark prompt, no modification
- `few_shot` — 2 real dataset samples prepended as Q+A examples
- `cot` — chain-of-thought system suffix; native thinking enabled for Qwen3/Gemma4

Outputs per task/mode:
- `{output_dir}/{task}_{mode}_responses.jsonl` — same schema as `run_inference_benchmarks.py`
- `{output_dir}/fewshot_examples_{task}.json` — which samples were used as few-shot examples

```bash
python run_prompt_sensitivity.py \
  --model_path "meta-llama/Llama-3.3-70B-Instruct" \
  --use_vllm \
  --tasks mcq rcm vsp ate ckt \
  --prompt_modes zero_shot few_shot cot \
  --n_test 100 \
  --n_shot 2 \
  --seed 42 \
  --output_dir outputs/prompt_sensitivity_llama33

# API run (thinking flag has no effect for API endpoints)
python run_prompt_sensitivity.py \
  --use_api \
  --api_endpoint "https://api.openai.com/v1/chat/completions" \
  --api_model "gpt-4" \
  --tasks mcq seceval \
  --prompt_modes zero_shot few_shot \
  --output_dir outputs/prompt_sensitivity_gpt4
```

---

### `run_redsage_lighteval.py` — RedSage logprob scoring via LightEval

Reproduces RedSage MCQ evaluation using the upstream LightEval implementation with
likelihood/logprob-based scoring. Comparing these results against generative scoring
from `run_inference_benchmarks.py` demonstrates the logprob vs. generative scoring
failure mode ($\mathcal{F}_1(\mathcal{A})$): the same model can score 45.7% generatively
vs. 86.6% under logprob on the same items. All prompt and scoring logic lives in
`external/RedSage/eval/cybersecurity_benchmarks.py`.

**Requires LightEval setup first:**
```bash
cd external/RedSage/eval/lighteval && pip install -e . && cd ../../../../
pip install cvss aenum
```

```bash
# All 5 subsets, logprob scoring (default)
python run_redsage_lighteval.py \
  --model RISys-Lab/RedSage-Qwen3-8B-DPO

# Generative (exact-match) variants
python run_redsage_lighteval.py \
  --model RISys-Lab/RedSage-Qwen3-8B-DPO \
  --task-mode generative

# vLLM backend, smoke test
python run_redsage_lighteval.py vllm \
  --model RISys-Lab/RedSage-Qwen3-8B-DPO \
  --max-samples 5

# Specific subsets only
python run_redsage_lighteval.py \
  --model RISys-Lab/RedSage-Qwen3-8B-DPO \
  --subsets cybersecurity_skills,cybersecurity_tools_kali
```

## Relationship to the Unified Pipeline

| Concern | Original pipeline | Unified pipeline |
|---------|-----------------|-----------------|
| Inference | `run_inference_benchmarks.py` | `../unified-benchmark-pipeline/run_inference_benchmarks.py` |
| Regex baseline | `evaluate.py` | — (intentionally not standardized) |
| LLM judge | `run_evaluate_llm_judge.py` | `../unified-benchmark-pipeline/run_evaluate_llm_judge.py` |
| Batch runner | `run_batch_evaluate.py` | `run_judge_all_models.sh` |
| Prompt study | `run_prompt_sensitivity.py` | — (failure mode measurement) |
| RedSage logprob | `run_redsage_lighteval.py` | — (failure mode §$\mathcal{A}$1) |

Running the same model outputs through both pipelines is how we measure the rank shifts
reported in Table 4 of the paper.
