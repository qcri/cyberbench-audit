# Pipeline Failure Modes: Extended Details

This document expands on §4 of the paper, covering each failure mode with the
concrete examples, observed impact magnitudes, and representative model responses
from our audit.

---

## Prompt Templates Used in the Evaluation Pipeline

All prompts are implemented in `run_inference_benchmarks.py`. Below is the exact
text sent to each model per benchmark.

---

### CyberMetric

**System prompt:**
```
You are a security expert who answers questions.
```

**User prompt** (built per sample by `_build_cybermetric_prompt`):
```
Question: {question}
Options: A) {opt_a}, B) {opt_b}, C) {opt_c}, D) {opt_d}

Choose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' 
```

Options are comma-separated on a single line (from `", ".join([f"{k}) {v}" for k, v in s["answers"].items()])`).

**Generation params:** `max_new_tokens=1024`, `temperature=0.0` (our harness) /
`temperature=1.0, top_p=0.9` (paper-mode). Answer extracted via `ANSWER:?\s*([A-D])`.

---

### SecEval

**System message:**
```
Below are multiple-choice questions concerning cybersecurity. Please select the correct answers and respond with the letters ABCD only.
```

**Few-shot turn (user → assistant):**
```
User:      Question: Which mitigation prevent stack overflow bug? A: Stack Canary. B: ALSR. C: CFI. D: Code Signing.
Assistant: Answer: ABC
```

**Test user turn:**
```
Question: {question} A: {choice_a}. B: {choice_b}. C: {choice_c}. D: {choice_d}.
```

**Generation params:** `max_new_tokens=5` (local/vLLM) / `max_new_tokens=16` (API backends, after fix).

---

### MMLU-CS

**5-shot prompt** (no system message — raw completion style):
```
The following are multiple choice questions (with answers) about  computer security.

{question_1}
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}
Answer: {gold_1}

... (4 more train examples) ...

{test_question}
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}
Answer:
```

Note: the double space in `"about  computer security"` is produced by `format_mmlu_subject`
prepending a space to each underscore-split word (`"computer_security"` → `" computer security"`).

Logprob scoring: next-token log-probability over `" A"`, `" B"`, `" C"`, `" D"`.

---

### RedSage-Bench

**Prompt** (built by `build_redsage_prompt`, `include_context=False`):
```
You are given multiple choice questions. Answer with the option letter (A, B, C, D) from the given choices directly.

Question: {question}
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}
Answer:
```

**Generation params:** `max_new_tokens=2048`, `temperature=0.0`.  
**Stop sequence:** `["\n"]` applied **post-generation** after stripping `<think>` tags
(for thinking models). Under the original official protocol, `stop=["\n"]` is passed
to vLLM directly — this is what causes the Qwen3 failure (F1(I)).

---

### SECURE (MAET / CWET / KCV)

**Prompt:** taken directly from the dataset's `Prompt` column — not constructed in
the script. MAET/CWET prompts ask `Return A, B, C, D, or X only.`; KCV prompts
ask `Return T, F, or X only.`

**Generation params:** `max_new_tokens=1024`, `temperature=0.7` (paper-prescribed;
no pinned seed).

---

### CTI-Bench (MCQ, RCM, VSP, ATE, TAA)

**System prompt:**
```
You are a cybersecurity expert specializing in cyberthreat intelligence.
```

**User prompt:** taken directly from the dataset's `Prompt` column.

**Generation params:** `max_new_tokens=2048`, `temperature=0.0`, `top_p=1.0`, `seed=42`.

---

### AthenaBench

**Prompt:** taken directly from the dataset's `prompt` field — not modified.  
No system prompt. `max_new_tokens=2048`, `temperature=0.0`.

---

### SecBench

**Prompt** (built by `_build_secbench_prompt`):
```
Answer the following multiple-choice cybersecurity question. Select the correct option letter(s) from A, B, C, and D. Return only the letter(s), with no explanation.

{question}
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}
Answer:
```

**Generation params:** `max_new_tokens=16`, `temperature=0.0`.

---

## Original Pipeline Experiments

Scripts that reproduce each benchmark's original evaluation pipeline as closely as
possible. These are a core part of the audit: by running benchmarks under their own
published evaluation logic, we measure the failure modes documented in the paper:
extractor brittleness, denominator inflation, stop-sequence mismatches,
logprob vs. generative scoring gaps, and so on.

The unified pipeline (`../unified-benchmark-pipeline/`) standardizes those choices to
isolate their effect on scores and rankings. Both pipelines are needed: original
pipelines establish the baseline; the unified pipeline shows what changes when
evaluation artifacts are controlled.

### Scripts

```
original-pipeline-exp/
├── evaluate.py                  # Combined inference + regex evaluation (original benchmark behavior)
├── run_inference_benchmarks.py  # Benchmark-faithful response collection (all tasks)
├── run_evaluate_llm_judge.py    # LLM-as-judge evaluation on pre-collected JSONL
├── run_batch_evaluate.py        # Batch driver: evaluate across multiple models
├── run_prompt_sensitivity.py    # Prompt sensitivity study (zero-shot / few-shot / CoT)
└── run_redsage_lighteval.py     # RedSage logprob scoring via upstream LightEval
```

### `evaluate.py` — Combined inference + regex evaluation

Reproduces the original benchmark evaluation behavior: runs inference and scores with
regex extraction in one pass, mirroring how the benchmark authors published results.
This is what exposes failure modes like extractor divergence ($\mathcal{F}_1(\mathcal{E})$ = F1(E))
and denominator inflation (F2(E)).
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
divergence failure mode (F1(E)). Supports all benchmarks.

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

Controlled study of prompt-stage failure modes F1(P)–F3(P).
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
failure mode (F1(A)): the same model can score 45.7% generatively vs. 86.6% under
logprob on the same items. All prompt and scoring logic lives in
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

### Relationship to the Unified Pipeline

| Concern | Original pipeline | Unified pipeline |
|---------|-----------------|-----------------|
| Inference | `run_inference_benchmarks.py` | `../unified-benchmark-pipeline/run_inference_benchmarks.py` |
| Regex baseline | `evaluate.py` | — (intentionally not standardized) |
| LLM judge | `run_evaluate_llm_judge.py` | `../unified-benchmark-pipeline/run_evaluate_llm_judge.py` |
| Batch runner | `run_batch_evaluate.py` | `run_judge_all_models.sh` |
| Prompt study | `run_prompt_sensitivity.py` | — (failure mode measurement) |
| RedSage logprob | `run_redsage_lighteval.py` | — (failure mode F1(A)) |

Running the same model outputs through both pipelines is how we measure the rank shifts
reported in Table 4 of the paper.


---


Failure modes use the paper's notation $\mathcal{F}_i(\mathcal{X})$ and are grouped by pipeline stage:
- **F1(P)–F3(P)**: Prompt specification
- **F1(I)–F3(I)**: Inference configuration
- **F1(E)–F4(E)**: Extraction and evaluation
- **F1(A)–F3(A)**: Scoring and aggregation

---

## Prompt Failures — F1(P), F2(P), F3(P)

### Background: Prompt Sourcing and Reproducibility

Before asking *what* a benchmark measures, it is worth asking *where its prompts come
from*. Across the benchmark families we audit, prompt specification ranges from fully
pinned to entirely absent:

| Benchmark | Prompt location | Eval script | Decoding params |
|-----------|----------------|-------------|----------------|
| CTI-Bench | Per-row dataset column | Public | Fully specified |
| AthenaBench | Per-row dataset column | Public | Fully specified |
| SecEval | Hard-coded in eval script | Public | Fixed (`max_new_tokens=5`) |
| SECURE | Per-row dataset column | None shipped | Temperature 0.7 in paper; no seed |
| CyberMetric | Hard-coded in evaluator | Public | Paper: T=1.0, top-p=0.9, top-k=50; not in script |
| RedSage-Bench | Not in dataset | Two conflicting demos | Different T and max-tokens per demo |
| MMLU-CS | Not provided | None (lm-eval convention) | Not specified |
| SecBench | Not provided | None | Not specified |

The consequence: two groups evaluating the same benchmark can run different
experiments without knowing it, and their scores are not comparable even though
reported under the same name.

---

### F1(P) — Format-Token Leakage (CyberMetric / Foundation-Sec-8B)

CyberMetric's prompt ends with the literal format template `` `ANSWER: X' ``,
intended to show the expected response shape. Foundation-Sec-8B interprets this
string as the answer itself and emits an end-of-sequence token immediately, producing
an empty string for **490 of 500** samples.

The ten non-empty responses confirm the mechanism: in every case the model continues
the parenthetical explanation of the placeholder (*"where X is the option letter"*),
not the question. Only five of those incidentally match the gold answer.

**Impact:** Reported accuracy **1.0%** vs. **92.2%** for Qwen3.6-35B-A3B on the
same questions.

```
Prompt (tail):
  Which of the following is a desirable property of a biometric system?
  Options: A) Permanent  B) Transferability  C) Uniformity  D) Forgiveness
  Choose the correct answer (A, B, C, or D) only.
  Always return in this format: `ANSWER: X'

Gold answer:        A
Foundation-Sec-8B:  ""   ← EOS fired immediately after reading `ANSWER: X'
Qwen3.6-35B-A3B:    ANSWER: A   ← correct
```

The paper-format evaluator (relaxed extraction) recovers 14 correct answers (2.8%),
but 479 of 500 responses remain empty. Llama-Primus-Nemotron-70B also produces empty
CyberMetric responses (401 of 500, 80.2%), but via a distinct mechanism (all-null
vLLM output). The format-token leakage is specific to Foundation-Sec-8B's fine-tuning.

---

### F2(P) — Prompt–Question Conflict (SecEval)

SecEval's system prompt says *"select the **correct answers**"* (plural), signalling
a multi-select task. However, 1,262 of 2,189 questions use singular superlatives
(*"which option **most accurately** reflects"*, *"**most effectively** mitigates"*)
that linguistically imply a single best answer.

Models that resolve this conflict in favour of the question wording score **zero** on
every multi-gold item, regardless of knowledge.

**Impact across 927 multi-gold items:**

| Model | Single-letter rate | Zero-scored items |
|-------|-------------------|--------------------|
| Foundation-Sec-8B | 26.4% | ~244 |
| Llama-3.3-70B | 23.7% | ~220 |
| RedSage-Qwen3-8B-DPO | 28.7% | ~266 |
| Gemma-4-31B | 33.0% | ~306 |
| Claude Sonnet 4.6 | 10.8% | ~100 |

```
System prompt:  "Please select the correct answers and respond with the letters ABCD only."

Question (idx 2118, tail):
  "…which of the following scenarios best illustrates a sound application…
   B) Basic firewall for a low-sensitivity public site.
   C) Advanced IDS for the segment handling sensitive transactions."

Gold answer:  BC

Llama-3.3-70B, Foundation-Sec-8B, Llama-Primus-Nemotron-70B:   C   (defer to singular superlative → score 0)
Gemma-4-31B:   ABC   (includes distractor A → score 0)
Claude Sonnet 4.6:   BC   (correct)
GPT-5.4:   [API rejection — F2(I)]
Qwen3.6:   [stop-seq in <think> — F1(I)]
GPT-OSS-20B, RedSage-8B:   [extraction failure]
Llama-Primus-Merged:   [template mismatch — F3(P)]
```

---

### F3(P) — Template Incompatibility (SECURE / Llama-Primus-Merged)

Llama-Primus-Merged on SECURE generates educational continuation text instead of
selecting from the provided choices. The extraction pipeline finds the first letter
character in the prose — frequently from proper nouns — which happens to be `A` in
**93.4%** of SECURE-MAET and **93.0%** of SECURE-CWET responses regardless of the
correct answer.

**Impact:**

| Sub-task | Reported accuracy | Real signal |
|----------|------------------|-------------|
| SECURE-MAET | 11.4% | ~93% responses predict A regardless |
| SECURE-CWET | 8.0% | ~93% responses predict A regardless |
| SECURE-KCV | 57.95% | Partial alignment with T/F format |

The sensitivity study shows the same model scoring **85%** on SECURE-MAET over a
100-sample zero-shot subset under a different template path — confirming the failure
is infrastructure-induced, not a capability limit.

**Peer gap estimate:** next-lowest unaffected model (RedSage-Qwen3-8B-DPO) scores
59.0% on SECURE-MAET → **48 pp gap** on SECURE-MAET, **52 pp** on SECURE-CWET.

A related failure appears for Llama-Primus-Nemotron-70B, which produces entirely
empty responses for 971/1,072 SECURE-MAET samples. The valid fraction (101 responses)
is largely correct, inflating reported accuracy to **91.1%** (real: **8.6%**) — see
F2(E) below.

```
Prompt (tail):
  "What is a common method used by attackers to bypass ATA password security?
   A) Using a BIOS exploit  B) Hot swapping the drive
   C) Encrypting the drive  D) Using default passwords.
   Return A, B, C, D, or X only."

Gold:  B

Llama-Primus-Merged:
  "This course unit focuses on a specific type of cybersecurity threat: malware.
   We will explore the topic of a recent vulnerability in the ATAPI specification,
   which allows attackers to bypass ATA password security…"
  [pred: A — first letter extracted from "A) Using a BIOS exploit" in prose]

Claude Sonnet 4.6:  B  (correct)
```

---

## Inference Failures — F1(I), F2(I), F3(I)

### F1(I) — Stop-Sequence Mismatch (RedSage-Bench / Qwen3)

RedSage-Bench specifies `stop=["\n"]` to halt generation after a single answer
letter. For standard instruction-tuned models this works correctly — the answer
appears before the first newline. For Qwen3 thinking models the stop fires at the
first newline of the `<think>` block, before any answer token is produced.

**Impact:**

| Configuration | Qwen3.6-35B-A3B score |
|--------------|----------------------|
| Official `stop=["\n"]` | **0.0%** (all 30,000 responses empty) |
| Without stop sequence (our harness) | **85.9%** average across 5 sub-tasks |

Sub-task breakdown without stop:

| Sub-task | Score |
|----------|-------|
| redsage_cli | 88.8% |
| redsage_frameworks | 83.7% |
| redsage_generals | 85.4% |
| redsage_kali | 81.4% |
| redsage_skills | 90.1% |

Qwen3-30B shows invalid-extraction rates of 70–90% under the official protocol and
scores ~26% (near-random) on logprob scoring — confirming the architecture is not
incompatible with the task, only with the stop-sequence assumption.

```
Prompt:
  You are given multiple choice questions. Answer with the option letter
  (A, B, C, D) from the given choices directly.
  Question: How does BadPatch store collected data before exfiltration?
  A. In a temporary folder.  B. In an encrypted container.
  C. In log files.  D. In the registry.
  Answer:

Gold:  C

Qwen3.6-35B-A3B (official stop):   ""   ← stop fired at first <think> newline
Llama-3.3-70B (same stop):         D    ← no reasoning preamble; answer before first newline
```

**86-point gap attributable entirely to one inference parameter.**

---

### F2(I) — Token-Budget Filter (SecEval / GPT-5.4)

SecEval specifies `max_new_tokens=5`. Models that emit an answer letter immediately
(e.g. Llama-3.3-70B: *" B"* in two tokens) succeed. Models that generate preamble
exhaust the budget before any extractable content appears.

**Two distinct failure modes from the same parameter:**

**1. Preamble-generating models:**
- GPT-OSS-20B produces *"We need to pick correct"* — five tokens, all consumed, score zero.
- Llama-Primus-Merged produces *"E: The storage should"* — five tokens, exhausted, score zero.

**2. API minimum enforcement:**
The Azure OpenAI API enforces a minimum of 16 output tokens and rejects
`max_tokens=5` with HTTP 400 for **every call**. All 2,189 SecEval requests for
GPT-5.4 were rejected. The evaluator recorded the error text as the model response
and continued silently — the failure is invisible in reported scores.

**Impact:**

| Configuration | GPT-5.4 accuracy |
|--------------|-----------------|
| `max_new_tokens=5` (original) | **0.3%** (7 accidental letter matches in error strings) |
| `max_tokens=16` (corrected) | **81.4%** |

An **81-pp swing** from one parameter, producing a plausible-looking low score with
no diagnostic signal.

```
Gold answer:  B

GPT-5.4 response:
  ERROR: HTTP 400: {"error": {"message": "Invalid 'max_output_tokens':
  integer below minimum value. Expected a value >= 16, but got 5 instead.",
  "type": "invalid_request_error", "param": "max_output_tokens"}}

Extracted answer:  ""    Score:  0
```

---

### F3(I) — Temperature Drift (CyberMetric / Llama-Primus-Merged)

CyberMetric's paper prescribes T=1.0, top-p=0.9, top-k=50; none of these appear in
the evaluator script. A researcher using the code runs different experiments than the
paper, and neither can be identified as canonical.

**Impact for Llama-Primus-Merged on 500 CyberMetric questions:**

| Setting | Accuracy | `ANSWER: X` pattern rate |
|---------|----------|--------------------------|
| Greedy (T=0) | **17.2%** | 18% of responses |
| Paper temp (T=1.0) | **57.2%** | 80% of responses |

**40-pp gap** on identical questions. The model's cybersecurity knowledge does not
change; what changes is whether the extractor can parse the response. Six unaffected
peer models show ≤3 pp difference between the two settings on the same questions.

```
Documented (paper):  temperature 1.0, top-p 0.9, top-k 50
Enacted in script:   no temperature, no top-p, no top-k, no seed
```

---

## Extraction and Evaluation Failures — F1(E), F2(E), F3(E), F4(E)

### F1(E) — Extractor Divergence (CyberMetric / Gemma-4-31B)

CyberMetric's extractor searches for `ANSWER: [A-D]` (a concrete letter).
Gemma-4-31B echoes the format placeholder literally, producing
`` `ANSWER: X' `` repeated — the literal character X, never a substituted letter.
The extractor finds no match in **95% of responses**.

**Impact:**

| Evaluator | Gemma-4-31B | Peer median (6 models) |
|-----------|-------------|----------------------|
| Standard | **4.0%** | **91.9%** |
| Paper-format | **4.8%** | ≤3 pp shift |

**~88 pp gap**; consistent near-zero under both evaluator variants confirms the
failure is in output style, not extractor permissiveness. Gemma scores 85.4% on
Athena-VSP and 26% on CTI-TAA, ruling out a general capability deficit.

---

### F2(E) — Denominator Inflation

CTI-Bench RCM, AthenaBench RCM, and SECURE sub-tasks report `correct / valid`
instead of `correct / total`. The policy is sound when failures are occasional and
random. It fails when a model's output style is systematically incompatible with
the extractor.

**Observed inflation:**

| Model | Task | Reported accuracy | Real accuracy | Inflation factor |
|-------|------|------------------|---------------|-----------------|
| Gemma-4-31B | CTI-RCM | **100.0%** (2/2 valid) | **0.2%** (2/1,000) | ×500 |
| Llama-Primus-Nemotron-70B | SECURE-CWET | **100.0%** (91/91 valid) | **9.4%** (91/964) | ×11 |
| Llama-Primus-Nemotron-70B | SECURE-MAET | **91.1%** (101/1,072 valid) | **8.6%** (92/1,072) | ~×11 |

The denominator policy converts an output-style incompatibility into an apparent
accuracy advantage.

---

### F3(E) — Cross-Benchmark Metric Inversion (CTI-VSP vs. Athena-VSP)

CTI-Bench and AthenaBench both implement CVSS v3.1 base-score prediction from CVE
descriptions, but under incompatible metric conventions:

- **CTI-Bench:** raw mean absolute deviation (MAD) — lower is better
- **AthenaBench:** `max(0, 1 − MAD/R) × 100` where R=7.7 — higher is better

Applying both to the same 10 models on the same responses produces rank inversions of
**up to 5 positions**.

| Model | CTI-VSP rank (MAD) | Athena-VSP rank (%) |
|-------|-------------------|---------------------|
| RedSage-Qwen3-8B-DPO | **3rd** (MAD=1.15) | **8th** (73.5%) |
| Llama-3.3-70B | **9th** | **6th** |

A researcher selecting CTI-Bench's metric would conclude RedSage outperforms
Llama-3.3 by six rank positions; one selecting AthenaBench's would conclude they
are roughly tied. The disagreement is entirely from the evaluation convention.

---

### F4(E) — Reasoning–Extraction Conflict

Many extractors anchor to the **final line** of the model response. When models
prefix answers with chain-of-thought, correct identifiers appear mid-response and
are discarded.

**Prompt-mode sensitivity on `athena_ate` (Claude Sonnet 4.6):**

| Mode | Accuracy |
|------|----------|
| Zero-shot | **49.0%** |
| Chain-of-thought | **89.0%** |
| Gap | **40 pp** |

**Mechanism example (CTI-ATE idx 2 / GPT-OSS-20B):**

```
Prompt (tail):  "…Extract all MITRE ATT&CK techniques…
                 Ensure the final line contains only the technique IDs, separated by commas."

Gold:  T1027, T1055, T1059, T1071, T1105, T1140, T1518

GPT-OSS-20B reasoning body (lines 10–18):
  "…downloader → T1105…  C2 via HTTP → T1071…
   command-line PE execution → T1059…  AES decryption → T1140…"
   (4 of 7 gold techniques correctly identified)

Final line (line 20):  T1567   ← model diverged in reasoning loop
Extracted:             T1567   ← extractor anchors to final line
Score:  0                      ← T1059, T1071, T1105, T1140 in body ignored
```

---

## Aggregation Failures — F1(A), F2(A), F3(A)

### F1(A) — Logprob vs. Generative Scoring

Logit-based selection scores the model's token-level log-probability over answer
choices, avoiding text extraction entirely. Generative evaluation requires the model
to produce a parseable response. The two modes can disagree substantially:

| Model / Task | Generative | Logprob | Gap |
|-------------|-----------|---------|-----|
| Qwen3.6-35B-A3B / MMLU-CS | higher | — | **23 pp** |
| Llama-Primus-Merged / MMLU-CS | lower | — | **17 pp** |
| Gemma-4-31B / RedSage | 45.7% | 86.6% | **+41 pp** (logprob) |
| Qwen3.6-35B-A3B / RedSage | 85.9% | 59.2% | **−27 pp** (generative) |

The direction of the gap is model-specific: Gemma's prose outputs defeat regex
extraction, so logprob wins; Qwen3.6's thinking architecture defeats the stop-sequence
assumption, so generative wins without the stop.

---

### F2(A) — TAA Metric Drift (CTI-TAA vs. Athena-TAA)

Threat Actor Attribution is implemented under two incompatible definitions:

- **CTI-Bench:** Correct + Plausible (alias-graph BFS over Malpedia / MITRE)
- **AthenaBench:** Binary correctness; no alias graph; plausible = zero

**Per-model gap on same responses:**

| Model | CTI score | Athena score | Gap |
|-------|----------|-------------|-----|
| Llama-3.3-70B | 70% | **0%** | 70 pp |
| GPT-OSS-20B | 64% | 5% | 59 pp |
| GPT-5.4 | 86% | 33% | 53 pp |
| Qwen3.6-35B-A3B | 62% | 14% | 48 pp |
| Claude Sonnet 4.6 | 94% | 47% | 47 pp |
| RedSage-Qwen3-8B-DPO | 20% | 21% | −1 pp ← only unaffected model |

A researcher selecting CTI's metric concludes TAA is tractable (best model 86–94%);
one selecting Athena's concludes it is unsolved (best model <50%). Identical outputs,
different conclusions.

---

### F3(A) — Aggregation Inconsistency

The label *accuracy* appears across benchmarks for quantities computed differently:

| Benchmark | Denominator | Notes |
|-----------|-------------|-------|
| SecEval | correct / total | |
| CTI-Bench RCM | correct / valid | excludes unparseable responses |

**Gemma-4-31B example:**

| Benchmark | Reported accuracy | Real accuracy (correct/total) |
|-----------|------------------|-------------------------------|
| CTI-RCM | **100.0%** (2/2 valid) | **0.2%** (2/1,000) |
| SecEval | **9.5%** | 9.5% (consistent denominator) |

Apparent cross-benchmark gap: **90.5 pp**
Real cross-benchmark gap: **9.3 pp**

The 90.5 pp apparent gap is attributable entirely to denominator policy. Both
benchmarks label their output "accuracy"; neither specifies which denominator was used.

---

## Summary: Impact by Failure Mode

| Code | Stage | Benchmark(s) | Max observed impact |
|------|-------|-------------|-------------------|
| F1(P) | Prompt | CyberMetric | ~90 pp (Foundation-Sec-8B) |
| F2(P) | Prompt | SecEval | 33% items answered incorrectly by design |
| F3(P) | Prompt | SECURE | **48–52 pp** gap (Primus-Merged) |
| F1(I) | Inference | RedSage | **86 pp** swing (Qwen3.6) |
| F2(I) | Inference | SecEval | **81 pp** swing (GPT-5.4) |
| F3(I) | Inference | CyberMetric | **40 pp** gap (Primus-Merged) |
| F1(E) | Extraction | CyberMetric | ~88 pp gap (Gemma-4-31B) |
| F2(E) | Extraction | CTI-RCM, SECURE | ×500 denominator inflation |
| F3(E) | Extraction | CTI-VSP, Athena-VSP | 5-rank inversion |
| F4(E) | Extraction | CTI-ATE, Athena-ATE | **40 pp** prompt-mode sensitivity |
| F1(A) | Aggregation | MMLU-CS, RedSage | **41 pp** logprob vs. generative |
| F2(A) | Aggregation | CTI-TAA, Athena-TAA | **70 pp** gap (Llama-3.3-70B) |
| F3(A) | Aggregation | CTI-RCM, SecEval | **90 pp** apparent gap from denominator |
