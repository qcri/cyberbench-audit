# Original-Pipeline Walkthrough Notebooks

A runnable tour of the empirical experiments behind
*"Benchmark Scores Are Pipeline-Dependent: A Reliability Audit of Cybersecurity LLM Benchmarks."*

Each notebook loads pre-computed outputs from the scripts in `../` and renders the evidence
for each failure mode without re-running inference. Light steps recompute instantly;
heavy steps (GPU / API) fall back to cached artifacts automatically.

---

## Notebook map

| Notebook | Stage | Failure modes covered | Paper section |
|---|---|---|---|
| [`00_overview.ipynb`](00_overview.ipynb) | — | Framing, model inventory, run order | §3 |
| [`01_results_table.ipynb`](01_results_table.ipynb) | 𝒜 Aggregation | Full 10-model × 25-task results; LaTeX Table 2 | §5 |
| [`02_failure_modes.ipynb`](02_failure_modes.ipynb) | 𝒫 𝓘 𝓔 𝒜 | 15 failure-mode headline numbers (Table 3) | §4 |
| [`03_prompt_sensitivity.ipynb`](03_prompt_sensitivity.ipynb) | 𝒫 𝓔 | F1(P)–F3(P) format leakage, conflict, template; F4(E) CoT/extraction conflict | §4.2, §4.4 |
| [`04_inference_config.ipynb`](04_inference_config.ipynb) | 𝓘 | F1(I) stop-sequence, F2(I) token-budget, F3(I) temperature drift | §4.3 |
| [`05_logprob_vs_generative.ipynb`](05_logprob_vs_generative.ipynb) | 𝒜 | F1(A) logprob vs. generative, F2(A) TAA metric drift | §4.5 |

---

## The measurement-pipeline framing

Every benchmark is modelled as a 5-stage pipeline:

```
Dataset 𝒟  ·  Prompt 𝒫  ·  Inference 𝓘  ·  Extraction/scoring 𝓔  ·  Aggregation 𝒜
```

A reported score `S_b(m)` is **conditional on all five stages**, not an intrinsic model
property. The 15 failure modes below quantify what happens when individual stages are varied
while keeping the model outputs fixed.

---

## Failure-mode summary (Table 3 from the paper)

| ID | Stage | Failure mode | Max observed impact |
|---|---|---|---|
| F1(D) | Dataset | Limited capability coverage | 4/8 benchmarks ≥97% knowledge-oriented |
| F2(D) | Dataset | Gold-label correctness | 23.9% confirmed label errors (manual audit) |
| F1(P) | Prompt | Format-token leakage | ~90 pp gap — Foundation-Sec-8B on CyberMetric |
| F2(P) | Prompt | Prompt–question conflict | Up to 33% of SecEval multi-answer items answered as single |
| F3(P) | Prompt | Template incompatibility | 48 pp gap — Primus-Merged on SECURE |
| F1(I) | Inference | Stop-sequence mismatch | **86 pp swing** — Qwen3.6-35B on RedSage-Bench |
| F2(I) | Inference | Token-budget filter | **81 pp recovery** — GPT-5.4 on SecEval |
| F3(I) | Inference | Temperature drift | 40 pp gap — Primus-Merged on CyberMetric |
| F1(E) | Extraction | Extractor divergence | ~88 pp gap — Gemma-4-31B on CyberMetric |
| F2(E) | Extraction | Denominator inflation | **×500** — Gemma-4-31B on CTI-RCM (100% → 0.2%) |
| F3(E) | Extraction | Metric-direction mismatch | Up to 5-rank inversion — CTI-VSP vs. Athena-VSP |
| F4(E) | Extraction | Reasoning–extraction conflict | 40 pp prompt-mode sensitivity — ATE with CoT |
| F1(A) | Aggregation | Logprob vs. generative scoring | 41 pp gap on identical items — RedSage |
| F2(A) | Aggregation | Task-level metric drift | 70 pp gap — Llama-3.3-70B on TAA |
| F3(A) | Aggregation | Aggregation inconsistency | 90 pp apparent cross-benchmark gap from denominator |

---

## Notebook details

### [`00_overview.ipynb`](00_overview.ipynb) — Framing and model inventory

Introduces the measurement-pipeline framing and lists the 10 evaluated models
with their task counts from `outputs_final/`. Shows the run order and explains
how the original pipeline notebooks relate to the unified-pipeline analysis.

**Models:** Claude Sonnet 4.6 · GPT-5.4 · Gemma-4-31B · Qwen3.6-35B ·
Llama-3.3-70B · GPT-OSS-20B · Primus-Nemotron-70B · Primus-Merged-8B ·
Foundation-Sec-8B · RedSage-Qwen3-8B-DPO

---

### [`01_results_table.ipynb`](01_results_table.ipynb) — Full results (paper Table 2)

Renders the complete 10-model × 25-task accuracy table across all 8 benchmarks
(MMLU-CS, SecEval, SECURE ×3, CTI-Bench ×5, AthenaBench ×6, CyberMetric,
RedSage-Bench ×5, SecBench). Includes:
- Best-per-task highlighting (bold)
- Artifact annotations (†) where scores are known to be pipeline-inflated
- API-model markers (✦) for GPT-5.4 and Claude Sonnet 4.6
- GPT-5.4 SecEval corrected score: **81.4%** (original 0.3% was F2(I))
- RedSage three-section split: generative · logprob · generative-EM

---

### [`02_failure_modes.ipynb`](02_failure_modes.ipynb) — 15 failure-mode headlines

One cell per headline failure mode from Table 3. Each cell computes the
key number directly from stored result files — no re-inference needed.

| Cell | Failure mode | What it shows |
|---|---|---|
| F1(P) | Format-token leakage | Foundation-Sec-8B CyberMetric vs. peer median (~90 pp gap) |
| F2(E) | Denominator inflation | All-model CTI-RCM table; Gemma ×500 (100% → 0.2%) |
| F3(I) | Temperature drift | Bar chart + table: paper T=1.0 vs. unified T=0 per model |
| F2(A)+F3(E) | TAA metric drift + VSP direction mismatch | CTI vs. Athena side-by-side |

**Key data point — F2(E):** Gemma-4-31B produced only 2 parseable responses out of
1,000 on CTI-RCM, both correct → **100% reported, 0.2% real, ×500 inflation**.
Primus-Nemotron-70B shows ×8 inflation on the same task.

---

### [`03_prompt_sensitivity.ipynb`](03_prompt_sensitivity.ipynb) — Prompt failures + CoT extraction

Covers the controlled prompt-sensitivity study from `run_prompt_sensitivity.py`:
100 seeded samples per task under zero-shot / 2-shot few-shot / chain-of-thought,
all other pipeline stages held fixed.

**F1(P) — Format-token leakage (CyberMetric / Foundation-Sec-8B)**
Foundation-Sec-8B copies the literal `` `ANSWER: X' `` placeholder rather than
filling it. 490/500 responses are empty strings. Reported accuracy: **1.0%**
vs. peer median **91.9%** on the same items.

**F2(P) — Prompt–question conflict (SecEval)**
System prompt says "select the correct **answers**" (plural); 1,262/2,189 items
use singular superlatives. Models that resolve in favour of the question wording
score zero on every multi-gold item.

| Model | Single-letter rate on 927 multi-gold items |
|---|---|
| Gemma-4-31B | 33.0% |
| RedSage-Qwen3-8B-DPO | 28.7% |
| Foundation-Sec-8B | 26.4% |
| Llama-3.3-70B | 23.7% |
| Claude Sonnet 4.6 | 10.8% |

**F3(P) — Template incompatibility (SECURE / Primus-Merged-8B)**
Primus-Merged generates explanation prose; extractor reads the first letter
character → **A** in 93.4% of SECURE-MAET responses regardless of gold answer.
Gap vs. next unaffected peer: **48 pp** on MAET, **52 pp** on CWET.

**F4(E) — Reasoning–extraction conflict (AthenaBench ATE)**
Standard extractor reads only the final line. Under CoT, correct ATT&CK
identifiers appear in the reasoning body but are not repeated at the end.
Gap on `athena_ate` for Claude Sonnet 4.6: **zero-shot 49.0% → CoT 89.0%
(+40 pp)** once a CoT-aware extractor is applied.

Four styled sensitivity tables (ZS / FS / CoT / Δ) show per-model, per-task scores.

---

### [`04_inference_config.ipynb`](04_inference_config.ipynb) — Inference failures

All three inference-stage failures are reproduced directly from stored result files
without any re-inference.

**F2(I) — Token-budget filter (SecEval / GPT-5.4)**
SecEval specifies `max_new_tokens=5`. The Azure OpenAI API enforces a minimum
of 16 tokens and rejected every GPT-5.4 request with HTTP 400. The evaluator
recorded error strings as responses — the failure is invisible in raw scores.

| Configuration | GPT-5.4 SecEval accuracy |
|---|---|
| `max_new_tokens=5` (original) | **0.3%** (7 accidental letter matches in error strings) |
| `max_tokens=16` (corrected) | **81.4%** |

**F1(I) — Stop-sequence mismatch (RedSage-Bench / Qwen3.6-35B-A3B)**
`stop=["\n"]` fires at the first newline of Qwen3's `<think>` block, before
any answer token is produced. All 30,000 responses are empty strings.
Removing the stop sequence recovers **85.9%** average across 5 sub-tasks.

| Sub-task | Score without stop |
|---|---|
| redsage_cli | 88.8% |
| redsage_frameworks | 83.7% |
| redsage_generals | 85.4% |
| redsage_kali | 81.4% |
| redsage_skills | 90.1% |

**F3(I) — Temperature drift (CyberMetric / Primus-Merged-8B)**
Paper prescribes T=1.0, top-p=0.9, top-k=50; none appear in the released
evaluator script. For Primus-Merged: **17.2% (T=0) → 57.2% (T=1.0), −40 pp**.
Six unaffected peer models show ≤3 pp difference between the two settings.

---

### [`05_logprob_vs_generative.ipynb`](05_logprob_vs_generative.ipynb) — Aggregation failures

**F1(A) — Logprob vs. generative scoring (RedSage-Bench)**
`run_redsage_lighteval.py` ran RedSage in two modes on the same items:
log-probability over choice tokens (LightEval default) and generative
exact-match. The gap is model-specific — neither method is uniformly preferable.

| Model | Generative | Logprob | Gap |
|---|---|---|---|
| Gemma-4-31B | 45.7% | **86.6%** | +41 pp (logprob) |
| Qwen3.6-35B-A3B | **85.9%** | 59.2% | −27 pp (generative) |

Gemma's prose outputs defeat regex extraction so logprob wins; Qwen3.6's
thinking architecture defeats the stop-sequence assumption so generative wins.

**F2(A) — TAA metric drift (CTI-TAA vs. Athena-TAA)**
Both benchmarks implement threat-actor attribution, but under incompatible rules:
CTI-Bench awards Correct+Plausible via alias-graph BFS; AthenaBench uses
strict binary correctness with no alias resolution.

| Model | CTI-TAA C+P | Athena-TAA | Gap |
|---|---|---|---|
| Llama-3.3-70B | 70% | 0% | **70 pp** |
| GPT-5.4 | 86% | 33% | 53 pp |
| Claude Sonnet 4.6 | 94% | 47% | 47 pp |
| RedSage-Qwen3-8B-DPO | 20% | 21% | −1 pp (only unaffected model) |

---

## Scripts that generate the outputs

The notebooks load from pre-computed files written by these scripts in `../`:

| Script | What it does | Failure modes it exposes |
|---|---|---|
| `run_inference_benchmarks.py` | Benchmark-faithful response collection (all 24 sub-tasks) | baseline for all stages |
| `evaluate.py` | Combined inference + regex evaluation (original benchmark behavior) | F1(E), F2(E) |
| `run_evaluate_llm_judge.py` | LLM-as-judge evaluation on pre-collected JSONL | F1(E) extractor divergence |
| `run_prompt_sensitivity.py` | Zero-shot / few-shot / CoT controlled study | F1(P)–F3(P), F4(E) |
| `run_redsage_lighteval.py` | RedSage MCQ via LightEval (logprob + generative) | F1(A) |

Data outputs land in `outputs_final/` (canonical 10-model result tree) and
`outputs/` (sensitivity runs, lighteval results). Notebooks load from both.

---

## Run

```bash
# Install deps (once)
pip install -r requirements-notebooks.txt

# Interactive
jupyter lab   # open any notebook, run top-to-bottom
```

- **Light steps** (loading pre-collected outputs) recompute instantly from
  `outputs_final/`; if central storage is not mounted they fall back with a note.
- **Heavy steps** (GPU inference, API calls) are **off by default** — cached
  artifacts load automatically. To recompute:
  ```bash
  SAYF_NB_REGEN=1 jupyter lab   # needs GPU/local model and/or API key
  ```

### Headless execution

```bash
./run_all.sh   # nbconvert --execute over all six notebooks, SAYF_NB_REGEN=0
```

Or via SLURM (used internally):

```bash
sbatch ../execute_notebooks.sh
```

---

## Editing

Notebooks are generated from [`build.py`](build.py) — the editable source of all
cell content. After changing it:

```bash
python build.py   # re-emits all six .ipynb files
```

Shared helpers (`OUTPUTS_FINAL`, `PAPER_MODELS`, `run_mod`, `show_*`, `heavy`)
live in [`nbtools.py`](nbtools.py).

---

## Relationship to the unified-pipeline notebooks

| | Original pipeline (`this directory`) | Unified pipeline (`../../test_unifiednotebooks/`) |
|---|---|---|
| Purpose | Reproduce each benchmark's published evaluation logic | Standardise extraction, denominator, metric, token budget |
| Data source | `outputs_final/` per-model result trees | Same + sensitivity JSONL |
| Failure modes | Shows **what breaks** in the original pipelines | Shows **what changes** under standardisation |
| Rank shifts | Baseline rankings (Table 2) | Post-standardisation shifts (Table 5) |

Running the same model outputs through both pipelines is how the rank shifts in
Table 5 of the paper are measured. Under standardisation, **7 of 10 models shift
by at least 3 ranks on at least one benchmark**.
