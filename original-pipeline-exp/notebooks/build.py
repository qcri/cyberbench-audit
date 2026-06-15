"""Generate the original-pipeline walkthrough notebooks from the cell content
defined here.  Run:  python build.py

Design rule (matches the unified-analysis notebook vibe):
  - Notebooks are **thin** — they call existing scripts via ``run_mod`` and
    display cached artifacts with ``show_*``.  No analysis logic lives here.
  - Each topic is one markdown cell (why) + one code cell (run / show).
  - Heavy steps (GPU / API) are gated by ``heavy()``; light steps always run.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETUP = """\
import nbtools as nb
from nbtools import (
    show_df, show_fig, show_md, show_tex,
    run_mod, run_live, heavy,
    REGEN, OUTPUTS_FINAL, OUTPUTS_DIR, PIPELINE_ROOT,
)
import pandas as pd
from IPython.display import display, HTML
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.1f}'.format)

# The 10 models evaluated in the paper (Table 2).
PAPER_MODELS = {
    'Claude Sonnet 4.6':   'claude-sonnet-4-6-cyberxpert',
    'GPT-5.4':             'gpt-5.4',
    'Gemma-4-31B':         'gemma-4-31B-it',
    'Qwen3.6-35B':         'Qwen3.6-35B-A3B',
    'Llama-3.3-70B':       'Llama-3.3-70B-Instruct',
    'GPT-OSS-20B':         'gpt-oss-20b',
    'Primus-Nemotron-70B': 'Llama-Primus-Nemotron-70B',
    'Primus-Merged-8B':    'Llama-Primus-Merged',
    'Foundation-Sec-8B':   'Foundation-Sec-8B-Instruct',
    'RedSage-Qwen3-8B':    'RedSage-Qwen3-8B-DPO',
}

print(f"Data root: {OUTPUTS_FINAL.name}/  — {'OK' if OUTPUTS_FINAL.exists() else 'MISSING'}")\
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


def setup_cell():
    return code(SETUP)


# ─────────────────────────────── 00 — overview ───────────────────────────────
NB00 = [
    md("""\
# Original-pipeline walkthrough

A guided, runnable tour of the empirical experiments behind *"Benchmark Scores
Are Pipeline-Dependent: A Reliability Audit of Cybersecurity LLM Benchmarks."*

Each notebook calls the existing analysis scripts and renders pre-computed
outputs. No logic is duplicated here — the scripts are the source of truth.

## How to use
- Run cells top-to-bottom. Light steps (loading results, plotting) recompute
  instantly from `outputs_final/`. If the central storage isn't mounted they
  fall back gracefully with a note.
- **Heavy steps are off by default.** GPU inference and API calls are pre-computed;
  stored outputs load instantly.

## The measurement-pipeline framing

Every benchmark is modelled as a 5-stage pipeline:

```
Dataset 𝒟  ·  Prompt 𝒫  ·  Inference 𝓘  ·  Extraction/scoring 𝓔  ·  Aggregation 𝒜
```

A reported score is conditional on *all five stages*, not an intrinsic model
property. The experiments below hold 𝒟 fixed and vary each other stage.

## Notebook map
| Notebook | Theme | Paper section |
|---|---|---|
| `01_results_table` | Full 10-model × 25-task results (paper Table 2) + LaTeX table | §5 |
| `02_failure_modes` | 15 failure-mode headline numbers | §4 |
| `03_prompt_sensitivity` | Zero-shot vs few-shot vs CoT across 10 models (paper Table 2) | §4.4 F4(E) |
| `04_inference_config` | Stop-sequence, token-budget, temperature | §4.3 F1–3(I) |
| `05_logprob_vs_generative` | RedSage logprob vs generative; TAA drift | §4.5 F1–3(A) |

## How results were collected

**Inference** — `run_inference_benchmarks.py` collected model responses for all 10 models
across 8 benchmarks (23 tasks). Open-weight models were served locally via vLLM;
proprietary models (GPT-5.4, Claude Sonnet 4.6) were queried through their APIs.

**Evaluation** — `evaluate.py` applied benchmark-faithful scoring to collected responses
(task-specific extractors, exact match, CWE regex, CVSS MAD, etc.), producing
`{task}_result.json` and `{task}_detail.jsonl` per model.
`run_evaluate_llm_judge.py` ran the unified LLM-as-judge extraction for open-ended tasks
(ATE, TAA, RCM) where regex-based extraction is insufficient.

**Prompt sensitivity** — `run_prompt_sensitivity.py` re-ran 100 seeded samples per task
under zero-shot, 2-shot few-shot, and chain-of-thought prompting for all 10 models.

**RedSage logprob** — `run_redsage_lighteval.py` ran the official LightEval logprob
and generative exact-match evaluation on RedSage-Bench.

All outputs are pre-computed and stored in the repository — notebooks load directly
from stored files, no inference or API calls required.
"""),
    setup_cell(),
    md("### Evaluated models"),
    code("""\
PAPER_DIR_NAMES = set(PAPER_MODELS.values())
if OUTPUTS_FINAL.exists():
    rows = []
    for label, dir_name in PAPER_MODELS.items():
        p = OUTPUTS_FINAL / dir_name
        n_tasks = len(list((p / 'eval').glob('*_result.json'))) if (p / 'eval').exists() else 0
        rows.append({'Model': label, 'Tasks evaluated': n_tasks})
    inv = pd.DataFrame(rows)
    display(inv.style
            .hide(axis='index')
            .set_caption('Models evaluated in the paper — task count from outputs_final/.'))
else:
    print("[missing] Result files not found.")\
"""),
]



# ──────────────────────────── 01 — results table ─────────────────────────────
NB01 = [
    md("""\
# 01 · Master results table

Per-(model, task) accuracy across the 10 models and 25 tasks reported in the paper (Table 2).
All scores are read from pre-computed evaluation files.
"""),
    setup_cell(),
    md("### Results table — 10 models × 25 tasks (paper Table 2)"),
    code("""\
import sys, json, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, str(PIPELINE_ROOT))
from load_results import load_all_scores

all_scores = load_all_scores(OUTPUTS_FINAL)

# Paper model order
MODEL_ORDER = [
    ('Claude Sonnet 4.6',  'claude-sonnet-4-6-cyberxpert'),
    ('GPT-5.4',            'gpt-5.4'),
    ('Llama-3.3-70B',      'Llama-3.3-70B-Instruct'),
    ('Qwen3.6-35B',        'Qwen3.6-35B-A3B'),
    ('Primus-Nemotron-70B','Llama-Primus-Nemotron-70B'),
    ('RedSage-Qwen3-8B',   'RedSage-Qwen3-8B-DPO'),
    ('Foundation-Sec-8B',  'Foundation-Sec-8B-Instruct'),
    ('GPT-OSS-20B',        'gpt-oss-20b'),
    ('Primus-Merged',      'Llama-Primus-Merged'),
    ('Gemma-4-31B',        'gemma-4-31B-it'),
]

# GPT-5.4 SecEval: use corrected score (81.4%), not the broken 0.3%
from pathlib import Path
import re as _re
rerun = OUTPUTS_DIR / 'eval_results' / 'gpt-5.4_seceval_rerun' / 'seceval_result.json'
if rerun.exists():
    all_scores['gpt-5.4']['seceval'] = json.loads(rerun.read_text()).get('primary_score')

scores = {label: dict(all_scores.get(d, {})) for label, d in MODEL_ORDER}

# Load LightEval logprob + generative EM scores
LP_MAP  = {'cybersecurity_tools_cli':'lp_cli','cybersecurity_knowledge_frameworks':'lp_fw',
            'cybersecurity_knowledge_generals':'lp_gen','cybersecurity_tools_kali':'lp_kali',
            'cybersecurity_skills':'lp_skills','_average':'lp_avg'}
GEM_MAP = {'cybersecurity_tools_cli':'gem_cli','cybersecurity_knowledge_frameworks':'gem_fw',
            'cybersecurity_knowledge_generals':'gem_gen','cybersecurity_tools_kali':'gem_kali',
            'cybersecurity_skills':'gem_skills','_average':'gem_avg'}
for label, dir_name in MODEL_ORDER:
    le_dir = OUTPUTS_FINAL / dir_name / 'lighteval'
    for fname, kmap in [('redsage_logprob_result.json', LP_MAP),
                        ('redsage_generative_result.json', GEM_MAP)]:
        fpath = le_dir / fname
        if not fpath.exists(): continue
        for rkey, vals in json.loads(fpath.read_text()).get('results', {}).items():
            m = _re.search(':([^|]+)[|]', rkey)
            if m and m.group(1) in kmap:
                scores[label][kmap[m.group(1)]] = round(vals.get('acc', 0) * 100, 1)

# Task groups: (task_key, display_label, metric_note)
TASK_GROUPS = [
    ('CTI-Bench', [
        ('mcq',     'MCQ',     'Acc %'),
        ('rcm',     'RCM',     'Acc %'),
        ('vsp',     'VSP',     'MAD ↓'),
        ('ate',     'ATE',     'Acc %'),
        ('cti_taa', 'TAA',     'C+P %'),
    ]),
    ('AthenaBench', [
        ('ckt',        'CKT', 'Acc %'),
        ('rms',        'RMS', 'F1 %'),
        ('taa',        'TAA', 'Acc %'),
        ('athena_ate', 'ATE', 'Acc %'),
        ('athena_rcm', 'RCM', 'Acc %'),
        ('athena_vsp', 'VSP', 'MAD-norm %'),
    ]),
    ('SECURE', [
        ('secure_maet', 'MAET', 'Acc %'),
        ('secure_cwet', 'CWET', 'Acc %'),
        ('secure_kcv',  'KCV',  'Acc %'),
    ]),
    ('General', [
        ('seceval',          'SecEval',        'Acc %'),
        ('cybermetric',      'CyberMetric det','Acc %'),
        ('cybermetric_paper','CyberMetric samp','Acc %'),
        ('mmlu-cs',          'MMLU-CS gen',    'Acc %'),
        ('mmlu-cs-logprobs', 'MMLU-CS logp',   'Acc %'),
        ('secbench',         'SecBench',        'Acc %'),
    ]),
    ('RedSage-Bench (generative)', [
        ('redsage_cli',        'CLI',    'Acc %'),
        ('redsage_frameworks', 'FW',     'Acc %'),
        ('redsage_generals',   'GEN',    'Acc %'),
        ('redsage_kali',       'Kali',   'Acc %'),
        ('redsage_skills',     'Skills', 'Acc %'),
    ]),
    ('RedSage-Bench (logprob — official)', [
        ('lp_cli',    'CLI',   'Acc %'),
        ('lp_fw',     'FW',    'Acc %'),
        ('lp_gen',    'GEN',   'Acc %'),
        ('lp_kali',   'Kali',  'Acc %'),
        ('lp_skills', 'Skills','Acc %'),
        ('lp_avg',    'Avg.',  'Acc %'),
    ]),
    ('RedSage-Bench (generative EM)', [
        ('gem_cli',    'CLI',   'Acc %'),
        ('gem_fw',     'FW',    'Acc %'),
        ('gem_gen',    'GEN',   'Acc %'),
        ('gem_kali',   'Kali',  'Acc %'),
        ('gem_skills', 'Skills','Acc %'),
        ('gem_avg',    'Avg.',  'Acc %'),
    ]),
]

LOWER_BETTER = {'vsp'}
ARTIFACTS = {('Gemma-4-31B','rcm'), ('Primus-Nemotron-70B','secure_maet'),
             ('Primus-Nemotron-70B','secure_cwet'), ('Primus-Nemotron-70B','secure_kcv')}
API_MODELS = {'Claude Sonnet 4.6', 'GPT-5.4'}

model_labels = [m for m, _ in MODEL_ORDER]

# Build two parallel structures: numeric values + display strings
index_tuples, num_rows, str_rows, best_mask, artifact_mask = [], [], [], [], []
for bench, tasks in TASK_GROUPS:
    for task_key, task_name, metric in tasks:
        index_tuples.append((bench, f'{task_name}  [{metric}]'))
        row_vals = [scores[m].get(task_key) for m in model_labels]
        valid = [v for v in row_vals if v is not None]
        best  = (min(valid) if task_key in LOWER_BETTER else max(valid)) if valid else None
        prec  = 2 if task_key in LOWER_BETTER else 1
        num_rows.append(row_vals)
        str_rows.append([f'{v:.{prec}f}' if v is not None else '—' for v in row_vals])
        best_mask.append([v is not None and best is not None and abs(v - best) < 0.05
                          for v in row_vals])
        artifact_mask.append([(m, task_key) in ARTIFACTS for m in model_labels])

idx = pd.MultiIndex.from_tuples(index_tuples, names=['Benchmark', 'Task [Metric]'])
col_labels = [f'{c} ✦' if c in API_MODELS else c for c in model_labels]

# Display DataFrame with artifact markers baked into values
display_vals = []
for str_row, art_row in zip(str_rows, artifact_mask):
    display_vals.append([f'{s} †' if a and s != '—' else s
                         for s, a in zip(str_row, art_row)])

df_display = pd.DataFrame(display_vals, index=idx, columns=col_labels)

# Apply bold CSS to best-per-row cells
import numpy as np
best_arr = np.array(best_mask)

def bold_best(data):
    styles = pd.DataFrame('', index=data.index, columns=data.columns)
    for i in range(len(best_arr)):
        for j in range(len(best_arr[i])):
            if best_arr[i, j]:
                styles.iloc[i, j] = 'font-weight: bold'
    return styles

styled = df_display.style.apply(bold_best, axis=None)
print("bold = best per task  |  † = metric artifact  |  ✦ = closed/API model")
print("GPT-5.4 SecEval: corrected score 81.4% (original 0.3% was F2(I) token-budget failure)")
display(styled)\
"""),
    md("### Per-benchmark average scores — model overview"),
    code("""\
import numpy as np

# One representative score per benchmark per model (mean of accuracy tasks within each)
# Uses unified harness scores; VSP excluded (MAD scale); CyberMetric uses T=0 (unified)
BENCH_TASKS = {
    'CTI-Bench':   ['mcq', 'rcm', 'ate', 'cti_taa'],
    'AthenaBench': ['ckt', 'taa', 'athena_ate', 'athena_rcm', 'athena_vsp'],
    'SECURE':      ['secure_maet', 'secure_cwet', 'secure_kcv'],
    'SecEval':     ['seceval'],
    'CyberMetric': ['cybermetric'],
    'MMLU-CS':     ['mmlu-cs'],
    'SecBench':    ['secbench'],
    'RedSage':     ['redsage_cli','redsage_frameworks','redsage_generals',
                    'redsage_kali','redsage_skills'],
}

avg_rows = []
for label, _ in MODEL_ORDER:
    row = {'Model': label}
    for bench, tasks in BENCH_TASKS.items():
        vals = [scores[label].get(t) for t in tasks if scores[label].get(t) is not None]
        row[bench] = round(np.mean(vals), 1) if vals else None
    avg_rows.append(row)

avg_df = pd.DataFrame(avg_rows).set_index('Model').apply(pd.to_numeric, errors='coerce')

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(avg_df, cmap='YlOrRd', annot=True, fmt='.0f',
            annot_kws={'size': 10}, linewidths=0.5, vmin=0, vmax=100, ax=ax,
            linecolor='white')
ax.set_title('Average accuracy (%) per benchmark — 10 models (paper Table 2)', fontsize=13, pad=12)
ax.set_xlabel('Benchmark', fontsize=10)
ax.set_ylabel('')
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.yticks(fontsize=9)
plt.tight_layout(); plt.show()
print("Note: Per-benchmark averages over accuracy tasks. VSP excluded (MAD scale).")
print("CyberMetric = unified T=0. GPT-5.4 SecEval = corrected score (81.4%).")\
"""),
    md("### Known metric artifacts — denominator inflation"),
    code("""\
from load_results import real_score

ARTIFACTS = {
    ('Gemma-31B',  'rcm'):         '998/1000 invalid → reported 100%, real 0.2% (×500)',
    ('Nemo-70B',   'secure_maet'): '971/1072 empty → MAET inflated',
    ('Nemo-70B',   'secure_cwet'): '873/964 empty → reported 100%, real 9.4% (×11)',
}
art_rows = []
for (m, t), note in ARTIFACTS.items():
    art, real = real_score(m, t)
    art_rows.append({
        'Model': m, 'Task': t,
        'Reported (correct/valid)': f"{art:.1f}%" if art else '—',
        'Real (correct/total)':     f"{real:.1f}%" if real else '—',
        'Note': note,
    })
art_df = pd.DataFrame(art_rows)
print(art_df.to_string(index=False))\
"""),
    md("### LaTeX results table — `gen_table.py`\nThe `tab:main_results` block used in the paper. Note: `gen_table.py` reads all 12 entries in `outputs_final/` (including Fanar and Qwen3-30B) as written by the paper authors; the caption reflects the full 12-model run used in the paper."),
    code("""\
import subprocess, sys
result = subprocess.run(
    [sys.executable, str(PIPELINE_ROOT / 'gen_table.py')],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(result.stdout)
else:
    print("[error]", result.stderr[:500])\
"""),
]


# ──────────────────────────── 02 — failure modes ─────────────────────────────
NB02 = [
    md("""\
# 02 · Benchmark-Level Failure Modes (Table 3)

15 recurring failure modes across the 5-stage measurement pipeline, matching
the paper's Table 3 exactly. Each number is computed from stored result files.

| Stage | ID | Failure mode | Observed impact |
|---|---|---|---|
| **𝒟 Dataset** | F1(D) | Limited capability coverage | 4/8 benchmarks ≥97% knowledge-oriented |
| | F2(D) | Gold-label correctness | 23.9% confirmed label errors (manual audit) |
| **𝒫 Prompt** | F1(P) | Format-token leakage | ~90pp gap from answer-template copying |
| | F2(P) | Prompt–question conflict | Up to 33% of multi-answer items answered as single-answer |
| | F3(P) | Template incompatibility | 48pp gap from chat-template mismatch |
| **𝓘 Inference** | F1(I) | Stop-sequence mismatch | 86pp swing from one stop-sequence parameter |
| | F2(I) | Token-budget filter | 81pp recovery after increasing token budget |
| | F3(I) | Temperature drift | 40pp gap between documented and enacted decoding |
| **𝓔 Extraction** | F1(E) | Extractor divergence | ~88pp gap from regex/output mismatch |
| | F2(E) | Denominator inflation | Up to ×500 inflation under valid-only scoring |
| | F3(E) | Metric-direction mismatch | Up to five-rank inversion on related scoring tasks |
| | F4(E) | Reasoning–extraction conflict | Up to 40pp prompt-mode sensitivity |
| **𝒜 Aggregation** | F1(A) | Logprob vs generative scoring | Up to 41pp gap on identical items |
| | F2(A) | Task-level metric drift | Up to 70pp gap from partial-credit rules |
| | F3(A) | Aggregation inconsistency | Up to 100pp discrepancy under mixed denominators |

*F1(D) and F2(D) required LLM-judge classification and manual label audit — not reproducible from stored result files.*
"""),
    setup_cell(),
    md("""\
### F1(P) — Format-token leakage (CyberMetric)

Foundation-Sec-8B scores ~1% on CyberMetric; peer median is ~89%.
The gap (~88pp) arises because the `ANSWER: X` prompt template is copied
literally by the model, producing unparseable outputs.
"""),
    code("""\
import json, numpy as np
from load_results import load_all_scores, MODEL_DIRS

scores = load_all_scores(OUTPUTS_FINAL)
# Get CyberMetric unified judge scores for all paper models
cyber = {label: scores.get(dir_name, {}).get('cybermetric')
         for label, dir_name in PAPER_MODELS.items()}
cyber = {k: v for k, v in cyber.items() if v is not None}

affected = cyber.get('Foundation-Sec-8B')
peers    = [v for k, v in cyber.items() if k != 'Foundation-Sec-8B']
peer_med = round(float(np.median(peers)), 1)

f1p_df = pd.DataFrame([
    {'Model': 'Foundation-Sec-8B', 'CyberMetric (%)': affected,  'Role': 'Affected'},
    {'Model': 'Peer median',       'CyberMetric (%)': peer_med,  'Role': 'Peer'},
    {'Model': 'Gap',               'CyberMetric (%)': peer_med - affected, 'Role': 'Gap (pp)'},
])
display(f1p_df.style
        .hide(axis='index')
        .format({'CyberMetric (%)': '{:.1f}'})
        .set_caption('F1(P): Format-token leakage — Foundation-Sec-8B copies the ANSWER: X template literally (paper reports ~90pp gap).'))\
"""),
    md("""\
### F2(E) — Denominator inflation (CTI-Bench RCM)

The original evaluator uses `correct / valid` (valid = parseable responses).
One model produces only 2 parseable outputs out of 1,000 — both correct —
yielding 100% under valid-only scoring vs 0.2% under correct/total (×500).
"""),
    code("""\
from load_results import real_score, inflation_ratio

rows = []
for label, dir_name in PAPER_MODELS.items():
    art, real = real_score(dir_name, 'rcm')
    if art is not None:
        infl = inflation_ratio(dir_name, 'rcm')
        rows.append({'Model': label,
                     'Reported correct/valid (%)': round(art, 1),
                     'Real correct/total (%)':     round(real, 1),
                     'Inflation':                  f'×{infl}' if infl and infl > 1 else '×1'})

denom_df = pd.DataFrame(rows).sort_values('Reported correct/valid (%)', ascending=False).reset_index(drop=True)
display(denom_df.style
        .hide(axis='index')
        .format({'Reported correct/valid (%)': '{:.1f}%', 'Real correct/total (%)': '{:.1f}%'})
        .map(lambda v: 'color:red;font-weight:bold' if isinstance(v, str) and v.startswith('×') and v != '×1' else '')
        .set_caption('F2(E): Denominator inflation — CTI-Bench RCM reported correct/valid vs true correct/total. Red = inflated denominator.'))\
"""),
    md("""\
### F3(I) — Temperature drift (CyberMetric)

CyberMetric's paper prescribes T=1.0, top-p=0.9; the released evaluator
uses T=0 (greedy). For Primus-Merged-8B this produces a **40pp gap**.
Both runs stored: `cybermetric_paper_result.json` (T=1.0) vs
`cybermetric_result.json` (T=0).
"""),
    code("""\
from load_results import load_all_scores
import matplotlib.pyplot as plt, numpy as np

scores = load_all_scores(OUTPUTS_FINAL)
rows = []
for label, dir_name in PAPER_MODELS.items():
    p = scores.get(dir_name, {}).get('cybermetric_paper')
    u = scores.get(dir_name, {}).get('cybermetric')
    if p is not None and u is not None:
        rows.append({'Model': label,
                     'Paper T=1.0 (%)': round(p, 1),
                     'Unified T=0 (%)': round(u, 1),
                     'Δ (pp)':          round(u - p, 1)})

temp_df = pd.DataFrame(rows).sort_values('Δ (pp)').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(temp_df)); w = 0.35
ax.bar(x - w/2, temp_df['Paper T=1.0 (%)'].values, w, label='Paper T=1.0', color='#4C72B0')
ax.bar(x + w/2, temp_df['Unified T=0 (%)'].values, w, label='Unified T=0', color='#DD8452')
ax.set_xticks(x)
ax.set_xticklabels(temp_df['Model'].values, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('CyberMetric accuracy (%)')
ax.set_title('F3(I): Temperature drift — paper T=1.0 vs unified T=0')
ax.legend(); plt.tight_layout(); plt.show()
display(temp_df.style
        .hide(axis='index')
        .format({'Paper T=1.0 (%)': '{:.1f}%', 'Unified T=0 (%)': '{:.1f}%', 'Δ (pp)': '{:+.1f}'})
        .map(lambda v: 'color:red;font-weight:bold' if isinstance(v,(int,float)) and v <= -10 else '', subset=['Δ (pp)'])
        .set_caption('F3(I): Temperature drift — paper T=1.0 vs unified T=0. Red = gap ≥10pp.'))\
"""),
    md("""\
### F2(A) + F3(E) — TAA metric drift and VSP direction mismatch

**F2(A):** CTI-Bench TAA awards Correct+Plausible via alias-graph BFS;
AthenaBench TAA uses binary correctness. Same models, opposite conventions.

**F3(E):** CTI-Bench VSP reports raw MAD (lower=better); AthenaBench VSP
normalises to a percentage (higher=better). Applied to same CVSS vectors.
"""),
    code("""\
from load_results import load_all_scores

scores = load_all_scores(OUTPUTS_FINAL)

# F2(A): TAA drift
taa_rows = []
for label, dir_name in PAPER_MODELS.items():
    cti = scores.get(dir_name, {}).get('cti_taa')
    ath = scores.get(dir_name, {}).get('taa')
    if cti is not None and ath is not None:
        taa_rows.append({'Model': label, 'CTI-TAA C+P (%)': round(cti,1),
                         'Athena-TAA (%)': round(ath,1), 'Gap (pp)': round(cti-ath,1)})

taa_df = pd.DataFrame(taa_rows).sort_values('Gap (pp)', ascending=False).reset_index(drop=True)
display(taa_df.style
        .hide(axis='index')
        .format({'CTI-TAA C+P (%)': '{:.1f}%', 'Athena-TAA (%)': '{:.1f}%', 'Gap (pp)': '{:+.1f}'})
        .map(lambda v: 'color:red;font-weight:bold' if isinstance(v,(int,float)) and v > 20 else '', subset=['Gap (pp)'])
        .set_caption('F2(A): TAA metric drift — CTI-Bench C+P partial credit vs AthenaBench binary. Red gap ≥20pp.'))

# F3(E): VSP direction mismatch
vsp_rows = []
for label, dir_name in PAPER_MODELS.items():
    cti = scores.get(dir_name, {}).get('vsp')
    ath = scores.get(dir_name, {}).get('athena_vsp')
    if cti is not None and ath is not None:
        vsp_rows.append({'Model': label, 'CTI-VSP MAD (↓)': round(cti,2),
                         'Athena-VSP % (↑)': round(ath,1)})

vsp_df = pd.DataFrame(vsp_rows).reset_index(drop=True)
display(vsp_df.style
        .hide(axis='index')
        .format({'CTI-VSP MAD (↓)': '{:.2f}', 'Athena-VSP % (↑)': '{:.1f}%'})
        .set_caption('F3(E): VSP metric-direction mismatch — CTI reports raw MAD (↓ better); Athena normalises to % (↑ better).'))\
"""),
]


# ──────────────────────── 03 — prompt sensitivity ────────────────────────────
NB03 = [
    md("""\
# 03 · Prompt sensitivity — $\\mathcal{F}(\\mathcal{P})$ + $\\mathcal{F}_4(\\mathcal{E})$

`run_prompt_sensitivity.py` ran a controlled study across three prompt modes
(zero-shot, 2-shot few-shot, chain-of-thought) on 100 seeded samples per task,
holding all other pipeline components fixed.

Aggregated scores are in `outputs/sensitivity_eval/scores.json`
(10 models × 24 tasks × 3 modes). Per-model response JSONL files live in
`outputs/sensitivity_{model}/` for the per-response prompt-failure analysis.
"""),
    setup_cell(),
    md("### Load aggregated sensitivity scores"),
    code("""\
import json, pandas as pd

SCORES_FILE = OUTPUTS_DIR / 'sensitivity_eval' / 'scores.json'
with open(SCORES_FILE) as f:
    raw = json.load(f)

# Only the 10 paper models — fanar2_27b is in the data but not in the paper
MODEL_LABELS = {
    'claude_sonnet_4_6':   'Claude Sonnet 4.6',
    'gpt_5_4':             'GPT-5.4',
    'gemma4_31b':          'Gemma-4-31B',
    'qwen3_35b':           'Qwen3.6-35B',
    'llama33_70b':         'Llama-3.3-70B',
    'gpt_oss_20b':         'GPT-OSS-20B',
    'nemotron_70b':        'Primus-Nemotron-70B',
    'llama_primus_merged': 'Primus-Merged-8B',
    'foundation_sec_8b':   'Foundation-Sec-8B',
    'redsage_qwen3_8b':    'RedSage-Qwen3-8B',
}
records = []
for model, tasks in raw.items():
    if model not in MODEL_LABELS: continue  # skip Fanar and any non-paper models
    if not isinstance(tasks, dict): continue
    for task, modes in tasks.items():
        if not isinstance(modes, dict): continue
        records.append({'model': MODEL_LABELS[model], 'task': task,
                        'zero_shot': modes.get('zero_shot'),
                        'few_shot':  modes.get('few_shot'),
                        'cot':       modes.get('cot')})
df = pd.DataFrame(records).dropna()
df['cot_delta'] = df['cot'] - df['zero_shot']
print(f"{df['model'].nunique()} models  ×  {df['task'].nunique()} tasks")\
"""),
    md("### CoT vs zero-shot delta heatmap — F4(E) signature"),
    code("""\
import matplotlib.pyplot as plt, seaborn as sns

pivot = df.pivot_table(index='model', columns='task', values='cot_delta')
pivot = pivot.reindex(sorted(pivot.columns), axis=1)

fig, ax = plt.subplots(figsize=(22, 6))
sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-70, vmax=70,
            annot=True, fmt='.0f', annot_kws={'size': 7},
            linewidths=0.3, ax=ax)
ax.set_title('CoT − zero-shot delta (pp). Red = CoT hurts. Blue = CoT helps.')
ax.set_xlabel(''); ax.set_ylabel('')
plt.xticks(rotation=40, ha='right', fontsize=8)
plt.tight_layout(); plt.show()\
"""),
    md("""\
### F4(E) case studies — CoT effect by task type

**CoT hurts** on `ckt` and `secbench`: the MCQ extractor (`evaluate.py`) reads only the
final line for a letter answer (A/B/C/D). Under CoT, models often place the answer letter
inside the reasoning body — the final line contains a conclusion sentence, not the letter,
so the extractor scores it wrong.

**CoT helps** on `ate` and `athena_ate`: the ATE extractor (`ctibench_format_ate` in
`evaluate.py`) tries the final line first but **falls back to a full-text scan** for MITRE
T-IDs if the last line is empty. Under CoT, models mention technique IDs throughout the
reasoning trace — the full-text fallback finds them even when the last line lacks IDs.

The same CoT output scores differently depending purely on which extractor is applied.
"""),
    code("""\
import matplotlib.pyplot as plt
import numpy as np

focus  = ['ckt', 'ate', 'athena_ate', 'secbench', 'rcm', 'rcm_2021']
modes  = ['zero_shot', 'few_shot', 'cot']
mlabels = ['ZS', 'FS', 'CoT']
colors  = plt.cm.tab10.colors

fig, axes = plt.subplots(1, len(focus), figsize=(20, 5))
for ax, task in zip(axes, focus):
    tdf = df[df['task'] == task].reset_index(drop=True)
    for i, row in tdf.iterrows():
        ax.plot(mlabels, [row[m] for m in modes],
                marker='o', color=colors[i % 10],
                label=row['model'], lw=1.5, ms=4)
    ax.set_title(task, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)' if task == focus[0] else '')
    ax.grid(axis='y', alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8,
           bbox_to_anchor=(0.5, -0.14))
fig.suptitle('F4(E): CoT hurts tasks with positional extractors (ckt, secbench);'
             ' helps tasks with LLM judge (ate, athena_ate)', fontsize=10)
plt.tight_layout(); plt.show()\
"""),
    md("### F2(P): Prompt–question conflict — single-letter rate on SecEval  *(light · response JSONL)*"),
    code("""\
import json, re

for short, full_dir in [
    ('foundation_sec_8b', 'sensitivity_foundation_sec_8b'),
    ('llama33_70b',       'sensitivity_llama33_70b'),
    ('claude_sonnet_4_6', 'sensitivity_claude_sonnet_4_6'),
]:
    jsonl = OUTPUTS_DIR / full_dir / 'seceval_zero_shot_responses.jsonl'
    if jsonl.exists():
        rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        single = sum(1 for r in rows
                     if len(set(re.findall(r'[A-D]', str(r.get('model_response',''))))) == 1)
        label = MODEL_LABELS.get(short, short)
        print(f"F2(P) {label:<30}: {single}/{len(rows)} single-letter ({100*single/len(rows):.1f}%)")
    else:
        print(f"[missing] {jsonl.name} — run run_prompt_sensitivity.py for {short}")\
"""),
    md("""\
### Sensitivity table — Zero-shot / Few-shot / CoT per model × task

**Experimental setup:** `run_prompt_sensitivity.py` evaluated 100 randomly-seeded samples
per task under three prompt configurations, holding all other pipeline components fixed
(same model, same decoding parameters, same token budgets, same extractor):

- **ZS (Zero-shot):** Original benchmark prompt, no demonstrations
- **FS (Few-shot):** 2 held-out in-distribution examples prepended to the prompt
- **CoT (Chain-of-thought):** Extended reasoning budget appended to the system prompt

All three modes were evaluated on the **same 100 samples per task** (seeded split),
so differences between columns are attributable purely to prompt specification (𝒫).

Δ = CoT − ZS. **Negative Δ** = CoT hurts; **Positive Δ** = CoT helps.
"""),
    code("""\
# Pivot: models as rows, tasks as columns
# Two tables: zero-shot scores, and CoT delta (CoT − zero-shot)

TASK_ORDER = [
    ('mcq','MCQ'),('rcm','RCM'),('ate','ATE'),('taa','TAA'),
    ('ckt','CKT'),('rms','RMS'),('athena_ate','ATE-A'),('athena_rcm','RCM-A'),('athena_vsp','VSP-A'),
    ('secure_maet','MAET'),('secure_cwet','CWET'),('secure_kcv','KCV'),
    ('seceval','SecEval'),('cybermetric','CyberMet'),('mmlu-cs','MMLU'),('secbench','SecBench'),
    ('redsage_cli','RS-CLI'),('redsage_frameworks','RS-FW'),('redsage_generals','RS-GEN'),
    ('redsage_kali','RS-Kali'),('redsage_skills','RS-Skills'),
]
BENCH_COLS = {
    'MCQ':'CTI','RCM':'CTI','ATE':'CTI','TAA':'CTI',
    'CKT':'Ath','RMS':'Ath','ATE-A':'Ath','RCM-A':'Ath','VSP-A':'Ath',
    'MAET':'SEC','CWET':'SEC','KCV':'SEC',
    'SecEval':'Gen','CyberMet':'Gen','MMLU':'Gen','SecBench':'Gen',
    'RS-CLI':'RS','RS-FW':'RS','RS-GEN':'RS','RS-Kali':'RS','RS-Skills':'RS',
}

task_keys  = [k for k, _ in TASK_ORDER if k in df['task'].unique()]
task_names = [n for k, n in TASK_ORDER if k in df['task'].unique()]
model_list = df['model'].unique()

def make_mode_table(col):
    rows = {}
    for model in model_list:
        row = {}
        for key, name in zip(task_keys, task_names):
            r = df[(df['model'] == model) & (df['task'] == key)]
            if not r.empty:
                v = r.iloc[0][col]
                row[name] = round(float(v), 1) if v is not None and str(v) != 'nan' else None
            else:
                row[name] = None
        rows[model] = row
    result = pd.DataFrame(rows).T
    result.index.name = 'Model'
    midx = pd.MultiIndex.from_tuples(
        [(BENCH_COLS.get(n,'?'), n) for n in task_names], names=['Benchmark','Task']
    )
    result.columns = midx
    return result

zs_df    = make_mode_table('zero_shot')
fs_df    = make_mode_table('few_shot')
cot_df   = make_mode_table('cot')
delta_df = make_mode_table('cot_delta')

def show_table(tbl, title):
    fmt = lambda v: f'{v:+.1f}' if isinstance(v,(int,float)) and 'delta' in title.lower() else (f'{v:.1f}' if isinstance(v,(int,float)) else '—')
    print(title)
    display(tbl.fillna('—').style
            .format(lambda v: f'{v:+.1f}' if isinstance(v,(int,float)) and 'Δ' in title else (f'{v:.1f}' if isinstance(v,(int,float)) else '—'),
                    na_rep='—')
            .set_table_styles([
                {'selector':'th','props':[('font-size','9px'),('padding','3px 5px'),('text-align','center'),('white-space','nowrap')]},
                {'selector':'td','props':[('font-size','9px'),('padding','2px 5px'),('text-align','center')]},
                {'selector':'caption','props':[('font-weight','bold'),('font-size','11px'),('padding','6px 0')]},
            ])
            .set_caption(title))

show_table(zs_df,    'Zero-shot (ZS) accuracy %')
show_table(fs_df,    'Few-shot (FS) accuracy %')
show_table(cot_df,   'Chain-of-thought (CoT) accuracy %')
show_table(delta_df, 'Δ = CoT − Zero-shot (pp)  |  negative = CoT hurts  |  positive = CoT helps')\
"""),
]


# ─────────────────────────── 04 — inference config ───────────────────────────
NB04 = [
    md("""\
# 04 · Inference configuration — $\\mathcal{F}(\\mathcal{I})$

Three single-parameter choices caused score swings of 40–86 pp, all recoverable
from stored result files without re-running inference.
"""),
    setup_cell(),
    md("""\
### F2(I): Token-budget filter — SecEval / GPT-5.4

SecEval specifies `max_new_tokens=5`. For GPT-5.4 via Azure this fell below the
API minimum, causing silent HTTP 400 failures on every call.
Both runs are stored: `gpt-5.4_final` (0.32%) and `gpt-5.4_seceval_rerun` (81.36%).
"""),
    code("""\
import json, pandas as pd
from IPython.display import display
from load_results import EVAL_RESULTS

orig_p  = EVAL_RESULTS / 'gpt-5.4_final'         / 'seceval_result.json'
fixed_p = EVAL_RESULTS / 'gpt-5.4_seceval_rerun'  / 'seceval_result.json'

orig_s  = json.loads(orig_p.read_text()).get('primary_score')  if orig_p.exists()  else None
fixed_s = json.loads(fixed_p.read_text()).get('primary_score') if fixed_p.exists() else None

if orig_s is not None and fixed_s is not None:
    f2i_df = pd.DataFrame([
        {'Condition': 'Original run  (max_new_tokens=5)', 'SecEval score (%)': orig_s,  'Δ vs corrected (pp)': None},
        {'Condition': 'Corrected budget',                  'SecEval score (%)': fixed_s, 'Δ vs corrected (pp)': fixed_s - orig_s},
    ])
    display(f2i_df.style
            .hide(axis='index')
            .format({'SecEval score (%)': '{:.1f}%', 'Δ vs corrected (pp)': lambda v: f'+{v:.1f}' if v is not None else '—'})
            .background_gradient(subset=['SecEval score (%)'], cmap='Blues', vmin=0, vmax=100)
            .map(lambda v: 'color:green;font-weight:bold' if isinstance(v, float) and v > 50 else '', subset=['Δ vs corrected (pp)'])
            .set_caption('F2(I): Token-budget filter — GPT-5.4 SecEval. Original 5-token budget caused silent HTTP 400 failures.'))
else:
    print(f"[missing] orig={orig_p.exists()}  fixed={fixed_p.exists()}")\
"""),
    md("""\
### F1(I): Stop-sequence mismatch — RedSage-Bench / Qwen3.6-35B

Official `stop=["\\n"]` fires at the first newline of Qwen3's `<think>` block,
before any answer token. Our harness drops the stop; the corrected scores are stored in the repository.
"""),
    code("""\
import json, pandas as pd
from IPython.display import display
from load_results import load_all_scores

scores = load_all_scores(OUTPUTS_FINAL)
qwen_dir = 'Qwen3.6-35B-A3B'
redsage_tasks = sorted(t for t in scores.get(qwen_dir, {}) if t.startswith('redsage_'))
if redsage_tasks:
    task_rows = [{'Task': t, 'Score (%)': scores[qwen_dir][t]}
                 for t in redsage_tasks if scores[qwen_dir].get(t) is not None]
    vals = [r['Score (%)'] for r in task_rows]
    avg = sum(vals) / len(vals)
    task_rows.append({'Task': 'Average — our harness (no stop sequence)', 'Score (%)': avg})
    f1i_df = pd.DataFrame(task_rows)
    display(f1i_df.style
            .hide(axis='index')
            .format({'Score (%)': '{:.1f}%'})
            .background_gradient(subset=['Score (%)'], cmap='Blues', vmin=0, vmax=100)
            .apply(lambda s: ['font-weight:bold;border-top:2px solid #aaa' if i == len(s)-1 else '' for i in range(len(s))], subset=['Score (%)'])
            .set_caption('F1(I): Qwen3.6-35B-A3B RedSage scores — our harness drops the official stop=[\"\\\\n\"] that fires inside the <think> block.'))
else:
    print(f"[missing] {qwen_dir} not in outputs_final/")\
"""),
    md("""\
### F3(I): Temperature drift — CyberMetric / Primus-Merged

CyberMetric's paper prescribes T=1.0; the released evaluator uses T=0.
Both runs stored: `cybermetric_paper_result.json` (T=1.0) vs
`cybermetric_result.json` (T=0). Primus-Merged: **57.2% → 17.2%, −40 pp**.
"""),
    code("""\
from load_results import load_all_scores

scores = load_all_scores(OUTPUTS_FINAL)
rows = []
for label, dir_name in PAPER_MODELS.items():
    p = scores.get(dir_name, {}).get('cybermetric_paper')
    u = scores.get(dir_name, {}).get('cybermetric')
    if p is not None and u is not None:
        rows.append({'Model': label,
                     'Paper T=1.0 (%)': round(p, 1),
                     'Unified T=0 (%)': round(u, 1),
                     'Δ (pp)':          round(u - p, 1)})

temp_df = pd.DataFrame(rows).sort_values('Δ (pp)').reset_index(drop=True)
import matplotlib.pyplot as plt, numpy as np
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(temp_df)); w = 0.35
ax.bar(x - w/2, temp_df['Paper T=1.0 (%)'].values, w, label='Paper T=1.0', color='#4C72B0')
ax.bar(x + w/2, temp_df['Unified T=0 (%)'].values, w, label='Unified T=0', color='#DD8452')
ax.set_xticks(x)
ax.set_xticklabels(temp_df['Model'].values, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('CyberMetric accuracy (%)')
ax.set_title('F3(I): Temperature drift — CyberMetric (paper T=1.0 vs unified T=0)')
ax.legend(); plt.tight_layout(); plt.show()
display(temp_df.style
        .hide(axis='index')
        .format({'Paper T=1.0 (%)': '{:.1f}%', 'Unified T=0 (%)': '{:.1f}%', 'Δ (pp)': '{:+.1f}'})
        .background_gradient(subset=['Paper T=1.0 (%)', 'Unified T=0 (%)'], cmap='Blues', vmin=0, vmax=100)
        .applymap(lambda v: 'color:red;font-weight:bold' if isinstance(v,(int,float)) and v <= -10 else '', subset=['Δ (pp)'])
        .set_caption('F3(I): Temperature drift — paper T=1.0 vs unified T=0. Red = gap ≥10pp.'))\
"""),
]


# ─────────────────────── 05 — logprob vs generative ──────────────────────────
NB05 = [
    md("""\
# 05 · Logprob vs generative — $\\mathcal{F}(\\mathcal{A})$

`run_redsage_lighteval.py` ran RedSage MCQ in two modes:
- **Logprob** — log-likelihood over choice tokens (LightEval default)
- **Generative** — model generates text; prefix/exact-match extracts the letter

Results: `outputs/redsage_lighteval_{model}_{full|generative_full}/results/`

**F1(A):** Same model, same items, 45.7% generative vs 86.6% logprob (+41 pp).
Direction is model-specific — neither method is uniformly preferable.
"""),
    setup_cell(),
    md("### Load LightEval logprob and generative scores"),
    code("""\
import json, re, pandas as pd

SUBSETS = [
    'cybersecurity_knowledge_frameworks',
    'cybersecurity_knowledge_generals',
    'cybersecurity_skills',
    'cybersecurity_tools_cli',
    'cybersecurity_tools_kali',
]

def extract_lighteval_scores(run_dir):
    jsons = list(run_dir.rglob('results_*.json'))
    if not jsons: return {}
    d = json.loads(jsons[0].read_text())
    out = {}
    for key, vals in d.get('results', {}).items():
        m = re.search(':([^|]+)[|]', key)
        if m:
            out[m.group(1)] = round((vals.get('acc') or 0) * 100, 1)
    return out

# Only load logprob/generative runs for the 10 paper model directories
PAPER_DIR_NAMES = set(PAPER_MODELS.values())
rows = []
for d in sorted(OUTPUTS_DIR.iterdir()):
    name = d.name
    if not (name.startswith('redsage_lighteval_') and
            name.endswith('_full') and 'generative' not in name):
        continue
    model_dir = name.replace('redsage_lighteval_', '').replace('_full', '')
    if model_dir not in PAPER_DIR_NAMES:
        continue  # skip Fanar and Qwen3-30B
    # Use the paper display name
    label = next((k for k, v in PAPER_MODELS.items() if v == model_dir), model_dir)
    lp  = extract_lighteval_scores(d)
    gen = extract_lighteval_scores(OUTPUTS_DIR / name.replace('_full', '_generative_full'))
    for s in SUBSETS:
        if lp.get(s) and gen.get(s):
            rows.append({'model': label, 'subset': s,
                         'logprob': lp[s], 'generative': gen[s],
                         'delta': lp[s] - gen[s]})

from IPython.display import display
comp = pd.DataFrame(rows)
agg  = comp.groupby('model')[['logprob', 'generative', 'delta']].mean().round(1)
agg  = agg.sort_values('delta', ascending=False)
print(f"Loaded {comp['model'].nunique()} models × {comp['subset'].nunique()} subsets")
display_agg = agg.reset_index().rename(columns={
    'model': 'Model', 'logprob': 'Logprob (%)', 'generative': 'Generative (%)', 'delta': 'Δ (pp)'
})
display(display_agg.style
        .hide(axis='index')
        .format({'Logprob (%)': '{:.1f}%', 'Generative (%)': '{:.1f}%', 'Δ (pp)': '{:+.1f}'})
        .background_gradient(subset=['Logprob (%)', 'Generative (%)'], cmap='Blues', vmin=0, vmax=100)
        .map(lambda v: 'color:red;font-weight:bold' if isinstance(v,(int,float)) and v < 0 else
                       ('color:green;font-weight:bold' if isinstance(v,(int,float)) and v > 20 else ''), subset=['Δ (pp)'])
        .set_caption('F1(A): Mean logprob vs generative accuracy per model (RedSage subsets). Δ = logprob − generative.'))\
"""),
    md("### F1(A): Side-by-side + per-model gap"),
    code("""\
import matplotlib.pyplot as plt, numpy as np, seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x = np.arange(len(agg)); w = 0.35
ax.bar(x - w/2, agg['logprob'],    w, label='Log-prob',   color='#4C72B0')
ax.bar(x + w/2, agg['generative'], w, label='Generative', color='#DD8452')
ax.set_xticks(x)
ax.set_xticklabels(agg.index, rotation=35, ha='right', fontsize=8)
ax.set_ylabel('Mean accuracy (%) across RedSage subsets')
ax.set_title('F1(A): Logprob vs Generative — RedSage-Bench')
ax.legend()

ax = axes[1]
colors = ['#d62728' if v < 0 else '#2ca02c' for v in agg['delta']]
ax.barh(agg.index, agg['delta'], color=colors)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Logprob − Generative (pp)')
ax.set_title('Gap: green = logprob higher, red = generative higher')

plt.tight_layout(); plt.show()

max_row = comp.loc[comp['delta'].idxmax()]
min_row = comp.loc[comp['delta'].idxmin()]
extremes = pd.DataFrame([
    {'Direction': 'Largest logprob advantage',    'Model': max_row['model'], 'Subset': max_row['subset'],
     'Logprob (%)': max_row['logprob'], 'Generative (%)': max_row['generative'], 'Δ (pp)': max_row['delta']},
    {'Direction': 'Largest generative advantage', 'Model': min_row['model'], 'Subset': min_row['subset'],
     'Logprob (%)': min_row['logprob'], 'Generative (%)': min_row['generative'], 'Δ (pp)': min_row['delta']},
])
display(extremes.style
        .hide(axis='index')
        .format({'Logprob (%)': '{:.1f}%', 'Generative (%)': '{:.1f}%', 'Δ (pp)': '{:+.1f}'})
        .set_caption('Extreme cases: per-subset logprob vs generative gap.'))\
"""),
    md("### F2(A): TAA metric drift — CTI-Bench (C+P) vs AthenaBench (binary)"),
    code("""\
import json, pandas as pd
from load_results import load_all_scores

scores = load_all_scores(OUTPUTS_FINAL)
rows = []
for label, dir_name in PAPER_MODELS.items():
    cti = scores.get(dir_name, {}).get('cti_taa')
    ath = scores.get(dir_name, {}).get('taa')
    if cti is not None and ath is not None:
        rows.append({'Model': label, 'CTI-TAA C+P (%)': cti,
                     'Athena-TAA (%)': ath, 'Gap (pp)': round(cti - ath, 1)})

taa = pd.DataFrame(rows).sort_values('Gap (pp)', ascending=False).reset_index(drop=True)
import matplotlib.pyplot as plt, numpy as np
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(taa)); w = 0.35
ax.bar(x - w/2, taa['CTI-TAA C+P (%)'].values, w, label='CTI-TAA C+P', color='#4C72B0')
ax.bar(x + w/2, taa['Athena-TAA (%)'].values,  w, label='Athena-TAA',  color='#DD8452')
ax.set_xticks(x)
ax.set_xticklabels(taa['Model'].values, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Accuracy (%)')
ax.set_title('F2(A): TAA metric drift — CTI-Bench (C+P) vs AthenaBench (binary)')
ax.legend(); plt.tight_layout(); plt.show()
display(taa.style
        .hide(axis='index')
        .format({'CTI-TAA C+P (%)': '{:.1f}%', 'Athena-TAA (%)': '{:.1f}%', 'Gap (pp)': '{:+.1f}'})
        .background_gradient(subset=['CTI-TAA C+P (%)', 'Athena-TAA (%)'], cmap='Blues', vmin=0, vmax=100)
        .map(lambda v: 'color:red;font-weight:bold' if isinstance(v,(int,float)) and v > 20 else '', subset=['Gap (pp)'])
        .set_caption('F2(A): TAA metric drift — CTI-Bench uses C+P partial credit; AthenaBench uses binary. Red gap ≥20pp.'))\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
NOTEBOOKS = {
    "00_overview.ipynb":            NB00,
    "01_results_table.ipynb":       NB01,
    "02_failure_modes.ipynb":       NB02,
    "03_prompt_sensitivity.ipynb":  NB03,
    "04_inference_config.ipynb":    NB04,
    "05_logprob_vs_generative.ipynb": NB05,
}

KERNEL_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}


def to_source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True) or [""]


def build_notebook(cells: list[dict]) -> dict:
    out_cells = []
    for i, c in enumerate(cells):
        cell = dict(c)
        cell["id"] = f"cell-{i:02d}"
        cell["source"] = to_source_lines(cell["source"])
        out_cells.append(cell)
    return {"cells": out_cells, "metadata": KERNEL_META, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        nb = build_notebook(cells)
        (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {name} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
