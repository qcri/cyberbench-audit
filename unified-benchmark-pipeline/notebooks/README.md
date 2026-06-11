# Analysis walkthrough notebooks

A friendly, runnable tour of the paper's analysis (`../analysis/`). Each notebook
**reuses** the analysis modules and renders the already-computed artifacts in
`../analysis/reports/` — no analysis logic is duplicated here.

| Notebook | Theme |
|---|---|
| `00_overview.ipynb` | Framing, how-to, `reports/` inventory, run order |
| `01_results_table.ipynb` | Master accuracy table + secondary metrics (F1, MAD) + LaTeX |
| `02_judge_agreement.ipynb` | LLM-judge reliability (Cohen's κ across judges) |
| `03_gold_errors_verification.ipynb` | Suspect gold labels + search-grounded verification |
| `04_capability_coverage.ipynb` | Knowledge-vs-Analytical coverage |
| `05_redundancy_correlation_embeddings.ipynb` | Cross-task redundancy, effective dimensions |

## Run

```bash
pip install -r requirements-notebooks.txt     # + the analysis deps
jupyter lab        # open any notebook, run top-to-bottom
```

- **Light** steps (results table, judge agreement, correlation, aggregates,
  figures) recompute live in seconds–minutes; if `../outputs/` isn't mounted they
  fall back to the cached `reports/` artifact automatically.
- **Heavy** steps are **off by default**: embeddings (GPU), and the verification
  / K-A judges (Azure API). To recompute them, launch with the regen flag:
  ```bash
  SAYF_NB_REGEN=1 jupyter lab          # needs a GPU and/or an Azure API key
  ```

## Headless execution (CI / reproduce)

```bash
./run_all.sh        # nbconvert --execute over all six, SAYF_NB_REGEN=0
```

On the cluster, run it on a compute node (the login node blocks installs and
heavy I/O) — see `run_all.sh`.

## Editing

The notebooks are generated from `build.py` (the editable source of cell
content). After changing it: `python build.py` to re-emit the `.ipynb`.
Shared helpers (`setup`, `show_df`/`show_fig`/`show_md`, `run_live`, `heavy`) live
in `nbtools.py`.
