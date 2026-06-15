"""Shared helpers for the original-pipeline walkthrough notebooks.

Mirrors the unified-analysis nbtools API exactly so the two notebook sets
feel identical to a reviewer.

Execution model:
  - **Light** steps run via :func:`run_mod` — import the script module and
    call ``main()``.  If the import or call fails (missing deps, central
    storage not mounted) the cell prints a graceful note.
  - **Heavy** steps (GPU inference, Azure API) are skipped unless
    ``SAYF_NB_REGEN=1``; :func:`heavy` gates them and renders cached artifacts.

Data roots:
  - ``OUTPUTS_FINAL`` — canonical 12-model result tree built by
    ``build_outputs_final.py``; used by ``load_results.py``.
  - ``OUTPUTS_DIR``   — local outputs/ (sensitivity_eval, lighteval runs, etc.)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# Notebooks live in test_notebooks/notebooks/
# BenchmarkingSecBenchmarks/ is two levels up
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent / "BenchmarkingSecBenchmarks"
OUTPUTS_FINAL = PIPELINE_ROOT / "outputs_final"
OUTPUTS_DIR   = PIPELINE_ROOT / "outputs"
REPORTS_DIR   = PIPELINE_ROOT  # tex / figure outputs written here by the scripts

REGEN = os.environ.get("SAYF_NB_REGEN", "0") == "1"

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPORTS_DIR / p


def show_df(path, n: int | None = None, **read_kw):
    """Read a CSV / JSONL under REPORTS_DIR and display it."""
    import pandas as pd
    from IPython.display import display

    p = _resolve(path)
    if not p.exists():
        print(f"[missing: {p}] — rerun the generating cell or set SAYF_NB_REGEN=1.")
        return None
    if str(path).endswith(".jsonl"):
        import json
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(p, **read_kw)
    display(df.head(n) if n else df)
    return df


def show_fig(path, width: int = 820):
    """Display a PNG / PDF under REPORTS_DIR (graceful note if missing)."""
    from IPython.display import Image, display

    p = _resolve(path)
    if not p.exists():
        print(f"[missing figure: {p}] — set SAYF_NB_REGEN=1 and rerun the generating cell.")
        return
    display(Image(filename=str(p), width=width))


def show_md(path):
    """Render a markdown / text file inline."""
    from IPython.display import Markdown, display

    p = _resolve(path)
    if not p.exists():
        print(f"[missing: {p}]")
        return
    display(Markdown(p.read_text()))


def show_tex(path):
    """Print a .tex file as a code block."""
    p = _resolve(path)
    if not p.exists():
        print(f"[missing: {p}]")
        return
    print(p.read_text())


def run_live(label: str, fn) -> bool:
    """Run a light step live; fall back gracefully on failure."""
    try:
        fn()
        print(f"[{label}] completed.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[{label}] unavailable ({type(e).__name__}: {e}); using cached outputs.")
        return False


def run_mod(label: str, module_name: str, func: str = "main",
            argv: list[str] | None = None) -> bool:
    """Import an analysis / generation module and call ``func`` (default ``main``).

    Neutralises ``sys.argv`` so argparse modules see their defaults.
    Falls back gracefully if the module or its deps aren't available.
    """
    import importlib

    def _call():
        old_argv = sys.argv
        sys.argv = [module_name, *(argv or [])]
        try:
            mod = importlib.import_module(module_name)
            getattr(mod, func)()
        finally:
            sys.argv = old_argv

    return run_live(label, _call)


def heavy(label: str) -> bool:
    """Return True only when SAYF_NB_REGEN=1; otherwise explain the skip."""
    if REGEN:
        return True
    print(
        f"[{label}] heavy step skipped — rendering cached artifact. "
        "Set SAYF_NB_REGEN=1 to recompute (needs GPU / API key)."
    )
    return False
