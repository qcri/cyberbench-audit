"""Shared helpers for the analysis walkthrough notebooks.

Importing this module puts the pipeline root on ``sys.path`` so ``import
analysis.*`` resolves exactly as ``python -m analysis.X`` does. The analysis
modules resolve ``reports/`` and ``outputs/`` relative to their own ``__file__``,
so no working-directory change is needed.

Execution model:
  - **Light** steps (results table, judge agreement, correlation, aggregates,
    all ``make_*_plots``) are recomputed live via :func:`run_live`; if the live
    run can't proceed (e.g. ``outputs/`` not mounted), it falls back to the
    cached ``reports/`` artifact with a note.
  - **Heavy** steps (``embed`` = GPU; ``verify`` / ``classify_ka`` = API) are
    skipped unless ``SAYF_NB_REGEN=1``; :func:`heavy` gates them and otherwise
    we render the cached artifact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless-safe plotting (figure cells regenerate plots from cached CSVs).
os.environ.setdefault("MPLBACKEND", "Agg")

PIPELINE_ROOT = Path(__file__).resolve().parent.parent  # unified-benchmark-pipeline/
REPORTS_DIR = PIPELINE_ROOT / "analysis" / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Heavy (GPU/API) steps are off by default; set SAYF_NB_REGEN=1 to recompute.
REGEN = os.environ.get("SAYF_NB_REGEN", "0") == "1"

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPORTS_DIR / p


def show_df(path, n: int | None = None, **read_kw):
    """Read a CSV under reports/ and display it (optionally the first ``n`` rows)."""
    import pandas as pd
    from IPython.display import display

    df = pd.read_csv(_resolve(path), **read_kw)
    display(df.head(n) if n else df)
    return df


def show_fig(path, width: int = 820):
    """Display a figure under reports/figures/ (graceful note if missing)."""
    from IPython.display import Image, display

    p = path if Path(path).is_absolute() else FIGURES_DIR / path
    if not Path(p).exists():
        print(f"[missing figure: {p}] — set SAYF_NB_REGEN=1 and rerun the generating cell.")
        return
    display(Image(filename=str(p), width=width))


def show_md(path):
    """Render a markdown report under reports/ inline."""
    from IPython.display import Markdown, display

    display(Markdown(_resolve(path).read_text()))


def run_live(label: str, fn) -> bool:
    """Run a light analysis step live; on failure, fall back to cached reports/."""
    try:
        fn()
        print(f"[{label}] recomputed live.")
        return True
    except Exception as e:  # noqa: BLE001 — notebooks should never hard-fail here
        print(f"[{label}] live recompute unavailable ({type(e).__name__}: {e}); using cached reports/.")
        return False


def run_mod(label: str, module_name: str, func: str = "main", argv: list[str] | None = None) -> bool:
    """Import an analysis module and call ``func`` (default ``main``), tolerantly.

    - The import happens inside the try/except, so a module whose (optional) deps
      aren't installed never crashes the notebook — it falls back to the cached
      ``reports/`` artifact rendered by the next cell.
    - ``sys.argv`` is temporarily neutralized to ``[module] + argv`` so modules
      whose ``main()`` uses ``argparse`` see their defaults (a bare Jupyter kernel
      otherwise leaves ``-f <connection>.json`` in argv and argparse errors).
    """
    import importlib

    def _call():
        old_argv = sys.argv
        sys.argv = [module_name, *(argv or [])]
        try:
            getattr(importlib.import_module(module_name), func)()
        finally:
            sys.argv = old_argv

    return run_live(label, _call)


def heavy(label: str) -> bool:
    """Return True if a heavy (GPU/API) step should run; else explain the skip."""
    if REGEN:
        return True
    print(
        f"[{label}] heavy step skipped — rendering cached artifact. "
        f"Set SAYF_NB_REGEN=1 to recompute (needs GPU/API key)."
    )
    return False
