"""Shared helpers for the hydrodynamics forward-test methods-paper figures.

Every script caches its simulation output under ``data/`` (``numpy`` ``.npz``)
and regenerates the figure from that cache unless ``--rerun`` is passed, so the
plots can be re-styled without re-running any simulation.

``astronomix`` is installed non-editably in site-packages, so run the scripts
with the repo on PYTHONPATH and GPU selection through autocvd, e.g.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/hydrodynamics/double_blast.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def rerun_requested() -> bool:
    return "--rerun" in sys.argv


# consistent solver colours across the hydro figures
FV_HLL_COLOR = "#1f77b4"    # blue
FV_HLLC_COLOR = "#2ca02c"   # green
FV_AM_HLLC_COLOR = "#9467bd"  # purple
FD_COLOR = "#d62728"        # red
EXACT_COLOR = "black"
