"""Shared helpers for the MHD forward-test methods-paper figures.

Every script in ``paper_plots/mhd`` caches its simulation output under
``data/`` (``jnp.savez``) and regenerates the figure from that cache unless
``--rerun`` is passed, so the plots can be re-styled without re-running any
simulation.

``astronomix`` is installed non-editably in site-packages, so run the scripts
with the repo on ``PYTHONPATH`` to pick up this worktree, e.g.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/mhd/orszag_tang.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# repo root (this worktree) so the cached arena results can be located
REPO_ROOT = HERE.parent.parent
ARENA_RESULTS = REPO_ROOT / "arena" / "results"

# --- consistent solver naming / styling across all MHD figures ------------
FD_LABEL = "FD"
FV_HLL_LABEL = "FV (HLL, minmod)"

FD_COLOR = "#d62728"      # red — finite-difference WENO
FV_HLL_COLOR = "#1f77b4"  # blue — finite-volume HLL


def rerun_requested() -> bool:
    return "--rerun" in sys.argv


def mhd_registered_variables(solver_mode, dimensionality=3):
    """Variable registry for an MHD run, so figures index ``final_state`` by
    name (``rv.density_index`` etc.) instead of hard-coded integers.

    ``solver_mode`` (FINITE_VOLUME / FINITE_DIFFERENCE) is what determines the
    state layout, so the registry is rebuilt for whichever backend produced the
    cached array being plotted.
    """
    from astronomix import SimulationConfig, get_registered_variables

    config = SimulationConfig(
        mhd=True, dimensionality=dimensionality, solver_mode=solver_mode
    )
    return get_registered_variables(config)
