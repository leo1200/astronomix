"""Shared helpers for the MHD forward-test methods-paper figures.

Every script in ``examples/scripts/forward/mhd`` caches its simulation output under
``data/`` (``jnp.savez``) and regenerates the figure from that cache unless
``--rerun`` is passed, so the plots can be re-styled without re-running any
simulation.

``astronomix`` is installed non-editably in site-packages, so run the scripts
with the repo on ``PYTHONPATH`` to pick up this worktree, e.g.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/orszag_tang.py
"""

# general
import sys
from pathlib import Path

# astronomix constants
from astronomix.plotting_helpers import paper_style


# The figures and their input caches live next to this helper module, so all
# paths are resolved relative to it rather than the (arbitrary) working
# directory the scripts are launched from.
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Consistent solver naming / styling shared across all MHD figures, so the
# finite-difference and finite-volume curves look the same in every panel.
# ---------------------------------------------------------------------------
FD_LABEL = "FD"
FV_HLL_LABEL = "FV (HLL, minmod)"

FD_COLOR = paper_style.FD_COLOR
FV_COLOR = paper_style.FV_COLOR
FV_HLLE_COLOR = paper_style.FV_HLLE_COLOR


def rerun_requested() -> bool:
    """Return whether the user passed ``--rerun`` on the command line."""
    return "--rerun" in sys.argv


def mhd_registered_variables(solver_mode, dimensionality=3):
    """Build the variable registry for an MHD run.

    The registry lets figures index ``final_state`` by name
    (``rv.density_index`` etc.) instead of by hard-coded integers. Because the
    ``solver_mode`` (FINITE_VOLUME / FINITE_DIFFERENCE) determines the state
    layout, the registry is rebuilt for whichever backend produced the cached
    array being plotted.

    Args:
        solver_mode: The solver mode (FINITE_VOLUME or FINITE_DIFFERENCE) whose
            state layout the registry should describe.
        dimensionality: The spatial dimensionality of the run.

    Returns:
        The RegisteredVariables instance describing the state-array layout.
    """
    # Imported locally so that importing this helper module stays cheap and
    # does not pull in the full astronomix package for callers that only need
    # the path constants above.
    from astronomix import SimulationConfig, get_registered_variables

    config = SimulationConfig(
        mhd=True,
        dimensionality=dimensionality,
        solver_mode=solver_mode,
    )
    return get_registered_variables(config)
