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
