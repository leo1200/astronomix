"""Shared helpers for the hydrodynamics forward-test methods-paper figures.

Provides the shared cache/figure directories, the ``--rerun`` command-line
check and the consistent solver colours used across the double-blast, Sedov,
shock-tube and sound-wave figures. Every generator caches its simulation output
under ``data/`` (numpy ``.npz``) and regenerates its figure from that cache
unless ``--rerun`` is passed, so the plots can be re-styled without re-running
any simulation.

``astronomix`` is installed non-editably in site-packages, so run the scripts
with the repo on PYTHONPATH and GPU selection through autocvd, e.g.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/double_blast.py
"""

# general
import sys
from pathlib import Path

# astronomix constants
from astronomix.plotting_helpers import paper_style

# The generators live alongside this helper; anchor the cache and figure
# directories to that location so they resolve the same regardless of cwd.
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def rerun_requested() -> bool:
    """Return whether ``--rerun`` was passed on the command line.

    Returns:
        True when the caller asked to recompute the simulation caches rather
        than reuse the cached ``.npz`` output.
    """
    return "--rerun" in sys.argv


# The solver colours of all figures of the paper, so the same solver reads as
# the same colour from one figure to the next.
FV_HLL_COLOR = paper_style.FV_HLLE_COLOR
FV_HLLC_COLOR = paper_style.FV_COLOR
FD_COLOR = paper_style.FD_COLOR
EXACT_COLOR = paper_style.REFERENCE_COLOR
