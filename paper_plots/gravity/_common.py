"""Shared helpers for the self-gravity methods-paper figures.

Every script in ``paper_plots/gravity`` imports the scheme definitions,
consistent labels/styles, the Pallas backend configuration and a couple of
plotting utilities from here, so the four figures stay visually and
terminologically consistent.

``astronomix`` is installed non-editably in site-packages, so run the scripts
with the repo on ``PYTHONPATH`` to pick up this worktree, e.g.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/gravity/jeans_convergence.py

Each script caches its simulation output under ``data/`` (``jnp.savez``) and
regenerates the figure from that cache unless ``--rerun`` is passed, so plots
can be re-styled without re-running any simulation.
"""

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from astronomix.option_classes.simulation_config import (
    SECOND_ORDER_CONSERVATIVE,
    FINITE_DIFFERENCE,
    PALLAS,
    SIMPLE_SOURCE,
    FOURTH_ORDER_CONSERVATIVE,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# --- consistent scheme naming across all gravity figures -------------------
# These are the three finite-difference self-gravity source-term schemes
# compared throughout the self-gravity section of the methods paper.
SCHEME_LABELS = {
    SIMPLE_SOURCE: "FD, simple source",
    SECOND_ORDER_CONSERVATIVE: "FD, flux-based source",
    FOURTH_ORDER_CONSERVATIVE: "FD, corrected flux-based source",
}


class Scheme(NamedTuple):
    """A self-gravity source-term scheme with its consistent plot styling.

    ``solver_mode`` defaults to finite difference (the three FD source-term
    schemes shared by every figure). ``label_override`` lets a figure add a
    differently-named scheme (e.g. the finite-volume comparison in the Evrard
    energy plot) without disturbing the shared FD ``SCHEME_LABELS``.
    """

    self_gravity_version: int
    marker: str
    linestyle: str
    color: str
    linewidth: float = 2.0
    solver_mode: int = FINITE_DIFFERENCE
    label_override: Optional[str] = None

    @property
    def label(self) -> str:
        return self.label_override or SCHEME_LABELS[self.self_gravity_version]


# Order and styling shared by all figures. Colours/markers are fixed per
# scheme so a reader can match curves across the four panels.
SCHEMES = [
    Scheme(SIMPLE_SOURCE, marker="o", linestyle=":", color="C0", linewidth=2.5),
    Scheme(SECOND_ORDER_CONSERVATIVE, marker="s", linestyle="-.", color="C1", linewidth=2.0),
    Scheme(FOURTH_ORDER_CONSERVATIVE, marker="^", linestyle="--", color="C2", linewidth=2.0),
]

# The scheme the convergence-order fit is reported for.
FIT_SCHEME_VERSION = FOURTH_ORDER_CONSERVATIVE


# --- Pallas backend --------------------------------------------------------
# The Pallas/Triton FD backend is ~10x faster than native JAX for these
# forward runs. The WENO Pallas kernel only engages when every spatial
# dimension is divisible by the matching ``pallas_block_shape`` entry,
# otherwise it silently falls back to native JAX. Resolutions used by the
# convergence scripts are multiples of 16, so a fixed (4, 4, 8) block divides
# both the jeans (N, N/2, N/2) and the N^3 slab/collapse grids.
PALLAS_BLOCK_SHAPE = (4, 4, 8)


def pallas_config_kwargs(block_shape=PALLAS_BLOCK_SHAPE) -> dict:
    """Keyword arguments enabling the Pallas/Triton FD backend."""
    return dict(
        backend=PALLAS,
        pallas_block_shape=block_shape,
        pallas_use_triton=True,
        pallas_interpret=False,
    )


def block_shape_for(dims, cap=8) -> tuple:
    """Largest power-of-two block (per axis, capped) that divides ``dims``.

    A robust fallback when resolutions are not all multiples of 16 — keeps the
    Pallas WENO kernel engaged for any grid.
    """

    def largest_pow2_div(n):
        b = 1
        while b * 2 <= cap and n % (b * 2) == 0:
            b *= 2
        return b

    return tuple(largest_pow2_div(int(n)) for n in dims)


# --- analysis helpers ------------------------------------------------------
def fit_power_law(n, error, floor_factor=3.0):
    """Least-squares fit of ``error ~ A * N**p`` in log-log space.

    The fit is restricted to the pre-saturation range: points whose error sits
    above ``floor_factor * min(error)`` (so points that have flattened at the
    machine-precision / scheme floor are excluded from the order estimate).

    Returns ``(p, A, n_fit, error_fit)`` where ``n_fit`` / ``error_fit`` are the
    points that entered the fit (sorted by ``n``).
    """
    n = np.asarray(n, dtype=float)
    error = np.asarray(error, dtype=float)
    order = np.argsort(n)
    n, error = n[order], error[order]

    floor = floor_factor * float(np.min(error))
    mask = error > floor
    if mask.sum() < 2:  # not enough unsaturated points; fall back to all
        mask = np.ones_like(error, dtype=bool)

    n_fit, error_fit = n[mask], error[mask]
    p, log_a = np.polyfit(np.log(n_fit), np.log(error_fit), 1)
    return float(p), float(np.exp(log_a)), n_fit, error_fit


def apply_paper_style():
    """A light, consistent matplotlib style for the gravity figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.markersize": 6,
        }
    )


def integer_log_ticks(ax, resolutions, axis="x"):
    """Label a log axis directly with the integer cell counts used."""
    from matplotlib.ticker import NullLocator

    resolutions = sorted(int(r) for r in np.unique(np.asarray(resolutions)))
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(NullLocator())
    target.set_minor_locator(NullLocator())
    if axis == "x":
        ax.set_xticks(resolutions)
        ax.set_xticklabels([str(r) for r in resolutions])
    else:
        ax.set_yticks(resolutions)
        ax.set_yticklabels([str(r) for r in resolutions])
