"""Shared helpers for the self-gravity methods-paper figures.

Every script in ``examples/scripts/forward/self_gravity`` imports the scheme
definitions, consistent labels and styles and a couple of plotting utilities
from here, so the figures stay visually and terminologically consistent.

Run the scripts with the repository on ``PYTHONPATH`` to pick up this checkout,
e.g.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/self_gravity/jeans_convergence.py

Each script caches its simulation output under ``data/`` and regenerates the
figure from that cache unless ``--rerun`` is passed, so plots can be re-styled
without re-running any simulation.
"""

# general
from pathlib import Path

# typing
from typing import NamedTuple

# numerics
import numpy as np

# astronomix constants
from astronomix import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLC,
    NATIVE_JAX,
    PALLAS,
    VL2,
)
from astronomix.option_classes.simulation_config import (
    ATHENA_SECOND_ORDER,
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)
from astronomix.plotting_helpers.paper_style import (
    ATHENA_COUPLING_COLOR,
    FLUX_BASED_COLOR,
    FLUX_BASED_CORRECTED_COLOR,
    FLUX_CORRECTED_WORK_COLOR,
    GUIDE_COLOR,
    COLUMN_WIDTH,
    SIMPLE_SOURCE_COLOR,
    SINGLE_PANEL_WIDTH,
    TEXT_WIDTH,
    TIME_CENTRED_WORK_COLOR,
    TWO_COLUMN,
    layout_path,
)

# astronomix containers
from astronomix import BackendConfig

# astronomix functions
from astronomix.plotting_helpers.paper_style import (
    figure_legend,
    use_paper_style,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# These are the three finite-difference self-gravity couplings compared
# throughout the self-gravity section of the methods paper, named as there,
# and the second-order finite-volume scheme (VL2) with Athena++'s coupling.
SCHEME_LABELS = {
    SIMPLE_SOURCE: "simple source",
    SECOND_ORDER_CONSERVATIVE: "flux-based",
    FOURTH_ORDER_CONSERVATIVE: "flux-based corrected",
    ATHENA_SECOND_ORDER: "FV (VL2): Athena++ coupling",
}


class Scheme(NamedTuple):
    """A self-gravity coupling with its consistent plot styling.

    ``work_flux_correction`` adds the flux-corrected gravitational work
    (``GravityConfig.work_flux_correction``) to a conservative coupling.
    ``positivity_preserving`` marks the "+ PP-WENO" line of the smooth tests,
    which otherwise run without positivity-preserving WENO; the collapse runs
    always use it. The ``ATHENA_SECOND_ORDER`` coupling selects the
    finite-volume VL2 scheme, optionally with the time-centred energy source
    of Mullen, Hanawa & Gammie (2021) (``time_centred_energy_source``).
    """

    self_gravity_version: int
    marker: str
    linestyle: str
    color: str
    linewidth: float = 1.5
    work_flux_correction: bool = False
    positivity_preserving: bool = False
    time_centred_energy_source: bool = False

    @property
    def finite_volume(self) -> bool:
        """Whether the scheme is the finite-volume VL2 scheme."""
        return self.self_gravity_version == ATHENA_SECOND_ORDER

    @property
    def label(self) -> str:
        """The legend label."""
        label = SCHEME_LABELS[self.self_gravity_version]
        if self.work_flux_correction:
            label += " + flux-corrected work"
        if self.positivity_preserving:
            label += " + PP-WENO"
        if self.time_centred_energy_source:
            label += " + time-centered work"
        return label

    @property
    def key(self) -> str:
        """A unique key of the scheme for the cached data."""
        key = str(self.self_gravity_version)
        if self.work_flux_correction:
            key += "_fct"
        if self.positivity_preserving:
            key += "_pp"
        if self.time_centred_energy_source:
            key += "_tc"
        return key


# Order and styling shared by all figures. Colours (those of all figures of
# the paper) and markers are fixed per scheme so a reader can match curves
# across the panels.
SIMPLE_SCHEME = Scheme(SIMPLE_SOURCE, marker="o", linestyle=":", color=SIMPLE_SOURCE_COLOR, linewidth=1.8)
FLUX_SCHEME = Scheme(SECOND_ORDER_CONSERVATIVE, marker="s", linestyle="-.", color=FLUX_BASED_COLOR)
CORRECTED_SCHEME = Scheme(FOURTH_ORDER_CONSERVATIVE, marker="^", linestyle="--", color=FLUX_BASED_CORRECTED_COLOR)
CORRECTED_FCT_SCHEME = Scheme(
    FOURTH_ORDER_CONSERVATIVE,
    marker="v",
    linestyle="-",
    color=FLUX_CORRECTED_WORK_COLOR,
    linewidth=1.2,
    work_flux_correction=True,
)

#: The second-order finite-volume scheme (VL2, PLM, HLLC) with Athena++'s
#: coupling, and with the time-centred energy source of Mullen et al. (2021).
FINITE_VOLUME_SCHEME = Scheme(
    ATHENA_SECOND_ORDER,
    marker="D",
    linestyle=(0, (3, 1, 1, 1, 1, 1)),
    color=ATHENA_COUPLING_COLOR,
)
FINITE_VOLUME_TIME_CENTRED_SCHEME = Scheme(
    ATHENA_SECOND_ORDER,
    marker="P",
    linestyle=(0, (8, 3)),
    color=TIME_CENTRED_WORK_COLOR,
    time_centred_energy_source=True,
)

#: The three couplings of the smooth tests (Jeans waves, slab advection).
SCHEMES = [SIMPLE_SCHEME, FLUX_SCHEME, CORRECTED_SCHEME]

#: The smooth tests add the finite-volume scheme.
SMOOTH_TEST_SCHEMES = SCHEMES + [FINITE_VOLUME_SCHEME]

#: The Courant numbers of the smooth tests: the WENO-Z scheme alone runs at
#: 1.5; with positivity-preserving WENO the bound of its proof, 0.75.
SMOOTH_TEST_COURANT_NUMBER = 1.5
POSITIVITY_PRESERVING_COURANT_NUMBER = 0.75

#: The Courant number of the finite-volume VL2 scheme in 3D, AthenaPK's (and
#: Athena++'s) value for its 3D tests.
FINITE_VOLUME_COURANT_NUMBER = 0.3

#: The collapse figures add the flux-corrected work to the corrected coupling.
COLLAPSE_SCHEMES = [SIMPLE_SCHEME, FLUX_SCHEME, CORRECTED_SCHEME, CORRECTED_FCT_SCHEME]
CONSERVATIVE_SCHEMES = [FLUX_SCHEME, CORRECTED_SCHEME, CORRECTED_FCT_SCHEME]

#: The 128^3 collapse (energy budget and radial profiles) adds the
#: finite-volume scheme with both energy sources.
FINITE_VOLUME_SCHEMES = [FINITE_VOLUME_SCHEME, FINITE_VOLUME_TIME_CENTRED_SCHEME]
COLLAPSE_COMPARISON_SCHEMES = COLLAPSE_SCHEMES + FINITE_VOLUME_SCHEMES

#: The couplings the collapse figures of the paper show. The flux-corrected work
#: and the time-centred energy source of the finite-volume scheme are run and
#: cached with the others but only quoted in the text.
FIGURE_COLLAPSE_SCHEMES = [SIMPLE_SCHEME, FLUX_SCHEME, CORRECTED_SCHEME, FINITE_VOLUME_SCHEME]
FIGURE_CONSERVATIVE_SCHEMES = [FLUX_SCHEME, CORRECTED_SCHEME]


def solver_options(scheme: Scheme) -> dict:
    """
    The ``SimulationConfig`` options that select a scheme's solver: the
    finite-difference baseline (WENO-Z, Pallas, positivity-preserving WENO as
    the scheme says) or the finite-volume VL2 scheme (PLM, HLLC; native, as
    its gravity coupling runs on the native backend only).

    Args:
        scheme: The self-gravity scheme.

    Returns:
        The keyword arguments for ``SimulationConfig``.
    """
    if scheme.finite_volume:
        return dict(
            backend_config=BackendConfig(backend=NATIVE_JAX),
            solver_mode=FINITE_VOLUME,
            time_integrator=VL2,
            riemann_solver=HLLC,
        )
    return dict(
        backend_config=BackendConfig(backend=PALLAS),
        solver_mode=FINITE_DIFFERENCE,
        weno_z=True,
        weno_positivity_preserving=scheme.positivity_preserving,
    )


def gravity_options(scheme: Scheme) -> dict:
    """
    The ``GravityConfig`` options of a scheme's coupling. The finite-volume
    scheme uses Athena++'s Poisson kernel, the discrete Laplacian (it only
    acts on periodic solves).

    Args:
        scheme: The self-gravity scheme.

    Returns:
        The keyword arguments for ``GravityConfig`` (without ``self_gravity``
        and the boundary options).
    """
    return dict(
        self_gravity_version=scheme.self_gravity_version,
        work_flux_correction=scheme.work_flux_correction,
        poisson_discrete_laplacian=scheme.finite_volume,
        time_centred_energy_source=scheme.time_centred_energy_source,
    )


def smooth_test_courant_number(scheme: Scheme) -> float:
    """The Courant number of a scheme in the smooth tests."""
    if scheme.finite_volume:
        return FINITE_VOLUME_COURANT_NUMBER
    if scheme.positivity_preserving:
        return POSITIVITY_PRESERVING_COURANT_NUMBER
    return SMOOTH_TEST_COURANT_NUMBER


# --- analysis helpers ------------------------------------------------------
def fit_power_law(n, error, floor_factor=3.0):
    """Least-squares fit of ``error ~ A * N**p`` in log-log space.

    The fit is restricted to the pre-saturation range: points whose error sits
    above ``floor_factor * min(error)`` (so points that have flattened at the
    machine-precision / scheme floor are excluded from the order estimate).

    Args:
        n: The resolutions (cell counts) at which the error was measured.
        error: The measured error for each resolution.
        floor_factor: Multiple of the minimum error below which points are
            treated as saturated and excluded from the fit.

    Returns:
        ``(p, A, n_fit, error_fit)`` where ``p`` is the fitted exponent, ``A``
        the prefactor, and ``n_fit`` / ``error_fit`` are the points that
        entered the fit (sorted by ``n``).
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


def observed_orders(step_sizes, errors):
    """Observed convergence orders between successive refinements.

    Args:
        step_sizes: The step sizes (time steps or cell sizes), any order.
        errors: The error per step size.

    Returns:
        ``log(e_k / e_{k+1}) / log(h_k / h_{k+1})`` from the coarsest to the
        finest pair.
    """
    step_sizes = np.asarray(step_sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    order = np.argsort(step_sizes)[::-1]
    step_sizes, errors = step_sizes[order], errors[order]
    return np.log(errors[:-1] / errors[1:]) / np.log(step_sizes[:-1] / step_sizes[1:])


def apply_paper_style():
    """The common style of the figures of the paper (``paper_style``)."""
    use_paper_style()


def integer_log_ticks(ax, resolutions, axis="x"):
    """Label a log axis directly with the integer cell counts used.

    Args:
        ax: The matplotlib axes whose ticks are being set.
        resolutions: The cell counts to place as tick labels.
        axis: Which axis to relabel, either ``"x"`` or ``"y"``.
    """
    # Local import keeps matplotlib off the import path for callers that only
    # use the (matplotlib-free) analysis helpers.
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


def save_figure(fig, stem, dpi=300, extensions=("svg",)):
    """Save a figure under ``figures/`` (svg, the format the paper includes,
    unless more formats are requested).

    Args:
        fig: The matplotlib figure.
        stem: The file name without extension.
        dpi: The resolution of a png and of rasterized layers.
        extensions: The file formats to write.

    Returns:
        The path of the svg file.
    """
    for extension in extensions:
        fig.savefig(layout_path(FIG_DIR / f"{stem}.{extension}"), dpi=dpi)
    return layout_path(FIG_DIR / f"{stem}.svg")
