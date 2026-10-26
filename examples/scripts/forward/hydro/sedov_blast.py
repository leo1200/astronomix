"""Sedov-Taylor blast wave — paper figure (3x3 layout at 256^3).

Generates ``figures/sedov_blast_<N>.png`` (and a ``.pdf`` copy): radial
profiles (density, |v|, pressure) of the spherical Sedov-Taylor blast at
t = 0.1 compared with the exact self-similar solution, for three solvers, one
per row, all at the same resolution:

    row 0:  FV (VL2, HLLE)
    row 1:  FV (VL2, HLLC)
    row 2:  FD (WENO-Z)

The finite-volume rows use AthenaPK's VL2 scheme with the safeguards AthenaPK
uses for strong shocks: the first-order flux correction and the density and
pressure floors. The finite-difference row uses WENO-Z weights alone (the
test needs no positivity-preserving WENO). ``--without-fofc`` runs the
finite-volume rows without these safeguards (cached and plotted separately, as
``sedov_blast_<N>_without_fofc``) to show whether they are needed.

Each run is cached separately under ``data/sedov_<key>_<N>.npz`` (binned radial
profiles and their spread, a scatter subsample and a central density slice),
so the figure can be re-styled without re-running any simulation. Pass
``--rerun`` to recompute. Besides the figure, the script prints per run the
extrema of the final state and the largest spread of |v| within a radial bin
around the shock front, which measures the carbuncle instability of the HLLC
solver.

Resolution can be overridden for a quick smoke test:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/sedov_blast.py --res 32 --rerun

Default run (the paper figure):

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/sedov_blast.py --rerun

ExactPack (https://github.com/lanl/ExactPack) provides the exact solution.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import sys

# jax
import jax.numpy as jnp
import jax.random as jr

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# exact solution
from exactpack.solvers.sedov.sedov import Sedov

# astronomix constants
from astronomix import (
    CARTESIAN,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLL,
    HLLC,
    PERIODIC_BOUNDARY,
    VL2,
)
from astronomix.option_classes.simulation_config import POSITIVITY_HARD_FLOOR
from astronomix.plotting_helpers.paper_style import (
    GUIDE_COLOR,
    TEXT_WIDTH,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
)

# astronomix functions
from astronomix import (
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
    construct_primitive_state,
)
from astronomix.plotting_helpers.paper_style import (
    figure_legend,
    use_paper_style,
)


# shared hydro-figure helpers
from _common import (
    DATA_DIR,
    EXACT_COLOR,
    FIG_DIR,
    FD_COLOR,
    FV_HLLC_COLOR,
    FV_HLL_COLOR,
    rerun_requested,
)

# ---- physical setup (matches tests/hydro_tests/sedov3D.py) ----------------
T_END = 0.1
GAMMA = 5.0 / 3.0
E_EXPLOSION = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1e-4
# Injection region: a well-resolved sphere (~8 cells radius at 256^3) with a
# smooth tanh taper of ~2 cells, instead of a single-cell sharp top-hat. The
# taper removes the grid-imprint noise the sharp injection leaves near the
# origin, and the explosion pressure is renormalised (below) so the total
# deposited thermal energy is exactly E_EXPLOSION regardless of the smoothing.
R_EXPLOSION = 0.03
SMOOTH_CELLS = 2.0
NUM_BINS = 200
NUM_SCATTER = 80_000

# CFL numbers: AthenaPK's 3D value for the finite-volume VL2 scheme; the
# finite-difference run keeps the C_cfl = 0.4 this test always used.
FINITE_VOLUME_CFL = 0.3
FINITE_DIFFERENCE_CFL = 0.4

# AthenaPK's density and pressure floors (dfloor / pfloor) for the
# finite-volume runs, as in its Sedov validation case; far below the smallest
# density (~1e-5 at the centre) and pressure (1e-4) of the exact solution.
DENSITY_FLOOR = 1e-8
PRESSURE_FLOOR = 1e-8

# The radial shell around the exact shock radius (in units of it) in which
# the spread of |v| within a radial bin measures the carbuncle instability.
SHOCK_SHELL = (0.85, 1.05)

# One entry per solver row in the figure, as
# (key, label, color, solver_mode, riemann_solver). The finite-difference
# scheme has no Riemann solver.
RUNS = [
    ("fv_vl2_hlle", "FV (VL2, HLLE)", FV_HLL_COLOR, FINITE_VOLUME, HLL),
    ("fv_vl2_hllc", "FV (VL2, HLLC)", FV_HLLC_COLOR, FINITE_VOLUME, HLLC),
    ("fd", "FD (WENO-Z)", FD_COLOR, FINITE_DIFFERENCE, None),
]


def _resolution_from_argv():
    """Return the requested cubic resolution.

    Returns:
        The integer following a ``--res`` command-line flag, or the default
        256 (the paper resolution) when the flag is absent.
    """
    if "--res" in sys.argv:
        return int(sys.argv[sys.argv.index("--res") + 1])
    return 256


def _strong_shock_safeguards_from_argv():
    """Return whether the finite-volume runs use AthenaPK's safeguards.

    Returns:
        False when ``--without-fofc`` was passed (no first-order flux
        correction and no floors), True otherwise.
    """
    return "--without-fofc" not in sys.argv


def make_config(solver_mode, riemann_solver, num_cells, strong_shock_safeguards):
    """Build the simulation configuration for one Sedov solver row.

    Args:
        solver_mode: The solver family, FINITE_VOLUME or FINITE_DIFFERENCE.
        riemann_solver: The Riemann solver of the finite-volume scheme (HLL
            or HLLC); not read for the finite-difference scheme.
        num_cells: The number of cells per dimension (cubic domain).
        strong_shock_safeguards: Whether the finite-volume scheme uses
            AthenaPK's first-order flux correction and its density and
            pressure floors.

    Returns:
        The assembled :class:`SimulationConfig`.
    """
    # The blast stays inside the unit box until t = 0.1, so the boundaries do
    # not matter; periodic ones are the cheapest for both schemes.
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    common_settings = dict(
        geometry=CARTESIAN,
        solver_mode=solver_mode,
        dimensionality=3,
        num_cells=num_cells,
        exact_end_time=True,
        progress_bar=True,
        mhd=False,
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
    )
    if solver_mode == FINITE_DIFFERENCE:
        return SimulationConfig(
            weno_z=True,
            **common_settings,
        )
    if strong_shock_safeguards:
        positivity_config = PositivityConfig(per_step_mode=POSITIVITY_HARD_FLOOR)
    else:
        positivity_config = PositivityConfig()
    return SimulationConfig(
        time_integrator=VL2,
        riemann_solver=riemann_solver,
        first_order_flux_correction=strong_shock_safeguards,
        positivity_config=positivity_config,
        **common_settings,
    )


def simulate(solver_mode, riemann_solver, num_cells, strong_shock_safeguards):
    """Run one 3D Sedov simulation and reduce it to radial diagnostics.

    Args:
        solver_mode: The solver family, FINITE_VOLUME or FINITE_DIFFERENCE.
        riemann_solver: The Riemann solver of the finite-volume scheme.
        num_cells: The number of cells per dimension (cubic domain).
        strong_shock_safeguards: Whether the finite-volume scheme uses the
            first-order flux correction and the floors.

    Returns:
        A dict of numpy arrays holding the binned radial means and standard
        deviations (density, |v|, pressure), a reproducible scatter subsample
        of the raw cells, the central density slice, the extrema of the final
        state and the maximum domain radius — everything the figure and the
        printed diagnostics need, cached to disk.
    """

    # -------------------------------------------------------------
    # ============ ↓ Initial condition (energy injection) ↓ =======
    # -------------------------------------------------------------

    config = make_config(solver_mode, riemann_solver, num_cells, strong_shock_safeguards)
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (num_cells, num_cells, num_cells)
    density = jnp.ones(shape) * RHO_AMBIENT
    zeros = jnp.zeros(shape)

    # Smoothly-tapered spherical injection weight in [0, 1] (≈1 inside
    # R_EXPLOSION, tanh taper of SMOOTH_CELLS cells), then renormalise the
    # over-pressure so the deposited thermal energy above ambient,
    #   E = Σ (p - p_amb)/(γ-1) · ΔV ,
    # is exactly E_EXPLOSION (independent of the taper / resolution).
    dx = 1.0 / num_cells  # box_size = 1.0
    smooth_width = SMOOTH_CELLS * dx
    radius = helper_data.r
    weight = 0.5 * (1.0 - jnp.tanh((radius - R_EXPLOSION) / smooth_width))
    cell_volume = dx**3
    delta_p = E_EXPLOSION * (GAMMA - 1.0) / (jnp.sum(weight) * cell_volume)
    gas_pressure = P_AMBIENT + delta_p * weight

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, initial_state.shape)

    if solver_mode == FINITE_DIFFERENCE:
        params = SimulationParams(t_end=T_END, gamma=GAMMA, C_cfl=FINITE_DIFFERENCE_CFL)
    else:
        params = SimulationParams(
            t_end=T_END,
            gamma=GAMMA,
            C_cfl=FINITE_VOLUME_CFL,
            minimum_density=DENSITY_FLOOR,
            minimum_pressure=PRESSURE_FLOOR,
        )

    # -------------------------------------------------------------
    # ============ ↑ Initial condition (energy injection) ↑ =======
    # -------------------------------------------------------------

    result = time_integration(initial_state, config, params, registered_variables)

    # -------------------------------------------------------------
    # ================= ↓ Radial reduction / binning ↓ ============
    # -------------------------------------------------------------

    radius_flat = helper_data.r.flatten()
    density_flat = result[registered_variables.density_index].flatten()
    pressure_flat = result[registered_variables.pressure_index].flatten()
    velocity_index = registered_variables.velocity_index
    velocity_magnitude = jnp.sqrt(
        result[velocity_index.x] ** 2
        + result[velocity_index.y] ** 2
        + result[velocity_index.z] ** 2
    ).flatten()

    # Bin every cell by radius and average each field within a bin. Empty bins
    # would divide by zero, so their count is forced to one (their mean stays 0).
    domain_max_r = float(jnp.max(radius_flat))
    bins = jnp.linspace(0, domain_max_r, NUM_BINS + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_index = jnp.clip(
        jnp.searchsorted(bins, radius_flat, side="right") - 1, 0, NUM_BINS - 1
    )

    counts = jnp.zeros(NUM_BINS).at[bin_index].add(1.0)
    counts = jnp.where(counts == 0, 1.0, counts)

    def binned_mean_and_spread(field):
        """Return the per-bin mean and standard deviation of ``field``."""
        mean = jnp.zeros(NUM_BINS).at[bin_index].add(field) / counts
        mean_of_square = jnp.zeros(NUM_BINS).at[bin_index].add(field**2) / counts
        spread = jnp.sqrt(jnp.maximum(mean_of_square - mean**2, 0.0))
        return mean, spread

    mean_rho, spread_rho = binned_mean_and_spread(density_flat)
    mean_v, spread_v = binned_mean_and_spread(velocity_magnitude)
    mean_p, spread_p = binned_mean_and_spread(pressure_flat)

    # Reproducible scatter subsample of the raw cells for the background cloud.
    perm = jr.permutation(jr.PRNGKey(42), radius_flat.shape[0])[:NUM_SCATTER]

    # -------------------------------------------------------------
    # ================= ↑ Radial reduction / binning ↑ ============
    # -------------------------------------------------------------

    return dict(
        bin_centers=np.asarray(bin_centers),
        mean_rho=np.asarray(mean_rho),
        mean_v=np.asarray(mean_v),
        mean_p=np.asarray(mean_p),
        spread_rho=np.asarray(spread_rho),
        spread_v=np.asarray(spread_v),
        spread_p=np.asarray(spread_p),
        r_scatter=np.asarray(radius_flat[perm]),
        rho_scatter=np.asarray(density_flat[perm]),
        v_scatter=np.asarray(velocity_magnitude[perm]),
        p_scatter=np.asarray(pressure_flat[perm]),
        density_slice=np.asarray(result[registered_variables.density_index, :, :, num_cells // 2]),
        min_density=np.asarray(jnp.min(density_flat)),
        max_velocity=np.asarray(jnp.max(velocity_magnitude)),
        min_pressure=np.asarray(jnp.min(pressure_flat)),
        all_finite=np.asarray(jnp.all(jnp.isfinite(result))),
        domain_max_r=np.asarray(domain_max_r),
    )


def run_key(key, solver_mode, strong_shock_safeguards):
    """Return the cache key of a run, marking finite-volume runs without the
    strong-shock safeguards."""
    if solver_mode == FINITE_VOLUME and not strong_shock_safeguards:
        return f"{key}_without_fofc"
    return key


def cache_path(key, num_cells):
    """Return the ``.npz`` cache path for run ``key`` at ``num_cells``^3."""
    return DATA_DIR / f"sedov_{key}_{num_cells}.npz"


def get_run(key, solver_mode, riemann_solver, num_cells, strong_shock_safeguards, rerun):
    """Load the cached diagnostics for one run, computing them if needed.

    Args:
        key: The short run identifier used in the cache filename.
        solver_mode: The solver family, FINITE_VOLUME or FINITE_DIFFERENCE.
        riemann_solver: The Riemann solver of the finite-volume scheme.
        num_cells: The number of cells per dimension (cubic domain).
        strong_shock_safeguards: Whether the finite-volume scheme uses the
            first-order flux correction and the floors.
        rerun: When True, recompute and overwrite the cache instead of reusing it.

    Returns:
        The dict of radial diagnostics produced by :func:`simulate`.
    """
    path = cache_path(run_key(key, solver_mode, strong_shock_safeguards), num_cells)
    if path.exists() and not rerun:
        return dict(np.load(path))
    print(f"running Sedov {key} at {num_cells}^3 ...")
    data = simulate(solver_mode, riemann_solver, num_cells, strong_shock_safeguards)
    np.savez(path, **data)
    print(f"  cached -> {path}")
    return data


def exact_solution(domain_max_r):
    """Evaluate the exact self-similar Sedov solution out to ``domain_max_r``.

    Args:
        domain_max_r: The maximum radius to sample the analytic solution over.

    Returns:
        A tuple ``(r_exact, sol)`` of the sample radii and the ExactPack
        solution object (fields ``density``, ``velocity``, ``pressure``).
    """
    solver = Sedov(geometry=3, eblast=E_EXPLOSION / RHO_AMBIENT, gamma=GAMMA, omega=0.0)
    r_exact = np.linspace(0.0, domain_max_r, 500)
    sol = solver(r=r_exact, t=T_END)
    return r_exact, sol


def print_diagnostics(label, run_data, r_exact, sol):
    """
    Print the extrema of a run's final state and the carbuncle measures: the
    largest |v| of any cell and the largest standard deviation of |v| within
    a radial bin around the shock front (``SHOCK_SHELL``), both relative to
    the exact post-shock velocity.

    Args:
        label: The run's label.
        run_data: The run's cached diagnostics.
        r_exact: The radii of the exact solution.
        sol: The exact solution.
    """
    exact_velocity = np.asarray(sol["velocity"])
    post_shock_velocity = exact_velocity.max()
    shock_radius = r_exact[np.argmax(exact_velocity)]
    bin_centers = run_data["bin_centers"]
    in_shell = (bin_centers > SHOCK_SHELL[0] * shock_radius) & (
        bin_centers < SHOCK_SHELL[1] * shock_radius
    )
    shell_velocity_spread = run_data["spread_v"][in_shell].max() / post_shock_velocity
    print(
        f"{label}: min density {float(run_data['min_density']):.3e}, "
        f"min pressure {float(run_data['min_pressure']):.3e}, "
        f"finite {bool(run_data['all_finite'])}, "
        f"max |v| {float(run_data['max_velocity']) / post_shock_velocity:.3f} and largest "
        f"|v| spread at the shock front {shell_velocity_spread:.3f} of the post-shock velocity"
    )


def plot(num_cells, strong_shock_safeguards, rerun):
    """Draw the Sedov figure (one solver per row, three fields per column).

    Args:
        num_cells: The number of cells per dimension (cubic domain).
        strong_shock_safeguards: Whether the finite-volume rows use the
            first-order flux correction and the floors.
        rerun: When True, recompute each run's cache before plotting.
    """
    use_paper_style()
    fig, axes = plt.subplots(
        len(RUNS),
        3,
        figsize=(TEXT_WIDTH, 1.85 * len(RUNS) + 0.35),
        sharex=True,
        layout="constrained",
    )

    col_labels = ["density", r"$|v|$", "pressure"]

    for row_index, (key, label, color, solver_mode, riemann_solver) in enumerate(RUNS):
        run_data = get_run(
            key,
            solver_mode,
            riemann_solver,
            num_cells,
            strong_shock_safeguards,
            rerun,
        )
        domain_max_r = float(run_data["domain_max_r"])
        r_exact, sol = exact_solution(domain_max_r)
        print_diagnostics(label, run_data, r_exact, sol)

        # Per column: the scatter cloud, the binned mean, the exact profile, and
        # whether the y-axis is logarithmic (pressure spans several decades).
        profiles = [
            ("rho", run_data["rho_scatter"], run_data["mean_rho"], sol["density"], False),
            ("v", run_data["v_scatter"], run_data["mean_v"], sol["velocity"], False),
            ("p", run_data["p_scatter"], run_data["mean_p"], sol["pressure"], True),
        ]
        for col_index, (_, scatter, mean, exact, log_y) in enumerate(profiles):
            ax = axes[row_index, col_index]
            ax.scatter(
                run_data["r_scatter"],
                scatter,
                color="lightgray",
                alpha=0.5,
                s=1,
                rasterized=True,
                label="simulation (sampled)",
            )
            ax.plot(
                run_data["bin_centers"],
                mean,
                "-",
                color=color,
                label="binned mean",
            )
            ax.plot(
                r_exact,
                exact,
                "--",
                color=EXACT_COLOR,
                linewidth=1.2,
                label="exact solution",
            )
            ax.set_xlim(0, domain_max_r)
            if log_y:
                ax.set_yscale("log")
                ax.set_ylim(bottom=P_AMBIENT / 2)
            else:
                ax.set_ylim(0, None)
            ax.set_ylabel(col_labels[col_index])
        # The solver label goes on the left, outside the axes (a row label, not
        # an axes title), rotated to run alongside the row.
        axes[row_index, 0].annotate(
            label,
            xy=(0, 0.5),
            xytext=(-axes[row_index, 0].yaxis.labelpad - 12, 0),
            xycoords=axes[row_index, 0].yaxis.label,
            textcoords="offset points",
            ha="right",
            va="center",
            rotation=90,
        )

    for col_index in range(3):
        axes[-1, col_index].set_xlabel("radius")

    # The binned mean carries the colour of the scheme of its row, so its
    # legend entry is drawn in a neutral grey.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles = [
        Line2D([], [], color=GUIDE_COLOR) if label == "binned mean" else handle
        for handle, label in zip(handles, labels)
    ]
    figure_legend(fig, handles, labels, markerscale=4)

    figure_name = f"sedov_blast_{num_cells}"
    if not strong_shock_safeguards:
        figure_name += "_without_fofc"
    out = FIG_DIR / f"{figure_name}.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    plot(
        _resolution_from_argv(),
        _strong_shock_safeguards_from_argv(),
        rerun_requested(),
    )
