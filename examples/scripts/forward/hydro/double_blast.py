"""Double blast-wave problem (Woodward & Colella 1984) — paper figure.

Generates ``figures/double_blast.pdf`` (and a ``.png`` copy): the gas density
at t = 0.038 for four solver configurations,

    - FV (VL2, HLLE)       at   400 cells
    - FV (VL2, HLLC)       at   400 cells
    - FD (WENO-Z)          at   400 cells
    - FV (VL2, HLLC)       at 10000 cells   (reference)

The finite-volume runs use AthenaPK's VL2 scheme (van Leer predictor-corrector,
PLM with van Leer slopes); the finite-difference runs use WENO-Z weights (the
test needs no positivity-preserving WENO). The 10000-cell run serves as a
converged reference; the three 400-cell runs show how the different schemes
resolve the colliding-blast structure at coarse resolution.

Run with the repo on PYTHONPATH (GPU picked via autocvd):

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/double_blast.py [--rerun]
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    CARTESIAN,
    REFLECTIVE_BOUNDARY,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLL,
    HLLC,
    VL2,
)
from astronomix.plotting_helpers.paper_style import TEXT_WIDTH, TWO_COLUMN, SINGLE_PANEL_WIDTH, layout_path

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings1D,
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

BOX_SIZE = 1.0
T_END = 0.038
GAMMA = 1.4

# The finite-volume runs use AthenaPK's one-dimensional CFL number, the
# finite-difference runs the C_cfl = 0.4 this test always used.
FINITE_VOLUME_CFL = 0.5
FINITE_DIFFERENCE_CFL = 0.4

CACHE = DATA_DIR / "double_blast.npz"

# One entry per curve in the figure, as
# (key, label, solver_mode, riemann_solver, num_cells, color, style).
# (The Riemann solver entry of the finite-difference run is not read.)
RUNS = [
    ("fv_vl2_hlle_400", "FV (VL2, HLLE), 400", FINITE_VOLUME, HLL, 400, FV_HLL_COLOR, "marker"),
    ("fv_vl2_hllc_400", "FV (VL2, HLLC), 400", FINITE_VOLUME, HLLC, 400, FV_HLLC_COLOR, "marker"),
    ("fd_400", "FD (WENO-Z), 400", FINITE_DIFFERENCE, HLLC, 400, FD_COLOR, "marker"),
    (
        "fv_vl2_hllc_10000",
        "FV (VL2, HLLC), 10000 (reference)",
        FINITE_VOLUME,
        HLLC,
        10000,
        EXACT_COLOR,
        "line",
    ),
]


def simulate(solver_mode, riemann_solver, num_cells):
    """Run one double-blast simulation and return its final density profile.

    Args:
        solver_mode: The solver family, FINITE_VOLUME or FINITE_DIFFERENCE.
        riemann_solver: The Riemann solver to use (e.g. HLL, HLLC).
        num_cells: The number of grid cells along the 1D domain.

    Returns:
        A tuple ``(positions, density)`` of numpy arrays: the cell-centre
        positions and the final gas density at ``T_END``.
    """
    is_finite_difference = solver_mode == FINITE_DIFFERENCE
    common_settings = dict(
        geometry=CARTESIAN,
        solver_mode=solver_mode,
        boundary_settings=BoundarySettings1D(
            left_boundary=REFLECTIVE_BOUNDARY,
            right_boundary=REFLECTIVE_BOUNDARY,
        ),
        box_size=BOX_SIZE,
        num_cells=num_cells,
        dimensionality=1,
        mhd=False,
    )
    if is_finite_difference:
        config = SimulationConfig(
            weno_z=True,
            **common_settings,
        )
        cfl = FINITE_DIFFERENCE_CFL
    else:
        config = SimulationConfig(
            time_integrator=VL2,
            riemann_solver=riemann_solver,
            **common_settings,
        )
        cfl = FINITE_VOLUME_CFL

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # Woodward & Colella initial condition: uniform density at rest with two
    # high-pressure regions near the reflecting walls, quiescent in between.
    positions = helper_data.geometric_centers
    density = jnp.ones_like(positions)
    velocity_x = jnp.zeros_like(positions)
    gas_pressure = jnp.where(
        positions > 0.9, 100.0, jnp.where(positions < 0.1, 1000.0, 0.01)
    )

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        gas_pressure=gas_pressure,
    )

    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, gamma=GAMMA, C_cfl=cfl)

    final_state = time_integration(initial_state, config, params, registered_variables)
    final_density = final_state[registered_variables.density_index]
    final_pressure = final_state[registered_variables.pressure_index]
    print(
        f"  min density {float(jnp.min(final_density)):.3e}, "
        f"min pressure {float(jnp.min(final_pressure)):.3e}, "
        f"finite: {bool(jnp.all(jnp.isfinite(final_state)))}"
    )
    return np.asarray(positions), np.asarray(final_density)


def run_and_cache():
    """Run every configuration in ``RUNS``, cache the profiles and return them.

    Returns:
        A dict of numpy arrays keyed ``<key>_r`` / ``<key>_rho`` per run, as
        stored in the ``.npz`` cache.
    """
    cached_profiles = {}
    for key, label, solver_mode, riemann_solver, num_cells, _, _ in RUNS:
        print(f"running {label} ...")
        positions, density = simulate(solver_mode, riemann_solver, num_cells)
        cached_profiles[f"{key}_r"] = positions
        cached_profiles[f"{key}_rho"] = density
    np.savez(CACHE, **cached_profiles)
    print(f"cached -> {CACHE}")
    return cached_profiles


def print_reference_errors(data):
    """
    Print, for every coarse run, its mean absolute density difference from the
    reference run on the plotted interval 0.4 <= x <= 1 and its peak density
    there against the reference peak.

    Args:
        data: The dict of cached profiles (see :func:`run_and_cache`).
    """
    reference_key = RUNS[-1][0]
    reference_positions = data[f"{reference_key}_r"]
    reference_density = data[f"{reference_key}_rho"]
    in_window = reference_positions >= 0.4
    print(f"reference peak density {reference_density[in_window].max():.3f}")
    for key, label, _, _, _, _, style in RUNS:
        if style != "marker":
            continue
        positions = data[f"{key}_r"]
        density = data[f"{key}_rho"]
        in_window = positions >= 0.4
        reference_at_positions = np.interp(positions, reference_positions, reference_density)
        mean_difference = np.mean(np.abs(density - reference_at_positions)[in_window])
        print(
            f"{label}: mean |rho - rho_ref| = {mean_difference:.4f}, "
            f"peak density {density[in_window].max():.3f}"
        )


def plot(data):
    """Draw the double-blast density figure from cached profiles.

    Args:
        data: The dict of cached profiles (see :func:`run_and_cache`).
    """
    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_PANEL_WIDTH if TWO_COLUMN else 0.75 * TEXT_WIDTH, 3.2), layout="constrained")

    for key, label, _, _, _, color, style in RUNS:
        positions = data[f"{key}_r"]
        density = data[f"{key}_rho"]
        # The coarse 400-cell runs are drawn as scatter markers so individual
        # cells stay visible; the 10000-cell reference is drawn as a line.
        if style == "marker":
            ax.plot(
                positions,
                density,
                label=label,
                color=color,
                marker=".",
                markersize=3,
                linestyle="None",
            )
        else:
            ax.plot(positions, density, label=label, color=color, linewidth=1.0)

    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    figure_legend(fig, ncols=1 if TWO_COLUMN else 2, markerscale=2)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 7)

    out = layout_path(FIG_DIR / "double_blast.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    if rerun_requested() or not CACHE.exists():
        data = run_and_cache()
    else:
        data = dict(np.load(CACHE))
    print_reference_errors(data)
    plot(data)
