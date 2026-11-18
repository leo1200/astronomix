"""Advected self-gravitating slab: L1 density-error convergence.

Figure: ``figures/slab_error_convergence.{svg,pdf,png}``.

Advects a self-gravitating slab in equilibrium for one period and measures the
L1 density error vs resolution for the three finite-difference self-gravity
couplings. The scheme is the finite-difference baseline for smooth flow
(WENO-Z weights, admissible face state, Pallas backend, float64).
The second-order finite-volume scheme (VL2, PLM, HLLC, native backend,
C_cfl = 0.3) runs with Athena++'s coupling and Poisson kernel (discrete
Laplacian). The convergence order of the corrected flux-based coupling is
reported from a direct power-law fit and the x-axis is labelled directly with
the cell counts.

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/self_gravity/slab_convergence.py

Pass ``--rerun`` to re-run the simulations; otherwise the cached results in
``data/slab_convergence.npz`` are used.
``--rerun --schemes 3`` reruns only the finite-volume scheme and keeps the
cached errors of the others.
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse

# jax
import jax
import jax.numpy as jnp

# Use 64-bit precision for a clean error measurement; must be enabled before any
# array is created.
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import PERIODIC_ROLL

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
    construct_primitive_state,
    finalize_config,
)


# figure helpers
from _common import (
    CORRECTED_SCHEME,
    DATA_DIR,
    FINITE_VOLUME_SCHEME,
    SMOOTH_TEST_SCHEMES,
    SINGLE_PANEL_WIDTH,
    TEXT_WIDTH,
    TWO_COLUMN,
    apply_paper_style,
    figure_legend,
    fit_power_law,
    gravity_options,
    integer_log_ticks,
    save_figure,
    smooth_test_courant_number,
    solver_options,
)

RESOLUTIONS = [16, 32, 64, 96, 128, 192, 256]
NUM_PERIODS = 1
DATA_FILE = DATA_DIR / "slab_convergence.npz"
FIGURE_STEM = "slab_error_convergence"

# Slab parameters (see tests/self_gravity_tests/advection.py).
G = 1 / (4 * jnp.pi)
RHO_0 = 1.0
P_0 = 6.0
EPS = 0.3
V_VEC = jnp.array([0.6, 0.6, 0.6])
K_VEC = jnp.array([2 / 3, 2 / 3, 2 / 3])
GAMMA = 5 / 3


def initial_slab(x):
    """Equilibrium self-gravitating slab initial condition.

    Args:
        x: The cell-centre coordinates, shape ``(3, nx, ny, nz)``.

    Returns:
        ``(rho, (vx, vy, vz), P, w)``: the density, velocity components, gas
        pressure and the advection angular frequency ``w``.
    """
    kx = K_VEC[0] * x[0] + K_VEC[1] * x[1] + K_VEC[2] * x[2]
    k_squared = jnp.sum(K_VEC**2)
    w = K_VEC[0] * V_VEC[0] + K_VEC[1] * V_VEC[1] + K_VEC[2] * V_VEC[2]

    rho = RHO_0 * (1 + EPS * jnp.cos(kx) + EPS**2 / 3 * jnp.cos(2 * kx))
    vx = V_VEC[0] * jnp.ones_like(x[0])
    vy = V_VEC[1] * jnp.ones_like(x[1])
    vz = V_VEC[2] * jnp.ones_like(x[2])
    P = P_0 + 4 * jnp.pi * G * EPS * RHO_0**2 / k_squared * (
        (1 - EPS**2 / 12) * jnp.cos(kx)
        + EPS / 3 * jnp.cos(2 * kx)
        + EPS**2 / 12 * jnp.cos(3 * kx)
        + EPS**3 / 144 * jnp.cos(4 * kx)
    )
    return rho, (vx, vy, vz), P, w


def slab_l1_error(num_cells, scheme):
    """L1 density error after ``NUM_PERIODS`` advection periods.

    Advects the equilibrium slab for a whole number of periods and compares the
    final density to the (identical) initial density.

    Args:
        num_cells: The resolution ``N`` (isotropic cubic grid).
        scheme: The self-gravity scheme (coupling, solver and whether
            positivity-preserving WENO is on).

    Returns:
        The mean absolute (L1) density error at the final time.
    """
    box_size = float(2 * jnp.pi * jnp.max(1 / K_VEC))

    config = SimulationConfig(
        **solver_options(scheme),
        gravity_config=GravityConfig(
            self_gravity=True,
            **gravity_options(scheme),
        ),
        boundary_handling=PERIODIC_ROLL,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=3,
        box_size=box_size,
        num_cells=num_cells,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        progress_bar=False,
        return_snapshots=False,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    x = jnp.transpose(helper_data.geometric_centers, (3, 0, 1, 2))
    initial_rho, initial_v, initial_P, w = initial_slab(x)
    period = 2 * jnp.pi / w

    params = SimulationParams(
        C_cfl=smooth_test_courant_number(scheme),
        t_end=NUM_PERIODS * period,
        gamma=GAMMA,
        gravitational_constant=G,
    )

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=initial_rho,
        velocity_x=initial_v[0],
        velocity_y=initial_v[1],
        velocity_z=initial_v[2],
        gas_pressure=initial_P,
    )
    config = finalize_config(config, initial_state.shape)

    final_state = time_integration(initial_state, config, params, registered_variables)
    error = jnp.mean(
        jnp.abs(final_state[registered_variables.density_index] - initial_rho)
    )
    return float(jax.block_until_ready(error))


def run_and_cache(schemes):
    """
    Measure the L1 density error over the resolution sweep and cache it.

    Schemes that are not run keep their cached errors, so a single scheme can
    be added or rerun without repeating the others.

    Args:
        schemes: The schemes to run.
    """
    errors = {scheme.key: [] for scheme in schemes}
    for num_cells in RESOLUTIONS:
        for scheme in schemes:
            error = slab_l1_error(num_cells, scheme)
            errors[scheme.key].append(error)
            print(
                f"  N={num_cells:4d}  {scheme.label:44s}  L1 = {error:.6e}", flush=True
            )

    save_kwargs = {"resolutions": np.array(RESOLUTIONS)}
    if DATA_FILE.exists():
        cached = np.load(DATA_FILE)
        for key in cached.files:
            if key != "resolutions" and np.array_equal(cached["resolutions"], RESOLUTIONS):
                save_kwargs[key] = cached[key]
    for key, scheme_errors in errors.items():
        save_kwargs[f"errors_{key}"] = np.array(scheme_errors)
    np.savez(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def plot():
    """Draw the L1 error convergence figure with the fits of the corrected
    coupling and the finite-volume scheme, and reference slopes."""
    data = np.load(DATA_FILE)
    resolutions = data["resolutions"]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(SINGLE_PANEL_WIDTH, 4.4) if TWO_COLUMN else (0.75 * TEXT_WIDTH, 3.7), layout="constrained")

    for scheme in SMOOTH_TEST_SCHEMES:
        errs = data[f"errors_{scheme.key}"]
        ax.plot(
            resolutions,
            errs,
            marker=scheme.marker,
            linestyle=scheme.linestyle,
            color=scheme.color,
            linewidth=scheme.linewidth,
            label=scheme.label,
        )

    # Direct power-law fits of the corrected flux-based coupling and of the
    # finite-volume scheme.
    for fit_scheme, fit_color in ((CORRECTED_SCHEME, "black"), (FINITE_VOLUME_SCHEME, "0.45")):
        scheme_order, prefactor, n_fit, _ = fit_power_law(
            resolutions,
            data[f"errors_{fit_scheme.key}"],
        )
        n_line = np.array([n_fit.min(), n_fit.max()])
        ax.plot(
            n_line,
            prefactor * n_line**scheme_order,
            color=fit_color,
            linestyle="-",
            linewidth=0.9,
            zorder=5,
            label=rf"fit: $\propto N^{{{scheme_order:.2f}}}$ ({fit_scheme.label})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    integer_log_ticks(ax, resolutions, axis="x")
    ax.set_xlabel("number of cells per dimension $N$")
    ax.set_ylabel(r"$L_1$ density error")
    # The curves and fits leave no free corner, so the legend sits
    # below the axes.
    figure_legend(fig, ncols=1 if TWO_COLUMN else 2)
    corrected_order, _, _, _ = fit_power_law(resolutions, data[f"errors_{CORRECTED_SCHEME.key}"])
    print(f"Saved -> {save_figure(fig, FIGURE_STEM)}   (flux-based corrected order p = {corrected_order:.3f})")
    for scheme in SMOOTH_TEST_SCHEMES:
        scheme_order, _, _, _ = fit_power_law(resolutions, data[f"errors_{scheme.key}"])
        print(f"  {scheme.label}: fitted order {scheme_order:.2f}")


def main():
    """Parse arguments, (re)run or load the cache, then draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="re-run simulations (otherwise use cache)"
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=None,
        help="with --rerun, run only the schemes with these cache keys (e.g. 3 for "
        "the finite-volume scheme) and keep the cached errors of the others",
    )
    args = parser.parse_args()

    schemes = SMOOTH_TEST_SCHEMES
    if args.schemes is not None:
        schemes = [scheme for scheme in SMOOTH_TEST_SCHEMES if scheme.key in args.schemes]
    if args.rerun or not DATA_FILE.exists():
        run_and_cache(schemes)
    else:
        print(f"Using cached results in {DATA_FILE} (pass --rerun to recompute).")
    plot()


if __name__ == "__main__":
    main()
