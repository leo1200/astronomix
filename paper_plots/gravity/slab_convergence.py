"""Advected self-gravitating slab: L1 density-error convergence.

Figure: ``figures/slab_error_convergence.svg``.

Advects a self-gravitating slab in equilibrium for one period and measures the
L1 density error vs resolution for the three finite-difference self-gravity
source-term schemes. The convergence order of the corrected flux-based source
scheme is reported from a direct power-law fit and the x-axis is labelled
directly with the cell counts.

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/gravity/slab_convergence.py

Pass ``--rerun`` to re-run the simulations; otherwise the cached results in
``data/slab_convergence.npz`` are used.
"""

# ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    FINITE_DIFFERENCE,
    GravityConfig,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    finalize_config,
)
from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators

from _common import (
    DATA_DIR,
    FIG_DIR,
    FIT_SCHEME_VERSION,
    SCHEMES,
    SCHEME_LABELS,
    apply_paper_style,
    fit_power_law,
    integer_log_ticks,
    pallas_config_kwargs,
)

RESOLUTIONS = [16, 32, 64, 96, 128, 192, 256]
NUM_PERIODS = 1
DATA_FILE = DATA_DIR / "slab_convergence.npz"
FIG_FILE = FIG_DIR / "slab_error_convergence.svg"

# Slab parameters (see tests/self_gravity_tests/advection.py).
G = 1 / (4 * jnp.pi)
RHO_0 = 1.0
P_0 = 6.0
EPS = 0.3
V_VEC = jnp.array([0.6, 0.6, 0.6])
K_VEC = jnp.array([2 / 3, 2 / 3, 2 / 3])
GAMMA = 5 / 3


def initial_slab(x):
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


def slab_l1_error(num_cells, self_gravity_version):
    """L1 density error after ``NUM_PERIODS`` advection periods."""
    box_size = float(2 * jnp.pi * jnp.max(1 / K_VEC))

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=self_gravity_version,
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
        **pallas_config_kwargs(),
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    x = jnp.transpose(helper_data.geometric_centers, (3, 0, 1, 2))
    initial_rho, initial_v, initial_P, w = initial_slab(x)
    period = 2 * jnp.pi / w

    params = SimulationParams(
        C_cfl=1.5,
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


def run_and_cache():
    errors = {v: [] for v in SCHEME_LABELS}
    for num_cells in RESOLUTIONS:
        for scheme in SCHEMES:
            err = slab_l1_error(num_cells, scheme.self_gravity_version)
            errors[scheme.self_gravity_version].append(err)
            print(
                f"  N={num_cells:4d}  {scheme.label:32s}  L1 = {err:.6e}", flush=True
            )

    save_kwargs = {"resolutions": np.array(RESOLUTIONS)}
    for version, errs in errors.items():
        save_kwargs[f"errors_{version}"] = np.array(errs)
    jnp.savez(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def plot():
    data = np.load(DATA_FILE)
    resolutions = data["resolutions"]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    for scheme in SCHEMES:
        errs = data[f"errors_{scheme.self_gravity_version}"]
        ax.plot(
            resolutions,
            errs,
            marker=scheme.marker,
            linestyle=scheme.linestyle,
            color=scheme.color,
            linewidth=scheme.linewidth,
            label=scheme.label,
        )

    fit_errs = data[f"errors_{FIT_SCHEME_VERSION}"]
    p, A, n_fit, _ = fit_power_law(resolutions, fit_errs)
    n_line = np.array([n_fit.min(), n_fit.max()])
    ax.plot(
        n_line,
        A * n_line**p,
        color="k",
        linestyle="-",
        linewidth=1.2,
        zorder=5,
        label=rf"fit: $\propto N^{{{p:.2f}}}$ ({SCHEME_LABELS[FIT_SCHEME_VERSION]})",
    )

    # Reference power-law slopes (the indicators from the original advection
    # test), placed in the lower-left as a visual guide alongside the direct fit.
    add_power_law_indicators(
        ax=ax,
        anchor=(20, 1e-5),
        exponents=[-2, -5],
        x_span=2.0,
        scales=[1.0, 1.0],
        x_label="N",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    integer_log_ticks(ax, resolutions, axis="x")
    ax.set_xlabel("number of cells per dimension $N$")
    ax.set_ylabel("L1 density error")
    ax.set_title("Advected slab: L1 density error convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_FILE)
    print(f"Saved -> {FIG_FILE}   (corrected flux-based source order p = {p:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="re-run simulations (otherwise use cache)"
    )
    args = parser.parse_args()

    if args.rerun or not DATA_FILE.exists():
        run_and_cache()
    else:
        print(f"Using cached results in {DATA_FILE} (pass --rerun to recompute).")
    plot()


if __name__ == "__main__":
    main()
