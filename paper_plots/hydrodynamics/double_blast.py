"""Double blast-wave problem (Woodward & Colella 1984) — paper figure.

Density at t = 0.038 for four solver configurations:

    - FV (HLL, minmod)   at   400 cells
    - FV (HLLC, minmod)  at   400 cells
    - FD (WENO)          at   400 cells
    - FV (HLL, minmod)   at 10000 cells   (reference)

The 10000-cell HLL run serves as a converged reference; the three 400-cell
runs show how the different schemes resolve the colliding-blast structure at
coarse resolution.

Run with the repo on PYTHONPATH (GPU picked via autocvd):

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/hydrodynamics/double_blast.py [--rerun]
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    finalize_config,
    time_integration,
)
from astronomix import CARTESIAN, REFLECTIVE_BOUNDARY
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    FINITE_DIFFERENCE,
    HLL,
    HLLC,
    MINMOD,
    NATIVE_JAX,
    PALLAS,
    BoundarySettings1D,
)

from _common import DATA_DIR, FIG_DIR, rerun_requested
from _common import FV_HLL_COLOR, FV_HLLC_COLOR, FD_COLOR

BOX_SIZE = 1.0
T_END = 0.038
GAMMA = 1.4

CACHE = DATA_DIR / "double_blast.npz"

# (key, label, solver_mode, riemann_solver, num_cells, color, style)
RUNS = [
    ("fv_hll_400", "FV (HLL, minmod), 400", FINITE_VOLUME, HLL, 400, FV_HLL_COLOR, "marker"),
    ("fv_hllc_400", "FV (HLLC, minmod), 400", FINITE_VOLUME, HLLC, 400, FV_HLLC_COLOR, "marker"),
    ("fd_400", "FD, 400", FINITE_DIFFERENCE, HLL, 400, FD_COLOR, "marker"),
    ("fv_hll_10000", "FV (HLL, minmod), 10000 (reference)", FINITE_VOLUME, HLL, 10000, "0.3", "line"),
]


def simulate(solver_mode, riemann_solver, num_cells):
    is_fd = solver_mode == FINITE_DIFFERENCE
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=solver_mode,
        boundary_settings=BoundarySettings1D(
            left_boundary=REFLECTIVE_BOUNDARY, right_boundary=REFLECTIVE_BOUNDARY
        ),
        first_order_fallback=False,
        riemann_solver=riemann_solver,
        limiter=MINMOD,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        dimensionality=1,
        mhd=False,
        # FD/WENO uses the Pallas backend (bit-compatible with native JAX).
        backend=PALLAS if is_fd else NATIVE_JAX,
        pallas_block_shape=(4, 1, 1),
        pallas_use_triton=True,
        pallas_interpret=False,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    r = helper_data.geometric_centers
    rho = jnp.ones_like(r)
    u = jnp.zeros_like(r)
    p = jnp.where(r > 0.9, 100.0, jnp.where(r < 0.1, 1000.0, 0.01))

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=u,
        gas_pressure=p,
    )

    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, gamma=GAMMA)

    final_state = time_integration(initial_state, config, params, registered_variables)
    rho_final = final_state[registered_variables.density_index]
    return np.asarray(r), np.asarray(rho_final)


def run_and_cache():
    out = {}
    for key, label, mode, rs, ncells, _, _ in RUNS:
        print(f"running {label} ...")
        r, rho = simulate(mode, rs, ncells)
        out[f"{key}_r"] = r
        out[f"{key}_rho"] = rho
    np.savez(CACHE, **out)
    print(f"cached -> {CACHE}")
    return out


def plot(data):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for key, label, _, _, _, color, style in RUNS:
        r = data[f"{key}_r"]
        rho = data[f"{key}_rho"]
        if style == "marker":
            ax.plot(r, rho, label=label, color=color, marker=".", markersize=3, linestyle="None")
        else:
            ax.plot(r, rho, label=label, color=color, lw=1.5)

    ax.set_xlabel("position")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 7)

    plt.tight_layout()
    out = FIG_DIR / "double_blast.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    if rerun_requested() or not CACHE.exists():
        data = run_and_cache()
    else:
        data = dict(np.load(CACHE))
    plot(data)
