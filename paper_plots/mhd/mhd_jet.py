"""Magnetically driven jet (3D MHD) — paper figure, finite-difference solver.

A magnetic tower launched from a magnetized central region in an initially
uniform medium (vector-potential initial condition), evolved with the
finite-difference (WENO + constrained-transport) solver.  This is the
FD-only version of the jet test.

The figure is a density slice through the jet axis at t = 5.0.  The full 3D
density field is cached under ``data/`` so the figure can be re-sliced /
re-styled without re-running the (expensive) 256^3 simulation.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/mhd/mhd_jet.py [--res N] [--rerun]
"""

import sys

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax
import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_registered_variables,
    construct_primitive_state,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    OPEN_BOUNDARY,
    PALLAS,
    BoundarySettings,
    BoundarySettings1D,
    finalize_config,
    GravityConfig,
    PositivityConfig,
)
from astronomix.initial_condition_generation.magnetic_field_from_vector_potential import (
    setup_magnetic_fields_from_vector_potential,
)

from _common import DATA_DIR, FIG_DIR, rerun_requested, mhd_registered_variables

GAMMA = 5.0 / 3.0
BOX_SIZE = 24.0
T_END = 5.0
C_CFL = 0.8
RHO_0 = 1.0
P_0 = 1.0
A0 = 20.0


def _res():
    if "--res" in sys.argv:
        return int(sys.argv[sys.argv.index("--res") + 1])
    return 256


def cache_path(num_cells):
    return DATA_DIR / f"mhd_jet_fd_{num_cells}.npz"


def simulate(num_cells):
    grid_spacing = BOX_SIZE / num_cells
    center = BOX_SIZE / 2.0

    config = SimulationConfig(
        positivity_config=PositivityConfig(default_positivity_protection=True),
        solver_mode=FINITE_DIFFERENCE,
        # FD/WENO runs ~10x faster through the Pallas (Triton) backend;
        # bit-compatible with native JAX.
        backend=PALLAS,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
        grid_spacing=grid_spacing,
        mhd=True,
        progress_bar=True,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
            BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
            BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
    )
    rv = get_registered_variables(config)

    def jet_vector_potential(X, Y, Z):
        r = jnp.sqrt((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)
        A_x = -jnp.exp(-r**2) * (Y - center)
        A_y = jnp.exp(-r**2) * (X - center)
        A_z = 0.5 * A0 * jnp.exp(-r**2)
        return A_x, A_y, A_z

    B_x, B_y, B_z, bxb, byb, bzb = setup_magnetic_fields_from_vector_potential(
        config=config, vector_potential_func=jet_vector_potential
    )

    shape = (num_cells, num_cells, num_cells)
    rho = jnp.ones(shape) * RHO_0
    zeros = jnp.zeros(shape)
    p = jnp.ones(shape) * P_0

    params = SimulationParams(
        C_cfl=C_CFL,
        dt_max=0.1,
        t_end=T_END,
        gamma=GAMMA,
        minimum_density=1e-2 * RHO_0,
        minimum_pressure=1e-2 * P_0,
    )

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=rv,
        density=rho,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=p,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
    )

    config = finalize_config(config, initial_state.shape)
    final_state = time_integration(initial_state, config, params, rv)

    density = np.asarray(final_state[rv.density_index])
    return density


def get_run(num_cells, rerun):
    path = cache_path(num_cells)
    if path.exists() and not rerun:
        return np.load(path)["density"]
    print(f"running FD MHD jet at {num_cells}^3 ...")
    density = simulate(num_cells)
    np.savez(path, density=density)
    print(f"  cached -> {path}")
    return density


def plot(num_cells, rerun):
    density = get_run(num_cells, rerun)
    y_index = num_cells // 2

    # slice through the jet axis (x-z plane at mid-y)
    sl = density[:, y_index, :].T

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(
        sl, origin="lower", extent=(0, BOX_SIZE, 0, BOX_SIZE), cmap="YlOrRd"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label="density")

    plt.tight_layout()
    out = FIG_DIR / f"mhd_jet_fd_{num_cells}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    num_cells = _res()
    plot(num_cells, rerun_requested())
