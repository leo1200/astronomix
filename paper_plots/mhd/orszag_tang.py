"""Orszag-Tang vortex (2D MHD) — paper figure.

Left panel:  finite-difference (WENO) density at t = 3.0 (shown as the example
             field).
Right panel: density profile along the cut y = 0.625 pi for both
             FV (HLL, minmod) and FD (WENO), with the Pang & Wu (2024)
             reference overlaid.

Both solvers run at 200^2 in a periodic box of size 2 pi.  Final states are
cached under ``data/`` so the figure can be re-styled without re-running.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/mhd/orszag_tang.py [--rerun]
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
# dimensionality-aware interface-field initializer (the top-level one assumes 3D)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    FINITE_DIFFERENCE,
    HLL,
    MINMOD,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

from _common import (
    DATA_DIR,
    FIG_DIR,
    rerun_requested,
    FD_LABEL,
    FV_HLL_LABEL,
    FD_COLOR,
    FV_HLL_COLOR,
    mhd_registered_variables,
)

GAMMA = 5.0 / 3.0
BOX_SIZE = 2.0 * np.pi
RES_LOW = 200
RES_HIGH = 1024
T_END = 3.0
Y_CUT = 0.625 * np.pi

PERIODIC_2D = BoundarySettings(
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def simulate(solver_mode, num_cells):
    is_fd = solver_mode == FINITE_DIFFERENCE
    config = SimulationConfig(
        mhd=True,
        solver_mode=solver_mode,
        dimensionality=2,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        limiter=MINMOD,
        riemann_solver=HLL,
        progress_bar=True,
        boundary_settings=PERIODIC_2D,
        # FD/WENO runs ~10x faster through the Pallas backend (bit-compatible).
        # The block must divide the grid; (8, 8, 1) divides both 200 and 400.
        backend=PALLAS if is_fd else NATIVE_JAX,
        pallas_block_shape=(8, 8, 1),
        pallas_use_triton=True,
        pallas_interpret=False,
    )
    rv = get_registered_variables(config)

    dx = BOX_SIZE / num_cells
    x = jnp.linspace(dx / 2, BOX_SIZE - dx / 2, num_cells)
    X, Y = jnp.meshgrid(x, x, indexing="ij")

    rho = jnp.ones_like(X) * GAMMA**2
    P = jnp.ones_like(X) * GAMMA
    V_x = -jnp.sin(Y)
    V_y = jnp.sin(X)
    B_x = -jnp.sin(Y)
    B_y = jnp.sin(2 * X)
    B_z = jnp.zeros_like(X)

    if solver_mode == FINITE_DIFFERENCE:
        bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z, config.dimensionality)
    else:
        bxb, byb, bzb = None, None, None

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=rv,
        density=rho,
        velocity_x=V_x,
        velocity_y=V_y,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
        gas_pressure=P,
    )

    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, C_cfl=0.4, gamma=GAMMA)

    final_state = time_integration(initial_state, config, params, rv)
    return np.asarray(final_state), np.asarray(x)


def cache_path(name, num_cells):
    return DATA_DIR / f"orszag_tang_{name}_{num_cells}.npz"


def get_run(name, solver_mode, num_cells, rerun):
    path = cache_path(name, num_cells)
    if path.exists() and not rerun:
        d = np.load(path)
        return d["final_state"], d["x"]
    print(f"running Orszag-Tang {name} at {num_cells}^2 ...")
    final_state, x = simulate(solver_mode, num_cells)
    np.savez(path, final_state=final_state, x=x)
    print(f"  cached -> {path}")
    return final_state, x


def plot(rerun):
    # left panel: high-resolution FD field; right panel: resolution/solver
    # comparison of the density cut.
    fd_hi, x_hi = get_run("fd", FINITE_DIFFERENCE, RES_HIGH, rerun)
    fd_lo, x_lo = get_run("fd", FINITE_DIFFERENCE, RES_LOW, rerun)
    fv_lo, x_fv = get_run("fv_hll", FINITE_VOLUME, RES_LOW, rerun)

    rv_fd = mhd_registered_variables(FINITE_DIFFERENCE, dimensionality=2)
    rv_fv = mhd_registered_variables(FINITE_VOLUME, dimensionality=2)

    def cut(state, rv, x):
        y_index = int(np.argmin(np.abs(x - Y_CUT)))
        return state[rv.density_index, :, y_index]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: high-resolution FD density field (rasterized for SVG size) ---
    density_fd = fd_hi[rv_fd.density_index]
    im = axs[0].imshow(
        density_fd.T, origin="lower",
        extent=(0, BOX_SIZE, 0, BOX_SIZE), cmap="viridis",
        rasterized=True,
    )
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_aspect("equal", adjustable="box")
    cax = make_axes_locatable(axs[0]).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label="density")
    # mark the profile cut
    axs[0].axhline(Y_CUT, color="white", ls="--", lw=1.0, alpha=0.8)

    # --- right: density cut, FV 200 and FD 200 vs the FD 1024 reference ---
    axs[1].plot(x_hi, cut(fd_hi, rv_fd, x_hi),
                color="black", lw=1.6, label=f"{FD_LABEL}, {RES_HIGH} (reference)")
    axs[1].plot(x_fv, cut(fv_lo, rv_fv, x_fv),
                color=FV_HLL_COLOR, label=f"{FV_HLL_LABEL}, {RES_LOW}")
    axs[1].plot(x_lo, cut(fd_lo, rv_fd, x_lo),
                color=FD_COLOR, ls="--", label=f"{FD_LABEL}, {RES_LOW}")

    axs[1].set_xlabel("x")
    axs[1].set_ylabel("density")
    axs[1].legend(fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "orszag_tang.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    plot(rerun_requested())
