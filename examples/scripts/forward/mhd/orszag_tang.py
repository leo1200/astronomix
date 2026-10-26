"""Orszag-Tang vortex (2D MHD) — paper figure.

Left panel:  finite-difference density at t = 3.0 on 1024^2 cells.
Right panel: density profile along the cut y = 0.625 pi for the
             finite-volume and finite-difference schemes at 200^2 against the
             finite-difference 1024^2 result.

The finite-volume scheme is AthenaPK's VL2 (van Leer predictor-corrector, PLM
with van Leer slopes, HLLD, GLM divergence cleaning); the finite-difference
scheme uses WENO-Z weights (no positivity-preserving WENO). Both run at
C_cfl = 0.4 in a periodic box of size 2 pi (Pang & Wu 2024, Example 5.2).
The finite-difference scheme also runs in the setup of the supersonic
turbulence (positivity-preserving WENO, upwind dissipation of the CT edge EMFs
at the default strength, C_cfl = 0.75) at both resolutions; its 200^2 cut is
drawn, and its deviations from the plain finite-difference runs are printed.
Final states are cached under ``data/`` so the figure can be re-styled
without re-running.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/orszag_tang.py [--rerun]
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
    FINITE_VOLUME,
    FINITE_DIFFERENCE,
    HLLD,
    PALLAS,
    PERIODIC_BOUNDARY,
    VL2,
)
from astronomix.plotting_helpers.paper_style import (
    REFERENCE_COLOR,
    TEXT_WIDTH,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    BackendConfig,
)

# astronomix functions
from astronomix import (
    get_registered_variables,
    finalize_config,
    time_integration,
    initialize_interface_fields,
    construct_primitive_state,
)
from astronomix.plotting_helpers.paper_style import (
    attach_colorbar,
    figure_legend,
    use_paper_style,
)

# shared figure helpers
from _common import DATA_DIR
from _common import FIG_DIR
from _common import rerun_requested
from _common import FD_COLOR
from _common import FV_COLOR


GAMMA = 5.0 / 3.0
BOX_SIZE = 2.0 * np.pi
RES_LOW = 200
RES_HIGH = 1024
T_END = 3.0
Y_CUT = 0.625 * np.pi

# Both schemes run at the C_cfl = 0.4 of this test (also AthenaPK's value for
# its 2D Orszag-Tang problem).
C_CFL = 0.4

FD_LABEL = "FD (WENO-Z)"
FD_TURBULENCE_LABEL = "FD (WENO-Z + PP + upwind EMF)"

#: C_cfl of the turbulence setup, the bound of the positivity proof.
C_CFL_TURBULENCE = 0.75
FV_LABEL = "FV (VL2, HLLD)"

PERIODIC_2D = BoundarySettings(
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def make_config(solver_mode, num_cells, turbulence_setup=False):
    """Build the Orszag-Tang configuration of one scheme.

    Args:
        solver_mode: The solver mode (FINITE_VOLUME or FINITE_DIFFERENCE).
        num_cells: The per-dimension resolution of the square grid.
        turbulence_setup: Finite difference only: add the positivity-preserving
            WENO and the upwind EMF dissipation (the supersonic turbulence
            setup).

    Returns:
        The (not yet finalized) simulation configuration.
    """
    # The Pallas block must divide the grid; (8, 8, 1) divides both 200 and
    # 1024. The finite-difference MHD kernels are three-dimensional only, so
    # the 2D finite-difference run computes its fluxes natively.
    common_settings = dict(
        mhd=True,
        solver_mode=solver_mode,
        dimensionality=2,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        progress_bar=True,
        boundary_settings=PERIODIC_2D,
        backend_config=BackendConfig(
            backend=PALLAS,
            pallas_block_shape=(8, 8, 1),
        ),
    )
    if solver_mode == FINITE_DIFFERENCE:
        return SimulationConfig(
            weno_z=True,
            weno_positivity_preserving=turbulence_setup,
            ct_upwind_dissipation=turbulence_setup,
            **common_settings,
        )
    return SimulationConfig(
        time_integrator=VL2,
        riemann_solver=HLLD,
        **common_settings,
    )


def simulate(solver_mode, num_cells, turbulence_setup=False):
    """Run the Orszag-Tang vortex at ``num_cells``^2 with the given solver.

    Sets up the standard Orszag-Tang initial condition in a periodic box of
    size ``2*pi`` and evolves it to ``T_END``.

    Args:
        solver_mode: The solver mode (FINITE_VOLUME or FINITE_DIFFERENCE).
        num_cells: The per-dimension resolution of the square grid.
        turbulence_setup: See ``make_config``.

    Returns:
        A tuple ``(density, x)`` of the final density field and the 1D cell-
        centre coordinate array, both as numpy arrays.
    """
    config = make_config(solver_mode, num_cells, turbulence_setup)
    registered_variables = get_registered_variables(config)

    # -------------------------------------------------------------
    # ============= ↓ Orszag-Tang initial condition ↓ ============
    # -------------------------------------------------------------

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

    # The finite-difference (constrained-transport) solver additionally needs
    # the staggered interface magnetic fields; the finite-volume VL2 scheme
    # keeps cell-centred fields and cleans their divergence (GLM).
    if solver_mode == FINITE_DIFFERENCE:
        bxb, byb, bzb = initialize_interface_fields(
            B_x,
            B_y,
            B_z,
            config.dimensionality,
        )
    else:
        bxb, byb, bzb = None, None, None

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
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
    params = SimulationParams(
        t_end=T_END,
        C_cfl=C_CFL_TURBULENCE if turbulence_setup else C_CFL,
        gamma=GAMMA,
    )

    # -------------------------------------------------------------
    # ============= ↑ Orszag-Tang initial condition ↑ ============
    # -------------------------------------------------------------

    final_state = time_integration(initial_state, config, params, registered_variables)
    density = final_state[registered_variables.density_index]
    print(
        f"  min density {float(jnp.min(density)):.3e}, "
        f"min pressure {float(jnp.min(final_state[registered_variables.pressure_index])):.3e}, "
        f"finite: {bool(jnp.all(jnp.isfinite(final_state)))}"
    )
    return np.asarray(density), np.asarray(x)


def cache_path(name, num_cells):
    """Return the cache path for the run labelled ``name`` at ``num_cells``^2."""
    return DATA_DIR / f"orszag_tang_{name}_{num_cells}.npz"


def get_run(name, solver_mode, num_cells, rerun, turbulence_setup=False):
    """Return an Orszag-Tang run's final density, from cache or by (re-)running.

    Args:
        name: The short cache label for this run (e.g. ``"fd"``, ``"fv_vl2"``).
        solver_mode: The solver mode used for the run.
        num_cells: The per-dimension resolution of the run.
        rerun: When True, ignore any cache and re-run the simulation.
        turbulence_setup: See ``make_config``.

    Returns:
        A tuple ``(density, x)`` of the final density field and coordinate
        array.
    """
    path = cache_path(name, num_cells)
    if path.exists() and not rerun:
        cached = np.load(path)
        return cached["density"], cached["x"]
    print(f"running Orszag-Tang {name} at {num_cells}^2 ...")
    density, x = simulate(solver_mode, num_cells, turbulence_setup)
    np.savez(path, density=density, x=x)
    print(f"  cached -> {path}")
    return density, x


def plot(rerun):
    """Render the Orszag-Tang two-panel figure.

    The left panel shows the high-resolution finite-difference density field;
    the right panel compares the density profile along the cut ``Y_CUT`` for
    the finite-volume and finite-difference schemes at ``RES_LOW`` against
    the finite-difference ``RES_HIGH`` result. Saves ``orszag_tang.svg`` (and
    a ``.png`` copy).

    Args:
        rerun: When True, force the underlying simulations to be re-run.
    """

    # -------------------------------------------------------------
    # ================= ↓ Load / run the states ↓ ================
    # -------------------------------------------------------------

    fd_hi, x_hi = get_run("fd", FINITE_DIFFERENCE, RES_HIGH, rerun)
    fd_lo, x_lo = get_run("fd", FINITE_DIFFERENCE, RES_LOW, rerun)
    fv_lo, x_fv = get_run("fv_vl2", FINITE_VOLUME, RES_LOW, rerun)
    turbulence_hi, _ = get_run("fd_turbulence", FINITE_DIFFERENCE, RES_HIGH, rerun, turbulence_setup=True)
    turbulence_lo, _ = get_run("fd_turbulence", FINITE_DIFFERENCE, RES_LOW, rerun, turbulence_setup=True)

    def cut(density, x):
        """Return the density profile along the horizontal cut ``Y_CUT``."""
        y_index = int(np.argmin(np.abs(x - Y_CUT)))
        return density[:, y_index]

    # The mean absolute difference of each 200^2 cut from the 1024^2 cut
    # (interpolated to the coarse cell centres) quantifies which scheme is
    # closer to the high-resolution solution.
    reference_cut = np.interp(x_lo, x_hi, cut(fd_hi, x_hi))
    for label, density, x in (
        (FV_LABEL, fv_lo, x_fv),
        (FD_LABEL, fd_lo, x_lo),
        (FD_TURBULENCE_LABEL, turbulence_lo, x_lo),
    ):
        cut_difference = np.mean(np.abs(cut(density, x) - reference_cut))
        print(f"{label}, {RES_LOW}: mean |cut - {RES_HIGH} cut| = {cut_difference:.4f}")
    # How far the turbulence setup moves the whole density field.
    for resolution, plain, turbulence in ((RES_LOW, fd_lo, turbulence_lo), (RES_HIGH, fd_hi, turbulence_hi)):
        print(f"{FD_TURBULENCE_LABEL} vs {FD_LABEL}, {resolution}: mean |density difference| = "
              f"{np.mean(np.abs(turbulence - plain)):.4f} (mean density {np.mean(plain):.3f})")

    # -------------------------------------------------------------
    # ================= ↑ Load / run the states ↑ ================
    # -------------------------------------------------------------

    use_paper_style()
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.0))

    # -------------------------------------------------------------
    # ============ ↓ Left: high-resolution FD field ↓ ============
    # -------------------------------------------------------------

    # Rasterize the image to keep the resulting SVG small.
    image = axs[0].imshow(
        fd_hi.T,
        origin="lower",
        extent=(0, BOX_SIZE, 0, BOX_SIZE),
        cmap="viridis",
        rasterized=True,
    )
    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel("$y$")
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].grid(False)
    attach_colorbar(fig, image, axs[0], label="density")

    # Mark where the profile cut is taken.
    axs[0].axhline(Y_CUT, color="white", ls="--", lw=1.0, alpha=0.8)

    # -------------------------------------------------------------
    # ============ ↑ Left: high-resolution FD field ↑ ============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =========== ↓ Right: density-cut comparison ↓ ==============
    # -------------------------------------------------------------

    # FV 200 and FD 200 are compared against the FD 1024 reference.
    axs[1].plot(
        x_hi,
        cut(fd_hi, x_hi),
        color=REFERENCE_COLOR,
        linewidth=1.2,
        label=f"{FD_LABEL}, ${RES_HIGH}^2$ (reference)",
    )
    axs[1].plot(
        x_fv,
        cut(fv_lo, x_fv),
        color=FV_COLOR,
        label=f"{FV_LABEL}, ${RES_LOW}^2$",
    )
    axs[1].plot(
        x_lo,
        cut(fd_lo, x_lo),
        color=FD_COLOR,
        linestyle="--",
        label=f"{FD_LABEL}, ${RES_LOW}^2$",
    )
    axs[1].plot(
        x_lo,
        cut(turbulence_lo, x_lo),
        color=FD_COLOR,
        linestyle=":",
        label=f"{FD_TURBULENCE_LABEL}, ${RES_LOW}^2$",
    )

    axs[1].set_xlabel("$x$")
    axs[1].set_ylabel("density")
    figure_legend(fig, ncols=2)

    # -------------------------------------------------------------
    # =========== ↑ Right: density-cut comparison ↑ ==============
    # -------------------------------------------------------------

    out = FIG_DIR / "orszag_tang.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    plot(rerun_requested())
