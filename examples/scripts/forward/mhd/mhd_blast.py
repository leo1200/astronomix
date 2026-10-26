"""3D MHD blast-wave test 1 (Seo & Ryu 2023, Sec. 3.7) — paper figures.

Two figures, both built from blast-wave simulations cached under ``data/``
(pass ``--rerun`` to force a recompute even when a cache exists):

1. ``mhd_blast_oscillations_comparison.png`` — a resolution-by-scheme grid of
   central density slices: the finite-volume VL2 scheme with the HLLD and the
   HLLE Riemann solver and the finite-difference scheme, to show which
   schemes produce post-shock oscillations.

2. ``mhd_blast_test1_<N>cells.png`` — the finite-difference result at ``N``^3 in
   the first two columns (density / kinetic-energy / magnetic-pressure /
   pressure slices) and, in the third (profile) column, the ``|B|^2`` and
   pressure profiles along the box diagonal with the finite-volume (VL2,
   HLLD) result and the finite-difference scheme in the setup of the
   supersonic turbulence (below) at the same resolution overlaid.

The finite-volume scheme is AthenaPK's VL2 (van Leer predictor-corrector, PLM
with van Leer slopes, GLM divergence cleaning) at C_cfl = 0.3 with the
safeguards AthenaPK uses for strong shocks, the first-order flux correction
and density and pressure floors. The finite-difference scheme uses WENO-Z
weights at C_cfl = 0.75 (the test needs no positivity-preserving WENO); it
also runs in the setup of the supersonic turbulence (positivity-preserving
WENO and the upwind dissipation of the CT edge EMFs at the default strength,
C_cfl = 0.75), whose central-slice deviation from the plain run is printed.
``--without-fofc`` additionally runs the finite-volume schemes without these
safeguards and prints how much the results differ, i.e. whether the
safeguards are needed.

Run with the repo on PYTHONPATH:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/mhd_blast.py [--res 128 256] [--rerun] [--without-fofc]
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import sys

# jax
import jax
# The blast is heavy at high resolution; single precision roughly halves
# compute and memory.
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    FINITE_DIFFERENCE,
    HLL,
    HLLD,
    PALLAS,
    PERIODIC_BOUNDARY,
    VL2,
)
from astronomix.option_classes.simulation_config import POSITIVITY_HARD_FLOOR
from astronomix.plotting_helpers.paper_style import (
    SEQUENTIAL_COLORMAP,
    TEXT_WIDTH,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
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
    reserve_colorbar_space,
    use_paper_style,
)

# shared figure helpers
from _common import DATA_DIR
from _common import FIG_DIR
from _common import rerun_requested
from _common import FD_COLOR
from _common import FV_COLOR


# -------------------------------------------------------------
# ================= ↓ Test-wide constants ↓ ==================
# -------------------------------------------------------------

BOX_SIZE = 1.0        # periodic box of size 1.0 (Seo & Ryu setup)
T_END = 0.02          # end time of the blast-wave test
B0 = 10.0             # background field strength, B = (B0/√2, B0/√2, 0)
R0 = 0.125            # inner blast radius (p = 100 inside)
R1 = 1.1 * R0         # outer blast radius (linear pressure ramp between R0/R1)

# Resolutions down the rows of the comparison grid. Default is the lighter
# [64, 128]; pass ``--res 128 256`` for the paper figures
# (mhd_blast_test1_256cells and the oscillation comparison at 128^3 / 256^3).
if "--res" in sys.argv:
    first_resolution_index = sys.argv.index("--res") + 1
    RESOLUTIONS = [
        int(argument) for argument in sys.argv[first_resolution_index:] if argument.isdigit()
    ] or [64, 128]
else:
    RESOLUTIONS = [64, 128]

# Run the finite-volume schemes also without the strong-shock safeguards.
WITHOUT_FOFC_CHECK = "--without-fofc" in sys.argv

# AthenaPK's density and pressure floors (dfloor / pfloor) for the
# finite-volume runs; far below the ambient density and pressure of 1.
DENSITY_FLOOR = 1e-6
PRESSURE_FLOOR = 1e-6

FD_LABEL = "FD (WENO-Z)"
FD_TURBULENCE_LABEL = "FD (WENO-Z + PP + upwind EMF)"
FV_LABEL = "FV (VL2, HLLD)"

# Solver-scheme columns of the oscillation comparison: cache/config name ->
# plot label. Each name resolves to a concrete configuration via
# ``SCHEME_CONFIGS`` below.
CONFIGS = [
    ("fv_vl2_hlld", "FV (VL2, HLLD)"),
    ("fv_vl2_hlle", "FV (VL2, HLLE)"),
    ("fd_mhd", FD_LABEL),
]


def _finite_volume_scheme(riemann_solver, strong_shock_safeguards):
    """Return the configuration and parameter overrides of a VL2 scheme.

    Args:
        riemann_solver: HLLD or HLL (AthenaPK's HLLE).
        strong_shock_safeguards: Whether to use AthenaPK's first-order flux
            correction and its density and pressure floors.

    Returns:
        A dict with the ``config`` (SimulationConfig) and ``params``
        (SimulationParams) overrides.
    """
    if strong_shock_safeguards:
        positivity_config = PositivityConfig(per_step_mode=POSITIVITY_HARD_FLOOR)
    else:
        positivity_config = PositivityConfig()
    return {
        "config": dict(
            solver_mode=FINITE_VOLUME,
            time_integrator=VL2,
            riemann_solver=riemann_solver,
            first_order_flux_correction=strong_shock_safeguards,
            positivity_config=positivity_config,
        ),
        "params": dict(
            C_cfl=0.3,
            minimum_density=DENSITY_FLOOR,
            minimum_pressure=PRESSURE_FLOOR,
        ),
    }


# Per-scheme solver settings: ``config`` holds SimulationConfig overrides and
# ``params`` holds SimulationParams overrides. The finite-volume schemes run
# at AthenaPK's 3D CFL number, the finite-difference scheme at C_cfl = 0.75.
SCHEME_CONFIGS = {
    "fv_vl2_hlld": _finite_volume_scheme(HLLD, strong_shock_safeguards=True),
    "fv_vl2_hlle": _finite_volume_scheme(HLL, strong_shock_safeguards=True),
    "fv_vl2_hlld_without_fofc": _finite_volume_scheme(HLLD, strong_shock_safeguards=False),
    "fv_vl2_hlle_without_fofc": _finite_volume_scheme(HLL, strong_shock_safeguards=False),
    "fd_mhd": {
        "config": dict(weno_z=True),
        "params": dict(C_cfl=0.75),
    },
    # The setup of the supersonic turbulence.
    "fd_mhd_turbulence": {
        "config": dict(weno_z=True, weno_positivity_preserving=True, ct_upwind_dissipation=True),
        "params": dict(C_cfl=0.75),
    },
}

# 3D periodic boundaries; ``finalize_config`` promotes this to the faster
# PERIODIC_ROLL handling automatically for a fully-periodic 3D box.
PERIODIC_3D = BoundarySettings(
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def make_config(config_name, num_cells):
    """Build the (not yet finalized) configuration of one scheme.

    Args:
        config_name: The scheme key, a key of ``SCHEME_CONFIGS``.
        num_cells: The per-dimension resolution of the run.

    Returns:
        The simulation configuration.
    """
    scheme_settings = dict(backend_config=BackendConfig(backend=PALLAS))
    scheme_settings.update(SCHEME_CONFIGS[config_name]["config"])
    return SimulationConfig(
        mhd=True,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        boundary_settings=PERIODIC_3D,
        progress_bar=True,
        **scheme_settings,
    )


def registered_variables_for(config_name):
    """Return the variable registry of a scheme's state layout.

    The layouts differ: the finite-difference state carries the interface
    magnetic field, the VL2 state the divergence-cleaning scalar psi.

    Args:
        config_name: The scheme key, a key of ``SCHEME_CONFIGS``.

    Returns:
        The RegisteredVariables of that scheme.
    """
    return get_registered_variables(make_config(config_name, RESOLUTIONS[-1]))


# -------------------------------------------------------------
# ================= ↑ Test-wide constants ↑ ==================
# -------------------------------------------------------------


def run_and_cache(config_name, num_cells):
    """Run the blast-wave test for one scheme and cache its central slice.

    Builds the Seo & Ryu blast initial condition at ``num_cells``^3, configures
    the solver according to ``SCHEME_CONFIGS[config_name]``, evolves it to
    ``T_END`` and writes the central ``z`` slice of the final state (all the
    figures show) together with the extrema of the whole final state to
    ``data/mhd_blast_<config_name>_<N>cells.npz``.

    Args:
        config_name: The scheme key (e.g. ``"fd_mhd"``), a key of
            ``SCHEME_CONFIGS``.
        num_cells: The per-dimension resolution of the run.

    Returns:
        The central slice ``final_state[:, :, :, num_cells // 2]`` as a numpy
        array.
    """

    # -------------------------------------------------------------
    # =============== ↓ Solver configuration ↓ ===================
    # -------------------------------------------------------------

    scheme = SCHEME_CONFIGS[config_name]
    config = make_config(config_name, num_cells)
    registered_variables = get_registered_variables(config)

    # -------------------------------------------------------------
    # =============== ↑ Solver configuration ↑ ===================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Blast-wave initial condition ↓ ==============
    # -------------------------------------------------------------

    # Cell-centre coordinates of the periodic box and the radial distance from
    # the box centre.  (helper_data.r is only populated for curvilinear
    # geometries, so the radius is built explicitly here.)
    dx = BOX_SIZE / num_cells
    x = jnp.linspace(dx / 2, BOX_SIZE - dx / 2, num_cells)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    centre = BOX_SIZE / 2
    r = jnp.sqrt((X - centre) ** 2 + (Y - centre) ** 2 + (Z - centre) ** 2)

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r)
    P = jnp.where(r <= R0, 100.0, P)
    P = jnp.where((r > R0) & (r <= R1), 1.0 + 99.0 * (R1 - r) / (R1 - R0), P)
    P = jnp.where(r > R1, 1.0, P)

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    B_x = jnp.ones_like(r) * (B0 / jnp.sqrt(2))
    B_y = jnp.ones_like(r) * (B0 / jnp.sqrt(2))
    B_z = jnp.zeros_like(r)

    # The finite-difference (constrained-transport) solver additionally needs
    # the staggered interface magnetic fields; the finite-volume solver does not.
    if config.solver_mode == FINITE_DIFFERENCE:
        bxb, byb, bzb = initialize_interface_fields(
            B_x, B_y, B_z, config.dimensionality
        )
    else:
        bxb, byb, bzb = None, None, None

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=V_x,
        velocity_y=V_y,
        velocity_z=V_z,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
        gas_pressure=P,
    )

    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, **scheme["params"])

    # -------------------------------------------------------------
    # ============ ↑ Blast-wave initial condition ↑ ==============
    # -------------------------------------------------------------

    print(f"running MHD blast {config_name} at {num_cells}^3 ...")
    final_state = time_integration(initial_state, config, params, registered_variables)
    minimum_density = float(jnp.min(final_state[registered_variables.density_index]))
    minimum_pressure = float(jnp.min(final_state[registered_variables.pressure_index]))
    all_finite = bool(jnp.all(jnp.isfinite(final_state)))
    print(
        f"  min density {minimum_density:.3e}, min pressure {minimum_pressure:.3e}, "
        f"finite: {all_finite}"
    )
    central_slice = np.asarray(final_state[..., num_cells // 2])
    path = DATA_DIR / f"mhd_blast_{config_name}_{num_cells}cells.npz"
    np.savez(
        path,
        central_slice=central_slice,
        minimum_density=minimum_density,
        minimum_pressure=minimum_pressure,
        all_finite=all_finite,
    )
    print(f"  cached -> {path}")
    return central_slice


def load_central_slice(config_name, num_cells):
    """Return the central slice of a blast-wave final state, from cache or by
    running the simulation.

    Args:
        config_name: The scheme key (e.g. ``"fd_mhd"``).
        num_cells: The per-dimension resolution of the run.

    Returns:
        The central slice of the final state, shape ``(num_vars, N, N)``.
    """
    path = DATA_DIR / f"mhd_blast_{config_name}_{num_cells}cells.npz"
    if path.exists():
        return np.load(path)["central_slice"]
    return run_and_cache(config_name, num_cells)


def report_safeguard_effect():
    """
    Print, per resolution and Riemann solver, how much the central density
    slice of the VL2 run with the first-order flux correction and floors
    differs from the run without them. Zero means the safeguards never acted
    on this slice (they only change a cell's update when it would otherwise
    become non-positive).
    """
    density_index = registered_variables_for("fv_vl2_hlld").density_index
    for num_cells in RESOLUTIONS:
        for safeguarded_name in ("fv_vl2_hlld", "fv_vl2_hlle"):
            plain_name = f"{safeguarded_name}_without_fofc"
            safeguarded = load_central_slice(safeguarded_name, num_cells)[density_index]
            plain = load_central_slice(plain_name, num_cells)[density_index]
            plain_cache = np.load(DATA_DIR / f"mhd_blast_{plain_name}_{num_cells}cells.npz")
            largest_difference = np.max(np.abs(safeguarded - plain))
            print(
                f"{safeguarded_name} at {num_cells}^3 without the safeguards: "
                f"finite {bool(plain_cache['all_finite'])}, "
                f"min density {float(plain_cache['minimum_density']):.3e}, "
                f"min pressure {float(plain_cache['minimum_pressure']):.3e}, "
                f"largest central-slice density difference {largest_difference:.3e}"
            )


def report_turbulence_setup_effect():
    """
    Print, per resolution, how far the finite-difference run in the setup of
    the supersonic turbulence lies from the plain finite-difference run: the
    largest and mean central-slice density differences and the central
    pressure peaks of both.
    """
    registered_variables = registered_variables_for("fd_mhd")
    density_index = registered_variables.density_index
    pressure_index = registered_variables.pressure_index
    for num_cells in RESOLUTIONS:
        plain = load_central_slice("fd_mhd", num_cells)
        turbulence = load_central_slice("fd_mhd_turbulence", num_cells)
        difference = np.abs(turbulence[density_index] - plain[density_index])
        print(
            f"{FD_TURBULENCE_LABEL} vs {FD_LABEL} at {num_cells}^3: central-slice density "
            f"difference max {difference.max():.3e}, mean {difference.mean():.3e}; "
            f"pressure peak {turbulence[pressure_index].max():.2f} vs {plain[pressure_index].max():.2f}"
        )


def make_oscillation_comparison():
    """Build the resolution-by-scheme oscillation-comparison figure.

    A grid of central density slices with resolution down the rows and solver
    scheme across the columns, to compare the post-shock structure of the
    finite-volume VL2 scheme (HLLD and HLLE) with the finite-difference
    scheme. Saves ``mhd_blast_oscillations_comparison.png``.
    """

    # -------------------------------------------------------------
    # ================= ↓ Panel grid layout ↓ ====================
    # -------------------------------------------------------------

    resolutions = RESOLUTIONS
    configs = CONFIGS

    use_paper_style()
    panel_size = 0.8 * TEXT_WIDTH / len(configs)
    fig, axs = plt.subplots(
        len(resolutions),
        len(configs),
        figsize=(panel_size * len(configs), panel_size * len(resolutions) + 0.3),
        squeeze=False,
    )

    # -------------------------------------------------------------
    # ================= ↑ Panel grid layout ↑ ====================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============= ↓ Density slices per panel ↓ =================
    # -------------------------------------------------------------

    # Fix the colour scale across every panel so the schemes are directly
    # comparable by eye.
    density_min, density_max = 0.0, 1.5

    for row_index, num_cells in enumerate(resolutions):
        for column_index, (config_name, title) in enumerate(configs):
            central_slice = load_central_slice(config_name, num_cells)
            registered_variables = registered_variables_for(config_name)
            density_slice = central_slice[registered_variables.density_index]
            image = axs[row_index, column_index].imshow(
                density_slice,
                vmin=density_min,
                vmax=density_max,
                cmap=SEQUENTIAL_COLORMAP,
            )
            if row_index == 0:
                axs[row_index, column_index].set_title(title)
            axs[row_index, column_index].set_xticks([])
            axs[row_index, column_index].set_yticks([])
            axs[row_index, column_index].grid(False)
            # One colorbar per row, of the height of the panels, on the right;
            # the other panels keep the same size.
            if column_index == len(configs) - 1:
                attach_colorbar(fig, image, axs[row_index, column_index], label="density")
            else:
                reserve_colorbar_space(axs[row_index, column_index])

    for row_index, num_cells in enumerate(resolutions):
        axs[row_index, 0].set_ylabel(f"{num_cells}$^3$ cells")

    # -------------------------------------------------------------
    # ============= ↑ Density slices per panel ↑ =================
    # -------------------------------------------------------------

    fig.tight_layout()

    out = FIG_DIR / "mhd_blast_oscillations_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def make_fd_with_profiles(num_cells):
    """Build the finite-difference slices + diagonal-profile figure.

    Columns 0-1 show the finite-difference result at ``num_cells``^3 (density,
    kinetic energy, magnetic pressure and pressure slices through the box
    centre). Column 2 shows the ``|B|^2`` and pressure profiles along the box
    diagonal with the finite-volume (VL2, HLLD) result overlaid for
    comparison. Saves ``mhd_blast_test1_{num_cells}cells.png``.

    Args:
        num_cells: The per-dimension resolution of the runs to plot.
    """

    # -------------------------------------------------------------
    # ============ ↓ Load states and extract slices ↓ ============
    # -------------------------------------------------------------

    fd_central_slice = load_central_slice("fd_mhd", num_cells)
    fv_central_slice = load_central_slice("fv_vl2_hlld", num_cells)

    extent = (0, BOX_SIZE, 0, BOX_SIZE)

    def slices(central_slice, registered_variables):
        """Return the central (density, pressure, |B|^2, v^2) slices of a
        state, indexed through ``registered_variables``."""
        magnetic_index = registered_variables.magnetic_index
        velocity_index = registered_variables.velocity_index
        density = central_slice[registered_variables.density_index]
        pressure = central_slice[registered_variables.pressure_index]
        b_sq = (
            central_slice[magnetic_index.x] ** 2
            + central_slice[magnetic_index.y] ** 2
            + central_slice[magnetic_index.z] ** 2
        )
        v_sq = (
            central_slice[velocity_index.x] ** 2
            + central_slice[velocity_index.y] ** 2
            + central_slice[velocity_index.z] ** 2
        )
        return density, pressure, b_sq, v_sq

    density, pressure, b_sq, v_sq = slices(
        fd_central_slice,
        registered_variables_for("fd_mhd"),
    )

    # -------------------------------------------------------------
    # ============ ↑ Load states and extract slices ↑ ============
    # -------------------------------------------------------------

    use_paper_style()
    # The panels are placed explicitly (in inches): two columns of square
    # slices, each with its colour bar, and a column with the two diagonal
    # profiles, which share their axis; the slices share the box of the setup
    # and carry no axis ticks, their field is the panel title.
    figure_height = 4.6
    fig = plt.figure(figsize=(TEXT_WIDTH, figure_height))
    slice_size = 1.6
    slice_box_width = 1.1 * slice_size  # slice, colour-bar gap and colour bar
    colorbar_labels = 0.32
    row_bottoms = (0.8 + slice_size + 0.32, 0.8)
    column_lefts = (0.08, 0.08 + slice_box_width + colorbar_labels + 0.15)
    profile_left = column_lefts[1] + slice_box_width + colorbar_labels + 0.62
    profile_width = TEXT_WIDTH - 0.06 - profile_left

    def axes_at(left, bottom, width, height):
        """An axes at a position given in inches."""
        return fig.add_axes(
            [left / TEXT_WIDTH, bottom / figure_height, width / TEXT_WIDTH, height / figure_height]
        )

    axs = np.empty((2, 3), dtype=object)
    for row, bottom in enumerate(row_bottoms):
        for column, left in enumerate(column_lefts):
            axs[row, column] = axes_at(left, bottom, slice_box_width, slice_size)
        axs[row, 2] = axes_at(profile_left, bottom, profile_width, slice_size)
    axs[0, 2].sharex(axs[1, 2])

    def imshow_panel(ax, field, label):
        """Draw one field slice as an image panel with an attached colour bar."""
        image = ax.imshow(field.T, origin="lower", extent=extent, cmap=SEQUENTIAL_COLORMAP)
        attach_colorbar(fig, image, ax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # -------------------------------------------------------------
    # ============= ↓ Columns 0-1: FD slices ↓ ===================
    # -------------------------------------------------------------

    imshow_panel(axs[0, 0], density, "density")
    imshow_panel(axs[0, 1], v_sq, r"$v^2$")
    imshow_panel(axs[1, 0], b_sq, r"$B^2$")
    imshow_panel(axs[1, 1], pressure, "pressure")

    # -------------------------------------------------------------
    # ============= ↑ Columns 0-1: FD slices ↑ ===================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ======== ↓ Column 2: diagonal profiles, FD vs FV ↓ =========
    # -------------------------------------------------------------

    diagonal = np.arange(num_cells)
    r_diagonal = np.sqrt(2.0) * diagonal * (BOX_SIZE / num_cells)

    _, fv_pressure_slice, fv_b_sq, _ = slices(
        fv_central_slice,
        registered_variables_for("fv_vl2_hlld"),
    )
    _, turbulence_pressure_slice, turbulence_b_sq, _ = slices(
        load_central_slice("fd_mhd_turbulence", num_cells),
        registered_variables_for("fd_mhd_turbulence"),
    )

    b_diagonal_fd = b_sq[diagonal, diagonal]
    b_diagonal_fv = fv_b_sq[diagonal, diagonal]
    axs[0, 2].plot(r_diagonal, b_diagonal_fd, color=FD_COLOR, label=FD_LABEL)
    axs[0, 2].plot(
        r_diagonal,
        b_diagonal_fv,
        color=FV_COLOR,
        linestyle="--",
        label=FV_LABEL,
    )
    axs[0, 2].plot(
        r_diagonal,
        turbulence_b_sq[diagonal, diagonal],
        color=FD_COLOR,
        linestyle=":",
        label=FD_TURBULENCE_LABEL,
    )
    axs[0, 2].set_ylabel(r"$B^2$")
    axs[0, 2].tick_params(labelbottom=False)

    pressure_diagonal_fd = pressure[diagonal, diagonal]
    pressure_diagonal_fv = fv_pressure_slice[diagonal, diagonal]
    axs[1, 2].plot(r_diagonal, pressure_diagonal_fd, color=FD_COLOR, label=FD_LABEL)
    axs[1, 2].plot(
        r_diagonal,
        pressure_diagonal_fv,
        color=FV_COLOR,
        linestyle="--",
        label=FV_LABEL,
    )
    axs[1, 2].plot(
        r_diagonal,
        turbulence_pressure_slice[diagonal, diagonal],
        color=FD_COLOR,
        linestyle=":",
        label=FD_TURBULENCE_LABEL,
    )
    axs[1, 2].set_ylabel("pressure")
    axs[1, 2].set_xlabel("diagonal")
    # The panels are placed explicitly, so the legend goes below them directly
    # (``figure_legend`` would rearrange them with the tight layout).
    handles, labels = axs[1, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, 0.0))

    # -------------------------------------------------------------
    # ======== ↑ Column 2: diagonal profiles, FD vs FV ↑ =========
    # -------------------------------------------------------------

    out = FIG_DIR / f"mhd_blast_test1_{num_cells}cells.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    # Run each simulation once, before the figures read the caches.
    scheme_names = [config_name for config_name, _ in CONFIGS] + ["fd_mhd_turbulence"]
    if WITHOUT_FOFC_CHECK:
        scheme_names += ["fv_vl2_hlld_without_fofc", "fv_vl2_hlle_without_fofc"]
    for num_cells in RESOLUTIONS:
        for config_name in scheme_names:
            cache = DATA_DIR / f"mhd_blast_{config_name}_{num_cells}cells.npz"
            if rerun_requested() or not cache.exists():
                run_and_cache(config_name, num_cells)
    if WITHOUT_FOFC_CHECK:
        report_safeguard_effect()
    report_turbulence_setup_effect()
    make_oscillation_comparison()
    make_fd_with_profiles(RESOLUTIONS[-1])
