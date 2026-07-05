# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# ruff: noqa: E402
# =======================

from typing import NamedTuple

# numerics
import jax

# # enable 64-bit precision for better error measurement
# jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

# astronomix classes
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes.simulation_config import (
    SECOND_ORDER_CONSERVATIVE,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLC_LM,
    FOURTH_ORDER_CONSERVATIVE,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    PositivityConfig,
    SnapshotSettings
)

# astronomix functions
from astronomix import get_helper_data
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix.option_classes.simulation_config import finalize_config
from astronomix import get_registered_variables
from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD, OSHER, 
    PERIODIC_BOUNDARY, REFLECTIVE_BOUNDARY, 
    BoundarySettings, BoundarySettings1D,
    DOUBLE_MINMOD,
    LAX_FRIEDRICHS,
    MUSCL,
    RK2_SSP,
    SIMPLE_SOURCE,
    SPLIT,
    UNSPLIT,
    DOUBLE_MINMOD,
    LAX_FRIEDRICHS,
    MUSCL,
    RK2_SSP,
    SIMPLE_SOURCE,
    SPLIT,
    UNSPLIT,
)

self_gravity_version_fv = SIMPLE_SOURCE
self_gravity_version_fd = SECOND_ORDER_CONSERVATIVE
# in the simple source term version, the energy
# error is first order in space and time
# so reducing the timestep does not help as
# it accumulates over time.

# simulation settings
gamma = 5/3

# spatial domain
box_size = 4.0

# animate (per-snapshot slice GIFs; off for the scheme-comparison figures)
animate = False

baseline_config_fd = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    runtime_debugging = False,
    progress_bar = True,
    gravity_config = GravityConfig(
        self_gravity = True,
        self_gravity_version = self_gravity_version_fd,
        poisson_manual_open_boundaries = True,
    ),
    positivity_config = PositivityConfig(default_positivity_protection = False),
    mhd = False, # True,
    dimensionality = 3,
    box_size = box_size,
    differentiation_mode = FORWARDS,
    boundary_settings=BoundarySettings(
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
    ),
    return_snapshots = False,
    snapshot_settings = SnapshotSettings(
        return_states = animate,
        return_final_state = True,
        return_total_energy = True,
        return_internal_energy = True,
        return_kinetic_energy = True,
        return_gravitational_energy = True
    ),
    num_snapshots = 60
)


baseline_config_fv = SimulationConfig(
    runtime_debugging = False,
    progress_bar = True,
    gravity_config = GravityConfig(
        self_gravity = True,
        self_gravity_version = self_gravity_version_fv,
        poisson_manual_open_boundaries = True,
    ),
    first_order_fallback = False,
    dimensionality = 3,
    box_size = box_size,
    split = UNSPLIT,
    differentiation_mode = FORWARDS,
    limiter = MINMOD,
    time_integrator = RK2_SSP,
    riemann_solver = HLLC,
    boundary_settings=BoundarySettings(
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
    ),
    return_snapshots = False,
    snapshot_settings = SnapshotSettings(
        return_states = animate,
        return_final_state=True,
        return_total_energy=True,
        return_internal_energy=True,
        return_kinetic_energy=True,
        return_gravitational_energy=True
    ),
    num_snapshots = 60
)

# -------------------------------------------------------------
# =================== ↓ Evrard's Collapse ↓ ===================
# -------------------------------------------------------------

# usually t_end = 1.5
def simulate_collapse(num_cells, t_end = 1.2, return_snapshots = True, solver_mode = FINITE_DIFFERENCE, self_gravity_version = None):

    if solver_mode == FINITE_DIFFERENCE:
        baseline_config = baseline_config_fd
    elif solver_mode == FINITE_VOLUME:
        baseline_config = baseline_config_fv
    else:
        raise ValueError("Invalid solver mode")

    print("👷 Setting up simulation...")
    # setup simulation config
    config = baseline_config._replace(
        num_cells = num_cells,
        return_snapshots = return_snapshots,
    )

    params = SimulationParams(
        t_end = t_end,
        # normally 1.5
        C_cfl = 0.4 if solver_mode == FINITE_DIFFERENCE else 0.4,
        dt_max = jnp.inf,
        minimum_density = 1e-5,
        minimum_pressure = 3e-6,
        # minimum_density = 1e-6,
        # minimum_pressure = 1e-6,
    )

    if self_gravity_version is not None:
        config = config._replace(
            gravity_config = config.gravity_config._replace(
                self_gravity_version = self_gravity_version
            )
        )

    if self_gravity_version == SIMPLE_SOURCE:
        # not necessary in the simple source term version
        config = config._replace(
            positivity_config = config.positivity_config._replace(
                default_positivity_protection = False
            )
        )

    if self_gravity_version in (SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE):
        # the conservative flux schemes need the reconstruction-level
        # positivity-preserving flux limiter to survive the violent cold collapse
        config = config._replace(
            positivity_config = config.positivity_config._replace(
                preserving_flux = True
            )
        )

    helper_data = get_helper_data(config)

    registered_variables = get_registered_variables(config)
  
    R = 1.0
    M = 1.0

    dx = config.box_size / (config.num_cells - 1)

    # initialize density field
    rho = jnp.where(helper_data.r <= R, M / (2 * jnp.pi * R ** 2 * helper_data.r), 1e-4)

    total_injected_mass = jnp.sum(jnp.where(helper_data.r <= R, rho, 0)) * dx ** 3
    print(f"Injected mass: {total_injected_mass}")

    # better ball edges
    # overlap_weights = (R + dx / 2 - helper_data.r) / dx
    # rho = jnp.where((helper_data.r > R - dx / 2) & (helper_data.r < R + dx / 2), rho * overlap_weights, rho)

    # Initialize velocity fields to zero
    v_x = jnp.zeros_like(rho)
    v_y = jnp.zeros_like(rho)
    v_z = jnp.zeros_like(rho)

    # initial thermal energy per unit mass = 0.05
    e = 0.05
    p = (gamma - 1) * rho * e
    print("min and max initial pressure:", jnp.min(p), jnp.max(p))

    p = jnp.where(p < params.minimum_pressure, params.minimum_pressure, p)

    B0 = 1e-4

    B_x = jnp.zeros_like(rho)
    B_y = jnp.zeros_like(rho)
    B_z = B0 * jnp.ones_like(rho)

    bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

    # Construct the initial primitive state for the 3D simulation.
    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = v_x,
        velocity_y = v_y,
        velocity_z = v_z,
        gas_pressure = p,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        interface_magnetic_field_x = bxb,
        interface_magnetic_field_y = byb,
        interface_magnetic_field_z = bzb,
    )

    config = finalize_config(config, initial_state.shape)

    return jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    ), config, params, helper_data, registered_variables

def energy_error_convergence(num_cells_list = [16, 32, 64, 128], only_plot = True):
    
    # error of the final total energy
    # compared to the initial total energy
    energy_errors_fd = []
    energy_errors_fv = []

    if not only_plot:

        for num_cells in num_cells_list:
            print(f"Running simulation for {num_cells} cells...")

            snapshots_fd, _, _, _, _ = simulate_collapse(num_cells, solver_mode = FINITE_DIFFERENCE)
            snapshots_fv, _, _, _, _ = simulate_collapse(num_cells, solver_mode = FINITE_VOLUME)

            initial_total_energy_fd = snapshots_fd.total_energy[0]
            final_total_energy_fd = snapshots_fd.total_energy[-1]
            energy_error_fd = jnp.abs(final_total_energy_fd - initial_total_energy_fd) / jnp.abs(initial_total_energy_fd)
            energy_errors_fd.append(energy_error_fd)

            initial_total_energy_fv = snapshots_fv.total_energy[0]
            final_total_energy_fv = snapshots_fv.total_energy[-1]
            energy_error_fv = jnp.abs(final_total_energy_fv - initial_total_energy_fv) / jnp.abs(initial_total_energy_fv)
            energy_errors_fv.append(energy_error_fv)

        jnp.savez(
            "collapse_energy_errors.npz",
            num_cells_list = num_cells_list,
            energy_errors_fd = energy_errors_fd,
            energy_errors_fv = energy_errors_fv
        )
    
    else:
        data = jnp.load("collapse_energy_errors.npz")
        num_cells_list = data["num_cells_list"]
        energy_errors_fd = data["energy_errors_fd"]
        energy_errors_fv = data["energy_errors_fv"]

    fig_convergence, ax_convergence = plt.subplots(1, 1, figsize=(6, 4))
    ax_convergence.plot(num_cells_list, energy_errors_fd, label="Finite Difference", marker='o')
    ax_convergence.plot(num_cells_list, energy_errors_fv, label="Finite Volume", marker='o')

    anchor = (20, 2e-1)

    add_power_law_indicators(
        ax=ax_convergence,
        anchor=anchor,
        exponents=[-1, -2],
        x_span=2.0,
        scales=[1.0, 1.0],
        x_label='N'
    )

    ax_convergence.set_xlabel("Number of Cells")
    ax_convergence.set_ylabel("Relative Energy Error")
    ax_convergence.set_xscale("log")
    ax_convergence.set_yscale("log")
    ax_convergence.set_title("Energy Error Convergence")
    ax_convergence.legend()
    fig_convergence.savefig(f"figures/collapse_energy_error_convergence.svg")

        

def resolution_study_collapse(resolution = 128):

    class TestSetup(NamedTuple):
        solver_mode: int
        self_gravity_version: int
        resolution: int
        line_style: str = '-'
        linewidth: float = 2.0

        def __str__(self):
            # FD for finite difference, FV for finite volume
            # simple source for SIMPLE_SOURCE
            # flux-based source for SECOND_ORDER_CONSERVATIVE
            # corrected flux-based source for FOURTH_ORDER_CONSERVATIVE
            # total string e.g.: FD, simple source
            solver_str = 'FD' if self.solver_mode == FINITE_DIFFERENCE else 'FV'
            gravity_str = 'simple source' if self.self_gravity_version == SIMPLE_SOURCE else 'flux-based source' if self.self_gravity_version == SECOND_ORDER_CONSERVATIVE else 'corrected flux-based source'
            resolution_str = f"N={self.resolution}"
            return f"{solver_str}, {gravity_str}, {resolution_str}"

    test_setups = [
        TestSetup(solver_mode = FINITE_DIFFERENCE, self_gravity_version = SIMPLE_SOURCE, resolution = resolution, line_style = ':'),
        TestSetup(solver_mode = FINITE_DIFFERENCE, self_gravity_version = FOURTH_ORDER_CONSERVATIVE, resolution = resolution, line_style = '--'),
    ]

    fig, (ax_energy_terms, ax_total_energy) = plt.subplots(2, 1, figsize=(10, 10))

    for test_setup in test_setups:

        print(f"Running simulation for {test_setup.resolution} cells...")

        snapshots, _, _, _, registered_variables = simulate_collapse(test_setup.resolution, solver_mode = test_setup.solver_mode, self_gravity_version = test_setup.self_gravity_version)
        total_energy = snapshots.total_energy
        internal_energy = snapshots.internal_energy
        kinetic_energy = snapshots.kinetic_energy
        gravitational_energy = snapshots.gravitational_energy
        time = snapshots.time_points

        ax_energy_terms.plot(time, total_energy, label="Total, " + str(test_setup), color = 'black', linestyle = test_setup.line_style)
        ax_energy_terms.plot(time, internal_energy, label="Internal, " + str(test_setup), color = 'green', linestyle = test_setup.line_style)
        ax_energy_terms.plot(time, kinetic_energy, label="Kinetic, " + str(test_setup), color = 'red', linestyle = test_setup.line_style)
        ax_energy_terms.plot(time, gravitational_energy, label="Gravitational, " + str(test_setup), color = 'blue', linestyle = test_setup.line_style)

        ax_energy_terms.set_xlabel("Time")
        ax_energy_terms.set_ylabel("Energy")

        rel_energy_error = jnp.abs(total_energy - total_energy[0]) / jnp.abs(total_energy[0])
        ax_total_energy.plot(time, rel_energy_error, label="Relative Energy Error, " + str(test_setup), linestyle = test_setup.line_style)

        ax_total_energy.set_yscale("log")
        ax_total_energy.set_xlabel("Time")
        ax_total_energy.set_ylabel("Relative Energy Error")

        if animate:
            fig_anim, ax_anim = plt.subplots(1, 3, figsize=(15, 5))
            states = snapshots.states
            mid_plane = test_setup.resolution // 2

            density_series = states[:, registered_variables.density_index, :, :, mid_plane]
            pressure_series = states[:, registered_variables.pressure_index, :, :, mid_plane]
            v_sq_series = (
                states[:, registered_variables.velocity_index.x, :, :, mid_plane] ** 2
                + states[:, registered_variables.velocity_index.y, :, :, mid_plane] ** 2
                + states[:, registered_variables.velocity_index.z, :, :, mid_plane] ** 2
            )

            def log_bounds(series):
                min_positive = jnp.min(jnp.where(series > 0, series, jnp.inf))
                min_positive = jnp.where(jnp.isfinite(min_positive), min_positive, 1e-30)
                max_value = jnp.max(series)
                max_value = jnp.maximum(max_value, min_positive * 10)
                return float(min_positive), float(max_value)

            density_vmin, density_vmax = log_bounds(density_series)
            pressure_vmin, pressure_vmax = log_bounds(pressure_series)
            v_sq_vmin, v_sq_vmax = log_bounds(v_sq_series)

            im_density = ax_anim[0].imshow(
                jnp.maximum(density_series[0], density_vmin),
                origin='lower',
                extent=(0, box_size, 0, box_size),
                norm=LogNorm(vmin=density_vmin, vmax=density_vmax),
            )
            ax_anim[0].set_title("Density")
            ax_anim[0].set_xlabel("x")
            ax_anim[0].set_ylabel("y")
            fig_anim.colorbar(im_density, ax=ax_anim[0])

            im_pressure = ax_anim[1].imshow(
                jnp.maximum(pressure_series[0], pressure_vmin),
                origin='lower',
                extent=(0, box_size, 0, box_size),
                norm=LogNorm(vmin=pressure_vmin, vmax=pressure_vmax),
            )
            ax_anim[1].set_title("Pressure")
            ax_anim[1].set_xlabel("x")
            ax_anim[1].set_ylabel("y")
            fig_anim.colorbar(im_pressure, ax=ax_anim[1])

            im_v_sq = ax_anim[2].imshow(
                jnp.maximum(v_sq_series[0], v_sq_vmin),
                origin='lower',
                extent=(0, box_size, 0, box_size),
                norm=LogNorm(vmin=v_sq_vmin, vmax=v_sq_vmax),
            )
            ax_anim[2].set_title("Velocity Squared")
            ax_anim[2].set_xlabel("x")
            ax_anim[2].set_ylabel("y")
            fig_anim.colorbar(im_v_sq, ax=ax_anim[2])

            fig_anim.suptitle(f"Collapse Slice Evolution (t = {float(time[0]):.4f})")

            def update(frame):
                im_density.set_data(jnp.maximum(density_series[frame], density_vmin))
                im_pressure.set_data(jnp.maximum(pressure_series[frame], pressure_vmin))
                im_v_sq.set_data(jnp.maximum(v_sq_series[frame], v_sq_vmin))
                fig_anim.suptitle(f"Collapse Slice Evolution (t = {float(time[frame]):.4f})")
                return im_density, im_pressure, im_v_sq

            anim = FuncAnimation(
                fig_anim,
                update,
                frames=density_series.shape[0],
                interval=120,
                blit=False,
            )
            anim.save(
                f"collapse_slice_evolution_{test_setup.resolution}.gif",
                fps=10,
            )

            plt.close(fig_anim)

    ax_energy_terms.set_ylim(-2.5, 2.5)
    # ax.set_ylim(-0.7, -0.4)
    # ax.set_xlim(0.8, 1.0)

    ax_energy_terms.legend(fontsize="xx-small", ncol = len(test_setups))
    ax_energy_terms.set_title("Resolution Study for Evrard's Collapse")

    ax_total_energy.legend(fontsize="x-small",)
    ax_total_energy.set_title("Relative Energy Error for Evrard's Collapse")

    plt.savefig(f"figures/collapse_resolution_study_{resolution}.svg")
    fig.savefig(f"figures/collapse_resolution_study_{resolution}.png", dpi=150)

def radial_profile_study(num_cells = 128, t_end = 0.8):

    # compare the non-conservative simple source against the energy-conserving
    # 4th-order scheme (with the positivity-preserving flux limiter)
    schemes = [
        ("simple source", SIMPLE_SOURCE, "C0"),
        ("4th-order conservative + PP flux", FOURTH_ORDER_CONSERVATIVE, "C1"),
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    for label, version, color in schemes:
        print(f"Running radial profile ({label}) for {num_cells} cells...")
        final_state, _, params, helper_data, registered_variables = simulate_collapse(
            num_cells, t_end = t_end, return_snapshots = False,
            solver_mode = FINITE_DIFFERENCE, self_gravity_version = version,
        )

        r = helper_data.r.flatten()
        ax1.scatter(r, final_state[registered_variables.density_index].flatten(),
                    label=label, s=1, color=color)

        v_r = -jnp.sqrt(
            final_state[registered_variables.velocity_index.x] ** 2
            + final_state[registered_variables.velocity_index.y] ** 2
            + final_state[registered_variables.velocity_index.z] ** 2
        )
        ax2.scatter(r, v_r.flatten(), label=label, s=1, color=color)

        entropy = (final_state[registered_variables.pressure_index].flatten()
                   / final_state[registered_variables.density_index].flatten() ** params.gamma)
        ax3.scatter(r, entropy, label=label, s=1, color=color)

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(1e-2, 6e-1); ax1.set_ylim(1e-2, 1e3)
    ax1.set_xlabel("r"); ax1.set_ylabel("Density"); ax1.legend(markerscale=6)

    ax2.set_xscale("log")
    ax2.set_xlim(1e-2, 6e-1)
    ax2.set_xlabel("r"); ax2.set_ylabel("Radial velocity"); ax2.legend(markerscale=6)

    ax3.set_xscale("log")
    ax3.set_xlim(4.0 / num_cells, 6e-1); ax3.set_ylim(0, 0.2)
    ax3.set_xlabel("r"); ax3.set_ylabel(r"$P / \rho^\gamma$"); ax3.legend(markerscale=6)

    fig.suptitle(f"Evrard collapse radial profiles (N={num_cells}, t={t_end})")
    plt.tight_layout()
    plt.savefig(f"figures/collapse_radial_profile_{num_cells}.png", dpi=150)


if __name__ == "__main__":
    resolution_study_collapse()
    radial_profile_study()

# energy_error_convergence(
#     num_cells_list = [16, 32, 64, 96, 128, 160]
# )


# -------------------------------------------------------------
# =================== ↑ Evrard's Collapse ↑ ===================
# -------------------------------------------------------------