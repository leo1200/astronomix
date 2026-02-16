# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

# numerics
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix classes
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes.simulation_config import (
    DONOR_ACCOUNTING,
    HLLC_LM,
    RIEMANN_SPLIT,
    RIEMANN_SPLIT_UNSTABLE,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings
)

# astronomix functions
from astronomix import get_helper_data
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix import get_registered_variables

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD, OSHER, 
    PERIODIC_BOUNDARY, REFLECTIVE_BOUNDARY, 
    BoundarySettings, BoundarySettings1D,
    DOUBLE_MINMOD,
    LAX_FRIEDRICHS,
    MUSCL,
    RK2_SSP,
    SIMPLE_SOURCE_TERM,
    SPLIT,
    UNSPLIT,
    DOUBLE_MINMOD,
    LAX_FRIEDRICHS,
    MUSCL,
    RK2_SSP,
    SIMPLE_SOURCE_TERM,
    SPLIT,
    UNSPLIT,
)

# simulation settings
gamma = 5/3

# spatial domain
box_size = 4.0

baseline_config = SimulationConfig(
    runtime_debugging = False,
    progress_bar = True,
    self_gravity = True,
    first_order_fallback = False,
    dimensionality = 3,
    box_size = box_size,
    split = SPLIT,
    differentiation_mode = FORWARDS,
    limiter = MINMOD,
    time_integrator = MUSCL,
    riemann_solver = HLLC,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(
            left_boundary = REFLECTIVE_BOUNDARY,
            right_boundary = REFLECTIVE_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = REFLECTIVE_BOUNDARY,
            right_boundary = REFLECTIVE_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = REFLECTIVE_BOUNDARY,
            right_boundary = REFLECTIVE_BOUNDARY
        )
    ),
    return_snapshots = True,
    snapshot_settings = SnapshotSettings(
        return_states = False,
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

def simulate_collapse(num_cells, t_end = 3.0, self_gravity_version = RIEMANN_SPLIT, return_snapshots = True):

    print("👷 Setting up simulation...")
    # setup simulation config
    config = baseline_config._replace(
        num_cells = num_cells,
        return_snapshots = return_snapshots,
        self_gravity_version = self_gravity_version
    )

    helper_data = get_helper_data(config)

    params = SimulationParams(
        t_end = t_end,
        C_cfl = 0.4,
    )

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

    # Construct the initial primitive state for the 3D simulation.
    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = v_x,
        velocity_y = v_y,
        velocity_z = v_z,
        gas_pressure = p
    )

    config = finalize_config(config, initial_state.shape)

    return jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    ), config, params, helper_data, registered_variables

def stringify_self_gravity_version(version):
    if version == SIMPLE_SOURCE_TERM:
        return "non-conservative"
    elif version == RIEMANN_SPLIT_UNSTABLE or version == RIEMANN_SPLIT:
        return "conservative"
    else:
        return "unknown"

configurations = [
    (128, SIMPLE_SOURCE_TERM),
    (256, SIMPLE_SOURCE_TERM),
    (128, RIEMANN_SPLIT_UNSTABLE),
]

fig_profile, axes_profile = plt.subplots(1, 3, figsize=(12, 4))
fig_energy, ax_energy = plt.subplots(1, 1, figsize=(10, 5))

for num_cells, self_gravity_version in configurations:
    print(f"Running simulation for {num_cells} cells with {stringify_self_gravity_version(self_gravity_version)}...")
    snapshots, _, _, helper_data, registered_variables = simulate_collapse(num_cells, self_gravity_version = self_gravity_version, t_end = 3.0)
    total_energy = snapshots.total_energy
    internal_energy = snapshots.internal_energy
    kinetic_energy = snapshots.kinetic_energy
    gravitational_energy = snapshots.gravitational_energy
    time = snapshots.time_points
    ax_energy.plot(time, total_energy, label="total, N = " + str(num_cells) + "³, " + stringify_self_gravity_version(self_gravity_version), linestyle = '-' if self_gravity_version == SIMPLE_SOURCE_TERM else '--')
    ax_energy.plot(time, internal_energy, label="internal, N = " + str(num_cells) + "³, " + stringify_self_gravity_version(self_gravity_version), linestyle = '-' if self_gravity_version == SIMPLE_SOURCE_TERM else '--')
    ax_energy.plot(time, kinetic_energy, label="kinetic, N = " + str(num_cells) + "³, " + stringify_self_gravity_version(self_gravity_version), linestyle = '-' if self_gravity_version == SIMPLE_SOURCE_TERM else '--')
    ax_energy.plot(time, gravitational_energy, label="gravitational, N = " + str(num_cells) + "³, " + stringify_self_gravity_version(self_gravity_version), linestyle = '-' if self_gravity_version == SIMPLE_SOURCE_TERM else '--')
    ax_energy.set_xlabel("time")
    ax_energy.set_ylabel("energy")

    snapshots, helper_data, registered_variables = None, None, None # free memory

    snapshots, _, _, helper_data, registered_variables = simulate_collapse(num_cells, self_gravity_version = self_gravity_version, t_end = 0.8)
    final_state = snapshots.final_state
    ax = axes_profile[0]
    ax.scatter(helper_data.r.flatten(), final_state[registered_variables.density_index].flatten(), label="N = " + str(num_cells) + "³, " + stringify_self_gravity_version(self_gravity_version), s = 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 6e-1)
    ax.set_ylim(1e-2, 1e3)
    ax.set_xlabel("r")
    ax.set_ylabel("density")

    v_r = -jnp.sqrt(final_state[registered_variables.velocity_index.x] ** 2 + final_state[registered_variables.velocity_index.y] ** 2 + final_state[registered_variables.velocity_index.z] ** 2)
    ax = axes_profile[1]
    ax.scatter(helper_data.r.flatten(), v_r.flatten(), label="N = " + str(num_cells) + "³, " + stringify_self_gravity_version(self_gravity_version), s = 1)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 6e-1)
    ax.set_xlabel("r")
    ax.set_ylabel("velocity")

    ax = axes_profile[2]
    ax.scatter(helper_data.r.flatten(), final_state[registered_variables.pressure_index].flatten() / final_state[registered_variables.density_index].flatten() ** gamma, label="N = " + str(num_cells) + "³, " + stringify_self_gravity_version(self_gravity_version), s = 1)
    ax.set_xlim(1e-2, 6e-1)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("r")
    ax.set_ylabel(r"P / $\rho^\gamma$")
    ax.set_xscale("log")

fig_energy.suptitle("Evrard's collapse energy evolution")
ax_energy.legend(fontsize="x-small", ncol=len(configurations))
fig_energy.tight_layout()
fig_energy.savefig("collapse_energy_evolution_comparison.svg")

fig_profile.suptitle("Evrard's collapse radial profiles, t = 0.8")
for ax in axes_profile:
    ax.legend(loc = "lower left")
fig_profile.tight_layout()
fig_profile.savefig("collapse_radial_profiles_comparison.png", dpi = 800)