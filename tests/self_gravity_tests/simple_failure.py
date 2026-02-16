# ==== GPU selection ====
from autocvd import autocvd
from matplotlib.colors import LogNorm
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
    HALF_SPLIT,
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

# simulate and plot pressure slice
t_end = 0.0739
num_cells = 64

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Half-split version
snapshots, config, params, helper_data, registered_variables = simulate_collapse(num_cells, t_end = t_end, self_gravity_version = HALF_SPLIT)
final_state = snapshots.final_state
pressure = final_state[registered_variables.pressure_index]
im1 = axs[0].imshow(pressure[:, :, num_cells // 2], extent=(0, box_size, 0, box_size), norm=LogNorm(vmin=1e-8, vmax=1e-1))
fig.colorbar(im1, ax=axs[0], label="Pressure")
axs[0].set_title("pressure slice, half-split scheme")

# Riemann-split version
snapshots, config, params, helper_data, registered_variables = simulate_collapse(num_cells, t_end = t_end, self_gravity_version = RIEMANN_SPLIT)
final_state = snapshots.final_state
pressure = final_state[registered_variables.pressure_index]
im2 = axs[1].imshow(pressure[:, :, num_cells // 2], extent=(0, box_size, 0, box_size), norm=LogNorm(vmin=1e-8, vmax=1e-1))
fig.colorbar(im2, ax=axs[1], label="Pressure")
axs[1].set_title("pressure slice, riemann-split scheme")

# remove x and y axis ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(f"pressure slices at t = {t_end} for simulations with {num_cells}³ cells")
plt.tight_layout()
plt.savefig("simple_failure_pressure_slices.png", dpi=800)
plt.close()