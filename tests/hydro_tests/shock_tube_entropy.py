# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# ruff: noqa: E402
# =======================

from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings1D, SnapshotSettings

import jax.numpy as jnp

# constants
from astronomix import CARTESIAN

# astronomix option structures
from astronomix import SimulationConfig
from astronomix import SimulationParams

# simulation setup
from astronomix import get_helper_data
from astronomix import finalize_config
from astronomix import get_registered_variables
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state

# time integration, core function
from astronomix import time_integration

# plotting
import matplotlib.pyplot as plt

params = SimulationParams(
    t_end = 0.2, # the typical value for a shock test
)
box_size = 1.0

def simulate(num_cells):
    config = SimulationConfig(
        geometry = CARTESIAN,
        solver_mode = FINITE_DIFFERENCE,
        box_size = box_size,
        num_cells = num_cells,
        return_snapshots = True,
        num_snapshots = 20,
        boundary_settings = BoundarySettings1D(
            left_boundary = PERIODIC_BOUNDARY,
            right_boundary = PERIODIC_BOUNDARY
        ),
        snapshot_settings = SnapshotSettings(
            return_states = True
        )
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # setup the shock initial fluid state in terms of rho, u, p
    
    r = helper_data.geometric_centers
    rectangle_mask = (r > 0.4) & (r < 0.6)
    rho = jnp.where(rectangle_mask, 1.0, 0.125)
    u = jnp.zeros_like(r)
    p = jnp.where(rectangle_mask, 1.0, 0.1)

    # get initial state
    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )

    config = finalize_config(config, initial_state.shape)

    result = time_integration(initial_state, config, params, registered_variables)

    return result, config, registered_variables, helper_data

result, config, registered_variables, helper_data = simulate(num_cells = 10000)

# calculate the total entropy ln(p / rho^gamma) and plot it
gamma = params.gamma
# the first axis is the snapshot axis
densities = result.states[:, registered_variables.density_index]
pressures = result.states[:, registered_variables.pressure_index]
entropies = jnp.log(pressures / densities**gamma)
# total entropy
total_entropies = jnp.sum(entropies * densities, axis=1) * config.grid_spacing

# plot the entropy evolution
fig, ax = plt.subplots()
ax.plot(jnp.linspace(0, params.t_end, config.num_snapshots), total_entropies)
ax.set_xlabel("Time")
ax.set_ylabel("Total Entropy")
ax.set_title("Shock Tube Test: Total Entropy Evolution")
fig.savefig("shock_tube_entropy.png")

# plot the final state
final_density = densities[-1]
final_pressure = pressures[-1]
final_entropy = entropies[-1]
fig, ax = plt.subplots(3, 1, figsize=(8, 12))
ax[0].plot(helper_data.geometric_centers, final_density)
ax[0].set_title("Final Density")
ax[1].plot(helper_data.geometric_centers, final_pressure)
ax[1].set_title("Final Pressure")
ax[2].plot(helper_data.geometric_centers, final_entropy)
ax[2].set_title("Final Entropy")
fig.savefig("shock_tube_final_state.png")