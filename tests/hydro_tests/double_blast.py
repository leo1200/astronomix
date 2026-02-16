# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

from astronomix.option_classes.simulation_config import DOUBLE_MINMOD, HLLC, MINMOD, SUPERBEE, BoundarySettings1D

# 64-bit floating point precision
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# constants
from astronomix import CARTESIAN, REFLECTIVE_BOUNDARY

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

def limiter_to_string(limiter):
    """Convert a limiter to a string for plotting."""
    if limiter == MINMOD:
        return "Minmod"
    elif limiter == DOUBLE_MINMOD:
        return "Double Minmod"
    elif limiter == SUPERBEE:
        return "Superbee"
    else:
        return str(limiter)

# ===================================================
# ============== ↓ astronomix simulation ↓ =============
# ===================================================

# Parameters for the Double Blast Wave problem
params = SimulationParams(
    t_end = 0.038,
    gamma = 1.4,
)
box_size = 1.0

def simulate(limiter, num_cells):
    config = SimulationConfig(
        geometry = CARTESIAN,
        boundary_settings = BoundarySettings1D(left_boundary = REFLECTIVE_BOUNDARY, right_boundary = REFLECTIVE_BOUNDARY),
        first_order_fallback = False,
        riemann_solver = HLLC,
        limiter = limiter,
        box_size = box_size,
        num_cells = num_cells,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # Setup the double blast wave initial fluid state in terms of rho, u, p
    # The domain is [0, 1].
    # For 0 <= x < 0.1: rho = 1.0, u = 0.0, p = 1000.0
    # For 0.1 <= x < 0.9: rho = 1.0, u = 0.0, p = 0.01
    # For 0.9 <= x <= 1.0: rho = 1.0, u = 0.0, p = 100.0
    
    r = helper_data.geometric_centers
    rho = jnp.ones_like(r)
    u = jnp.zeros_like(r)
    # Use nested jnp.where to define the three-part piecewise pressure
    p = jnp.where(r > 0.9, 100.0, jnp.where(r < 0.1, 1000.0, 0.01))

    # get initial state
    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )

    config = finalize_config(config, initial_state.shape)

    final_state = time_integration(initial_state, config, params, registered_variables)
    rho_final = final_state[registered_variables.density_index]
    u_final = final_state[registered_variables.velocity_index]
    p_final = final_state[registered_variables.pressure_index]

    return (
        r,
        rho_final, u_final, p_final,
    )

# ===================================================
# ============== ↑ astronomix simulation ↑ =============
# ===================================================

# Plotting the results
fig, axs = plt.subplots(1, 1, figsize=(8, 5))
fig.suptitle("Double Blast Wave Problem (Woodward & Colella, 1984)", fontsize=16)

# Plot the initial conditions on a high-resolution grid for clarity
r_init = jnp.linspace(0.0, box_size, 1000)
rho_init = jnp.ones_like(r_init)
u_init = jnp.zeros_like(r_init)
p_init = jnp.where(r_init > 0.9, 100.0, jnp.where(r_init < 0.1, 1000.0, 0.01))

# axs[0].plot(r_init, rho_init, 'k:', label='Initial')
# axs[1].plot(r_init, u_init, 'k:', label='Initial')
# axs[2].plot(r_init, p_init, 'k:', label='Initial')


# Simulation results
# This is a much harder problem, so we use more cells.
# We compare a diffusive limiter (Minmod) with a compressive one (Superbee)
# and show the effect of higher resolution.
parameter_combinations = [
    (MINMOD, 401),
    (DOUBLE_MINMOD, 401),
    (DOUBLE_MINMOD, 801),
    (DOUBLE_MINMOD, 10001),
]

for limiter, num_cells in parameter_combinations:
    r, rho_final, u_final, p_final = simulate(limiter, num_cells)
    label = f"{limiter_to_string(limiter)}, ({num_cells} cells)"

    # Use points for lower-res and lines for higher-res to make plots clearer
    if num_cells <= 401:
        axs.plot(r, rho_final, label=label, marker='.', markersize=2, linestyle='None')
    else:
        axs.plot(r, rho_final, label=label)


# Formatting for Density Plot
axs.set_xlabel('Position')
axs.set_ylabel('Density')
axs.set_title('Density')
axs.legend(loc='upper left')
axs.set_xlim(0.4, 1)
axs.set_ylim(0, 7) # Set y-limit to better see the structure, which peaks around 6


plt.tight_layout() # Adjust layout to make room for suptitle
plt.savefig('figures/double_blast.pdf', bbox_inches='tight')