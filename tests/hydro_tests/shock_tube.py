# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

# 64-bit floating point precision
import jax
jax.config.update("jax_enable_x64", True)

from astronomix.option_classes.simulation_config import DOUBLE_MINMOD, HLL, HLLC, MINMOD, OSHER, SUPERBEE


import jax.numpy as jnp

# constants
from astronomix import SPHERICAL, CARTESIAN

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
# Import tools for creating the zoom-in box
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def limiter_to_string(limiter):
    """Convert a limiter to a string for plotting."""
    if limiter == MINMOD:
        return "Minmod"
    elif limiter == DOUBLE_MINMOD:
        return "Double Minmod"
    elif limiter == SUPERBEE:
        return "Superbee"
    elif limiter == OSHER:
        return "Osher"
    else:
        return str(limiter)

# ===================================================
# ============== ↓ astronomix simulation ↓ =============
# ===================================================

params = SimulationParams(
    t_end = 0.2, # the typical value for a shock test
)
shock_pos = 0.5
box_size = 1.0

def simulate(limiter, num_cells):
    config = SimulationConfig(
        geometry = CARTESIAN,
        first_order_fallback = False,
        riemann_solver = HLLC,
        limiter = limiter,
        box_size = box_size,
        num_cells = num_cells,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # setup the shock initial fluid state in terms of rho, u, p
    
    r = helper_data.geometric_centers
    rho = jnp.where(r < shock_pos, 1.0, 0.125)
    u = jnp.zeros_like(r)
    p = jnp.where(r < shock_pos, 1.0, 0.1)

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

# ===================================================
# ================ ↓ exact solution ↓ ===============
# ===================================================

# import ExactPack solvers
from exactpack.solvers.riemann.ep_riemann import IGEOS_Solver
import numpy as np

# Use the same spatial domain and shock position as the simulation
xvec = np.linspace(0.0, box_size, 1000)
t_final = float(params.t_end)  # Use the simulation's end time

r_init = jnp.array(xvec)
rho_init = jnp.where(xvec < shock_pos, 1.0, 0.125)
u_init = jnp.zeros_like(xvec)
p_init = jnp.where(xvec < shock_pos, 1.0, 0.1)

# Sod shock tube problem with the same initial 
# conditions as the simulation
riem1_ig_soln = IGEOS_Solver(
    rl=1.0,   ul=0.0,   pl=1.0,  gl = params.gamma,
    rr=0.125, ur=0.0,   pr=0.1,  gr = params.gamma,
    xmin=0.0, xd0=shock_pos, xmax=box_size, t=t_final
)

riem1_ig_result = riem1_ig_soln._run(xvec, t_final)

rho_exact = riem1_ig_result['density']
u_exact = riem1_ig_result['velocity']
p_exact = riem1_ig_result['pressure']

# ===================================================
# ================ ↑ exact solution ↑ ===============
# ===================================================


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# initial
axs[0].plot(r_init, rho_init, label='initial')
axs[2].plot(r_init, p_init, label='initial')
axs[1].plot(r_init, u_init, label='initial')

# exact
axs[0].plot(xvec, rho_exact, label='exact', linestyle='--')
axs[1].plot(xvec, u_exact, label='exact', linestyle='--')
axs[2].plot(xvec, p_exact, label='exact', linestyle='--')

# simulation results
parameter_combinations = [
    (MINMOD, 101),
    (MINMOD, 401),
    (DOUBLE_MINMOD, 101),
    (DOUBLE_MINMOD, 401),
]

for limiter, num_cells in parameter_combinations:
    r, rho_final, u_final, p_final = simulate(limiter, num_cells)
    label = f"{limiter_to_string(limiter)}, ({num_cells} cells)"

    axs[0].plot(r, rho_final, label=label)
    axs[1].plot(r, u_final, label=label)
    axs[2].plot(r, p_final, label=label)

axs[0].set_xlabel('Position')
axs[0].set_ylabel('Density')
axs[0].set_title('Density')
axs[0].legend(loc = 'upper right')

# ==================== ZOOM IN BOX ADDITION ====================
# Define the zoom region for the contact discontinuity
x1, x2 = 0.6, 0.7
y1, y2 = 0.40, 0.50

# Create inset axes in the lower left corner of the density plot
axins = inset_axes(axs[0], width="40%", height="40%", loc='lower left')

# Re-plot all data from the main density plot onto the inset axes
for line in axs[0].get_lines():
    axins.plot(line.get_xdata(), line.get_ydata(),
               linestyle=line.get_linestyle(),
               color=line.get_color())

# Set the limits of the zoom-in box
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.tick_params(labelleft=False, labelbottom=False)
# no axis ticks in the inset plot
axins.xaxis.set_ticks([])


# Optional: Add grid and adjust tick font size for better readability
axins.grid(True, linestyle='--', alpha=0.6)
axins.tick_params(axis='both', which='major', labelsize=8)

# Draw a box around the region of interest on the main plot
# and connect it to the inset plot for clarity
mark_inset(axs[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")
# =============================================================

axs[1].set_xlabel('Position')
axs[1].set_ylabel('Velocity')
axs[1].set_title('Velocity')
axs[1].legend(loc = 'lower right')

axs[2].set_xlabel('Position')
axs[2].set_ylabel('Pressure')
axs[2].set_title('Pressure')
axs[2].legend(loc = 'lower left')

plt.savefig('figures/shock_tube.pdf', bbox_inches='tight')