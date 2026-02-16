# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

import time

import jax.numpy as jnp
from matplotlib import pyplot as plt
from jamr import _BufferedList, animation, plot_density, time_integration_fixed_stepsize


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
from astronomix.option_classes.simulation_config import (
    DOUBLE_MINMOD, HLL,
    HLLC, MINMOD, OSHER, SUPERBEE
)

# time integration, core function
from astronomix import time_integration

# ===================================================
# ================ ↓ jamr simulation ↓ ==============
# ===================================================

def jamr_simulate(base_num_cells = 20, buffer_size = 200, dt = 0.001, derefinement_tolerance = 0.5, refinement_tolerance = 5.0, maximum_refinement_level = 3):
    # jamr settings
    num_steps = int(0.2 / dt)  # the typical value for a shock test

    # create example initial fluid state
    shock_pos = 0.5
    r = jnp.linspace(0, 1, base_num_cells)
    dx_coarse = 1 / (base_num_cells - 1)
    rho = jnp.where(r < shock_pos, 1.0, 0.125)
    u = jnp.ones_like(r) * 0.0
    p = jnp.where(r < shock_pos, 1.0, 0.1)

    # create buffers
    fluid_buffer = jnp.full((3, buffer_size), 1.0)
    center_buffer = jnp.linspace(1, 2, buffer_size)
    volume_buffer = jnp.full((buffer_size,), 1.0)
    refinement_level_buffer = jnp.full((buffer_size,), 1.0)

    # fill buffers with initial fluid state
    fluid_buffer = fluid_buffer.at[:, 0:base_num_cells].set(jnp.stack([rho, u, p], axis=0))
    center_buffer = center_buffer.at[0:base_num_cells].set(r)
    volume_buffer = volume_buffer.at[0:base_num_cells].set(dx_coarse)
    refinement_level_buffer = refinement_level_buffer.at[0:base_num_cells].set(0)

    # create fluid data
    fluid_data = _BufferedList(
        jnp.vstack([fluid_buffer, center_buffer, volume_buffer, refinement_level_buffer]), base_num_cells
    )

    # first run to compile the JAX functions
    time_integration_fixed_stepsize(
        fluid_data,
        dt,
        num_steps,
        maximum_refinement_level,
        refinement_tolerance,
        derefinement_tolerance
    ).buffer.block_until_ready()

    # timing the simulation
    runtimes = []
    for _ in range(5):
        start_time = time.time()
        # run the simulation
        time_integration_fixed_stepsize(
            fluid_data, dt, num_steps, maximum_refinement_level, refinement_tolerance, derefinement_tolerance
        ).buffer.block_until_ready()
        end_time = time.time()
        runtimes.append(end_time - start_time)
    runtime = min(runtimes)
    print(runtime)

    # run animation
    final_fluid_data = time_integration_fixed_stepsize(fluid_data, dt, num_steps, maximum_refinement_level, refinement_tolerance, derefinement_tolerance)

    final_primitive_state_jamr = final_fluid_data.buffer[0:3, 0 : final_fluid_data.num_cells]
    final_center_buffer_jamr = final_fluid_data.buffer[3, 0 : final_fluid_data.num_cells]

    return (
        runtime,
        final_center_buffer_jamr,
        final_primitive_state_jamr[0],  # density
        final_primitive_state_jamr[1],  # velocity
        final_primitive_state_jamr[2],  # pressure
    )

# ===================================================
# ================ ↑ jamr simulation ↑ ==============
# ===================================================

# ===================================================
# ============== ↓ astronomix simulation ↓ =============
# ===================================================

shock_pos = 0.5
box_size = 1.0

params = SimulationParams(
    t_end = 0.2, # the typical value for a shock test
)

def astronomix_simulate(num_cells, dt):
    config = SimulationConfig(
        geometry = CARTESIAN,
        first_order_fallback = True,
        fixed_timestep = True,
        num_timesteps = int(0.2 / dt),
        riemann_solver = HLL,
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

    # compile the JAX functions
    time_integration(initial_state, config, params, registered_variables).block_until_ready()

    # time the simulation
    runtimes = []
    for _ in range(5):
        start_time = time.time()
        # run the simulation
        final_state = time_integration(initial_state, config, params, registered_variables).block_until_ready()
        end_time = time.time()
        runtimes.append(end_time - start_time)
    runtime = min(runtimes)
    print(runtime)
    
    final_state = time_integration(initial_state, config, params, registered_variables)

    rho_final = final_state[registered_variables.density_index]
    u_final = final_state[registered_variables.velocity_index]
    p_final = final_state[registered_variables.pressure_index]

    return (
        runtime,
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
from matplotlib import cm

def exact_solution(xvec = np.linspace(0.0, box_size, 1000)):

    xvec = np.array(xvec)

    # Use the same spatial domain and shock position as the simulation
    t_final = float(0.2)  # Use the simulation's end time

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

    return (
        jnp.array(xvec),
        jnp.array(rho_exact),
        jnp.array(u_exact),
        jnp.array(p_exact),
    )

# ===================================================
# ================ ↑ exact solution ↑ ===============
# ===================================================

def jamr_test(base_resolutions = [20, 40, 80, 120], dt = 0.0002):

    rho_accuracies = []
    buffer_sizes = []
    u_accuracies = []
    p_accuracies = []
    runtimes = []

    max_buffer_size = 650

    for base_num_cells in base_resolutions:

        _, r_jamr, _, _, _ = jamr_simulate(
            base_num_cells=base_num_cells,
            buffer_size=max_buffer_size,
            dt=dt
        )
        optimal_buffer_size = r_jamr.shape[0] + 10 # safety margin
        buffer_sizes.append(optimal_buffer_size)
        print(f"jamr simulation with {optimal_buffer_size} cells")

        # run a first time for jit compilation
        _, r_jamr, rho_jamr, _, _ = jamr_simulate(
            base_num_cells=base_num_cells,
            buffer_size=optimal_buffer_size,
            dt=dt
        )
        rho_jamr = rho_jamr.block_until_ready()

        runtime, r_jamr, rho_jamr, u_jamr, p_jamr = jamr_simulate(
            base_num_cells=base_num_cells,
            buffer_size=optimal_buffer_size,
            dt=dt
        )

        runtimes.append(runtime)

        # calculate accuracy
        r_exact, rho_exact, u_exact, p_exact = exact_solution(r_jamr)
        rho_error = jnp.abs(rho_jamr - rho_exact)
        u_error = jnp.abs(u_jamr - u_exact)
        p_error = jnp.abs(p_jamr - p_exact)
        rho_accuracy = jnp.mean(rho_error)
        u_accuracy = jnp.mean(u_error)
        p_accuracy = jnp.mean(p_error)
        rho_accuracies.append(rho_accuracy)
        u_accuracies.append(u_accuracy)
        p_accuracies.append(p_accuracy)

    return (
        buffer_sizes,
        runtimes,
        rho_accuracies,
        u_accuracies,
        p_accuracies,
    )

def astronomix_test(cell_nums = [40, 80, 160, 320], dt = 0.0002):
    rho_accuracies = []
    u_accuracies = []
    p_accuracies = []
    runtimes = []

    for num_cells in cell_nums:
        runtime, r_astronomix, rho_astronomix, u_astronomix, p_astronomix = astronomix_simulate(num_cells, dt)
        runtimes.append(runtime)

        # calculate accuracy
        r_exact, rho_exact, u_exact, p_exact = exact_solution(r_astronomix)
        rho_error = jnp.abs(rho_astronomix - rho_exact)
        u_error = jnp.abs(u_astronomix - u_exact)
        p_error = jnp.abs(p_astronomix - p_exact)
        rho_accuracy = jnp.mean(rho_error)
        u_accuracy = jnp.mean(u_error)
        p_accuracy = jnp.mean(p_error)
        rho_accuracies.append(rho_accuracy)
        u_accuracies.append(u_accuracy)
        p_accuracies.append(p_accuracy)

    return (
        runtimes,
        rho_accuracies,
        u_accuracies,
        p_accuracies,
    )

# # Run the jamr test
# buffer_sizes, jamr_runtimes, jamr_rho_accuracies, jamr_u_accuracies, jamr_p_accuracies = jamr_test(base_resolutions=jnp.arange(20, 120, 5), dt=0.0001)
# # Run the astronomix test
# astronomix_runtimes, astronomix_rho_accuracies, astronomix_u_accuracies, astronomix_p_accuracies = astronomix_test(cell_nums=buffer_sizes, dt=0.0001)

# # save the runtimes and accuracies to a file
# buffer_sizes = jnp.array(buffer_sizes)
# jamr_runtimes = jnp.array(jamr_runtimes)
# astronomix_runtimes = jnp.array(astronomix_runtimes)
# jamr_rho_accuracies = jnp.array(jamr_rho_accuracies)
# astronomix_rho_accuracies = jnp.array(astronomix_rho_accuracies)
# jamr_u_accuracies = jnp.array(jamr_u_accuracies)
# astronomix_u_accuracies = jnp.array(astronomix_u_accuracies)
# jamr_p_accuracies = jnp.array(jamr_p_accuracies)
# astronomix_p_accuracies = jnp.array(astronomix_p_accuracies)

# # save the results to a file
# jnp.savez('jamr_vs_astronomix_results.npz',
#     buffer_sizes=buffer_sizes,
#     jamr_runtimes=jamr_runtimes,
#     astronomix_runtimes=astronomix_runtimes,
#     jamr_rho_accuracies=jamr_rho_accuracies,
#     astronomix_rho_accuracies=astronomix_rho_accuracies,
#     jamr_u_accuracies=jamr_u_accuracies,
#     astronomix_u_accuracies=astronomix_u_accuracies,
#     jamr_p_accuracies=jamr_p_accuracies,
#     astronomix_p_accuracies=astronomix_p_accuracies
# )

# load the results from the file
results = jnp.load('jamr_vs_astronomix_results.npz')
buffer_sizes = results['buffer_sizes']
jamr_runtimes = results['jamr_runtimes']
astronomix_runtimes = results['astronomix_runtimes']
jamr_rho_accuracies = results['jamr_rho_accuracies']
astronomix_rho_accuracies = results['astronomix_rho_accuracies']
jamr_u_accuracies = results['jamr_u_accuracies']
astronomix_u_accuracies = results['astronomix_u_accuracies']
jamr_p_accuracies = results['jamr_p_accuracies']
astronomix_p_accuracies = results['astronomix_p_accuracies']

# print the maximum relative error difference of jamr to astronomix and
# where they occur

max_rel_rho_diff = jnp.max((astronomix_rho_accuracies - jamr_rho_accuracies) / astronomix_rho_accuracies * 100)
max_rel_rho_diff_idx = jnp.argmax((astronomix_rho_accuracies - jamr_rho_accuracies) / astronomix_rho_accuracies * 100)
print(f"Maximum relative error difference in density: {max_rel_rho_diff:.2f}% at buffer size {buffer_sizes[max_rel_rho_diff_idx]}")
max_rel_u_diff = jnp.max((astronomix_u_accuracies - jamr_u_accuracies) / astronomix_u_accuracies * 100)
max_rel_u_diff_idx = jnp.argmax((astronomix_u_accuracies - jamr_u_accuracies) / astronomix_u_accuracies * 100)
print(f"Maximum relative error difference in velocity: {max_rel_u_diff:.2f}% at buffer size {buffer_sizes[max_rel_u_diff_idx]}")
max_rel_p_diff = jnp.max((astronomix_p_accuracies - jamr_p_accuracies) / astronomix_p_accuracies * 100)
max_rel_p_diff_idx = jnp.argmax((astronomix_p_accuracies - jamr_p_accuracies) / astronomix_p_accuracies * 100)
print(f"Maximum relative error difference in pressure: {max_rel_p_diff:.2f}% at buffer size {buffer_sizes[max_rel_p_diff_idx]}")

# Get data for the top row plots (final state comparison)
r_exact, rho_exact, u_exact, p_exact = exact_solution()
_, r_jamr, rho_jamr, u_jamr, p_jamr = jamr_simulate(base_num_cells=40, buffer_size=152, dt=0.001)
_, r_astronomix, rho_astronomix, u_astronomix, p_astronomix = astronomix_simulate(152, 0.001)

num_cells_jamr = r_jamr.shape[0]
print(f"jamr simulation with {num_cells_jamr} cells")

# Create a new figure with a GridSpec layout
# The figure will have 3 rows. The top row has 3 plots. The middle and bottom rows each have one plot.
fig = plt.figure(figsize=(10, 13))
gs = fig.add_gridspec(3, 3)

# --- Top Row: Direct Comparison Plots ---
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
# Use a colorblind-friendly palette
colors = {
    'exact': '#000000',      # black
    'jamr': '#0072B2',       # blue
    'astronomix': '#D55E00',    # orange
    'jamr_density': '#0072B2',
    'jamr_pressure': '#56B4E9',  # lighter blue
    'astronomix_density': '#D55E00',
    'astronomix_pressure': '#E69F00', # lighter orange
}

# Density Plot (ax1)
ax1.plot(r_exact, rho_exact, label='Exact', linestyle='--', color=colors['exact'])
ax1.plot(r_jamr, rho_jamr, label='jamr', marker='o', markersize=2, linestyle='-', color=colors['jamr'])
ax1.plot(r_astronomix, rho_astronomix, label='astronomix', marker='x', markersize=2, linestyle='-', color=colors['astronomix'])
ax1.set_title('Density')
ax1.set_xlabel('Position')
ax1.set_ylabel('Density')
ax1.legend()

# Velocity Plot (ax2)
ax2.plot(r_exact, u_exact, label='Exact', linestyle='--', color=colors['exact'])
ax2.plot(r_jamr, u_jamr, label='jamr', marker='o', markersize=2, linestyle='-', color=colors['jamr'])
ax2.plot(r_astronomix, u_astronomix, label='astronomix', marker='x', markersize=2, linestyle='-', color=colors['astronomix'])
ax2.set_title('Velocity')
ax2.set_xlabel('Position')
ax2.set_ylabel('Velocity')
ax2.legend()

# Pressure Plot (ax3)
ax3.plot(r_exact, p_exact, label='Exact', linestyle='--', color=colors['exact'])
ax3.plot(r_jamr, p_jamr, label='jamr', marker='o', markersize=2, linestyle='-', color=colors['jamr'])
ax3.plot(r_astronomix, p_astronomix, label='astronomix', marker='x', markersize=2, linestyle='-', color=colors['astronomix'])
ax3.set_title('Pressure')
ax3.set_xlabel('Position')
ax3.set_ylabel('Pressure')
ax3.legend()

# --- Middle Row: Mean Error Plot ---
ax4 = fig.add_subplot(gs[1, :])
ax4.plot(buffer_sizes, astronomix_rho_accuracies, label='astronomix, density', marker='x', color=colors['astronomix_density'])
ax4.plot(buffer_sizes, astronomix_p_accuracies, label='astronomix, pressure', marker='x', color=colors['astronomix_pressure'])
ax4.plot(buffer_sizes, jamr_rho_accuracies, label='jamr, density', marker='o', color=colors['jamr_density'])
ax4.plot(buffer_sizes, jamr_p_accuracies, label='jamr, pressure', marker='o', color=colors['jamr_pressure'])
ax4.set_title('Mean Errors')
ax4.set_xlabel('buffer size (jamr), num cells (astronomix)')
ax4.set_ylabel('Mean Absolute Error')
ax4.legend()
ax4.grid(True)

# --- Bottom Row: Runtime Plot ---
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(buffer_sizes, jamr_runtimes, label='jamr', marker='o', color=colors['jamr'])
ax5.plot(buffer_sizes, astronomix_runtimes, label='astronomix', marker='x', color=colors['astronomix'])
ax5.set_title('Runtime')
ax5.set_xlabel('buffer size (jamr), num cells (astronomix)')
ax5.set_ylabel('Runtime in seconds')
ax5.legend()
ax5.grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('jamr_vs_astronomix.svg')


# create example initial fluid state
shock_pos = 0.5
N = 20
r = jnp.linspace(0, 1, N)
dx_coarse = 1 / (N - 1)
rho = jnp.where(r < shock_pos, 1.0, 0.125)
u = jnp.ones_like(r) * 0.0
p = jnp.where(r < shock_pos, 1.0, 0.1)

# create buffers
buffer_size = 200
fluid_buffer = jnp.full((3, buffer_size), 1.0)
center_buffer = jnp.linspace(1, 2, buffer_size)
volume_buffer = jnp.full((buffer_size,), 1.0)
refinement_level_buffer = jnp.full((buffer_size,), 1.0)

# fill buffers with initial fluid state
fluid_buffer = fluid_buffer.at[:, 0:N].set(jnp.stack([rho, u, p], axis=0))
center_buffer = center_buffer.at[0:N].set(r)
volume_buffer = volume_buffer.at[0:N].set(dx_coarse)
refinement_level_buffer = refinement_level_buffer.at[0:N].set(0)

# create fluid data
fluid_data = _BufferedList(
    jnp.vstack([fluid_buffer, center_buffer, volume_buffer, refinement_level_buffer]), N
)

derefinement_tolerance = 0.5
refinement_tolerance = 5.0

fluid_data_final = time_integration_fixed_stepsize(
    fluid_data, 0.001, 200, 3, refinement_tolerance, derefinement_tolerance
)

fig, ax = plt.subplots(figsize=(10, 6))
plot_density(ax, fluid_data_final)
plt.savefig('jamr_density_plot.svg')