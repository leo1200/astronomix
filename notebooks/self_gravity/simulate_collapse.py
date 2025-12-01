# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

# numerics
import jax
import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# astronomix classes
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes.simulation_config import BoundarySettings, BoundarySettings1D

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
    BoundarySettings, BoundarySettings1D
)

def simulate_collapse(num_cells):
    print("👷 Setting up simulation...")

    # simulation settings
    gamma = 5/3

    # spatial domain
    box_size = 4.0

    fixed_timestep = False
    dt_max = 0.001

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging = False,
        progress_bar = True,
        self_gravity = True,
        dimensionality = 3,
        box_size = box_size, 
        num_cells = num_cells,
        fixed_timestep = fixed_timestep,
        differentiation_mode = FORWARDS,
        limiter = MINMOD,
        riemann_solver = HLL,
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
        num_snapshots = 60
    )

    helper_data = get_helper_data(config)

    params = SimulationParams(
        t_end = 3.0,
        C_cfl = 0.4,
        dt_max = dt_max,
    )

    registered_variables = get_registered_variables(config)

        
    R = 1.0
    M = 1.0

    dx = config.box_size / (config.num_cells - 1)

    # initialize density field
    num_injection_cells = jnp.sum(helper_data.r <= R)
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

    return jax.block_until_ready(time_integration(initial_state, config, params, registered_variables))

def resolution_study():

    num_cells_list = [32, 64, 128, 192]
    line_styles = ['-', '--', '-.', ':']

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for num_cells, line_style in zip(num_cells_list, line_styles):
        snapshots = simulate_collapse(num_cells)
        total_energy = snapshots.total_energy
        internal_energy = snapshots.internal_energy
        kinetic_energy = snapshots.kinetic_energy
        gravitational_energy = snapshots.gravitational_energy
        time = snapshots.time_points
        ax.plot(time, total_energy, label="Total Energy, N = " + str(num_cells), color = 'black', linestyle = line_style)
        ax.plot(time, internal_energy, label="Internal Energy, N = " + str(num_cells), color = 'green', linestyle = line_style)
        ax.plot(time, kinetic_energy, label="Kinetic Energy, N = " + str(num_cells), color = 'red', linestyle = line_style)
        ax.plot(time, gravitational_energy, label="Gravitational Energy, N = " + str(num_cells), color = 'blue', linestyle = line_style)
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")

    ax.set_ylim(-2.5, 2.5)

    ax.legend(fontsize="x-small", ncol=len(num_cells_list))
    ax.set_title("Resolution Study for Evrard's Collapse")

    plt.savefig("collapse_resolution_study.svg")

resolution_study()