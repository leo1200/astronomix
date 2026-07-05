# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
import jax.numpy as jnp

# setup
from astronomix import SimulationConfig
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, OPEN_BOUNDARY, BoundarySettings, BoundarySettings1D
)
from astronomix import SimulationParams
from astronomix import get_registered_variables
from astronomix import construct_primitive_state

# main time integration function
from astronomix import time_integration

# config finalizing
from astronomix.option_classes.simulation_config import finalize_config

# new vector potential helper
from astronomix.initial_condition_generation.magnetic_field_from_vector_potential import setup_magnetic_fields_from_vector_potential

# plotting
import matplotlib.pyplot as plt

# simulation settings
gamma = 5/3

# spatial domain
box_size = 24.0
num_cells = 256
grid_spacing = box_size / num_cells
x_center = box_size / 2.0
y_center = box_size / 2.0
z_center = box_size / 2.0

# simulation config
config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    grid_spacing = grid_spacing,
    mhd = True,
    progress_bar = True,
    dimensionality = 3,
    box_size = box_size,
    num_cells = num_cells,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        )
    ),
)

# get the variable registry
registered_variables = get_registered_variables(config)

# time domain
C_CFL = 0.8

# initial hydro state constants
rho_0 = 1.0
p_0 = 1.0

# define the vector potential function
def jet_vector_potential(X, Y, Z):
    r = jnp.sqrt((X - x_center)**2 + (Y - y_center)**2 + (Z - z_center)**2)
    A0 = 20.0
    
    A_x = -jnp.exp(-r ** 2) * (Y - y_center)
    A_y = jnp.exp(-r ** 2) * (X - x_center)
    A_z = 0.5 * A0 * jnp.exp(-r ** 2)
    
    return A_x, A_y, A_z

# init b fields from vector potential
B_x, B_y, B_z, bxb, byb, bzb = setup_magnetic_fields_from_vector_potential(
    config=config,
    vector_potential_func=jet_vector_potential
)

# Set up primitive hydro arrays
rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0
u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0

# simulation params
params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = 0.1,
    t_end = 5.0,
    gamma = gamma,
    minimum_density = 1e-2 * rho_0,
    minimum_pressure = 1e-2 * p_0,
)

# construct primitive state
initial_state = construct_primitive_state(
    config = config,
    registered_variables=registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p,
    magnetic_field_x = B_x,
    magnetic_field_y = B_y,
    magnetic_field_z = B_z,
    interface_magnetic_field_x = bxb,
    interface_magnetic_field_y = byb,
    interface_magnetic_field_z = bzb,
)

config = finalize_config(config, initial_state.shape)

# Diagnostic print
B_mag = jnp.sqrt(B_x**2 + B_y**2 + B_z**2)
print(jnp.max(B_mag))
print(jnp.argmax(B_mag))

# Run the simulation
final_state = time_integration(initial_state, config, params, registered_variables)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
y_index = num_cells // 2
ax.imshow(final_state[registered_variables.density_index, :, y_index, :].T, cmap="YlOrRd")

fig.savefig("figures/mhd_jet3D_density.png", dpi=300)