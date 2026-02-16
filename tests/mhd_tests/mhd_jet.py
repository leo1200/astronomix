# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

# numerics
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.random import PRNGKey, uniform

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import MHD_JET_BOUNDARY, OPEN_BOUNDARY, SPLIT, finalize_config
from matplotlib.colors import LogNorm

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD,
    OSHER, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D
)

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0
num_cells = 500

# setup simulation config
config = SimulationConfig(
    progress_bar = True,
    mhd = True,
    dimensionality = 2,
    box_size = box_size, 
    num_cells = num_cells,
    differentiation_mode = BACKWARDS,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(left_boundary = OPEN_BOUNDARY, right_boundary = OPEN_BOUNDARY),
        BoundarySettings1D(left_boundary = MHD_JET_BOUNDARY, right_boundary = OPEN_BOUNDARY),
        BoundarySettings1D(left_boundary = OPEN_BOUNDARY, right_boundary = OPEN_BOUNDARY))
)

helper_data = get_helper_data(config)

params = SimulationParams(
    t_end = 0.001,
    C_cfl = 0.4
)

registered_variables = get_registered_variables(config)


# Set the random seed for reproducibility
key = PRNGKey(0)

# Grid size and configuration
num_cells = config.num_cells
grid_spacing = box_size / num_cells
x = jnp.linspace(grid_spacing / 2, box_size - grid_spacing / 2, num_cells)
y = jnp.linspace(grid_spacing / 2, box_size - grid_spacing / 2, num_cells)
X, Y = jnp.meshgrid(x, y, indexing="ij")

# Initialize state
rho = jnp.ones_like(X) * gamma * 0.1
P = jnp.ones_like(X)

V_x = jnp.zeros_like(X)
V_y = jnp.zeros_like(X)

B_0 = 200 ** 0.5
B_x = B_0 * jnp.ones_like(X)
B_y = jnp.zeros_like(X)
# B_x = jnp.zeros_like(X)
# B_y = jnp.zeros_like(X)
B_z = jnp.zeros_like(X)

initial_magnetic_field = jnp.stack([B_x, B_y, B_z], axis=0)

dx = 1 / (num_cells - 1)

initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = V_x,
    velocity_y = V_y,
    magnetic_field_x = B_x,
    magnetic_field_y = B_y,
    magnetic_field_z = B_z,
    gas_pressure = P
)

config = finalize_config(config, initial_state.shape)

final_state = time_integration(initial_state, config, params, registered_variables)

# save the final state to a file
jnp.savez('mhd_jet_final_state.npz', final_state=final_state)
# load the final state from the file
final_state = jnp.load('mhd_jet_final_state.npz')['final_state']

# plot the final state, just an imshow of the density
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
im = ax.imshow(
    final_state[registered_variables.density_index].T,
    origin='lower',
    extent=(0, box_size, 0, box_size),
    aspect='auto',
    cmap='viridis',
    norm=LogNorm()
)
ax.set_title('Density')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='Density')
plt.tight_layout()
plt.savefig('mhd_jet.png', dpi=300)