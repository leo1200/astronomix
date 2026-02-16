# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

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
from astronomix.option_classes.simulation_config import finalize_config

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD,
    OSHER, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D
)

# simulation settings
gamma = 5/3  # adiabatic index

# spatial domain
box_size = 2 * jnp.pi
num_cells = 200

# setup simulation config
config = SimulationConfig(
    progress_bar = True,
    mhd = True,
    dimensionality = 2,
    box_size = box_size, 
    num_cells = num_cells,
    differentiation_mode = BACKWARDS,
    limiter = MINMOD,
    riemann_solver = HLL,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY))
)

helper_data = get_helper_data(config)

params = SimulationParams(
    t_end = 3.0,
    C_cfl = 0.4,
    gamma = gamma,
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
rho = jnp.ones_like(X) * gamma ** 2
P = jnp.ones_like(X) * gamma

V_x = -jnp.sin(Y)
V_y = jnp.sin(X)

B_x = -jnp.sin(Y)
B_y = jnp.sin(2 * X)
# B_x = jnp.zeros_like(X)
# B_y = jnp.zeros_like(X)
B_z = jnp.zeros_like(X)

initial_magnetic_field = jnp.stack([B_x, B_y, B_z], axis=0)

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
jnp.savez("final_state_orszag_tang.npz", final_state=final_state)
# load the final state from the file
final_state = jnp.load("final_state_orszag_tang.npz")["final_state"]

# plot the final state, just an imshow of the density
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im = axs[0].imshow(final_state[registered_variables.density_index].T, origin='lower', extent=(0, box_size, 0, box_size), aspect='auto', cmap='viridis')
axs[0].set_title('density')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
# set the aspect ratio to be equal
axs[0].set_aspect('equal', adjustable='box')
# add colorbar with make_axes_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('density')

y_eval = 0.625 * jnp.pi
y_index = jnp.argmin(jnp.abs(y - y_eval))
print(min(jnp.abs(y - y_eval)), y_index)
axs[1].plot(x, final_state[0, :, y_index], label = "density (astronomix)")

# load the data from pang24.txt in the format 0.04201468733773461; 3.1120498020333187
import numpy as np
pang24_data = np.loadtxt("pang24.txt", delimiter=";")
# sort the data by the first column
pang24_data = pang24_data[np.argsort(pang24_data[:, 0])]
axs[1].plot(pang24_data[:, 0], pang24_data[:, 1], label = "density (Pang and Wu, 2024)", linestyle='--')

axs[1].set_xlabel("x")
axs[1].set_ylabel("density")
axs[1].set_title(r"density at y = 0.625$\pi$")
axs[1].legend()

plt.tight_layout()

plt.savefig("figures/orszag_tang.png", dpi=300)