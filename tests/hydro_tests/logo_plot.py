# ==== GPU selection ====
import pickle
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================


# numerics
import jax
import jax.numpy as jnp

# equinox
import equinox as eqx

# optax for optimization
import optax

# timing and progress
from timeit import default_timer as timer
from tqdm import tqdm
import os

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable # Import the necessary toolkit

# astronomix
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config

from astronomix.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, MINMOD,
    OSHER, PERIODIC_BOUNDARY, BoundarySettings,
    BoundarySettings1D
)

from astronomix._physics_modules._neural_net_force._neural_net_force import ForceNet
from astronomix._physics_modules._neural_net_force._neural_net_force_options import NeuralNetForceConfig, NeuralNetForceParams


# ===================================================
# =============== ↓ setup neural net ↓ ==============
# ===================================================

key = jax.random.PRNGKey(42)
model = ForceNet(key)
# Partition the model into its trainable arrays (parameters) and static components
neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

# ===================================================
# =============== ↑ setup neural net ↑ ==============
# ===================================================


# ===================================================
# =============== ↓ simulation setup ↓ ==============
# ===================================================

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0
num_cells = 128

# setup simulation config
config = SimulationConfig(
    progress_bar = False, # Disable internal progress bar for cleaner training output
    dimensionality = 2,
    box_size = box_size,
    num_cells = num_cells,
    differentiation_mode = BACKWARDS,
    boundary_settings=BoundarySettings(
        x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    ),
    limiter = MINMOD,
    return_snapshots = False,
    neural_net_force_config = NeuralNetForceConfig(
        neural_net_force = True,
        network_static = neural_net_static, # Pass static part of the network to the config
    )
)

helper_data = get_helper_data(config)

# This is a template for the simulation parameters.
# The network parameters within it will be updated during training.
neural_net_force_params = NeuralNetForceParams(
        network_params = neural_net_params,
)

t_end = 2.0

params = SimulationParams(
    t_end = t_end,
    C_cfl = 0.4,
    neural_net_force_params = neural_net_force_params,
    dt_max = 0.01,  # Set a maximum timestep for stability
)

registered_variables = get_registered_variables(config)
xm, ym = helper_data.geometric_centers[..., 0], helper_data.geometric_centers[..., 1]

# ===================================================
# ======== ↓ Define Target and Initial State ↓ ======
# ===================================================

# 1. Define the target state: a thicker, blockier "1"
stem = (xm > 0.42) & (xm < 0.58) & (ym > 0.2) & (ym < 0.8)
nose = (xm > 0.25) & (xm < 0.42) & (ym > 0.65) & (ym < 0.8)
base = (xm > 0.25) & (xm < 0.75) & (ym > 0.2) & (ym < 0.35)
is_one = stem | nose | base
target_density = jnp.where(is_one, 1.001, 1.0)

# 2. Define the initial state: a square in the center with the same area, density, AND slightly higher pressure
# Calculate the total area of the target "1"
num_target_cells = jnp.sum(is_one)
cell_area = (box_size / num_cells)**2
target_area = num_target_cells * cell_area

# Calculate the side length of a square with this area
side_length = jnp.sqrt(target_area)
center_x, center_y = box_size / 2.0, box_size / 2.0

# Define the square's boundaries
is_in_square = (
    (xm > center_x - side_length / 2) & (xm < center_x + side_length / 2) &
    (ym > center_y - side_length / 2) & (ym < center_y + side_length / 2)
)

# Set initial density based on the square
initial_rho = jnp.mean(target_density) * jnp.ones_like(xm)
u_x = jnp.zeros_like(xm)
u_y = jnp.zeros_like(ym)

# === NEW: Set higher pressure inside the square ===
p_background = 2.5
p_square = 2.5  # A little higher to make it expand (e.g., 20% higher)
initial_p = jnp.where(is_in_square, p_square, p_background)
# ================================================

# Construct the initial state object for the simulation
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = initial_rho,
    velocity_x = u_x,
    velocity_y = u_y,
    gas_pressure = initial_p  # Use the new pressure array
)
config = finalize_config(config, initial_state.shape)
# ===================================================
# ======== ↑ Define Target and Initial State ↑ ======
# ===================================================


# This function defines the forward pass of our model: from network parameters to final state.
# It takes only the trainable network parameters as input, which is ideal for optimization.
def force_pass(
    network_params_arrays: eqx.Module,
):
    """Runs the simulation with the given network parameters."""
    # Create the full parameter objects required by the simulation
    current_nnf_params = NeuralNetForceParams(network_params=network_params_arrays)
    current_sim_params = params._replace(neural_net_force_params=current_nnf_params)

    return time_integration(
        initial_state,
        config,
        current_sim_params,
        registered_variables,
    )

# ===================================================
# ================= ↑ Training Setup ↑ ==============
# ===================================================

# load the trained parameters if needed
with open("trained_params3.pkl", "rb") as f:
    trained_params = pickle.load(f)


# ===================================================
# =================== ↓ plotting ↓ ==================
# ===================================================

print("Running final simulation with trained parameters...")
# Run the simulation one last time with the final trained parameters
final_state = force_pass(trained_params)

# Reconstruct the full trained model to evaluate the force field for plotting
trained_model = eqx.combine(trained_params, neural_net_static)

def get_force_and_density(time):

    # =============== evaluate force net ================
    N = config.num_cells
    positions = helper_data.geometric_centers
    positions_flat = jnp.reshape(positions, (-1, 2))  # (N*N, 2)

    # Broadcast t_end to all positions and concatenate
    t_end_broadcast = jnp.full((positions_flat.shape[0], 1), time)
    positions_with_time = jnp.concatenate([positions_flat, t_end_broadcast], axis=1)  # (N*N, 3)

    # Evaluate the model
    forces_flat = jax.vmap(trained_model)(positions_with_time)  # (N*N, 2)
    forces = forces_flat.reshape(N, N, 2).transpose(2, 0, 1)
    # ===================================================

    # ================== get density ====================

    current_nnf_params = NeuralNetForceParams(network_params=trained_params)
    current_sim_params = params._replace(neural_net_force_params=current_nnf_params, t_end=time)

    final_state = time_integration(
        initial_state,
        config,
        current_sim_params,
        registered_variables,
    )

    return forces, final_state[0]

# Plot the target density
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(target_density, extent=(0, box_size, 0, box_size),
               origin='lower', cmap='viridis')
ax.set_title("Target Density")
ax.set_xlabel("x")
ax.set_ylabel("y")
# Create a divider for the axes and add a colorbar axes to it
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax, label='Density')
plt.savefig("target_density.png", dpi=300)

# plot the density and force field over time
times = [0.0, 1.8, 2.0]
fig, axs = plt.subplots(2, len(times), figsize=(15, 8))

for i, time in enumerate(times):
    forces, density = get_force_and_density(time)

    # --- Plot the density ---
    ax_density = axs[0, i]
    im = ax_density.imshow(density, extent=(0, box_size, 0, box_size),
                           origin='lower', cmap='viridis')
    ax_density.set_title(f"Density at t={time:.2f}")
    ax_density.set_xlabel("x")
    ax_density.set_ylabel("y")
    # Make colorbar the same height as the plot
    divider_density = make_axes_locatable(ax_density)
    cax_density = divider_density.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax_density, label='Density')


    # --- Plot the force field ---
    ax_force = axs[1, i]
    xm, ym = helper_data.geometric_centers[..., 0], helper_data.geometric_centers[..., 1]
    step = 5
    X = xm[::step, ::step]
    Y = ym[::step, ::step]
    U = forces[0][::step, ::step]
    V = forces[1][::step, ::step]
    magnitude = jnp.sqrt(U**2 + V**2)

    # Normalize vectors to unit length for uniform arrow size
    norm = jnp.sqrt(U**2 + V**2)
    U_norm = jnp.where(norm > 0, U / norm, 0)
    V_norm = jnp.where(norm > 0, V / norm, 0)

    q = ax_force.quiver(
        X, Y, U_norm, V_norm, magnitude,
        cmap='plasma', scale=30, clim=(magnitude.min(), magnitude.max())
    )
    ax_force.set_title(f"Force Field at t={time:.2f}")
    ax_force.set_xlabel("x")
    ax_force.set_ylabel("y")
    ax_force.set_xlim(0, box_size)
    ax_force.set_ylim(0, box_size)
    ax_force.set_aspect('equal')
    # Make colorbar the same height as the plot
    divider_force = make_axes_locatable(ax_force)
    cax_force = divider_force.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(q, cax=cax_force, label='Force Magnitude')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("density_and_force_field.png", dpi=300)

# ===================================================
# =================== ↑ plotting ↑ ==================
# ===================================================