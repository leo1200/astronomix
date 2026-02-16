# ==== GPU selection ====
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
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

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
    dt_max = 0.01,
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
# ================= ↓ Training Setup ↓ ==============
# ===================================================

@eqx.filter_jit
def loss_fn(network_params_arrays):
    """Calculates the difference between the final state and the target."""
    final_state = force_pass(network_params_arrays)
    final_density = final_state[0, :, :]
    # Mean Squared Error loss
    loss = jnp.mean((final_density - target_density)**2)
    return loss

# Set up the optimizer using optax
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(neural_net_params)

@eqx.filter_jit
def train_step(network_params_arrays, opt_state):
    """Performs one step of gradient descent."""
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(network_params_arrays)
    updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
    network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
    return network_params_arrays, opt_state, loss_value

# ===================================================
# ================= ↑ Training Setup ↑ ==============
# ===================================================


# ===================================================
# ================== ↓ Training Loop ↓ ==============
# ===================================================
print("Starting training with optax...")
num_steps = 5000
losses = []

# This variable will hold the trained parameters and be updated in the loop
trained_params = neural_net_params

# Timing
start_time = timer()

# The main training loop
pbar = tqdm(range(num_steps))
best_loss = float('inf')
best_params = trained_params
for step in pbar:
    trained_params, opt_state, loss = train_step(trained_params, opt_state)
    losses.append(loss)
    if loss < best_loss:
        best_loss = loss
        best_params = trained_params
    pbar.set_description(f"Step {step+1}/{num_steps} | Loss: {loss:.2e}")

# After training, use the best parameters found
trained_params = best_params

end_time = timer()
print(f"Training finished in {end_time - start_time:.2f} seconds.")
# ===================================================
# ================== ↑ Training Loop ↑ ==============
# ===================================================

# pickle the trained parameters for later use
import pickle
with open("trained_params4.pkl", "wb") as f:
    pickle.dump(trained_params, f)

# load the trained parameters if needed
with open("trained_params4.pkl", "rb") as f:
    trained_params = pickle.load(f)

# save the losses for later analysis with jnp.savez
jnp.savez("losses4.npz", losses = losses)

# ===================================================
# ================ ↓ Final Evaluation ↓ =============
# ===================================================
print("Running final simulation with trained parameters...")
# Run the simulation one last time with the final trained parameters
final_state = force_pass(trained_params)

# Reconstruct the full trained model to evaluate the force field for plotting
trained_model = eqx.combine(trained_params, neural_net_static)
# ===================================================
# ================ ↑ Final Evaluation ↑ =============
# ===================================================


# ===================================================
# =================== ↓ plotting ↓ ==================
# ===================================================

# Ensure the output directory exists
os.makedirs("figures", exist_ok=True)

# =============== evaluate force net ================
N = config.num_cells
positions = helper_data.geometric_centers
positions_flat = jnp.reshape(positions, (-1, 2))  # (N*N, 2)

# Broadcast t_end to all positions and concatenate
t_end_broadcast = jnp.full((positions_flat.shape[0], 1), t_end)
positions_with_time = jnp.concatenate([positions_flat, t_end_broadcast], axis=1)  # (N*N, 3)

# Evaluate the model
forces_flat = jax.vmap(trained_model)(positions_with_time)  # (N*N, 2)
forces = forces_flat.reshape(N, N, 2).transpose(2, 0, 1)
# ===================================================

fig = plt.figure(figsize=(24, 10))
gs = GridSpec(2, 4, figure=fig) # New grid layout
fig.suptitle("Force Term Training Results (with Optax)", fontsize=16)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[1, :]) # Loss curve takes the whole bottom row

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_aspect('equal', 'box')

# Plot 1: Initial Density
im1 = ax1.imshow(initial_rho.T, cmap="jet", origin="lower", extent=[0, box_size, 0, box_size])
cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical')
cbar1.set_label("Density", rotation=270, labelpad=15)
ax1.set_title("Initial Density")

# Plot 2: Final Density
im2 = ax2.imshow(final_state[0, :, :].T, cmap="jet", origin="lower", extent=[0, box_size, 0, box_size])
cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical')
cbar2.set_label("Density", rotation=270, labelpad=15)
ax2.set_title("Final Density (Trained)")

# Plot 3: Target Density
im3 = ax3.imshow(target_density.T, cmap="jet", origin="lower", extent=[0, box_size, 0, box_size])
cbar3 = plt.colorbar(im3, ax=ax3, orientation='vertical')
cbar3.set_label("Density", rotation=270, labelpad=15)
ax3.set_title("Target Density")

# Plot 4: Trained Force Field
step = 8 # Use a larger step for a cleaner quiver plot
ax4.quiver(
    xm[::step, ::step], ym[::step, ::step],
    forces[0, ::step, ::step], forces[1, ::step, ::step],
    color='black',
    angles='xy',
    scale_units='xy',
    scale = 10.0,
)
ax4.set_xlim(0, box_size)
ax4.set_ylim(0, box_size)
ax4.set_title("Trained Neural Net Force Field")

# Plot 5: Loss Curve
ax5.plot(losses)
ax5.set_xlabel("Training Step")
ax5.set_ylabel("Loss (MSE)")
ax5.set_title("Training Loss Curve")
ax5.set_yscale('log')
ax5.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/trained_force_optax.png", dpi=300)
print("Saved final plot to figures/trained_force_optax.png")
# ===================================================
# =================== ↑ plotting ↑ ==================
# ===================================================