# ==== GPU selection ====
from autocvd import autocvd
from matplotlib import animation
autocvd(num_gpus = 1, interval = 10)
# =======================

# numerics
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# astronomix
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

# equinox
import equinox as eqx

# optax for optimization
import optax

# CNN stuff
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector import CorrectorCNN
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import CNNMHDParams, CNNMHDconfig

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

# timing
from timeit import default_timer as timer
from tqdm import tqdm

# io
import numpy as np
import os

# ===================================================
# =============== ↓ simulation setup ↓ ==============
# ===================================================

snapshot_timepoints_train = jnp.array([0.0, 0.05, 0.1, 0.15, 0.2])
snapshot_timepoints = jnp.linspace(0.0, 0.2, 50)

# baseline config
baseline_config = SimulationConfig(
    progress_bar = True,
    mhd = True,
    dimensionality = 2,
    limiter = MINMOD,
    box_size = 1.0, 
    num_cells = 512,
    differentiation_mode = BACKWARDS,
    riemann_solver = HLL,
    exact_end_time = True,
    return_snapshots = True,
    use_specific_snapshot_timepoints = True,
    num_snapshots = len(snapshot_timepoints),
)

# common variable registry
registered_variables = get_registered_variables(baseline_config)

# baseline simulation parameters
params = SimulationParams(
    t_end = 0.2,
    C_cfl = 0.1,
    snapshot_timepoints = snapshot_timepoints
)

def get_blast_setup(num_cells):

    # dummy config for setup
    dummy_config = baseline_config._replace(
        num_cells = num_cells,
    )

    helper_data = get_helper_data(dummy_config)

    # Grid size and configuration
    num_cells = dummy_config.num_cells
    # --- Initial Conditions ---
    grid_spacing = dummy_config.box_size / dummy_config.num_cells
    x = jnp.linspace(grid_spacing / 2, dummy_config.box_size - grid_spacing / 2, dummy_config.num_cells)
    y = jnp.linspace(grid_spacing / 2, dummy_config.box_size - grid_spacing / 2, dummy_config.num_cells)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    r = helper_data.r

    # Initialize state
    rho = jnp.ones_like(X)
    P = jnp.ones_like(X) * 0.1
    r_inj = 0.1 * dummy_config.box_size
    p_inj = 10.0
    P = jnp.where(r**2 < r_inj**2, p_inj, P)

    V_x = jnp.zeros_like(X)
    V_y = jnp.zeros_like(X)

    B_0 = 1 / jnp.sqrt(2)
    B_x = B_0 * jnp.ones_like(X)
    B_y = B_0 * jnp.ones_like(X)
    B_z = jnp.zeros_like(X)

    initial_state = construct_primitive_state(
        config = dummy_config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = V_x,
        velocity_y = V_y,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        gas_pressure = P
    )

    return initial_state, helper_data


def downaverage_state(state: jnp.ndarray, target_shape: tuple[int, int]) -> jnp.ndarray:
    """
    Downaverages the spatial dimensions of a state array using block reshaping.

    This function is designed for a state array with the shape
    (NUM_VARS, H, W) and reduces it to (NUM_VARS, h, w) by
    averaging over non-overlapping blocks.

    Args:
        state: The input JAX array with shape (NUM_VARS, H, W).
        target_shape: A tuple (h, w) representing the desired output
                      spatial dimensions. H must be divisible by h, and W
                      must be divisible by w.

    Returns:
        The downaveraged JAX array with shape (NUM_VARS, h, w).
    """
    # 1. Get input and output dimensions
    num_vars, h_in, w_in = state.shape
    h_out, w_out = target_shape

    # 2. Assert that the downsampling is possible (dimensions are divisible)
    if h_in % h_out != 0 or w_in % w_out != 0:
        raise ValueError(
            f"Input shape {(h_in, w_in)} is not divisible by target shape {(h_out, w_out)}"
        )
    
    # 3. Calculate the block size (or downsampling factor)
    h_factor = h_in // h_out
    w_factor = w_in // w_out
    
    # 4. Reshape to create blocks and then take the mean
    # Original shape: (V, H, W)
    # Reshape to: (V, h_out, h_factor, w_out, w_factor)
    # This groups the original grid into blocks.
    reshaped = state.reshape(num_vars, h_out, h_factor, w_out, w_factor)
    
    # Take the mean over the block axes (h_factor and w_factor).
    # The axes are 2 and 4 in the reshaped array.
    downaveraged = reshaped.mean(axis=(2, 4))
    
    return downaveraged

# ===================================================
# =============== ↑ simulation setup ↑ ==============
# ===================================================

num_cells_high_res = 512
num_cells_low_res = 64

config_high_res = baseline_config._replace(
    num_cells = num_cells_high_res,
)
initial_state_high_res, helper_data_high_res = get_blast_setup(num_cells_high_res)
config_high_res = finalize_config(config_high_res, initial_state_high_res.shape)

config_low_res = baseline_config._replace(
    num_cells = num_cells_low_res,
)
_, helper_data_low_res = get_blast_setup(num_cells_low_res)
# get the initial low res state by downsampling the high res state
initial_state_low_res = downaverage_state(initial_state_high_res, (num_cells_low_res, num_cells_low_res))
config_low_res = finalize_config(config_low_res, initial_state_low_res.shape)

# run high resolution simulation
result_high_res = time_integration(initial_state_high_res, config_high_res, params, registered_variables)
states_high_res_downsampled = jax.vmap(downaverage_state, in_axes=(0, None), out_axes=0)(result_high_res.states, (num_cells_low_res, num_cells_low_res))

# run low resolution simulation
result_low_res = time_integration(initial_state_low_res, config_low_res, params, registered_variables)

# calculate the mean squared error between the high and low resolution results
# the result.states is of shape (num_states, num_vars, H, W)
mse_base = jnp.mean((states_high_res_downsampled - result_low_res.states) ** 2, axis=(1, 2, 3))
print(mse_base)

# ===================================================
# ============== ↓ CNN training setup ↓ =============
# ===================================================

model = CorrectorCNN(
    in_channels = registered_variables.num_vars,
    hidden_channels = 16,
    key = jax.random.PRNGKey(42)
)
neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

cnn_mhd_corrector_config = CNNMHDconfig(
    cnn_mhd_corrector = True,
    network_static = neural_net_static
)

cnn_mhd_corrector_params = CNNMHDParams(
    network_params = neural_net_params
)

config_low_res_cnn = config_low_res._replace(
    cnn_mhd_corrector_config = cnn_mhd_corrector_config,
)
params_low_res_cnn = params._replace(
    cnn_mhd_corrector_params = cnn_mhd_corrector_params,
)

@eqx.filter_jit
def loss_fn(network_params_arrays):
    """Calculates the difference between the final state and the target."""
    results_low_res = time_integration(
        initial_state_low_res,
        config_low_res_cnn,
        params_low_res_cnn._replace(
            cnn_mhd_corrector_params = cnn_mhd_corrector_params._replace(
                network_params = network_params_arrays
            )
        ),
        registered_variables
    )
    # Calculate the L2 loss between the final state and the target state
    loss = jnp.mean((results_low_res.states - states_high_res_downsampled) ** 2)
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
# ============== ↑ CNN training setup ↑ =============
# ===================================================

# ===================================================
# ================== ↓ Training Loop ↓ ==============
# ===================================================
# print("Starting training with optax...")
# num_steps = 2000
# losses = []

# # This variable will hold the trained parameters and be updated in the loop
# trained_params = neural_net_params

# # Timing
# start_time = timer()

# # The main training loop
# pbar = tqdm(range(num_steps))
# best_loss = float('inf')
# best_params = trained_params
# for step in pbar:
#     trained_params, opt_state, loss = train_step(trained_params, opt_state)
#     losses.append(loss)
#     if loss < best_loss:
#         best_loss = loss
#         best_params = trained_params
#     pbar.set_description(f"Step {step+1}/{num_steps} | Loss: {loss:.2e}")

# # After training, use the best parameters found
# trained_params = best_params

# end_time = timer()
# print(f"Training finished in {end_time - start_time:.2f} seconds.")

# # # save the trained parameters using pickle
import pickle
output_dir = "trained_models"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cnn_mhd_corrector_params.pkl")
# with open(output_path, "wb") as f:
#     pickle.dump(trained_params, f)

# load the trained parameters
with open(output_path, "rb") as f:
    trained_params = pickle.load(f)

# save the losses as npz
losses_output_path = os.path.join(output_dir, "cnn_mhd_corrector_losses.npz")
# np.savez(losses_output_path, losses=losses)
# load the losses
losses = np.load(losses_output_path)['losses']

# ===================================================
# ================== ↑ Training Loop ↑ ==============
# ===================================================

# run the low resolution simulation with the trained CNN corrector
result_low_res_cnn = time_integration(
    initial_state_low_res,
    config_low_res_cnn,
    params_low_res_cnn._replace(
        cnn_mhd_corrector_params = params_low_res_cnn.cnn_mhd_corrector_params._replace(
            network_params = trained_params
        )
    ),
    registered_variables
)

mse_trained = jnp.mean((states_high_res_downsampled - result_low_res_cnn.states) ** 2, axis=(1, 2, 3))

v_min = 0.0
v_max = 2.5

# Create figure and gridspec
fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
axs = []

# --- Top row: 3 density maps ---
for i in range(3):
    axs.append(fig.add_subplot(gs[0, i]))

extent = (0, config_low_res.box_size, 0, config_low_res.box_size)

# Initial images
im0 = axs[0].imshow(result_low_res.states[0, 0], extent=extent, origin='lower', cmap='viridis', animated=True, vmin=v_min, vmax=v_max)
axs[0].set_title("Low Res (No Correction)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04).set_label("Density")

im1 = axs[1].imshow(result_low_res_cnn.states[0, 0], extent=extent, origin='lower', cmap='viridis', animated=True, vmin=v_min, vmax=v_max)
axs[1].set_title("Low Res (With CNN Correction)")
axs[1].set_xlabel("x")
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04).set_label("Density")

im2 = axs[2].imshow(states_high_res_downsampled[0, 0], extent=extent, origin='lower', cmap='viridis', animated=True, vmin=v_min, vmax=v_max)
axs[2].set_title("High Res (Downsampled)")
axs[2].set_xlabel("x")
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04).set_label("Density")

# --- Bottom row: MSE over time ---
ax_mse = fig.add_subplot(gs[1, :])
time_array = snapshot_timepoints
mse_uncorrected = jnp.mean((states_high_res_downsampled - result_low_res.states) ** 2, axis=(1, 2, 3))
mse_corrected = mse_trained

ax_mse.plot(time_array, mse_uncorrected, label='No Correction')
ax_mse.plot(time_array, mse_corrected, label='CNN Corrected')

# Mark the training timepoints
for i, t in enumerate(snapshot_timepoints_train):
    label = "Training Timepoints" if i == 0 else None
    ax_mse.axvline(t, color='gray', linestyle='--', alpha=0.5, label=label)

time_line = ax_mse.axvline(time_array[0], color='red', linestyle='-', linewidth=2, label='Current Time')
ax_mse.set_xlabel("Time")
ax_mse.set_ylabel("MSE")
ax_mse.set_title("Mean Squared Error Over Time")
ax_mse.legend()

def update(frame):
    t = time_array[frame]

    # Update density maps
    im0.set_array(result_low_res.states[frame, 0])
    axs[0].set_title(f"Low Res (No Correction) (t = {t:.2f})")

    im1.set_array(result_low_res_cnn.states[frame, 0])
    axs[1].set_title(f"Low Res (With CNN Correction) (t = {t:.2f})")

    im2.set_array(states_high_res_downsampled[frame, 0])
    axs[2].set_title(f"High Res (Downsampled) (t = {t:.2f})")

    # Update red line (use a sequence!)
    time_line.set_xdata([t, t])

    return [im0, im1, im2, time_line]

plt.tight_layout()

# Create and save animation
ani = animation.FuncAnimation(
    fig, update, frames=len(time_array),
    interval=200, blit=True
)

save_path = os.path.join(output_dir, "animation.mp4")
ani.save(save_path, fps=10, dpi=200)

plt.close(fig)