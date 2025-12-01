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

# load the trained parameters if needed
with open("trained_params3.pkl", "rb") as f:
    neural_net_params = pickle.load(f)

# ===================================================
# =============== ↑ setup neural net ↑ ==============
# ===================================================


def cost_analyse_backprop(num_checkpoints):

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
        ),
        num_checkpoints = num_checkpoints,  # Number of checkpoints for backpropagation
    )

    helper_data = get_helper_data(config)

    # This is a template for the simulation parameters.
    # The network parameters within it will be updated during training.
    neural_net_force_params = NeuralNetForceParams(
            network_params = neural_net_params,
    )

    t_end = 2.0

    params = SimulationParams(
        dt_max = 0.01,
        t_end = t_end,
        C_cfl = 0.4,
        neural_net_force_params = neural_net_force_params
    )

    registered_variables = get_registered_variables(config)
    xm, ym = helper_data.geometric_centers[..., 0], helper_data.geometric_centers[..., 1]

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

    # run the simulation once to see how many timesteps it takes
    config_alt = config._replace(
        return_snapshots = True,  # Enable snapshots to see the evolution
        num_snapshots = 10
    )
    result = time_integration(
        initial_state,
        config_alt,
        params,
        registered_variables
    )
    print(f"Number of timesteps: {result.num_iterations}")

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

    @eqx.filter_jit
    def loss_fn(network_params_arrays):
        """Calculates the difference between the final state and the target."""
        final_state = force_pass(network_params_arrays)
        final_density = final_state[0, :, :]
        # Mean Squared Error loss
        loss = jnp.mean((final_density - target_density)**2)
        return loss

    @eqx.filter_jit
    def get_grads(network_params_arrays):
        """Performs one step of gradient descent."""
        vals, grads = eqx.filter_value_and_grad(loss_fn)(network_params_arrays)
        return vals, grads
    
    get_grads(neural_net_params)

    # time the gradient computation
    runtimes = []
    for _ in range(5):
        print("Timing gradient computation...")
        start = timer()
        vals, grad = get_grads(neural_net_params)
        vals.block_until_ready()  # Ensure the computation is complete
        end = timer()
        runtimes.append(end - start)
    # take mean runtime
    runtime = jnp.mean(jnp.array(runtimes))

    # get storage usage
    # print(jax.jit(f).lower(x).cost_analysis())
    storage = jax.jit(get_grads).lower(neural_net_params).compile().cost_analysis()['bytes accessed'] / (1024**2)  # in MB

    return storage, runtime

# storage, runtime = cost_analyse_backprop(100)
# print(f"Checkpoints: {100}, Storage: {storage:.2f} MB, Runtime: {runtime:.2f} s")

storages = []
runtimes = []

checkpoint_counts = jnp.arange(5, 200, 5)

for num_checkpoints in tqdm(checkpoint_counts, desc="Running simulations"):
    storage, runtime = cost_analyse_backprop(int(num_checkpoints))
    print(f"Checkpoints: {num_checkpoints}, Storage: {storage:.2f} MB, Runtime: {runtime:.2f} s")
    storages.append(storage)
    runtimes.append(runtime)

# save results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
storage_file = os.path.join(output_dir, "storages.npy")
runtime_file = os.path.join(output_dir, "runtimes.npy")
jnp.save(storage_file, jnp.array(storages))
jnp.save(runtime_file, jnp.array(runtimes))

# plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(checkpoint_counts, storages, marker='o')
ax[0].set_xlabel("Number of Checkpoints")
ax[0].set_ylabel("Storage (MB)")
ax[0].set_title("Storage vs. Number of Checkpoints")

ax[1].plot(checkpoint_counts, runtimes, marker='o', color='orange')
ax[1].set_xlabel("Number of Checkpoints")
ax[1].set_ylabel("Runtime (s)")
ax[1].set_title("Runtime vs. Number of Checkpoints")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "storage_runtime_analysis.svg"))