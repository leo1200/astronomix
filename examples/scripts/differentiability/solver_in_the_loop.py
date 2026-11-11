"""
Differentiability: solver-in-the-loop learned corrector for a 2D MHD blast.

We train a small CNN corrector wired into the astronomix solver so that a
low-resolution MHD blast simulation matches a downsampled high-resolution
reference. The corrector is applied every step inside the differentiable time
integration and its weights are optimized end-to-end with reverse-mode autodiff
and Adam, minimizing the MSE between the corrected low-res run and the target.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import os

# jax
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# neural networks and optimization
import equinox as eqx
import optax

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import BACKWARDS

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
    time_integration,
    get_helper_data,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
)

# CNN corrector and its per-step update (network_updates.py next to this script)
from network_updates import CorrectorCNN, cnn_mhd_corrector_update

# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure a differentiable 2D MHD blast (reverse-mode)
num_cells_high_res = int(os.environ.get("NUM_CELLS_HIGH_RES", 128))
num_cells_low_res = int(os.environ.get("NUM_CELLS_LOW_RES", 32))
snapshot_timepoints = jnp.array([0.05, 0.1, 0.15, 0.2])

baseline_config = SimulationConfig(
    progress_bar = False,
    mhd = True,
    dimensionality = 2,
    box_size = 1.0,
    num_cells = num_cells_high_res,
    differentiation_mode = BACKWARDS,
    exact_end_time = True,
    return_snapshots = True,
    use_specific_snapshot_timepoints = True,
    num_snapshots = len(snapshot_timepoints),
    # the training target and corrected run both need the full per-snapshot
    # states (opt-in since it is the biggest snapshot allocation)
    snapshot_settings = SnapshotSettings(return_states = True),
)

registered_variables = get_registered_variables(baseline_config)
params = SimulationParams(t_end = 0.2, C_cfl = 0.1, snapshot_timepoints = snapshot_timepoints)


def get_blast_setup(num_cells):
    """Build the magnetized blast initial primitive state on an ``num_cells`` grid."""
    dummy_config = baseline_config._replace(num_cells = num_cells)
    helper_data = get_helper_data(dummy_config)

    rho = jnp.ones((num_cells, num_cells))
    P = jnp.ones((num_cells, num_cells)) * 0.1
    r_inj = 0.1 * dummy_config.box_size
    P = jnp.where(helper_data.r**2 < r_inj**2, 10.0, P)

    V_x = jnp.zeros((num_cells, num_cells))
    V_y = jnp.zeros((num_cells, num_cells))

    B_0 = 1 / jnp.sqrt(2)
    B_x = B_0 * jnp.ones((num_cells, num_cells))
    B_y = B_0 * jnp.ones((num_cells, num_cells))
    B_z = jnp.zeros((num_cells, num_cells))

    initial_state = construct_primitive_state(
        config = dummy_config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = V_x,
        velocity_y = V_y,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        gas_pressure = P,
    )
    return initial_state


def downaverage_state(state, target_shape):
    """Block-average a (num_vars, H, W) state down to (num_vars, h, w)."""
    num_vars, h_in, w_in = state.shape
    h_out, w_out = target_shape
    reshaped = state.reshape(num_vars, h_out, h_in // h_out, w_out, w_in // w_out)
    return reshaped.mean(axis=(2, 4))


# high-resolution reference, downsampled to the low-res grid as the training target
initial_state_high_res = get_blast_setup(num_cells_high_res)
config_high_res = finalize_config(baseline_config, initial_state_high_res.shape)
result_high_res = time_integration(initial_state_high_res, config_high_res, params, registered_variables)
states_target = jax.vmap(downaverage_state, in_axes=(0, None))(
    result_high_res.states, (num_cells_low_res, num_cells_low_res)
)

# low-resolution run: initial state obtained by downsampling the high-res initial state
initial_state_low_res = downaverage_state(initial_state_high_res, (num_cells_low_res, num_cells_low_res))
config_low_res = finalize_config(
    baseline_config._replace(num_cells = num_cells_low_res), initial_state_low_res.shape
)
result_low_res = time_integration(initial_state_low_res, config_low_res, params, registered_variables)

# build the CNN corrector and split it into trainable params and static architecture
model = CorrectorCNN(
    in_channels = registered_variables.num_vars,
    hidden_channels = 16,
    key = jax.random.PRNGKey(42),
)
neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

# the corrector enters the solver through the per-step update hook; the
# trainable weights travel in the simulation parameters
config_low_res_cnn = config_low_res._replace(
    per_step_update = cnn_mhd_corrector_update(neural_net_static)
)
params_low_res_cnn = params._replace(per_step_update_params = neural_net_params)


@eqx.filter_jit
def loss_fn(network_params_arrays):
    """MSE between the corrected low-res run and the downsampled reference."""
    results_low_res = time_integration(
        initial_state_low_res,
        config_low_res_cnn,
        params_low_res_cnn._replace(per_step_update_params = network_params_arrays),
        registered_variables,
    )
    return jnp.mean((results_low_res.states - states_target) ** 2)


optimizer = optax.adam(1e-3)
opt_state = optimizer.init(neural_net_params)


@eqx.filter_jit
def train_step(network_params_arrays, opt_state):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(network_params_arrays)
    updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
    network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
    return network_params_arrays, opt_state, loss_value


# train the corrector end-to-end through the solver
trained_params = neural_net_params
losses = []
for step in range(int(os.environ.get("NUM_TRAINING_STEPS", 40))):
    trained_params, opt_state, loss = train_step(trained_params, opt_state)
    losses.append(float(loss))
    print(f"step {step:3d}: loss = {loss:.4e}")

# run the low-res simulation with the trained corrector for the comparison plot
result_low_res_cnn = time_integration(
    initial_state_low_res,
    config_low_res_cnn,
    params_low_res_cnn._replace(per_step_update_params = trained_params),
    registered_variables,
)

# plot final-snapshot density: uncorrected low-res, corrected low-res, reference
density_index = registered_variables.density_index
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].plot(losses)
axs[0].set_yscale("log")
axs[0].set_title("training loss")
for ax, data, title in zip(
    axs[1:],
    [
        result_low_res.states[-1, density_index],
        result_low_res_cnn.states[-1, density_index],
        states_target[-1, density_index],
    ],
    ["low res (no correction)", "low res (CNN corrected)", "high res (downsampled)"],
):
    ax.imshow(data.T, origin="lower", cmap="viridis")
    ax.set_title(title)
fig.savefig(figures_dir / "solver_in_the_loop.png", dpi=200, bbox_inches="tight")
