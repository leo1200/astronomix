"""Training a Solver in the loop model while using FINITE DIFFERENCE as the solver mode"""

from autocvd import autocvd

autocvd(num_gpus=1)


from jf1uids.data_classes.simulation_helper_data import HelperData

import os
from timeit import default_timer as timer
from typing import Tuple

import jax
from jax.experimental import checkify as jax_checkify
import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import PyTree, Array
from tqdm import tqdm
import math
from jf1uids import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    initialize_interface_fields,
)

from jf1uids._finite_difference._maths._differencing import finite_difference_int6
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    STATE_TYPE,
    BoundarySettings,
    BoundarySettings1D,
)
from jf1uids.time_stepping import time_integration
from jf1uids.variable_registry.registered_variables import RegisteredVariables
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import numpy as np


def get_initial_state_training(
    num_cells_high_res: int,
    downaverage_factor: int,
    snapshot_timepoints_train: jnp.ndarray,
) -> Tuple[
    Tuple[
        STATE_TYPE, SimulationConfig, SimulationParams, HelperData, RegisteredVariables
    ],
    Tuple[
        STATE_TYPE, SimulationConfig, SimulationParams, HelperData, RegisteredVariables
    ],
]:
    params = SimulationParams(
        C_cfl=1.5,
        t_end=float(snapshot_timepoints_train[-1]),
        snapshot_timepoints=snapshot_timepoints_train,
    )

    print("Setting periodic boundaries in all diretions.")
    config = SimulationConfig(
        num_cells=num_cells_high_res,
        solver_mode=FINITE_DIFFERENCE,
        differentiation_mode=BACKWARDS,
        mhd=True,
        dimensionality=3,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ),
        use_specific_snapshot_timepoints=True,
        return_snapshots=True,
        num_snapshots=len(snapshot_timepoints_train),
        num_checkpoints=100,
        progress_bar=True,
        runtime_debugging=False,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # setup the initial conditions

    r = helper_data.r
    r0 = 0.125
    r1 = 1.1 * r0

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r) * 1.0
    P = jnp.where(r <= r0, 100.0, P)
    P = jnp.where((r > r0) & (r <= r1), 1.0 + 99.0 * (r1 - r) / (r1 - r0), P)
    P = jnp.where(r > r1, 1.0, P)

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    B0 = 10

    B_x = B0 / jnp.sqrt(2)
    B_y = B0 / jnp.sqrt(2)
    B_z = 0.0

    B_x = jnp.ones_like(r) * B_x
    B_y = jnp.ones_like(r) * B_y
    B_z = jnp.ones_like(r) * B_z

    bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=V_x,
        velocity_y=V_y,
        velocity_z=V_z,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
        gas_pressure=P,
    )

    config = finalize_config(config, initial_state.shape)
    initial_state_low_res = downaverage(
        state=initial_state, downaverage_factor=downaverage_factor
    )
    config_low_res = config._replace(num_cells=config.num_cells // downaverage_factor)
    helper_data_low_res = get_helper_data(config_low_res)
    return (
        (initial_state, config, params, helper_data, registered_variables),
        (
            initial_state_low_res,
            config_low_res._replace(progress_bar=False),
            params,
            helper_data_low_res,
            registered_variables,
        ),
    )


def downaverage(state: jnp.ndarray, downaverage_factor: int) -> jnp.ndarray:
    """Downaverage spatial (and depth) dimensions by non-overlapping block averaging.

    This function accepts either:
      - unbatched input of shape (NUM_VARS, H, W, D)
      - batched input of shape (N, NUM_VARS, H, W, D)

    The downaverage_factor is an integer factor by which each spatial/depth
    dimension (H, W, D) is reduced:
        h_out = H // downaverage_factor
        w_out = W // downaverage_factor
        d_out = D // downaverage_factor

    Args:
        state: JAX ndarray with shape (NUM_VARS, H, W, D) or (N, NUM_VARS, H, W, D).
        downaverage_factor: integer factor > 0 that divides H, W and D.

    Returns:
        downaveraged array with shape:
            - (NUM_VARS, h_out, w_out, d_out) for unbatched input
            - (N, NUM_VARS, h_out, w_out, d_out) for batched input

    Raises:
        ValueError: if input ndim is not 4 or 5, or if spatial/depth dims are not divisible
                    by downaverage_factor.

    """
    downaverage_factor = int(downaverage_factor)
    if downaverage_factor <= 0:
        raise ValueError("downaverage_factor must be a positive integer")

    if state.ndim == 4:
        # (NUM_VARS, H, W, D)
        num_vars, H, W, D = state.shape
        if (
            (H % downaverage_factor) != 0
            or (W % downaverage_factor) != 0
            or (D % downaverage_factor) != 0
        ):
            raise ValueError(
                f"Spatial/depth dims {(H, W, D)} must be divisible by downaverage_factor={downaverage_factor}"
            )
        h_out = H // downaverage_factor
        w_out = W // downaverage_factor
        d_out = D // downaverage_factor

        # reshape into blocks and mean over block axes
        reshaped = state.reshape(
            num_vars,
            h_out,
            downaverage_factor,
            w_out,
            downaverage_factor,
            d_out,
            downaverage_factor,
        )
        # mean over the block axes (2, 4, 6)
        downaveraged = reshaped.mean(axis=(2, 4, 6))
        return downaveraged

    elif state.ndim == 5:
        # (N, NUM_VARS, H, W, D)
        N, num_vars, H, W, D = state.shape
        if (
            (H % downaverage_factor) != 0
            or (W % downaverage_factor) != 0
            or (D % downaverage_factor) != 0
        ):
            raise ValueError(
                f"Spatial/depth dims {(H, W, D)} must be divisible by downaverage_factor={downaverage_factor}"
            )
        h_out = H // downaverage_factor
        w_out = W // downaverage_factor
        d_out = D // downaverage_factor

        reshaped = state.reshape(
            N,
            num_vars,
            h_out,
            downaverage_factor,
            w_out,
            downaverage_factor,
            d_out,
            downaverage_factor,
        )
        # mean over the block axes (3, 5, 7)
        downaveraged = reshaped.mean(axis=(3, 5, 7))
        return downaveraged

    else:
        raise ValueError(
            f"Unsupported input ndim {state.ndim}. Expected 4 (NUM_VARS,H,W,D) or "
            f"5 (N,NUM_VARS,H,W,D)."
        )


def perturb_state(
    key: jax.random.PRNGKey, state: jnp.ndarray, noise_level: float = 0.01
) -> jnp.ndarray:
    """
    Adds noise to Hydro variables only.
    Idx map: 0:rho, 1-3:vel, 4:P, 5-10:Mag (cell+face)
    """
    # Create a mask: 1 for Hydro, 0 for Mag
    # Shape: (11, 1, 1, 1) so it broadcasts over H, W, D
    # Indices: 0 (rho), 1,2,3 (v), 4 (P) -> receive noise
    # Indices: 5,6,7 (B_cen), 8,9,10 (B_face) -> NO noise
    mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = mask[:, None, None, None]

    # Generate random noise
    noise = jax.random.normal(key, shape=state.shape) * noise_level * mask

    # Apply noise
    perturbed_state = state + noise

    # Update Density (Index 0)
    rho_safe = jnp.maximum(perturbed_state[0], 1e-4)
    perturbed_state = perturbed_state.at[0].set(rho_safe)

    # Update Pressure (Index 4)
    p_safe = jnp.maximum(perturbed_state[4], 1e-4)
    perturbed_state = perturbed_state.at[4].set(p_safe)

    return perturbed_state


def training_model(
    lr_scheduler,
    num_cells_high_res: int,
    downaverage_factor: int,
    snapshot_timepoints_train: jnp.ndarray,
    start_correction_time: float,
    epochs_per_time: list[int],
    hidden_channels: int,
    noise_level: float = 0.0,
    model_initiaization_scale: float = 0.1,
    compute_target: bool = False,
) -> Tuple[PyTree, PyTree]:
    simulation_bundle_high_res, simulation_bundle_low_res = get_initial_state_training(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        snapshot_timepoints_train=snapshot_timepoints_train,
    )
    if compute_target:
        result_high_res = time_integration(*simulation_bundle_high_res)
        states_high_res_downsampled = downaverage(
            result_high_res.states, downaverage_factor=downaverage_factor
        )
    else:
        states_high_res_downsampled = jnp.zeros(
            (
                11,
                num_cells_high_res // downaverage_factor,
                num_cells_high_res // downaverage_factor,
                num_cells_high_res // downaverage_factor,
            )
        )
        noise_level = 0.0

    (
        initial_state_low_res,
        config_low_res,
        params,
        helper_data_low_res,
        registered_variables,
    ) = simulation_bundle_low_res
    model = CorrectorCNN(
        in_channels=registered_variables.num_vars,
        hidden_channels=hidden_channels,
        hidden_layers=1,
        key=jax.random.PRNGKey(100),
        scale=model_initiaization_scale,
    )
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    # nan_path = "arena/data/cnn_mhd_corrector_params_NAN.pkl"
    # if os.path.exists(nan_path):
    #     with open(nan_path, "rb") as f:
    #         try:
    #             loaded_params = pickle.load(f)
    #             neural_net_params = loaded_params
    #             print(f"Loaded params from {nan_path}")
    #         except Exception as e:
    #             print(
    #                 f"Failed to load params from {nan_path}: {e}; continuing with init params"
    #             )
    # else:
    #     print("No saved NAN params found, using initialized params.")
    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        correct_from_beggining=False,
        start_correction_time=start_correction_time,
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    params_low_res = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    # Set up the optimizer using optax

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_scheduler),
    )

    opt_state = optimizer.init(neural_net_params)

    print("Starting training with optax...")
    losses = []

    # This variable will hold the trained parameters and be updated in the loop
    trained_params = neural_net_params

    # Timing
    start_time = timer()

    # The main training loop
    # pbar = tqdm(range(epochs))
    best_loss = float("inf")
    best_params = trained_params

    key = jax.random.PRNGKey(112)

    @eqx.filter_jit
    def train_step(
        network_params_arrays: PyTree,
        opt_state: optax.OptState,
        target_state_arg: Array,
        initial_state_arg: Array,
        config_arg: SimulationConfig,
        params_arg: SimulationParams,
        key_arg: Array,
        helper_data_arg: HelperData,
        registered_variables_arg: RegisteredVariables,
    ):
        noisy_initial_state = perturb_state(
            key_arg, initial_state_arg, noise_level=noise_level
        )

        def loss_fn(network_params_arrays):
            """Calculates the difference between the final state and the target."""
            results_low_res = time_integration(
                noisy_initial_state,
                config_arg,
                params_arg._replace(
                    cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                        network_params=network_params_arrays
                    )
                ),
                helper_data_arg,
                registered_variables_arg,
            )
            jax.debug.print(
                "final time {time}", time=jnp.sum(results_low_res.time_points)
            )

            bxb = results_low_res.states[
                registered_variables.interface_magnetic_field_index.x
            ]
            byb = results_low_res.states[
                registered_variables.interface_magnetic_field_index.y
            ]
            bzb = results_low_res.states[
                registered_variables.interface_magnetic_field_index.z
            ]
            divergence = jnp.abs(
                1.0
                / current_config.grid_spacing
                * (
                    finite_difference_int6(bxb, axis=0)
                    + finite_difference_int6(byb, axis=1)
                    + finite_difference_int6(bzb, axis=2)
                )
            )
            jax.debug.print("divergence: {div}", div=jnp.sum(divergence))
            # Calculate the L2 loss between the final state and the target state
            loss = jnp.mean((results_low_res.states - target_state_arg) ** 2)
            return loss

        """Performs one step of gradient descent."""
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(network_params_arrays)
        gradients_modulus = jnp.sqrt(
            sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
        ).astype(float)
        updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
        network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
        return network_params_arrays, opt_state, loss_value, gradients_modulus

    for i, epochs in enumerate(epochs_per_time):
        # Update outer loop variables
        current_end_time = snapshot_timepoints_train[i]

        # Update the config/params objects for this specific timeframe
        current_config = config_low_res._replace(num_snapshots=1)
        current_params_sim = params_low_res._replace(
            t_end=current_end_time,
            snapshot_timepoints=jnp.array([current_end_time]),
        )

        current_target = states_high_res_downsampled[i]

        # Reset key for this epoch block if needed, or keep evolving it
        key = jax.random.PRNGKey(16 + i)

        for step in range(epochs):
            key, subkey = jax.random.split(key)

            # 3. CALL THE FUNCTION
            # Notice we pass EVERYTHING that defines the physics here
            start_time_epoch = timer()
            trained_params, opt_state, loss, gradients_mod = train_step(
                trained_params,
                opt_state,
                current_target,
                initial_state_low_res,
                current_config,
                current_params_sim,
                subkey,
                helper_data_low_res,
                registered_variables,
            )
            if math.isnan(gradients_mod):
                print("Found nan in gradients")
                output_dir = "arena/data"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, "cnn_mhd_corrector_params_NAN_2.pkl"
                )
                with open(output_path, "wb") as f:
                    pickle.dump(trained_params, f)
                return neural_net_params, neural_net_static
                continue
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_params = trained_params

            print(
                f"End Time: {current_end_time:.3f} | Step {step + 1}/{epochs} | "
                f"Loss: {loss:.6f} | Time: {(timer() - start_time_epoch):.3f} | "
                f"Grads: {gradients_mod:.3f}"
            )
    # After training, use the best parameters found
    trained_params = best_params

    end_time = timer()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    output_dir = "arena/data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cnn_mhd_corrector_params.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(trained_params, f)
    jnp.savez(os.path.join(output_dir, "lossesX.npz"), losses=losses)

    return neural_net_params, neural_net_static


def plot_training(
    neural_net_params: PyTree,
    neural_net_static: PyTree,
    times_eval: jnp.ndarray,
    num_cells_high_res: int,
    downaverage_factor: int,
    snapshot_timepoints_train: float,
    start_correction_time: float,
    epochs_per_time: list[int],
):
    snapshot_timepoints_idx = []
    for t in snapshot_timepoints_train:
        if t not in times_eval:
            times_eval = jnp.sort(jnp.concatenate([times_eval, jnp.array([t])]))
        snapshot_timepoints_idx.append(int(jnp.argmax(times_eval == t)))

    num_cells_lr = num_cells_high_res // downaverage_factor

    simulation_bundle_high_res, simulation_bundle_low_res = get_initial_state_training(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        snapshot_timepoints_train=times_eval,
    )
    result_high_res = time_integration(*simulation_bundle_high_res)
    states_target_low_res = downaverage(
        result_high_res.states, downaverage_factor=downaverage_factor
    )
    (
        initial_state_low_res,
        config_low_res,
        params,
        helper_data_low_res,
        registered_variables,
    ) = simulation_bundle_low_res

    states_low_res_uncorrected = time_integration(*simulation_bundle_low_res).states

    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        start_correction_time=start_correction_time,
        correct_from_beggining=False,
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    params_low_res = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    states_low_res = time_integration(
        initial_state_low_res,
        config_low_res,
        params_low_res,
        helper_data_low_res,
        registered_variables,
    ).states

    final_state_target_low_res = states_target_low_res[snapshot_timepoints_idx[-1]]
    final_state_low_res_uncorrected = states_low_res_uncorrected[
        snapshot_timepoints_idx[-1]
    ]
    final_state_low_res = states_low_res[snapshot_timepoints_idx[-1]]

    losses_data = np.load("arena/data/lossesX.npz")
    losses = losses_data["losses"]

    # --- Create figure and layout ---
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 3, height_ratios=[1, 1, 1, 1])

    # First row: density images
    axs_density = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axs_magnetic = [fig.add_subplot(gs[1, i]) for i in range(3)]

    titles_density = [
        "Target State (Density)",
        "Final State before Training (Density)",
        "Final State after Training (Density)",
    ]
    titles_magnetic = [
        "Target State (|B|^2 along diagonal)",
        "Final State before Training (|B|^2 along diagonal)",
        "Final State after Training (|B|^2 along diagonal)",
    ]

    states = [
        final_state_target_low_res,
        final_state_low_res_uncorrected,
        final_state_low_res,
    ]

    l2_error_initial = jnp.mean(
        (final_state_low_res_uncorrected - final_state_target_low_res) ** 2
    )
    l2_errors_corrected = jnp.mean(
        (states_low_res - states_target_low_res) ** 2,
        axis=tuple(range(1, states_low_res.ndim)),
    )
    l2_errors_uncorrected = jnp.mean(
        (states_low_res_uncorrected - states_target_low_res) ** 2,
        axis=tuple(range(1, states_low_res.ndim)),
    )

    # Shared color scale
    vmin = min(jnp.min(s[registered_variables.density_index]) for s in states)
    vmax = max(jnp.max(s[registered_variables.density_index]) for s in states)

    for ax_density, ax_magnetic, state, title_density, title_magnetic in zip(
        axs_density, axs_magnetic, states, titles_density, titles_magnetic, strict=True
    ):
        im = ax_density.imshow(
            state[registered_variables.density_index, :, :, 32],
            extent=(0, config_low_res.box_size, 0, config_low_res.box_size),
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax_density.set_title(title_density)
        ax_density.set_aspect("equal", adjustable="box")
        divider = make_axes_locatable(ax_density)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label="Density")

        # Second row: magnetic field across the diagonal
        diag_indices = jnp.arange(0, num_cells_lr)
        b_squared = (
            state[registered_variables.magnetic_index.x] ** 2
            + state[registered_variables.magnetic_index.y] ** 2
            + state[registered_variables.magnetic_index.z] ** 2
        )
        B_diag = b_squared[diag_indices, diag_indices, num_cells_lr // 2]
        r_diag = jnp.sqrt((diag_indices) ** 2 + (diag_indices) ** 2) * (
            config_low_res.box_size / num_cells_lr
        )
        ax_magnetic.plot(r_diag, B_diag)
        ax_magnetic.set_ylabel("|B|^2")
        ax_magnetic.set_xlabel("diagonal")
        ax_magnetic.set_title(title_magnetic)

    # Third row: loss curve
    ax_loss = fig.add_subplot(gs[2, :])
    ax_loss.plot(losses, label="Training Loss")
    ax_loss.axhline(
        y=l2_error_initial,
        color="r",
        linestyle="--",
        label="Initial L2 Error (uncorrected)",
    )
    ax_loss.set_xlabel("Training Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss During Training")
    ax_loss.legend()

    # Fourth row: L2 error over time
    ax_errors = fig.add_subplot(gs[3, :])
    ax_errors.plot(
        times_eval, l2_errors_corrected, label="Corrected Integration", color="tab:blue"
    )
    ax_errors.plot(
        times_eval,
        l2_errors_uncorrected,
        label="Uncorrected Integration",
        color="tab:orange",
        linestyle="--",
    )

    for t, epochs in zip(snapshot_timepoints_train, epochs_per_time, strict=False):
        ax_errors.axvline(
            x=t,
            color="gray",
            linestyle=":",
            label=f"Training time / # {epochs}",
        )

    ax_errors.set_xlabel("Time")
    ax_errors.set_ylabel("L2 Error")
    ax_errors.set_yscale("log")
    ax_errors.set_title("Mean Squared Error Over Time")
    ax_errors.legend()

    plt.tight_layout()
    plt.savefig("arena/results/fd_mhd_optimization.png", dpi=400)


def training_model_checkify(
    lr_scheduler,
    num_cells_high_res: int,
    downaverage_factor: int,
    snapshot_timepoints_train: jnp.ndarray,
    start_correction_time: float,
    epochs_per_time: list[int],
    hidden_channels: int,
    noise_level: float,
    model_initialization_scale: float,
    compute_target: bool = True,
) -> Tuple[PyTree, PyTree]:
    # --- Initialization (Unchanged) ---
    simulation_bundle_high_res, simulation_bundle_low_res = get_initial_state_training(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        snapshot_timepoints_train=snapshot_timepoints_train,
    )
    if compute_target:
        result_high_res = time_integration(*simulation_bundle_high_res)
        states_high_res_downsampled = downaverage(
            result_high_res.states, downaverage_factor=downaverage_factor
        )
    else:
        states_high_res_downsampled = jnp.zeros(
            (
                11,
                num_cells_high_res // downaverage_factor,
                num_cells_high_res // downaverage_factor,
                num_cells_high_res // downaverage_factor,
            )
        )
        noise_level = 0.0
    (
        initial_state_low_res,
        config_low_res,
        params,
        helper_data_low_res,
        registered_variables,
    ) = simulation_bundle_low_res

    model = CorrectorCNN(
        in_channels=registered_variables.num_vars,
        hidden_channels=hidden_channels,
        hidden_layers=1,
        key=jax.random.PRNGKey(100),
        scale=model_initialization_scale,
    )
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    nan_path = "arena/data/cnn_mhd_corrector_params_NAN.pkl"
    if os.path.exists(nan_path):
        with open(nan_path, "rb") as f:
            try:
                loaded_params = pickle.load(f)
                neural_net_params = loaded_params
                print(f"Loaded params from {nan_path}")
            except Exception as e:
                print(
                    f"Failed to load params from {nan_path}: {e}; continuing with init params"
                )
    else:
        print("No saved NAN params found, using initialized params.")
    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        correct_from_beggining=False,
        start_correction_time=start_correction_time,
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config,
    )
    params_low_res = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_scheduler),
    )

    opt_state = optimizer.init(neural_net_params)

    print("Starting training with optax and checkify...")
    losses = []
    trained_params = neural_net_params
    start_time = timer()
    best_loss = float("inf")
    best_params = trained_params
    key = jax.random.PRNGKey(112)

    # --- Checkify Transformation ---

    def train_step_impl(
        network_params_arrays: PyTree,
        opt_state: optax.OptState,
        target_state_arg: Array,
        initial_state_arg: Array,
        config_arg,
        params_arg,
        key_arg: Array,
        helper_data_arg,
        registered_variables_arg,
    ):
        noisy_initial_state = perturb_state(
            key_arg, initial_state_arg, noise_level=noise_level
        )

        def loss_fn(network_params_arrays):
            """Calculates the difference between the final state and the target."""
            results_low_res = time_integration(
                noisy_initial_state,
                config_arg,
                params_arg._replace(
                    cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                        network_params=network_params_arrays
                    )
                ),
                helper_data_arg,
                registered_variables_arg,
            )
            jax.debug.print("Forward succesful {_}", _=jnp.mean(results_low_res.states))
            # Calculate the L2 loss
            loss = jnp.mean((results_low_res.states - target_state_arg) ** 2)

            # CHECKIFY: Explicit check on loss
            jax_checkify.check(jnp.isfinite(loss), "Loss became NaN or Inf!")

            return loss

        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(network_params_arrays)

        # CHECKIFY: Explicit check on gradients
        grad_norm = jnp.sqrt(
            sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
        )
        jax_checkify.check(jnp.isfinite(grad_norm), "Gradients became NaN or Inf!")

        updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
        network_params_arrays = eqx.apply_updates(network_params_arrays, updates)

        return network_params_arrays, opt_state, loss_value, grad_norm

    # Apply checkify transformation with float_checks enabled (catches /0, etc.)
    # filter_jit must wrap the checkified function, not be inside it
    errors = (
        jax_checkify.user_checks
        | jax_checkify.float_checks
        | jax_checkify.nan_checks
        | jax_checkify.div_checks
    )

    checkified_train_step = jax_checkify.checkify(train_step_impl, errors=errors)

    @eqx.filter_jit
    def train_step_jit(*args):
        return checkified_train_step(*args)

    # --- Training Loop ---

    for i, epochs in enumerate(epochs_per_time):
        current_end_time = snapshot_timepoints_train[i]
        current_config = config_low_res._replace(num_snapshots=1)
        current_params_sim = params_low_res._replace(
            t_end=current_end_time,
            snapshot_timepoints=jnp.array([current_end_time]),
        )
        current_target = states_high_res_downsampled[i]
        key = jax.random.PRNGKey(16 + i)

        for step in range(epochs):
            key, subkey = jax.random.split(key)
            start_time_epoch = timer()

            # 1. Call the checkified function
            # It returns (err, actual_outputs)
            err, (trained_params_next, opt_state_next, loss, gradients_mod) = (
                train_step_jit(
                    trained_params,
                    opt_state,
                    current_target,
                    initial_state_low_res,
                    current_config,
                    current_params_sim,
                    subkey,
                    helper_data_low_res,
                    registered_variables,
                )
            )

            if not err.get():
                # 🟢 SUCCESS CASE
                trained_params = trained_params_next
                opt_state = opt_state_next

                losses.append(loss)
                if loss < best_loss and i == len(epochs_per_time) - 1:
                    best_loss = loss
                    best_params = trained_params

                print(
                    f"End Time: {current_end_time:.3f} | Step {step + 1}/{epochs} | "
                    f"Loss: {loss:.6f} | Time: {(timer() - start_time_epoch):.3f} | "
                    f"Grads: {gradients_mod:.3f}"
                )
            else:
                err.throw()
                raise RuntimeError(
                    "Training aborted due to NaN/Inf detected by checkify."
                )

        else:
            continue

        break

    end_time = timer()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    output_dir = "arena/data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cnn_mhd_corrector_params.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(best_params, f)
    jnp.savez(os.path.join(output_dir, "lossesX.npz"), losses=losses)

    return best_params, neural_net_static


def finite_difference_blast_test1(training: bool = True, checkify: bool = True):
    epochs_per_time = [100]
    snapshot_timepoints_train = jnp.array([0.2])
    assert len(epochs_per_time) == len(snapshot_timepoints_train), (
        "len of epochs per time and times to train at"
    )
    total_epochs = np.sum(np.array(epochs_per_time))
    learning_rate = 1e-4
    peak_lr = 1e-3
    end_lr = 5e-6
    warmup_steps_fraction = 0.4
    warmup_steps = int(total_epochs * warmup_steps_fraction)
    decay_steps = total_epochs - warmup_steps
    downaverage_factor = 2
    start_correction_time = 0.05
    num_cells_high_res = 128
    hidden_channels = 16
    noise_level = 0.03
    model_initiaization_scale = 0.005

    # --- STYLIZED PRINT BLOCK ---
    print("=" * 60)
    print(f"{' CONFIGURATION ':^60}")
    print("=" * 60)
    print(f"{'--- Scheduler ---':^60}")
    print(f"Epochs per Time     : {epochs_per_time}")
    print(f"Snapshot Times      : {snapshot_timepoints_train}")
    print(f"Total Epochs        : {total_epochs}")
    print(f"Warmup / Decay      : {warmup_steps} / {decay_steps} steps")
    print(f"Learning Rate Sched : {learning_rate:.1e} → {peak_lr:.1e} → {end_lr:.1e}")
    print(f"\n{'--- Simulation Parameters ---':^60}")
    print(f"Resolution (High)   : {num_cells_high_res} cells")
    print(f"Downaverage Factor  : {downaverage_factor}x")
    print(f"Hidden Channels     : {hidden_channels}")
    print(f"Noise Level         : {noise_level}")
    print(f"Start Correction t  : {start_correction_time}")
    print(f"Model init scale    : {model_initiaization_scale}")
    print("=" * 60)
    # ----------------------------
    learning_rate_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate,
        peak_value=peak_lr,
        end_value=end_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )

    if training:
        if checkify is False:
            neural_net_params, neural_net_static = training_model(
                lr_scheduler=learning_rate_scheduler,
                num_cells_high_res=num_cells_high_res,
                downaverage_factor=downaverage_factor,
                snapshot_timepoints_train=snapshot_timepoints_train,
                start_correction_time=start_correction_time,
                epochs_per_time=epochs_per_time,
                hidden_channels=hidden_channels,
                noise_level=noise_level,
                model_initiaization_scale=model_initiaization_scale,
                compute_target=True,
            )
        else:
            neural_net_params, neural_net_static = training_model_checkify(
                lr_scheduler=learning_rate_scheduler,
                num_cells_high_res=num_cells_high_res,
                downaverage_factor=downaverage_factor,
                snapshot_timepoints_train=snapshot_timepoints_train,
                start_correction_time=start_correction_time,
                epochs_per_time=epochs_per_time,
                hidden_channels=hidden_channels,
                noise_level=noise_level,
                model_initialization_scale=model_initiaization_scale,
                compute_target=False,
            )
    else:
        model = CorrectorCNN(
            in_channels=11,
            hidden_channels=hidden_channels,
            key=jax.random.PRNGKey(100),
        )
        neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

        with open("arena/data/cnn_mhd_corrector_params.pkl", "rb") as f:
            neural_net_params = pickle.load(f)

    plot_training(
        neural_net_params=neural_net_params,
        neural_net_static=neural_net_static,
        times_eval=jnp.linspace(0.0, 0.3, 35, endpoint=True),
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        start_correction_time=start_correction_time,
        snapshot_timepoints_train=snapshot_timepoints_train,
        epochs_per_time=epochs_per_time,
    )


if __name__ == "__main__":
    finite_difference_blast_test1(training=True, checkify=False)
