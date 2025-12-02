"""Training a Solver in the loop model while using FINITE DIFFERENCE as the solver mode"""

from autocvd import autocvd

autocvd(num_gpus=1)

import os
from timeit import default_timer as timer
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree
from tqdm import tqdm

from jf1uids import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    initialize_interface_fields,
)
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
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    STATE_TYPE,
    BoundarySettings,
    BoundarySettings1D,
)
from jf1uids.time_stepping import time_integration
from jf1uids.variable_registry.registered_variables import RegisteredVariables


def get_initial_state_training(
    num_cells_high_res: int,
    downaverage_factor: int,
    t_end: float,
    snapshot_timepoints_train: jnp.ndarray,
    scale: float,
) -> Tuple[
    Tuple[STATE_TYPE, SimulationConfig, SimulationParams, RegisteredVariables],
    Tuple[STATE_TYPE, SimulationConfig, SimulationParams, RegisteredVariables],
]:
    params = SimulationParams(
        C_cfl=1.5, t_end=t_end, snapshot_timepoints=snapshot_timepoints_train
    )

    print("Setting periodic boundaries in all directions.")
    config = SimulationConfig(
        progress_bar=True,
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
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # setup the initial conditions

    r = helper_data.r
    r0 = 0.125 * scale
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
            config_low_res,
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


def training_model(
    epochs: int,
    num_cells_high_res: int,
    downaverage_factor: int,
    t_end: float,
    snapshot_timepoints_train: jnp.array,
) -> Tuple[PyTree, PyTree]:
    scales = [0.9, 1.1]
    training_states = []
    for scale in scales:
        simulation_bundle_high_res, simulation_bundle_low_res = (
            get_initial_state_training(
                num_cells_high_res=num_cells_high_res,
                downaverage_factor=downaverage_factor,
                t_end=t_end,
                snapshot_timepoints_train=snapshot_timepoints_train,
                scale=scale,
            )
        )
        result_high_res = time_integration(*simulation_bundle_high_res)
        states_high_res_downaverage = downaverage(
            result_high_res.states, downaverage_factor=downaverage_factor
        )
        (
            initial_state_low_res,
            config_low_res,
            params,
            helper_data_low_res,
            registered_variables,
        ) = simulation_bundle_low_res

        training_states.append(
            {
                "scale": scale,
                "hr_da_states": states_high_res_downaverage,
                "lr_initial_state": initial_state_low_res,
            }
        )

    config_low_res = config_low_res._replace(progress_bar=False)
    model = CorrectorCNN(
        in_channels=registered_variables.num_vars,
        hidden_channels=16,
        key=jax.random.PRNGKey(100),
    )
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True, network_static=neural_net_static
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    params_low_res = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    # Set up the optimizer using optax
    learning_rate = 1e-3
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(neural_net_params)

    @eqx.filter_jit
    def train_step(network_params_arrays, opt_state):
        """Performs one step of gradient descent."""
        accumulated_grads = jax.tree_util.tree_map(
            jnp.zeros_like, network_params_arrays
        )
        losses = jnp.zeros(len(training_states))
        for training_state in training_states:

            @eqx.filter_jit
            def loss_fn(network_params_arrays):
                """Calculates the difference between the final state and the target."""
                results_low_res = time_integration(
                    primitive_state=training_state["lr_initial_state"],
                    config=config_low_res,
                    params=params_low_res._replace(
                        cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                            network_params=network_params_arrays
                        )
                    ),
                    helper_data=helper_data_low_res,
                    registered_variables=registered_variables,
                )
                # Calculate the L2 loss between the final state and the target state
                loss = jnp.mean(
                    (results_low_res.states - training_state["hr_da_states"]) ** 2
                )
                return loss

            loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
                network_params_arrays
            )
            losses.append(loss_value)
            accumulated_grads = jax.tree_util.tree_map(
                lambda acc, g: acc + g, accumulated_grads, grads
            )
        updates, opt_state = optimizer.update(
            accumulated_grads, opt_state, network_params_arrays
        )
        network_params_arrays = eqx.apply_updates(network_params_arrays, updates)

        return network_params_arrays, opt_state, jnp.mean(jnp.stack(losses))

    print("Starting training with optax...")
    losses = []

    # This variable will hold the trained parameters and be updated in the loop
    trained_params = neural_net_params

    # Timing
    start_time = timer()

    # The main training loop
    pbar = tqdm(range(epochs))
    best_loss = float("inf")
    best_params = trained_params
    for step in pbar:
        trained_params, opt_state, loss = train_step(trained_params, opt_state)
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = trained_params
        pbar.set_description(f"Step {step + 1}/{epochs} | Loss: {loss:.2e}")

    # After training, use the best parameters found
    trained_params = best_params

    end_time = timer()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    input("Press Enter to continue...")

    # # # save the trained parameters using pickle
    import pickle

    output_dir = "arena/data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "adapative_cnn_mhd_corrector_params.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(trained_params, f)

    return neural_net_params, neural_net_static


def finite_difference_blast_test1(training: bool = True):
    downaverage_factor = 2
    if training:
        neural_net_params, neural_net_static = training_model(
            epochs=100,
            num_cells_high_res=128,
            downaverage_factor=downaverage_factor,
            t_end=0.2,
            snapshot_timepoints_train=jnp.array([0.2]),
        )


if __name__ == "__main__":
    finite_difference_blast_test1(True)
