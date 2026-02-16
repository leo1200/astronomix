import jax.numpy as jnp
import jax
from jaxtyping import Array
from numpy import array
from jf1uids.data_classes.simulation_helper_data import HelperData
import os
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData
from jf1uids.time_stepping import time_integration
from typing import Optional, Tuple

from jf1uids import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    initialize_interface_fields,
)

from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    STATE_TYPE,
    VAN_ALBADA,
    BoundarySettings,
    BoundarySettings1D,
)
from jf1uids.variable_registry.registered_variables import RegisteredVariables

from arena.arena_tests.solver_in_the_loop.timepoint_updater import (
    BACK_TO_FRONT,
    FRONT_TO_BACK,
)

from arena.arena_tests.solver_in_the_loop.plot_states_comparison import plot_states

import logging

logger = logging.getLogger(__name__)


def get_initial_state_training(
    num_cells_high_res: int,
    downaverage_factor: int,
    snapshot_timepoints_train: jnp.ndarray,
    t_end: Optional[float] = None,
    direction: int = BACK_TO_FRONT,
    c_cfl: float = 1.5,
    limiter: int = 0,
    old_version: bool = False,  # NOTE: REALLY BAD PRACTICE
) -> Tuple[
    Tuple[
        STATE_TYPE, SimulationConfig, SimulationParams, HelperData, RegisteredVariables
    ],
    Tuple[
        STATE_TYPE, SimulationConfig, SimulationParams, HelperData, RegisteredVariables
    ],
]:
    if direction == BACK_TO_FRONT:
        snapshot_timepoints = jnp.sort(snapshot_timepoints_train)
        if t_end is None:
            print(
                f"t_end not given using as t_end the last t {snapshot_timepoints_train[-1]}"
            )
            t_end = float(snapshot_timepoints_train[-1])
        else:
            if not jnp.any(snapshot_timepoints == t_end):
                snapshot_timepoints = jnp.append(snapshot_timepoints, t_end)
    elif direction == FRONT_TO_BACK:
        assert isinstance(t_end, float)
        snapshot_timepoints = jnp.array([t_end])
    else:
        raise ValueError("The direction given doesnt exist")

    logger.debug(f"snapshot timepoints {snapshot_timepoints}")

    params = SimulationParams(
        C_cfl=c_cfl,
        t_end=t_end,
        snapshot_timepoints=snapshot_timepoints,
    )

    print("Setting periodic boundaries in all directions.")
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
        num_snapshots=len(snapshot_timepoints),
        num_checkpoints=100,
        progress_bar=True,
        runtime_debugging=False,
        limiter=limiter,  # 0=MINMOD
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
    if not old_version:
        config_low_res = finalize_config(config_low_res, initial_state_low_res.shape)
    logger.info(
        f"Grid spacing high res {config.grid_spacing}, low res {config_low_res.grid_spacing}"
    )
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


def downaverage(state: jnp.ndarray, downaverage_factor: int) -> Array:
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


def perturb_state(key: Array, state: jnp.ndarray, noise_level: float = 0.01):
    mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[
        :, None, None, None
    ]
    noise = jax.random.normal(key, shape=state.shape) * noise_level * mask
    perturbed_state = state + noise
    perturbed_state = perturbed_state.at[0].set(jnp.maximum(perturbed_state[0], 0.0))
    perturbed_state = perturbed_state.at[4].set(jnp.maximum(perturbed_state[4], 0.0))
    return perturbed_state


def initialize_training_data(
    snapshot_timepoints_train: jnp.ndarray,
    t_end: float,
    direction: int,
    num_cells_high_res: int,
    downaverage_factor: int,
    start_correction_time: float,
    correct_from_beggining: bool = False,
    c_cfl: float = 1.5,
    limiter: int = 0,
) -> Tuple[
    Array,
    Tuple[Array, SimulationConfig, SimulationParams, HelperData, RegisteredVariables],
]:
    "Loads the target data (or if not computes it) and returns the target data with the low res bundle"
    if direction == BACK_TO_FRONT:
        filename = (
            "hr_states_"
            + "_".join([f"{int(t * 100)}" for t in snapshot_timepoints_train])
            + f"_{num_cells_high_res}"
            + f"_{int(c_cfl * 100)}"
            + f"_{limiter}"
        )
    elif direction == FRONT_TO_BACK:
        filename = (
            "hr_states_"
            + f"_{int(t_end * 100)}"
            + f"_{num_cells_high_res}"
            + f"_{int(c_cfl * 100)}"
            + f"_{limiter}"
        )
    else:
        raise ValueError("Direction given nonexistent")

    filepath = f"arena/data/{filename}.npy"
    simulation_bundle_high_res, simulation_bundle_low_res = get_initial_state_training(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        direction=direction,
        t_end=t_end,
        snapshot_timepoints_train=snapshot_timepoints_train,
        c_cfl=c_cfl,
        limiter=limiter,
    )
    if not os.path.exists(filepath):
        result_high_res = time_integration(*simulation_bundle_high_res)
        assert isinstance(result_high_res, SnapshotData)
        states_high_res_downsampled = downaverage(
            result_high_res.states, downaverage_factor
        )
        jnp.save(filepath, states_high_res_downsampled)
        print(f"Saved states to {filepath}")
    else:
        states_high_res_downsampled = jnp.load(filepath)
        print(f"Loaded from {filepath}")

    # NOTE: we do this to save some computational time by starting at the start correction time state
    if not correct_from_beggining:
        print(f"Preparing initial state at t {start_correction_time}")
        (
            initial_state_low_res,
            config_low_res,
            params,
            helper_data_low_res,
            registered_variables,
        ) = simulation_bundle_low_res

        initial_state_low_res = time_integration(
            primitive_state=initial_state_low_res,
            config=config_low_res._replace(
                return_snapshots=False,
                progress_bar=True,
                exact_end_time=True,
            ),
            params=params._replace(t_end=start_correction_time),
            registered_variables=registered_variables,
            helper_data=helper_data_low_res,
        )
    else:
        print("Using the model from the beggining of the simulation")

    assert isinstance(simulation_bundle_low_res[0], Array)
    # plot_states(
    #     [simulation_bundle_low_res[0], simulation_bundle_high_res[0]],
    #     [16, 32],
    #     fig_name="initial_states",
    #     model_name="optuna_params",
    #     titles=["lr", "hr"],
    # )
    return states_high_res_downsampled, simulation_bundle_low_res
