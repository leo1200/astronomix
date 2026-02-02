from autocvd import autocvd
import os

# autocvd(num_gpus=1)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

from arena.arena_tests.solver_in_the_loop import model_manager
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData
import jax.numpy as jnp
from jax import vmap
from jaxtyping import PyTree
import jax

from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from jf1uids.time_stepping import time_integration
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from arena.arena_tests.solver_in_the_loop.utils import (
    downaverage,
    get_initial_state_training,
    perturb_state,
)
from arena.arena_tests.solver_in_the_loop.loss import (
    loss_setup,
)
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from jf1uids import get_registered_variables
from jf1uids import SimulationConfig

from arena.arena_tests.solver_in_the_loop.model_manager import (
    ModelManager,
    model_loader,
)
import os
import equinox as eqx

from jf1uids.variable_registry import registered_variables
import logging

from jf1uids.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
)

from functools import partial

logger = logging.Logger(__name__)


def plot_training(
    neural_net_params: PyTree,
    neural_net_static: PyTree,
    times_eval: jnp.ndarray,
    num_cells_high_res: int,
    downaverage_factor: int,
    snapshot_timepoints_train: list[float],
    start_correction_time: float,
    epochs_per_time: list[int],
    model_name: str,
    correct_from_beggining: bool,
    cfl_target: float = 1.5,
    cfl: float = 1.5,
    limiter: int = 0,
):
    """
    Args:
        times_eval: times at which to evaluate the loss
        snapshot_timepoints_train: times used for training the model
    """

    model_manager = ModelManager(model_name=model_name)
    training_config = model_manager.load_training_config()

    snapshot_timepoints_idx = []
    # Populate the times_eval with the trained times
    for t in snapshot_timepoints_train:
        if t not in times_eval:
            times_eval = jnp.sort(jnp.concatenate([times_eval, jnp.array([t])]))

    # Get the index of the trained times
    for t in snapshot_timepoints_train:
        snapshot_timepoints_idx.append(int(jnp.argmax(times_eval == t)))

    num_cells_lr = num_cells_high_res // downaverage_factor

    simulation_bundle_high_res, simulation_bundle_low_res = get_initial_state_training(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        snapshot_timepoints_train=times_eval,
        c_cfl=cfl_target,
        limiter=limiter,
    )
    result_high_res = time_integration(*simulation_bundle_high_res)
    assert isinstance(result_high_res, SnapshotData)
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

    # NOTE: not sure if i should perturb the state (probably not)
    #
    # initial_state_low_res = perturb_state(
    #     key=jax.random.PRNGKey(100),
    #     state=initial_state_low_res,
    #     noise_level=training_config.noise_level,
    # )

    params_low_res = params._replace(C_cfl=cfl)
    states_low_res_uncorrected = time_integration(
        primitive_state=initial_state_low_res,
        config=config_low_res,
        params=params_low_res,
        helper_data=helper_data_low_res,
        registered_variables=registered_variables,
    ).states

    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        start_correction_time=start_correction_time,
        correct_from_beggining=correct_from_beggining,
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    params_low_res = params_low_res._replace(
        cnn_mhd_corrector_params=cnn_mhd_corrector_params
    )

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

    losses_data = np.load(f"arena/data/models/{model_name}/losses.npz")
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

    loss_fn_kwargs, loss_fn_factory = loss_setup(
        training_config=training_config,
        target_states=states_target_low_res[jnp.array(snapshot_timepoints_idx)],
    )
    loss_fn = partial(loss_fn_factory, **loss_fn_kwargs)

    l2_error_initial = float(
        loss_fn(final_state_low_res_uncorrected, final_state_target_low_res)
    )

    v_loss = vmap(loss_fn, in_axes=(0, 0))

    l2_errors_corrected = v_loss(states_low_res, states_target_low_res)
    l2_errors_uncorrected = v_loss(states_low_res_uncorrected, states_target_low_res)
    #
    # l2_error_initial = jnp.mean(
    #     (final_state_low_res_uncorrected - final_state_target_low_res) ** 2
    # )
    # l2_errors_corrected = jnp.mean(
    #     (states_low_res - states_target_low_res) ** 2,
    #     axis=tuple(range(1, states_low_res.ndim)),
    # )
    # l2_errors_uncorrected = jnp.mean(
    #     (states_low_res_uncorrected - states_target_low_res) ** 2,
    #     axis=tuple(range(1, states_low_res.ndim)),
    # )

    # Shared color scale
    vmin = float(
        min(
            jnp.min(s[registered_variables.density_index]).astype(float) for s in states
        )
    )
    vmax = float(
        max(
            jnp.max(s[registered_variables.density_index]).astype(float) for s in states
        )
    )

    for ax_density, ax_magnetic, state, title_density, title_magnetic in zip(
        axs_density, axs_magnetic, states, titles_density, titles_magnetic, strict=True
    ):
        im = ax_density.imshow(
            state[registered_variables.density_index, :, :, num_cells_lr // 2],
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
    for t, epochs in zip(snapshot_timepoints_train, epochs_per_time, strict=False):
        ax_loss.axvline(
            x=epochs,
            color="gray",
            linestyle=":",
            label=f"Training time {t}/ # {epochs}",
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
    plt.savefig(f"arena/data/models/{model_name}/plots/summary.png", dpi=400)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_name = "optuna_params"
    load_model = True
    load_model_nan = False

    assert os.path.exists(f"arena/data/models/{model_name}"), (
        "Model folder doesnt exist"
    )

    registered_variables = get_registered_variables(
        SimulationConfig(
            dimensionality=3,
            mhd=True,
            solver_mode=FINITE_DIFFERENCE,
        )
    )
    model_manager = ModelManager(model_name=model_name)
    model_manager.print_model_info()
    training_config = model_manager.load_training_config()
    sim_training_config = model_manager.load_simulation_config()

    # Initialize model
    model = CorrectorCNN(
        in_channels=registered_variables.num_vars,
        hidden_channels=training_config.hidden_channels,
        hidden_layers=training_config.hidden_layers,
        key=jax.random.PRNGKey(100),
        scale=training_config.model_initialization_scale,
    )
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    neural_net_params = model_loader(
        model_manager,
        neural_net_params,
        load_model=load_model,
        load_model_nan=load_model_nan,
    )

    plot_training(
        neural_net_params=neural_net_params,
        neural_net_static=neural_net_static,
        times_eval=jnp.linspace(0.0, 0.3, 30),
        num_cells_high_res=64,
        downaverage_factor=2,
        snapshot_timepoints_train=training_config.snapshot_timepoints_train,
        start_correction_time=sim_training_config.start_correction_time,
        epochs_per_time=training_config.epochs_per_time,
        correct_from_beggining=sim_training_config.correct_from_beggining,
        model_name=training_config.model_name,
        cfl=sim_training_config.c_cfl,
        cfl_target=sim_training_config.c_cfl_target,
        limiter=sim_training_config.limiter,
    )
