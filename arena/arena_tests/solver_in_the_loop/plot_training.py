if __name__ == "__main__":
    from autocvd import autocvd

    autocvd(num_gpus=1)

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

from astronomix.option_classes.simulation_params import SimulationParams
from arena.arena_tests.solver_in_the_loop import model_manager
from astronomix.data_classes.simulation_snapshot_data import SnapshotData
import jax.numpy as jnp
from jax import vmap
from jaxtyping import PyTree
from typing import List
from astronomix.variable_registry.registered_variables import StaticIntVector
import jax

from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from astronomix.time_stepping import time_integration
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
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from astronomix import get_registered_variables
from astronomix import SimulationConfig

from arena.arena_tests.solver_in_the_loop.model_manager import (
    ModelManager,
    load_legacy_params_into_current,
)
import os
import equinox as eqx

from astronomix.variable_registry import registered_variables
import logging
from arena.arena_tests.solver_in_the_loop.timepoint_updater import (
    FRONT_TO_BACK,
    BACK_TO_FRONT,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    YAXIS,
    XAXIS,
    ZAXIS,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
)

from functools import partial
import warnings
from astronomix._finite_difference._maths._differencing import finite_difference_int6
from astronomix._finite_difference._maths._interpolate import interp_face_to_center
from arena.arena_tests.solver_in_the_loop.plot_states_comparison import plot_states
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_training(
    neural_net_params: PyTree,
    neural_net_static: PyTree,
    times_eval: jnp.ndarray,
    num_cells_high_res: int,
    downaverage_factor: int,
    start_correction_time: float,
    model_name: str,
    correct_from_beggining: bool,
    cfl_target: float = 1.5,
    cfl: float = 1.5,
    limiter: int = 0,
    old_version: bool = False,
):
    """
    Args:
        times_eval: times at which to evaluate the loss
        snapshot_timepoints_train: times used for training the model
    """

    model_manager = ModelManager(model_name=model_name)
    training_config = model_manager.load_training_config()

    if training_config.direction == FRONT_TO_BACK:
        timepoints_train = [
            training_config.t_end - t for t in training_config.snapshot_timepoints_train
        ]
    else:
        timepoints_train = training_config.snapshot_timepoints_train

    snapshot_timepoints_idx = []
    # Populate the times_eval with the trained times
    for t in timepoints_train:
        if t not in times_eval:
            times_eval = jnp.sort(jnp.concatenate([times_eval, jnp.array([t])]))

    # Get the index of the trained times
    for t in timepoints_train:
        snapshot_timepoints_idx.append(int(jnp.argmax(times_eval == t)))

    epochs_total = np.zeros(len(training_config.epochs_per_time))
    for i, epochs in enumerate(training_config.epochs_per_time):
        epochs_total[i] = epochs_total[i - 1] + epochs

    num_cells_lr = num_cells_high_res // downaverage_factor

    simulation_bundle_high_res, simulation_bundle_low_res = get_initial_state_training(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        snapshot_timepoints_train=times_eval,
        c_cfl=cfl_target,
        limiter=limiter,
        old_version=old_version,
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
        config=config_low_res._replace(progress_bar=True),
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
        config_low_res._replace(progress_bar=True),
        params_low_res,
        helper_data_low_res,
        registered_variables,
    ).states

    final_state_target_low_res = states_target_low_res[snapshot_timepoints_idx[-1]]
    final_state_low_res_uncorrected = states_low_res_uncorrected[
        snapshot_timepoints_idx[-1]
    ]
    final_state_low_res = states_low_res[snapshot_timepoints_idx[-1]]

    losses_data = np.load(f"arena/data/models/single_problem/{model_name}/losses.npz")
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
    for t, epochs in zip(timepoints_train, epochs_total, strict=False):
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

    for t, epochs in zip(timepoints_train, epochs_total, strict=False):
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
    plt.savefig(Path(model_manager.base_dir) / model_name / "summary.png", dpi=400)


def plot_states_histogram(
    states: List[float],
    states_max: List[float],
    times: List[float],
    model_name: str,
):
    # corrections_array = np.array(corrections)
    states_array = np.array(states)
    states_max_array = np.array(states_max)

    times_total = np.zeros(len(times))
    for i, time in enumerate(times):
        times_total[i] = times_total[i - 1] + time

    reg_vars = get_registered_variables(
        SimulationConfig(
            num_cells=32, dimensionality=3, mhd=True, solver_mode=FINITE_DIFFERENCE
        )
    )
    assert isinstance(reg_vars.velocity_index, StaticIntVector)
    assert isinstance(reg_vars.magnetic_index, StaticIntVector)
    assert isinstance(reg_vars.interface_magnetic_field_index, StaticIntVector)
    velocity_states = np.sqrt(
        states_array[:, reg_vars.velocity_index[0]] ** 2
        + states_array[:, reg_vars.velocity_index[1]] ** 2
        + states_array[:, reg_vars.velocity_index[2]] ** 2
    )
    magnetic_states = np.sqrt(
        states_array[:, reg_vars.magnetic_index[0]] ** 2
        + states_array[:, reg_vars.magnetic_index[1]] ** 2
        + states_array[:, reg_vars.magnetic_index[2]] ** 2
    )
    interface_magnetic_states = np.sqrt(
        states_array[:, reg_vars.interface_magnetic_field_index[0]] ** 2
        + states_array[:, reg_vars.interface_magnetic_field_index[1]] ** 2
        + states_array[:, reg_vars.interface_magnetic_field_index[2]] ** 2
    )
    velocity_states_max = np.sqrt(
        states_max_array[:, reg_vars.velocity_index[0]] ** 2
        + states_max_array[:, reg_vars.velocity_index[1]] ** 2
        + states_max_array[:, reg_vars.velocity_index[2]] ** 2
    )
    magnetic_states_max = np.sqrt(
        states_max_array[:, reg_vars.magnetic_index[0]] ** 2
        + states_max_array[:, reg_vars.magnetic_index[1]] ** 2
        + states_max_array[:, reg_vars.magnetic_index[2]] ** 2
    )
    interface_magnetic_states_max = np.sqrt(
        states_max_array[:, reg_vars.interface_magnetic_field_index[0]] ** 2
        + states_max_array[:, reg_vars.interface_magnetic_field_index[1]] ** 2
        + states_max_array[:, reg_vars.interface_magnetic_field_index[2]] ** 2
    )
    histogram_states = [
        states_array[:, reg_vars.density_index],
        states_array[:, reg_vars.pressure_index],
        velocity_states,
        magnetic_states,
        interface_magnetic_states,
    ]
    histogram_states_max = [
        states_max_array[:, reg_vars.density_index],
        states_max_array[:, reg_vars.pressure_index],
        velocity_states_max,
        magnetic_states_max,
        interface_magnetic_states_max,
    ]
    titles = ["Density", "Pressure", "Velocity", "Magnetic", "Interface_magnetic"]

    fig = plt.figure(figsize=(20, 12))
    num_channels = len(histogram_states)
    bins = 50
    gs = GridSpec(3, num_channels, height_ratios=[1, 1, 1])

    axs_histograms_avg = [fig.add_subplot(gs[0, i]) for i in range(num_channels)]
    axs_histograms_max = [fig.add_subplot(gs[1, i]) for i in range(num_channels)]

    axs_time_evolution = fig.add_subplot(gs[2, :])

    for i, (state, state_max, title) in enumerate(
        zip(histogram_states, histogram_states_max, titles, strict=True)
    ):
        axs_histograms_avg[i].hist(state, bins=bins)
        axs_histograms_avg[i].set_title(f"Average {title}")

        axs_histograms_max[i].hist(state_max, bins=bins)
        axs_histograms_max[i].set_title(f"Max {title}")

        axs_time_evolution.plot(times_total, state, label=title)

    axs_time_evolution.set_title("Average states evolution")
    axs_time_evolution.set_ylabel("Average")
    axs_time_evolution.set_xlabel("Time")
    axs_time_evolution.legend()

    plt.savefig(f"arena/data/models/single_problem/{model_name}/plots/states_magnitude.png", dpi=400)
    pass


def model_output_figures(
    corrections: List[float],
    effective_corrections: List[float],
    states: List[float],
    times: List[float],
    model_name: str,
):
    reg_vars = get_registered_variables(
        SimulationConfig(
            num_cells=32, dimensionality=3, mhd=True, solver_mode=FINITE_DIFFERENCE
        )
    )
    corrections_array = np.array(corrections)
    e_corrections_array = np.array(effective_corrections)
    states_array = np.array(states)
    times_total = np.zeros(len(times))
    # times_delta = np.array(times)
    for i, time in enumerate(times):
        times_total[i] = times_total[i - 1] + time

    assert isinstance(reg_vars.velocity_index, StaticIntVector)
    assert isinstance(reg_vars.magnetic_index, StaticIntVector)
    assert isinstance(reg_vars.interface_magnetic_field_index, StaticIntVector)
    velocity_states = np.sqrt(
        states_array[:, reg_vars.velocity_index[0]] ** 2
        + states_array[:, reg_vars.velocity_index[1]] ** 2
        + states_array[:, reg_vars.velocity_index[2]] ** 2
    )
    velocity_corr = np.sqrt(
        corrections_array[:, reg_vars.velocity_index[0]] ** 2
        + corrections_array[:, reg_vars.velocity_index[1]] ** 2
        + corrections_array[:, reg_vars.velocity_index[2]] ** 2
    )
    velocity_e_corr = np.sqrt(
        e_corrections_array[:, reg_vars.velocity_index[0]] ** 2
        + e_corrections_array[:, reg_vars.velocity_index[1]] ** 2
        + e_corrections_array[:, reg_vars.velocity_index[2]] ** 2
    )

    velocity_e_corr_norm = velocity_e_corr / velocity_states

    velocity_corr_norm = velocity_corr / velocity_states

    magnetic_states = np.sqrt(
        states_array[:, reg_vars.magnetic_index[0]] ** 2
        + states_array[:, reg_vars.magnetic_index[1]] ** 2
        + states_array[:, reg_vars.magnetic_index[2]] ** 2
    )
    magnetic_corr = np.sqrt(
        corrections_array[:, reg_vars.magnetic_index[0]] ** 2
        + corrections_array[:, reg_vars.magnetic_index[1]] ** 2
        + corrections_array[:, reg_vars.magnetic_index[2]] ** 2
    )

    interface_magnetic_states = np.sqrt(
        states_array[:, reg_vars.interface_magnetic_field_index[0]] ** 2
        + states_array[:, reg_vars.interface_magnetic_field_index[1]] ** 2
        + states_array[:, reg_vars.interface_magnetic_field_index[2]] ** 2
    )
    magnetic_e_corr = np.sqrt(
        e_corrections_array[:, reg_vars.interface_magnetic_field_index[0]] ** 2
        + e_corrections_array[:, reg_vars.interface_magnetic_field_index[1]] ** 2
        + e_corrections_array[:, reg_vars.interface_magnetic_field_index[2]] ** 2
    )

    magnetic_e_corr_norm = magnetic_e_corr / interface_magnetic_states
    magnetic_corr_norm = magnetic_corr / magnetic_states

    # Histogram
    # Density
    density_states = states_array[:, reg_vars.density_index]
    density_corr = corrections_array[:, reg_vars.density_index]
    density_e_corr = e_corrections_array[:, reg_vars.density_index]
    density_corr_norm = density_corr / density_states
    density_e_corr_norm = density_e_corr / density_states
    # Pressure
    pressure_states = states_array[:, reg_vars.pressure_index]
    pressure_corr = corrections_array[:, reg_vars.pressure_index]
    pressure_e_corr = e_corrections_array[:, reg_vars.pressure_index]
    pressure_corr_norm = pressure_corr / pressure_states
    pressure_e_corr_norm = pressure_e_corr / pressure_states

    pressures = [
        pressure_corr,
        pressure_corr_norm,
        pressure_e_corr,
        pressure_e_corr_norm,
    ]
    densities = [
        density_corr,
        density_corr_norm,
        density_e_corr,
        density_e_corr_norm,
    ]
    magnetics = [
        magnetic_corr,
        magnetic_corr_norm,
        magnetic_e_corr,
        magnetic_e_corr_norm,
    ]
    velocitys = [
        velocity_corr,
        velocity_corr_norm,
        velocity_e_corr,
        velocity_e_corr_norm,
    ]

    titles = [
        "unnormalized",
        "normalized",
        "effective_unnormalized",
        "effective_normalized",
    ]

    ylabels = [
        "Correction",
        "Correction/state",
        "Correction",
        "Correction/state",
    ]

    for pressure, density, magnetic, velocity, title, ylabel in zip(
        pressures, densities, magnetics, velocitys, titles, ylabels, strict=True
    ):
        plot_corrections_figure(
            density_states=density_states,
            pressure_states=pressure_states,
            velocity_states_mag=velocity_states,
            magnetic_states_mag=interface_magnetic_states,
            density_corr=density,
            pressure_corr=pressure,
            velocity_corr_mag=velocity,
            magnetic_corr_mag=magnetic,
            times_total=times_total,
            bins=100,
            title_suffix=title,
            ylabel=ylabel,
            model_name=model_name,
        )


def plot_corrections_figure(
    *,
    density_states,
    pressure_states,
    velocity_states_mag,
    magnetic_states_mag,
    density_corr,
    pressure_corr,
    velocity_corr_mag,
    magnetic_corr_mag,
    times_total,
    bins,
    title_suffix,
    ylabel,
    model_name,
):
    fig, axes = plt.subplots(3, 1, figsize=(9, 15))
    ax_hist_states, ax_hist_corr, ax_time = axes

    # -----------------------
    # Histogram
    # -----------------------
    ax_hist_states.hist(
        density_states, bins=bins, density=False, alpha=0.6, label="Density"
    )
    ax_hist_states.hist(
        pressure_states, bins=bins, density=False, alpha=0.6, label="Pressure"
    )
    ax_hist_states.hist(
        velocity_states_mag, bins=bins, density=False, alpha=0.6, label="Velocity"
    )
    ax_hist_states.hist(
        magnetic_states_mag,
        bins=bins,
        density=False,
        alpha=0.6,
        label="Magnetic_interface",
    )

    ax_hist_states.set_title(f"Distribution of states average ({title_suffix})")
    ax_hist_states.set_xlabel("Correction value")
    ax_hist_states.set_ylabel("Probability density")
    ax_hist_states.legend()
    ax_hist_states.grid(alpha=0.3)

    ax_hist_corr.hist(
        density_corr, bins=bins, density=False, alpha=0.6, label="Density"
    )
    ax_hist_corr.hist(
        pressure_corr, bins=bins, density=False, alpha=0.6, label="Pressure"
    )
    ax_hist_corr.hist(
        velocity_corr_mag, bins=bins, density=False, alpha=0.6, label="Velocity"
    )
    ax_hist_corr.hist(
        magnetic_corr_mag,
        bins=bins,
        density=False,
        alpha=0.6,
        label="Magnetic_interface",
    )

    ax_hist_corr.set_title(f"Distribution of corrections average ({title_suffix})")
    ax_hist_corr.set_xlabel("Correction value")
    ax_hist_corr.set_ylabel("Probability density")
    ax_hist_corr.legend()
    ax_hist_corr.grid(alpha=0.3)

    # -----------------------
    # Time evolution
    # -----------------------
    ax_time.plot(times_total, density_corr, label="Density")
    ax_time.plot(times_total, pressure_corr, label="Pressure")

    ax_time.plot(times_total, velocity_corr_mag, label="Velocity")
    ax_time.plot(times_total, magnetic_corr_mag, label="Magnetic interface")

    ax_time.set_title(f"Model output ({title_suffix})")
    ax_time.set_xlabel("Times")
    ax_time.set_ylabel(ylabel)
    ax_time.legend()

    plt.tight_layout()
    plt.savefig(
        f"arena/data/models/single_problem/{model_name}/plots/model_output_{title_suffix}.png", dpi=400
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_name = "optuna_params"
    load_model = True
    load_model_nan = False
    legacy_mode = False

    if legacy_mode:
        warnings.warn(
            "Using the legacy version of the code (same grid_spacing)",
            DeprecationWarning,
            stacklevel=1,
        )
    assert os.path.exists(f"arena/data/models/single_problem/{model_name}"), (
        "Model folder doesnt exist"
    )

    registered_variables = get_registered_variables(
        SimulationConfig(
            dimensionality=3,
            mhd=True,
            solver_mode=FINITE_DIFFERENCE,
        )
    )
    params = SimulationParams()
    model_manager = ModelManager(model_name=model_name)
    model_manager.print_model_info()
    training_config = model_manager.load_training_config()

    corrections = []
    states = []
    times = []
    corrections_max = []
    states_max = []
    effective_corrections = []

    # TODO: change the corrected state calculation to the proper one used in the model
    def snapshot_callable(time, state, correction):
        corrections.append(jnp.mean(correction, axis=[1, 2, 3]))
        states.append(jnp.mean(state, axis=[1, 2, 3]))
        corrections_max.append(jnp.max(correction, axis=[1, 2, 3]))
        states_max.append(jnp.max(state, axis=[1, 2, 3]))
        times.append(time)

        original_state = state
        corrected_state = state

        # update the primitive corrected_state with the correction
        corrected_state = corrected_state.at[:5].add(correction[:5] * time)
        corrected_state = corrected_state.at[-3:].add(correction[-3:] * time)

        Bx_center = interp_face_to_center(corrected_state[-3], XAXIS)
        By_center = interp_face_to_center(corrected_state[-2], YAXIS)
        Bz_center = interp_face_to_center(corrected_state[-1], ZAXIS)

        corrected_state = corrected_state.at[registered_variables.magnetic_index.x].set(
            Bx_center
        )
        corrected_state = corrected_state.at[registered_variables.magnetic_index.y].set(
            By_center
        )
        corrected_state = corrected_state.at[registered_variables.magnetic_index.z].set(
            Bz_center
        )

        corrected_state = corrected_state.at[registered_variables.pressure_index].set(
            jnp.maximum(
                corrected_state[registered_variables.pressure_index],
                params.minimum_pressure,
            )
        )
        corrected_state = corrected_state.at[registered_variables.density_index].set(
            jnp.maximum(
                corrected_state[registered_variables.density_index],
                params.minimum_density,
            )
        )

        effective_corrections.append(
            jnp.mean(corrected_state - original_state, axis=[1, 2, 3])
        )

        pass

    # Initialize model
    model = CorrectorCNN(
        in_channels=registered_variables.num_vars,
        hidden_channels=training_config.hidden_channels,
        hidden_layers=training_config.hidden_layers,
        key=jax.random.PRNGKey(100),
        scale=training_config.model_initialization_scale,
        snapshot_callable=snapshot_callable,
    )

    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

    if legacy_mode:
        neural_net_params = load_legacy_params_into_current(
            model, path=f"arena/data/models/single_problem/{model_name}/model_params.pkl"
        )
        model_manager.save_model_params(params=neural_net_params)
        logger.info("Loaded model with the legacy pickle and saved it with eqx")
    else:
        neural_net_params = model_manager.load_model_params(like=neural_net_params)

    # neural_net_params = model_loader(
    #     model_manager,
    #     neural_net_params,
    #     load_model=load_model,
    #     load_model_nan=load_model_nan,
    # )

    model = eqx.combine(neural_net_params, neural_net_static)

    plot_training(
        neural_net_params=neural_net_params,
        neural_net_static=neural_net_static,
        times_eval=jnp.linspace(0.0, 0.3, 30),
        num_cells_high_res=64,
        downaverage_factor=2,
        start_correction_time=training_config.start_correction_time,
        correct_from_beggining=training_config.correct_from_beggining,
        model_name=training_config.model_name,
        cfl=training_config.c_cfl,
        cfl_target=training_config.c_cfl_target,
        limiter=training_config.limiter,
        old_version=False,
    )

    print(f"Simulation took {len(times)} steps")

    plot_states_histogram(
        states=states,
        states_max=states_max,
        times=times,
        model_name=training_config.model_name,
    )

    model_output_figures(
        corrections=corrections,
        effective_corrections=effective_corrections,
        states=states,
        times=times,
        model_name=model_name,
    )
