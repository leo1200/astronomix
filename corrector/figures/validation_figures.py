from typing import List, Optional
from corrector_src.utils.downaverage import downaverage
import jf1uids
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData
from matplotlib import animation
import matplotlib.pyplot as plt
from jf1uids.fluid_equations.registered_variables import StaticIntVector
from jf1uids.fluid_equations.total_quantities import (
    calculate_internal_energy,
    calculate_kinetic_energy,
    calculate_total_energy,
)
import jax
import jax.numpy as jnp
import numpy as np
import os
from corrector_src.utils.power_spectra_1d import pk_jax_1d
from corrector_src.loss.sgs_turb_loss import get_energy, make_loss_function
from corrector_src.data.dataset import SimulationBundle
from jf1uids.option_classes.simulation_config import SimulationConfig
from omegaconf import DictConfig, OmegaConf
from jf1uids._physics_modules._mhd._vector_maths import divergence3D


def energy_conservation_plots(
    config: DictConfig,
    hr_snapshot: SnapshotData,
    lr_snapshot: SnapshotData,
    lr_sol_snapshot: SnapshotData,
    sim_bundle_hr: SimulationBundle,
    sim_bundle_lr: SimulationBundle,
    loss_dict: Optional[dict] = None,
    folder: str = "corrector/figures",
    model_name: str = "fno",
    figure_name: str = "energy_conservation",
):
    # state, helper_data, gamma, config, registered_variables
    v_internal_energy = jax.vmap(
        calculate_internal_energy, in_axes=(0, None, None, None, None)
    )
    v_kinetic_energy = jax.vmap(calculate_kinetic_energy, in_axes=(0, None, None, None))
    v_total_energy = jax.vmap(
        calculate_total_energy, in_axes=(0, None, None, None, None, None)
    )
    energies = []
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for snapshot_data, sim_bundle, name, color in zip(
        [hr_snapshot, lr_snapshot, lr_sol_snapshot],
        [sim_bundle_hr, sim_bundle_lr, sim_bundle_lr],
        [
            f"{str(config.data.hr_res)}",
            f"{str(config.data.hr_res // config.data.downscaling_factor)}",
            f"{str(config.data.hr_res // config.data.downscaling_factor)} corrected",
        ],
        colors,
    ):
        internal_energies = v_internal_energy(
            snapshot_data.states,
            sim_bundle.helper,
            sim_bundle.params.gamma,
            sim_bundle.config,
            sim_bundle.reg_vars,
        )
        kinetic_energies = v_kinetic_energy(
            snapshot_data.states,
            sim_bundle.helper,
            sim_bundle.config,
            sim_bundle.reg_vars,
        )
        total_energies = v_total_energy(
            snapshot_data.states,
            sim_bundle.helper,
            sim_bundle.params.gamma,
            sim_bundle.params.gravitational_constant,
            sim_bundle.config,
            sim_bundle.reg_vars,
        )

        energies.append(
            {
                "name": name,
                "color": color,
                "internal_e": internal_energies,
                "kinetic_e": kinetic_energies,
                "total_e": total_energies,
                "times": snapshot_data.time_points,
            }
        )
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for energy in energies:
        c = energy["color"]

        # Internal energy
        ax[0].plot(
            energy["times"],
            energy["internal_e"],
            label=f"{energy['name']}",
            color=c,
        )

        # Kinetic energy
        ax[1].plot(
            energy["times"],
            energy["kinetic_e"],
            label=f"{energy['name']}",
            color=c,
        )

        # Total energy
        ax[2].plot(
            energy["times"],
            energy["total_e"],
            label=f"{energy['name']}",
            color=c,
        )
        ax[2].plot(
            energy["times"],
            energy["total_e"][0] - energy["total_e"],
            linestyle="dashed",
            color=c,
            label=f"{energy['name']} difference with initial",
        )
    if isinstance(loss_dict, dict):
        used_snapshot_times_training = loss_dict["loss_calculation_times"]
        for a in ax:
            for time in used_snapshot_times_training:
                a.axvline(time, color="gray", linestyle="--")

    # ---- Titles, labels, legend ----
    ax[0].set_title("Internal Energy")
    ax[1].set_title("Kinetic Energy")
    ax[2].set_title("Total Energy")

    for a in ax:
        a.legend()
        a.set_xlabel("Time")
        a.set_ylabel("Energy")
    plt.savefig(os.path.join(folder, model_name, figure_name + ".png"))
    plt.tight_layout()
    plt.show()


def losses_plots(
    data_config: OmegaConf,
    training_config: OmegaConf,
    hr_snapshot: SnapshotData,
    lr_snapshot: SnapshotData,
    lr_sol_snapshot: SnapshotData,
    sim_bundle_hr: SimulationBundle,
    loss_dict: dict,
    folder: str = "corrector/figures",
    model_name: str = "fno",
    figure_name: str = "losses",
):
    loss_function, compute_loss_from_components, active_loss_indices = (
        make_loss_function(training_config)
    )
    v_loss_fn = jax.vmap(loss_function, in_axes=(0, 0, None, None, None))
    hr_states_downscaled = downaverage(
        hr_snapshot.states, data_config.downscaling_factor
    )
    _, loss_lr_hr_corrected = v_loss_fn(
        lr_sol_snapshot.states,
        hr_states_downscaled,
        sim_bundle_hr.config,
        sim_bundle_hr.reg_vars,
        sim_bundle_hr.params,
    )

    _, loss_lr_hr_not_corrected = v_loss_fn(
        hr_states_downscaled,
        lr_snapshot.states,
        sim_bundle_hr.config,
        sim_bundle_hr.reg_vars,
        sim_bundle_hr.params,
    )

    n_losses = len(active_loss_indices.values())
    _, axs = plt.subplots(1, n_losses, figsize=(5 * n_losses, 5), sharey=False)
    if n_losses == 1:
        axs = [axs]
    used_snapshot_times_training = loss_dict["loss_calculation_times"]
    snapshot_times = np.array(sim_bundle_hr.params.snapshot_timepoints)

    _, axs = plt.subplots(
        1,
        len(active_loss_indices.values()),
        figsize=(5 * len(active_loss_indices.values()), 5),
    )

    color_corrected = "tab:blue"
    color_not_corrected = "tab:orange"

    for i, (name, weight) in active_loss_indices.items():
        axs[i].plot(
            snapshot_times,
            weight * loss_lr_hr_corrected[name],
            color=color_corrected,
            label="lr corrected - hr",
        )
        axs[i].plot(
            snapshot_times,
            weight * loss_lr_hr_not_corrected[name],
            color=color_not_corrected,
            label="lr - hr",
        )
        for time in used_snapshot_times_training:
            axs[i].axvline(time, color="gray", linestyle="--")

        axs[i].set_title(f"{name} loss")
        axs[i].set_xlabel("Snapshot Time")
        axs[i].set_ylabel("Loss")
        axs[i].legend()

    plt.savefig(os.path.join(folder, model_name, figure_name + ".png"))
    plt.tight_layout()
    plt.show()


def energy_spectra_validation(
    data_config: OmegaConf,
    hr_snapshot: SnapshotData,
    lr_snapshot: SnapshotData,
    lr_sol_snapshot: SnapshotData,
    sim_bundle_lr: SimulationBundle,
    folder: str = "corrector/figures",
    model_name: str = "fno",
    animate: bool = False,
    animation_name: str = "spectra",
    n_plots: int = 4,
):
    vget_energy = jax.vmap(get_energy, in_axes=(0, None, None, None))
    labels = ["low res", "high res", "low res sol"]
    states_list = (
        lr_snapshot.states,
        downaverage(
            hr_snapshot.states, downscale_factor=data_config.downscaling_factor
        ),
        lr_sol_snapshot.states,
    )
    energies = {}
    for states, label in zip(states_list, labels):
        energies[label] = vget_energy(
            states,
            sim_bundle_lr.config,
            sim_bundle_lr.reg_vars,
            sim_bundle_lr.params,
        )

    vpower = jax.vmap(pk_jax_1d, in_axes=(0, None, None))
    spectrums = []
    for label, energy in energies.items():
        k, Pk, _ = vpower(energy, 1.0, 0)
        spectrums.append(
            {
                "spectrum": Pk,
                "k": k,
                "label": label,
                "time_points": hr_snapshot.time_points,
            }
        )

    # --- Static plotting (non-animated mode) ---
    if not animate:
        time_points = spectrums[0]["time_points"]
        total_frames = len(time_points)
        if n_plots > total_frames:
            n_plots = total_frames

        # choose n_plots equally spaced frame indices
        frame_indices = np.linspace(0, total_frames - 1, n_plots, dtype=int)

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
        if n_plots == 1:
            axes = [axes]

        for ax, j in zip(axes, frame_indices):
            for spec in spectrums:
                ax.plot(spec["k"][j], spec["spectrum"][j], lw=2, label=spec["label"])

            # reference line k^-2
            k_ref = spectrums[0]["k"][j]
            P_ref = k_ref**-2
            ax.plot(k_ref, P_ref, "k--", label=r"$k^{-2}$")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("k")
            ax.set_title(f"t = {time_points[j]:.2f}")
            ax.set_ylim(1e-6, 4e-1)
            ax.grid(True, which="both", ls="--", lw=0.5)

        axes[0].set_ylabel("P(k)")
        axes[-1].legend()
        fig.suptitle("Energy spectra at selected time points", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(folder, model_name, f"{animation_name}_static.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.show()
        return

    # --- Animation mode ---
    fig, ax = plt.subplots(figsize=(8, 6))
    lines = []
    for spec in spectrums:
        (line,) = ax.plot([], [], lw=2, label=spec["label"])
        lines.append(line)

    k_ref = spectrums[0]["k"][0]
    P_ref = k_ref**-2
    (_,) = ax.plot(k_ref, P_ref, "k--", label=r"$k^{-2}$ reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.set_ylim(1e-6, 4e-1)
    ax.set_title("Power spectra for all resolutions")
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate_frame(j):
        for line, spec in zip(lines, spectrums):
            line.set_data(spec["k"][j], spec["spectrum"][j])
        return lines

    ani = animation.FuncAnimation(
        fig,
        animate_frame,
        init_func=init,
        frames=len(spectrums[0]["time_points"]),
        interval=100,
        blit=False,
    )

    os.makedirs(os.path.join(folder, model_name), exist_ok=True)
    ani.save(
        os.path.join(folder, model_name, animation_name + ".mp4"),
        writer=animation.FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.show()


def divergence_metric(B, divB, grid_spacing):
    div_rms = jnp.sqrt(jnp.mean(divB**2))
    B_rms = jnp.sqrt(jnp.mean(B**2))
    metric = div_rms / (B_rms / grid_spacing)
    return metric


def magnetic_field_validation(
    data_config: OmegaConf,
    hr_snapshot: SnapshotData,
    lr_snapshot: SnapshotData,
    lr_sol_snapshot: SnapshotData,
    sim_bundle_hr: SimulationBundle,
    sim_bundle_lr: SimulationBundle,
    loss_dict: dict,
    folder: str = "corrector/figures",
    model_name: str = "fno",
    figure_name: str = "divergence",
):
    hr_states_downscaled = downaverage(
        hr_snapshot.states, data_config.downscaling_factor
    )
    used_snapshot_times_training = loss_dict["loss_calculation_times"]
    v_div = jax.vmap(divergence3D, in_axes=[0, None])
    v_div_metric = jax.vmap(divergence_metric, in_axes=[0, 0, None])

    if isinstance(sim_bundle_lr.reg_vars.magnetic_index, StaticIntVector):
        magnetic_index = list(sim_bundle_lr.reg_vars.magnetic_index)
    else:
        raise ValueError("The snapshot given wasnt 2d nor 3d")

    divergence_lr_sol = v_div(
        lr_sol_snapshot.states[:, magnetic_index],
        sim_bundle_lr.config.grid_spacing,
    )
    divergence_lr_sol_metric = v_div_metric(
        lr_sol_snapshot.states[:, magnetic_index],
        divergence_lr_sol,
        sim_bundle_lr.config.grid_spacing,
    )

    divergence_lr = v_div(
        lr_snapshot.states[:, magnetic_index],
        sim_bundle_lr.config.grid_spacing,
    )

    divergence_lr_metric = v_div_metric(
        lr_snapshot.states[:, magnetic_index],
        divergence_lr,
        sim_bundle_lr.config.grid_spacing,
    )
    divergence_hr = v_div(
        hr_snapshot.states[:, magnetic_index],
        sim_bundle_hr.config.grid_spacing,
    )

    divergence_hr_metric = v_div_metric(
        hr_snapshot.states[:, magnetic_index],
        divergence_hr,
        sim_bundle_hr.config.grid_spacing,
    )
    _, axs = plt.subplots(1, 1, figsize=(8, 6), sharey=False)
    snapshot_times = np.array(sim_bundle_hr.params.snapshot_timepoints)

    color_corrected = "tab:blue"
    color_not_corrected = "tab:orange"
    color_hr = "tab:red"
    axs.plot(
        snapshot_times,
        divergence_lr_sol_metric,
        color=color_corrected,
        label="lr corrected",
    )
    axs.plot(
        snapshot_times,
        divergence_lr_metric,
        color=color_not_corrected,
        label="lr",
    )
    axs.plot(snapshot_times, divergence_hr_metric, color=color_hr, label="hr")
    for time in used_snapshot_times_training:
        axs.axvline(time, color="gray", linestyle="--")

    axs.set_title("Magnetic Field Divergence Metric")
    axs.set_xlabel("Snapshot Time")
    axs.set_ylabel("Metric (rms div / rms B)")
    axs.legend()

    plt.savefig(os.path.join(folder, model_name, figure_name + ".png"))
    plt.tight_layout()
    plt.show()


# def model_output_figures(
#     corrections: List[float],
#     states: List[float],
#     times: List[float],
#     folder: str = "corrector/figures",
#     model_name: str = "fno",
#     figure_name: str = "model_output",
# ):
#     fig, axes = plt.subplots(2, 1, figsize=(9, 10))
#     ax_hist, ax_time = axes
#     reg_vars = jf1uids.get_registered_variables(
#         SimulationConfig(num_cells=32, dimensionality=3, mhd=True)
#     )
#     corrections_array = np.array(corrections)
#     states_array = np.array(states)
#     times_total = np.zeros(len(times))
#     times_delta = np.array(times)
#     for i, time in enumerate(times):
#         times_total[i] = times_total[i - 1] + time
#         print(time, times_total[i])
#
#     assert isinstance(reg_vars.velocity_index, StaticIntVector)
#     assert isinstance(reg_vars.magnetic_index, StaticIntVector)
#     velocity_states_magnitude = (
#         np.sqrt(states_array[:, reg_vars.velocity_index[0]] ** 2)
#         + np.sqrt(states_array[:, reg_vars.velocity_index[1]] ** 2)
#         + np.sqrt(states_array[:, reg_vars.velocity_index[2]] ** 2)
#     )
#     velocity_corrections_magnitude = (
#         np.sqrt(corrections_array[:, reg_vars.velocity_index[0]] ** 2)
#         + np.sqrt(corrections_array[:, reg_vars.velocity_index[1]] ** 2)
#         + np.sqrt(corrections_array[:, reg_vars.velocity_index[2]] ** 2)
#     )
#     magnetic_states_magnitude = (
#         np.sqrt(states_array[:, reg_vars.magnetic_index[0]] ** 2)
#         + np.sqrt(states_array[:, reg_vars.magnetic_index[1]] ** 2)
#         + np.sqrt(states_array[:, reg_vars.magnetic_index[2]] ** 2)
#     )
#     magnetic_corrections_magnitude = (
#         np.sqrt(corrections_array[:, reg_vars.magnetic_index[0]] ** 2)
#         + np.sqrt(corrections_array[:, reg_vars.magnetic_index[1]] ** 2)
#         + np.sqrt(corrections_array[:, reg_vars.magnetic_index[2]] ** 2)
#     )
#
#     # Histogram
#     # Density
#     density_corr = corrections_array[:, reg_vars.density_index]
#     density_states = states_array[:, reg_vars.density_index]
#
#     # Pressure
#     pressure_corr = corrections_array[:, reg_vars.pressure_index]
#     pressures_states = states[:, reg_vars.pressure_index]
#
#     bins = 100
#
#     ax_hist.hist(
#         times_delta * density_corr / density_states,
#         bins=bins,
#         density=True,
#         alpha=0.6,
#         label="Density",
#     )
#
#     ax_hist.hist(
#         times_delta * pressure_corr / pressures_states,
#         bins=bins,
#         density=True,
#         alpha=0.6,
#         label="Pressure",
#     )
#
#     ax_hist.hist(
#         times_delta * velocity_corrections_magnitude / velocity_states_magnitude,
#         bins=bins,
#         density=True,
#         alpha=0.6,
#         label="Velocity",
#     )
#
#     ax_hist.hist(
#         times_delta * magnetic_corrections_magnitude / magnetic_states_magnitude,
#         bins=bins,
#         density=True,
#         alpha=0.6,
#         label="Magnetic",
#     )
#
#     ax_hist.set_title("Distribution of corrections (by variable)")
#     ax_hist.set_xlabel("Correction value")
#     ax_hist.set_ylabel("Probability density")
#     ax_hist.legend()
#     ax_hist.grid(alpha=0.3)
#
#     # Time evolution
#     # Density
#     ax_time.plot(
#         times_total,
#         times_delta
#         * corrections_array[:, reg_vars.density_index]
#         / states_array[:, reg_vars.density_index],
#         label="Density",
#     )
#     # Pessure
#     ax_time.plot(
#         times_total,
#         times_delta
#         * corrections_array[:, reg_vars.pressure_index]
#         / states_array[:, reg_vars.pressure_index],
#         label="Pressure",
#     )
#     # Velocity and Magnetic
#     mean_velocity_corrections = np.mean(velocity_corrections_magnitude, axis=0)
#     mean_velocity_states = np.mean(velocity_states_magnitude, axis=0)
#     mean_magnetic_corrections = np.mean(magnetic_corrections_magnitude, axis=0)
#     mean_magnetic_states = np.mean(magnetic_states_magnitude, axis=0)
#     ax_time.plot(
#         times_total,
#         times_delta * velocity_corrections_magnitude / velocity_states_magnitude,
#         label="Velocity",
#     )
#     ax_time.plot(
#         times_total,
#         times_delta * magnetic_corrections_magnitude / magnetic_states_magnitude,
#         label="Magnetic",
#     )
#     ax_time.set_title("Model output normalized")
#     ax_time.set_xlabel("Times")
#     ax_time.set_ylabel("Correction * t / state")
#     ax_time.legend()
#
#     plt.savefig(os.path.join(folder, model_name, figure_name + ".png"))
#     plt.tight_layout()
#     plt.show()
#


def model_output_figures(
    corrections: List[float],
    states: List[float],
    times: List[float],
    folder: str = "corrector/figures",
    model_name: str = "fno",
    figure_name: str = "model_output",
):
    reg_vars = jf1uids.get_registered_variables(
        SimulationConfig(num_cells=32, dimensionality=3, mhd=True)
    )
    corrections_array = np.array(corrections)
    states_array = np.array(states)
    times_total = np.zeros(len(times))
    times_delta = np.array(times)
    for i, time in enumerate(times):
        times_total[i] = times_total[i - 1] + time

    assert isinstance(reg_vars.velocity_index, StaticIntVector)
    assert isinstance(reg_vars.magnetic_index, StaticIntVector)
    velocity_states_magnitude = np.sqrt(
        states_array[:, reg_vars.velocity_index[0]] ** 2
        + states_array[:, reg_vars.velocity_index[1]] ** 2
        + states_array[:, reg_vars.velocity_index[2]] ** 2
    )
    velocity_corrections_magnitude = np.sqrt(
        corrections_array[:, reg_vars.velocity_index[0]] ** 2
        + corrections_array[:, reg_vars.velocity_index[1]] ** 2
        + corrections_array[:, reg_vars.velocity_index[2]] ** 2
    )

    velocity_corrections_norm = (
        times_delta * velocity_corrections_magnitude / velocity_states_magnitude
    )

    magnetic_states_magnitude = np.sqrt(
        states_array[:, reg_vars.magnetic_index[0]] ** 2
        + states_array[:, reg_vars.magnetic_index[1]] ** 2
        + states_array[:, reg_vars.magnetic_index[2]] ** 2
    )
    magnetic_corrections_magnitude = np.sqrt(
        corrections_array[:, reg_vars.magnetic_index[0]] ** 2
        + corrections_array[:, reg_vars.magnetic_index[1]] ** 2
        + corrections_array[:, reg_vars.magnetic_index[2]] ** 2
    )

    magnetic_corrections_norm = (
        times_delta * magnetic_corrections_magnitude / magnetic_states_magnitude
    )

    # Histogram
    # Density
    density_corr = corrections_array[:, reg_vars.density_index]
    density_states = states_array[:, reg_vars.density_index]
    density_corr_norm = times_delta * density_corr / density_states
    # Pressure
    pressure_corr = corrections_array[:, reg_vars.pressure_index]
    pressures_states = states_array[:, reg_vars.pressure_index]
    pressure_corr_norm = times_delta * pressure_corr / pressures_states

    plot_corrections_figure(
        density_corr=density_corr_norm,
        pressure_corr=pressure_corr_norm,
        velocity_corr_mag=velocity_corrections_norm,
        magnetic_corr_mag=magnetic_corrections_norm,
        times_total=times_total,
        bins=100,
        title_suffix="normalized",
        ylabel="Correction * t /state",
        save_path=os.path.join(folder, model_name, figure_name + ".png"),
    )

    plot_corrections_figure(
        density_corr=density_corr,
        pressure_corr=pressure_corr,
        velocity_corr_mag=velocity_corrections_magnitude,
        magnetic_corr_mag=magnetic_corrections_magnitude,
        times_total=times_total,
        bins=100,
        title_suffix="unnormalized",
        ylabel="Correction",
        save_path=os.path.join(
            folder, model_name, figure_name + "_unnormalized" + ".png"
        ),
    )


def plot_corrections_figure(
    *,
    density_corr,
    pressure_corr,
    velocity_corr_mag,
    magnetic_corr_mag,
    times_total,
    bins,
    title_suffix,
    ylabel,
    save_path,
):
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    ax_hist, ax_time = axes

    # -----------------------
    # Histogram
    # -----------------------
    ax_hist.hist(density_corr, bins=bins, density=False, alpha=0.6, label="Density")
    ax_hist.hist(pressure_corr, bins=bins, density=False, alpha=0.6, label="Pressure")
    ax_hist.hist(
        velocity_corr_mag, bins=bins, density=False, alpha=0.6, label="Velocity"
    )
    ax_hist.hist(
        magnetic_corr_mag, bins=bins, density=False, alpha=0.6, label="Magnetic"
    )

    ax_hist.set_title(f"Distribution of corrections ({title_suffix})")
    ax_hist.set_xlabel("Correction value")
    ax_hist.set_ylabel("Probability density")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    # -----------------------
    # Time evolution
    # -----------------------
    ax_time.plot(times_total, density_corr, label="Density")
    ax_time.plot(times_total, pressure_corr, label="Pressure")

    ax_time.plot(times_total, velocity_corr_mag, label="Velocity")
    ax_time.plot(times_total, magnetic_corr_mag, label="Magnetic")

    ax_time.set_title(f"Model output ({title_suffix})")
    ax_time.set_xlabel("Times")
    ax_time.set_ylabel(ylabel)
    ax_time.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
