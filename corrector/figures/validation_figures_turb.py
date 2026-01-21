from autocvd import autocvd

num_gpus = 1

autocvd(num_gpus=num_gpus)

from corrector_src.utils.downaverage import downaverage
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData
from jf1uids.option_classes.simulation_config import SimulationConfig
from matplotlib import animation
import matplotlib.pyplot as plt
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
from corrector_src.data.dataset import SimulationBundle, dataset
import Pk_library as PKL
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate

import equinox as eqx
from corrector_src.model.fno_hd_force_corrector import TurbulenceSGSForceCorrectorFNO
from corrector_src.model._corrector_options import (
    CorrectorConfig,
    CorrectorParams,
)


def energy_conservation_plots(
    config: SimulationConfig,
    hr_snapshot: SnapshotData,
    lr_snapshot: SnapshotData,
    lr_sol_snapshot: SnapshotData,
    sim_bundle_hr: SimulationBundle,
    sim_bundle_lr: SimulationBundle,
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
    sim_bundle_lr: SimulationBundle,
    # losses: np.lib.npyio.NpzFile,
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
    fig, axs = plt.subplots(1, n_losses, figsize=(5 * n_losses, 5), sharey=False)
    if n_losses == 1:
        axs = [axs]
    used_snapshot_times_training = loss_dict["loss_calculation_times"]
    snapshot_times = np.array(sim_bundle_hr.params.snapshot_timepoints)

    fig, axs = plt.subplots(
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
    sim_bundle_hr: SimulationBundle,
    sim_bundle_lr: SimulationBundle,
    folder: str = "corrector/figures",
    model_name: str = "fno",
    animate: bool = False,
    animation_name: str = "spectra",
    n_plots: int = 4,  # <-- new argument controlling how many static plots to make
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
        k, Pk, Nmodes = vpower(energy, 1.0, 0)
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
    (ref_line,) = ax.plot(k_ref, P_ref, "k--", label=r"$k^{-2}$ reference")

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


if __name__ == "__main__":
    model_name = "navy_blue"
    model_path = os.path.join("corrector/models/fno_turbulence", model_name)
    config_path = os.path.join("../..", model_path)

    with initialize(config_path=config_path, version_base="1.2"):
        config = compose(
            config_name="config",
        )
    dataset_turb = dataset(scenarios_to_use=config.data.scenarios, cfg_data=config.data)

    model_cfg = OmegaConf.to_container(config.models, resolve=True)
    model_specs = model_cfg.pop("_name_", None)

    key = jax.random.PRNGKey(config.training.rng_key)

    # ─── New Fields ───────────────────────────────────────────────────────
    model_cfg["postprocessing_floor"] = True
    # model_cfg["output_channels"] = 5

    model = instantiate(model_cfg, key=key)
    model = eqx.tree_deserialise_leaves(
        os.path.join(
            model_path,
            "fno.eqx",
        ),
        model,
    )

    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    trainable_params = sum(
        x.size
        for x in jax.tree_util.tree_leaves(eqx.filter(neural_net_params, eqx.is_array))
    )
    print(
        f" ✅ Initialized model '{model_name}' successfully with # of params {trainable_params}"
    )
    corrector_config = CorrectorConfig(
        corrector=True,
        network_static=neural_net_static,
        correct_from_beggining=False,
        start_correction_time=0.06,
    )
    corrector_params = CorrectorParams(network_params=neural_net_params)

    (
        sim_bundle_hr,
        sim_bundle_lr,
        hr_snapshot_data,
        lr_snapshot_data,
        lr_ml_snapshot_data,
    ) = dataset_turb.hr_lr_ml_states_integration(
        corrector_config=corrector_config, corrector_params=corrector_params
    )

    figures_folder = os.path.join("corrector/figures", model_name)
    os.makedirs(figures_folder, exist_ok=True)

    energy_conservation_plots(
        config=config,
        hr_snapshot=hr_snapshot_data,
        lr_snapshot=lr_snapshot_data,
        lr_sol_snapshot=lr_ml_snapshot_data,
        sim_bundle_hr=sim_bundle_hr,
        sim_bundle_lr=sim_bundle_lr,
        model_name=model_name,
    )

    losses_plots(
        data_config=config.data,
        training_config=config.training,
        hr_snapshot=hr_snapshot_data,
        lr_snapshot=lr_snapshot_data,
        lr_sol_snapshot=lr_ml_snapshot_data,
        sim_bundle_hr=sim_bundle_hr,
        sim_bundle_lr=sim_bundle_lr,
        model_name=model_name,
        loss_dict={"loss_calculation_times": [0.5, 1.0]},
    )
    energy_spectra_validation(
        data_config=config.data,
        hr_snapshot=hr_snapshot_data,
        lr_snapshot=lr_snapshot_data,
        lr_sol_snapshot=lr_ml_snapshot_data,
        sim_bundle_hr=sim_bundle_hr,
        sim_bundle_lr=sim_bundle_lr,
        model_name=model_name,
    )
