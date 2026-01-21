from autocvd import autocvd

num_gpus = 1

autocvd(num_gpus=num_gpus)

import jax
import jax.numpy as jnp
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate

import equinox as eqx
from corrector_src.model.cnn_mhd_model import CorrectorCNN
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CorrectorParams,
    CorrectorConfig,
)
from corrector.figures.validation_figures import (
    losses_plots,
    energy_conservation_plots,
    energy_spectra_validation,
    magnetic_field_validation,
    model_output_figures,
)
from corrector_src.data.dataset import dataset
from corrector_src.hydra.create_config import load_config

PROJECT_ROOT = os.getcwd()

if __name__ == "__main__":
    model_name = "grad_acc_mhd_blast"
    model_path = os.path.join("corrector/models", model_name)

    config = load_config(
        overrides=["models=cnn", "data=blast_cnn"],
        exp_name="mhd_blast_plots",
        create_exp_folder=False,
    )
    config.data.t_end = 0.5
    dataset_turb = dataset(scenarios_to_use=config.data.scenarios, cfg_data=config.data)

    model_cfg = OmegaConf.to_container(config.models, resolve=True)
    assert isinstance(model_cfg, dict)
    model_specs = model_cfg.pop("_name_", None)

    key = jax.random.PRNGKey(config.training.rng_key)

    # ─── New Fields ───────────────────────────────────────────────────────

    corrections = []
    states = []
    times = []

    def snapshot_callable(time, state, correction):
        corrections.append(jnp.mean(correction, axis=[1, 2, 3]))
        states.append(jnp.mean(state, axis=[1, 2, 3]))
        times.append(time)
        pass

    model_cfg["snapshot_callable"] = snapshot_callable

    model = instantiate(model_cfg, key=key)
    model = eqx.tree_deserialise_leaves(
        os.path.join(
            PROJECT_ROOT,
            model_path,
            "cnn_model.eqx",
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
        correct_from_beggining=True,
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
        corrector_config=corrector_config,
        corrector_params=corrector_params,
        rng_seed=112,
    )

    figures_folder = os.path.join(PROJECT_ROOT, "corrector/figures")
    os.makedirs(os.path.join(figures_folder, model_name), exist_ok=True)

    loss_dict = {"loss_calculation_times": config.data.snapshot_timepoints}
    magnetic_field_validation(
        data_config=config.data,
        hr_snapshot=hr_snapshot_data,
        lr_snapshot=lr_snapshot_data,
        lr_sol_snapshot=lr_ml_snapshot_data,
        sim_bundle_hr=sim_bundle_hr,
        sim_bundle_lr=sim_bundle_lr,
        model_name=model_name,
        loss_dict=loss_dict,
        folder=figures_folder,
    )
    energy_conservation_plots(
        config=config,
        hr_snapshot=hr_snapshot_data,
        lr_snapshot=lr_snapshot_data,
        lr_sol_snapshot=lr_ml_snapshot_data,
        sim_bundle_hr=sim_bundle_hr,
        sim_bundle_lr=sim_bundle_lr,
        model_name=model_name,
        loss_dict=loss_dict,
        folder=figures_folder,
    )

    losses_plots(
        data_config=config.data,
        training_config=config.training,
        hr_snapshot=hr_snapshot_data,
        lr_snapshot=lr_snapshot_data,
        lr_sol_snapshot=lr_ml_snapshot_data,
        sim_bundle_hr=sim_bundle_hr,
        model_name=model_name,
        loss_dict=loss_dict,
        folder=figures_folder,
    )
    energy_spectra_validation(
        data_config=config.data,
        hr_snapshot=hr_snapshot_data,
        lr_snapshot=lr_snapshot_data,
        lr_sol_snapshot=lr_ml_snapshot_data,
        sim_bundle_lr=sim_bundle_lr,
        model_name=model_name,
        folder=figures_folder,
    )
    model_output_figures(
        corrections=corrections,
        states=states,
        times=times,
        model_name=model_name,
        folder=figures_folder,
    )
