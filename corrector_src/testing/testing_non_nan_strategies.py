"""
so far ive found that i get quite some nans when training, this could be due to several reasons, so im gonna try find out why
11/10 im going to try to put a floor value to the pressure and density in the fno postprocessing
the values that im gonna hard code are taking from an optuna experiment that failed in the 1st iteration
"""

from autocvd import autocvd

autocvd(num_gpus=1)

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

import numpy as np
import jax.numpy as jnp
import jax

# corrector_src
from corrector_src.training.time_integration_w_training import (
    time_integration as time_integration_train,
)
from corrector_src.training.training_config import TrainingConfig, TrainingParams
from corrector_src.model.cnn_mhd_model import CorrectorCNN
from corrector_src.model.fno_hd_force_corrector import TurbulenceSGSForceCorrectorFNO
from corrector_src.model._corrector_options import (
    CorrectorConfig,
    CorrectorParams,
)
from corrector_src.loss.sgs_turb_loss import make_loss_function
from corrector_src.data.dataset import dataset
from corrector_src.training.early_stopper import EarlyStopper
from corrector_src.utils.printing_config_summary import print_full_config_summary

# other stuff
import equinox as eqx
import matplotlib.pyplot as plt
import optax
import time
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
import datetime


def load_config(
    config_path="../../configs",
    config_name="config",
    overrides=None,
    experiment_root="experiments",
    exp_name="turbulence_corrector",
    version_base="1.2",
):
    """Load a Hydra config but mimic the @hydra.main runtime behavior."""
    # overrides = overrides or []
    overrides = ["data=turbulence", "training=turbulence_optuna"]

    # --- Compose config manually ---
    with initialize(config_path=config_path, version_base=version_base):
        cfg = compose(config_name=config_name, overrides=overrides)

    # --- Create an experiment directory (like Hydra does) ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = cfg.get("experiment_name", exp_name)
    run_dir = os.path.join(experiment_root, exp_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # --- Save the composed config inside that folder ---
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    # --- Mimic Hydra’s runtime: change into the run directory ---
    os.chdir(run_dir)
    print(f"💾 Running in Hydra-style directory: {run_dir}")

    return cfg


early_stop = False
rng_seeds = [
    535108711,
    1006377286,  # 1143102980
]


def training_loop(cfg):
    print("Hydra run dir:", os.getcwd())
    print_full_config_summary(cfg.data)
    # epochs = cfg.training.epochs
    epochs = 2
    # loss_timesteps = jnp.array(cfg.data.snapshot_timepoints)
    loss_timesteps = jnp.array([0.5, 1.0])
    assert cfg.data.differentiation_mode == 1, (
        "differentiation_mode must be BACKWARDS (1)"
    )

    assert not (loss_timesteps > cfg.data.t_end).any(), (
        "found value greater than the end time in the snapshot timepoints"
    )

    # ─── Training Configuration ───────────────────────────────────────────

    training_config = TrainingConfig(accumulate_grads=True)
    training_params = TrainingParams(loss_calculation_times=loss_timesteps)
    cfg.training.mse_loss = 1  # 0.8071784768701379
    cfg.training.spectral_energy_loss = 0.05  # 0.6729643573801185
    cfg.training.rate_of_strain_loss = 0.2  # 0.541096280010971
    loss_function, compute_loss_from_components, active_loss_indices = (
        make_loss_function(cfg.training)
    )

    # ─── Creating Model And Optimizer ─────────────────────────────────────
    cfg.models.hidden_channels = 13
    cfg.models.n_fourier_layers = 2
    cfg.models.fourier_modes = 14
    cfg.models.shifting_modes = 1

    model_cfg = OmegaConf.to_container(cfg.models, resolve=True)
    model_name = model_cfg.pop("_name_", None)

    key = jax.random.PRNGKey(cfg.training.rng_key)

    # ─── New Fields ───────────────────────────────────────────────────────
    model_cfg["postprocessing_floor"] = True
    # model_cfg["output_channels"] = 5

    model = instantiate(model_cfg, key=key)

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
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(float(cfg.training.learning_rate)),
    )
    opt_state = optimizer.init(neural_net_params)

    snapshot_losses = []
    epoch_losses = []

    gt_cfg_data = cfg.data
    # gt_cfg_data.debug = False
    dataset_creator = dataset(gt_cfg_data.scenarios, gt_cfg_data)
    print(f" ✅ Using data {dataset_creator.scenario_list}")

    # ─── Early Stopping ───────────────────────────────────────────────────
    cfg.training.early_stopping = False
    if cfg.training.early_stopping:
        patience = 50
        print(f" 🥱 Using early stopper with patience {patience}")
        early_stopper = EarlyStopper(patience=patience)
        best_params = neural_net_params
    else:
        early_stopper = None

    if not cfg.data.generate_data_on_fly and len(gt_cfg_data.scenarios) == 1:
        (
            ground_truth,
            sim_bundle_train,
        ) = dataset_creator.train_initializator(
            corrector_config=corrector_config,
            corrector_params=corrector_params,
        )

    for i in range(epochs):
        if cfg.data.generate_data_on_fly:
            if i < len(rng_seeds):
                (
                    ground_truth,
                    sim_bundle_train,
                ) = dataset_creator.train_initializator(
                    corrector_config=corrector_config,
                    corrector_params=corrector_params,
                    rng_seed=rng_seeds[i],
                    verbose=True,
                )
            else:
                (
                    ground_truth,
                    sim_bundle_train,
                ) = dataset_creator.train_initializator(
                    corrector_config=corrector_config,
                    corrector_params=corrector_params,
                )
        else:
            sim_bundle_train.params = sim_bundle_train.params._replace(
                corrector_params=corrector_params
            )
        time_train = time.time()

        losses, new_network_params, opt_state, _ = time_integration_train(
            **sim_bundle_train.unpack_integrate(),
            optimizer=optimizer,
            loss_function=loss_function,
            opt_state=opt_state,
            target_data=ground_truth,
            training_config=training_config,
            training_params=training_params,
        )

        epoch_loss = np.sum(compute_loss_from_components(losses))
        time_train = time.time() - time_train

        if np.isnan(np.mean(losses)):
            print("nan found in loss, stopping the training")
            break
        # snapshot_losses.append(losses.flatten())
        snapshot_losses.append(losses)
        epoch_losses.append(epoch_loss)

        if early_stopper is not None:
            early_stop = early_stopper.early_stop(epoch_loss)
            if epoch_loss < early_stopper.min_validation_loss:
                best_params = new_network_params
            if early_stop:
                print("🚨 Early stopping 🚨")
                break

        corrector_params = corrector_params._replace(network_params=new_network_params)
        print(
            f" 🟡 Epoch {i} time_train {float(time_train):2f} loss {float(epoch_losses[-1].item()):2f}"
        )
    snapshot_losses = np.array(snapshot_losses)

    # snapshot_losses = snapshot_losses.flatten()
    plt.figure()
    if len(training_params.loss_calculation_times) > 1:
        for j, t in enumerate(training_params.loss_calculation_times):
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            for i, (name, weight) in active_loss_indices.items():
                ax.plot(weight * snapshot_losses[:, j, i], label=f"{name} (w={weight})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training Loss Components (t={t})")
            ax.legend()
            fig.savefig(f"components_loss_curve_t_{t}.png")
            plt.close(fig)

        # --- Combined plot with all t values ---
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for j, t in enumerate(training_params.loss_calculation_times):
            for i, (name, weight) in active_loss_indices.items():
                ax.plot(weight * snapshot_losses[:, j, i], label=f"{name}, t={t}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Components")
        ax.legend()
        fig.savefig("components_loss_curve.png")
        plt.close(fig)

    else:
        for i, (name, weight) in active_loss_indices.items():
            plt.plot(weight * snapshot_losses[:, i], label=name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Components")
        plt.legend()
        plt.tight_layout()
        plt.savefig("components_loss_curve.png")
        plt.close()

    plt.figure()
    plt.plot(epoch_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig("epoch_loss_curve.png")
    plt.close()

    if early_stopper is not None:
        final_model = eqx.combine(best_params, neural_net_static)
    else:
        final_model = eqx.combine(new_network_params, neural_net_static)

    eqx.tree_serialise_leaves(f"{model_name}.eqx", final_model)
    np.savez(
        "losses.npz",
        loss_calculation_times=np.array(training_params.loss_calculation_times),
        epoch_losses=np.array(epoch_losses),
        snapshot_losses=np.array(snapshot_losses),
    )


if __name__ == "__main__":
    config = load_config()
    training_loop(config)
