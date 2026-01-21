import os
import datetime
from omegaconf import OmegaConf
from hydra import initialize, compose


def load_config(
    config_path="../../configs",
    config_name="config",
    overrides=None,
    experiment_root="experiments",
    exp_name="turbulence_corrector",
    version_base="1.2",
    create_exp_folder=True,
):
    """Load a Hydra config but mimic the @hydra.main runtime behavior."""

    # --- Compose config manually ---
    with initialize(config_path=config_path, version_base=version_base):
        cfg = compose(config_name=config_name, overrides=overrides)

    # --- Save the composed config inside that folder and change working directory---
    if create_exp_folder:
        # --- Create an experiment directory (like Hydra does) ---
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = cfg.get("experiment_name", exp_name)
        run_dir = os.path.join(experiment_root, exp_name, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
        os.chdir(run_dir)
        print(f"💾 Running in Hydra-style directory: {run_dir}")

    return cfg
