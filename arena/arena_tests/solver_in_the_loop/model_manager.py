from typing import Optional
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
import datetime
from jaxtyping import PyTree
import jax.tree_util as jtu
import os
import numpy as np
import logging
import equinox as eqx
# ============================================================================
# MODEL MANAGEMENT CLASSES
# ============================================================================

logger = logging.Logger(__name__)


@dataclass
class TrainingConfig:
    epochs_per_time: list
    snapshot_timepoints_train: list
    model_name: str = "default"
    learning_rate: float = 0.005
    peak_lr: float = 0.08
    end_lr: float = 0.03
    warmup_steps_fraction: float = 0.4
    hidden_channels: int = 5
    hidden_layers: int = 2
    model_initialization_scale: float = 0.1
    noise_level: float = 0.05
    gradient_clip: float = 1.0
    loss_type: str = "norm_mse"
    use_interface: bool = False
    physics_weights: Optional[list] = None
    use_checkify: bool = False
    use_early_stopper: bool = True
    patience: int = 15
    direction: int = 1  # BACK TO FRONT
    num_cells_high_res: int = 64
    downaverage_factor: int = 2
    start_correction_time: float = 0.0
    correct_from_beggining: bool = True
    limiter: int = 4
    c_cfl: float = 1.0
    c_cfl_target: float = 0.8
    t_end: float = 0.2
    batch_size: int = 2
    conFIG: bool = False
    sgd: bool = False
    normalize_input: bool = False
    use_film_corrector: bool = False

    def to_dict(self):
        d = asdict(self)
        if hasattr(d["snapshot_timepoints_train"], "tolist"):
            d["snapshot_timepoints_train"] = d["snapshot_timepoints_train"].tolist()
        if hasattr(d["epochs_per_time"], "tolist"):
            d["epochs_per_time"] = d["epochs_per_time"].tolist()
        return d


@dataclass
class SimulationConfigTraining:
    num_cells_high_res: int = 64
    downaverage_factor: int = 2
    start_correction_time: float = 0.0
    correct_from_beggining: bool = True
    t_end: float = 0.2
    c_cfl: float = 1.5
    c_cfl_target: float = 0.8  # Rather messy the fact that the target data in the optuna was created with c_cfl = 0.8 and limiter 4
    limiter: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelMetadata:
    model_name: str
    created_at: str
    total_epochs: int
    final_epoch: int
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    training_time_seconds: Optional[float] = None
    performance_metric: Optional[float] = None
    succesful_training: bool = False
    early_stopped: Optional[bool] = None
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class ModelManager:
    def __init__(
        self, base_dir: str = "arena/data/models/single_problem", model_name: Optional[str] = None
    ):
        if model_name is None:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            model_name = f"model_{timestamp}"
            print(f"Using generated model name {model_name}")
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_model_directory(self) -> str:
        model_dir = self.base_dir / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "checkpoints").mkdir(exist_ok=True)
        (model_dir / "plots").mkdir(exist_ok=True)
        return self.model_name

    def save_training_config(self, config: TrainingConfig):
        path = self.base_dir / self.model_name / "training_config.json"
        with open(path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        print("✓ Saved training config")

    def save_simulation_config(self, config: SimulationConfigTraining):
        path = self.base_dir / self.model_name / "simulation_config.json"
        with open(path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        print("✓ Saved simulation config")

    def save_metadata(self, metadata: ModelMetadata):
        path = self.base_dir / self.model_name / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        print("✓ Saved metadata")

    def save_losses(self, losses):
        path = self.base_dir / self.model_name / "losses.npz"
        np.savez(path, losses=losses)
        print("✓ Saved losses")

    def load_training_config(self) -> TrainingConfig:
        path = self.base_dir / self.model_name / "training_config.json"
        with open(path, "r") as f:
            data = json.load(f)
        return TrainingConfig(**data)

    def save_model_params(self, params: PyTree, filename: str = "model_params.eqx"):
        path = self.base_dir / self.model_name / filename
        # Equinox leaf serialization is stable across static-field changes
        eqx.tree_serialise_leaves(path, params)
        print("✓ Saved model params")

    def load_model_params(self, like: PyTree, param_type: str = "normal"):
        if param_type == "normal":
            model_params = "model_params"
        elif param_type == "best":
            model_params = "best_model_params"
        elif param_type == "nan":
            model_params = "model_params_NAN"
        else:
            raise ValueError("param_type given is not defined")
        path = self.base_dir / self.model_name / f"{model_params}.eqx"
        return eqx.tree_deserialise_leaves(path, like=like)

    def load_best_model_params(self, like: PyTree):
        path = self.base_dir / self.model_name / "best_model_params.eqx"
        return eqx.tree_deserialise_leaves(path, like=like)

    def save_checkpoint(self, params: PyTree, epoch: int, name: str = None):
        checkpoint_dir = self.base_dir / self.model_name / "checkpoints"
        if name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}.eqx"
        else:
            checkpoint_name = f"{name}.eqx" if not name.endswith(".eqx") else name
        checkpoint_path = checkpoint_dir / checkpoint_name
        eqx.tree_serialise_leaves(checkpoint_path, params)

    def load_model_params_nan(self, like: PyTree):
        path = self.base_dir / self.model_name / "model_params_NAN.eqx"
        if os.path.isfile(path):
            return eqx.tree_deserialise_leaves(path, like=like)
        raise ValueError(f"Model {self.model_name} doesnt have nan params")

    def load_metadata(self) -> ModelMetadata:
        path = self.base_dir / self.model_name / "metadata.json"
        with open(path, "r") as f:
            data = json.load(f)
        return ModelMetadata(**data)

    def load_simulation_config(self) -> SimulationConfigTraining:
        path = self.base_dir / self.model_name / "simulation_config.json"
        with open(path, "r") as f:
            data = json.load(f)
        return SimulationConfigTraining(**data)

    def list_models(self):
        return sorted([d.name for d in self.base_dir.iterdir() if d.is_dir()])

    def print_model_info(self):
        print("=" * 70)
        print(f"MODEL: {self.model_name}")
        print("=" * 70)
        try:
            metadata = ModelMetadata(
                **json.load(open(self.base_dir / self.model_name / "metadata.json"))
            )
            training = self.load_training_config()
            print(f"Best Loss: {metadata.best_loss:.6e}")
            if metadata.performance_metric:
                print(f"Performance: {metadata.performance_metric:.6e}")
            print(f"Hidden: {training.hidden_channels}x{training.hidden_layers}")
            print(f"Resolution: {training.num_cells_high_res}")
            if metadata.notes != "":
                print(f"Description {metadata.notes}")
        except Exception as e:
            print(f"Error: {e}")
        print("=" * 70)


def load_legacy_params_into_current(model, path):
    # current model params template (new treedef)
    current_params = eqx.filter(model, eqx.is_array)

    # old saved params
    with open(path, "rb") as f:
        loaded_params = pickle.load(f)

    # reuse old leaves, but apply new treedef
    leaves, _ = jtu.tree_flatten(loaded_params)
    _, new_treedef = jtu.tree_flatten(current_params)

    if len(leaves) != len(jtu.tree_leaves(current_params)):
        raise ValueError(
            "Leaf count mismatch — architecture changed, not just static fields."
        )

    return jtu.tree_unflatten(new_treedef, leaves)


def model_loader(
    model_manager: ModelManager,
    neural_net_params: PyTree,
    load_model: bool = False,
    load_model_nan: bool = False,
    model: Optional[eqx.Module] = None,  # add this
):
    if load_model and not load_model_nan:
        # Prefer new .eqx
        try:
            neural_net_params = model_manager.load_model_params(like=neural_net_params)
        except FileNotFoundError:
            # fallback to legacy .pkl migration
            if model is None:
                raise ValueError(
                    "Must pass `model` to migrate legacy .pkl params"
                ) from FileNotFoundError
            legacy_path = (
                model_manager.base_dir / model_manager.model_name / "model_params.pkl"
            )
            neural_net_params = load_legacy_params_into_current(model, legacy_path)
        logger.info("✓ Loaded existing model")

    if load_model_nan:
        try:
            neural_net_params = model_manager.load_model_params_nan(
                like=neural_net_params
            )
        except FileNotFoundError:
            if model is None:
                raise ValueError(
                    "Must pass `model` to migrate legacy .pkl params"
                ) from FileNotFoundError
            legacy_path = (
                model_manager.base_dir
                / model_manager.model_name
                / "model_params_NAN.pkl"
            )
            neural_net_params = load_legacy_params_into_current(model, legacy_path)
        logger.info("✓ Loaded existing model nan params")
    return neural_net_params
