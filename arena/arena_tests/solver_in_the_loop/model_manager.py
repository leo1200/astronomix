from typing import Optional
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
import datetime
from jaxtyping import PyTree
import os
import numpy as np
import logging
# ============================================================================
# MODEL MANAGEMENT CLASSES
# ============================================================================

logger = logging.Logger(__name__)


@dataclass
class TrainingConfig:
    epochs_per_time: list
    snapshot_timepoints_train: list
    learning_rate: float = 0.005
    peak_lr: float = 0.08
    end_lr: float = 0.03
    warmup_steps_fraction: float = 0.5
    hidden_channels: int = 5
    hidden_layers: int = 2
    model_initialization_scale: float = 0.1
    noise_level: float = 0.05
    gradient_clip: float = 1.0
    loss_type: str = "norm_mse"
    use_interface: bool = False
    physics_weights: Optional[list] = None
    use_checkify: bool = False
    early_stopper: bool = True
    patience: int = 15
    direction: int = 1  # BACK TO FRONT

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
        self, base_dir: str = "arena/data/models", model_name: Optional[str] = None
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

    def save_model_params(self, params: PyTree, filename: str = "model_params.pkl"):
        path = self.base_dir / self.model_name / filename
        with open(path, "wb") as f:
            pickle.dump(params, f)
        print("✓ Saved model params")

    def save_checkpoint(self, params: PyTree, epoch: int, loss: float):
        checkpoint_dir = self.base_dir / self.model_name / "checkpoints"
        checkpoint_path = (
            checkpoint_dir / f"checkpoint_epoch_{epoch:04d}_loss_{loss:.6f}.pkl"
        )
        with open(checkpoint_path, "wb") as f:
            pickle.dump(params, f)

    def save_losses(self, losses):
        path = self.base_dir / self.model_name / "losses.npz"
        np.savez(path, losses=losses)
        print("✓ Saved losses")

    def load_training_config(self) -> TrainingConfig:
        path = self.base_dir / self.model_name / "training_config.json"
        with open(path, "r") as f:
            data = json.load(f)
        return TrainingConfig(**data)

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

    def load_model_params(self):
        path = self.base_dir / self.model_name / "model_params.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_model_params_nan(self):
        path = self.base_dir / self.model_name / "model_params_NAN.pkl"
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Model {self.model_name} doesnt have nan params")

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
            sim = self.load_simulation_config()
            print(f"\nBest Loss: {metadata.best_loss:.6e}")
            if metadata.performance_metric:
                print(f"Performance: {metadata.performance_metric:.6e}")
            print(f"Hidden: {training.hidden_channels}x{training.hidden_layers}")
            print(f"Resolution: {sim.num_cells_high_res}")
        except Exception as e:
            print(f"Error: {e}")
        print("=" * 70)


def model_loader(
    model_manager: ModelManager,
    neural_net_params: PyTree,
    load_model: bool = False,
    load_model_nan: bool = False,
):
    if load_model and not load_model_nan:
        neural_net_params = model_manager.load_model_params()
        logger.info("✓ Loaded existing model")

    if load_model_nan:
        neural_net_params = model_manager.load_model_params_nan()
        logger.info("✓ Loaded existing model nan params")
    return neural_net_params
