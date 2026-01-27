import math
import jax.numpy as jnp
from jaxtyping import Array
from typing import Optional
from arena.arena_tests.solver_in_the_loop.model_manager import TrainingConfig


def normalized_weighted_loss(
    pred_state: jnp.ndarray,
    target_state: jnp.ndarray,
    channel_normalizers: Optional[jnp.ndarray] = None,
    physics_weights: Optional[jnp.ndarray] = None,
    use_interface: bool = False,
):
    if physics_weights is None:
        physics_weights = jnp.ones(11)
    if channel_normalizers is not None:
        normalized_error = (pred_state - target_state) / channel_normalizers[
            :, None, None, None
        ]
    else:
        normalized_error = pred_state - target_state
    weighted_error = physics_weights[:, None, None, None] * (normalized_error**2)
    return jnp.mean(weighted_error[:8] if not use_interface else weighted_error)


def simple_mse_loss(pred_state: jnp.ndarray, target_state: jnp.ndarray):
    return jnp.mean((pred_state - target_state) ** 2)


def loss_setup(training_config: TrainingConfig, target_states: Array):
    if training_config.loss_type == "norm_mse":
        channel_normalizers = jnp.maximum(
            jnp.std(target_states, axis=(0, 2, 3, 4)), 1e-8
        )
        loss_fn_kwargs = {
            "channel_normalizers": channel_normalizers,
            "physics_weights": training_config.physics_weights,
            "use_interface": training_config.use_interface,
        }
        loss_fn_factory = normalized_weighted_loss
    else:
        loss_fn_kwargs = {}
        loss_fn_factory = simple_mse_loss
    return loss_fn_kwargs, loss_fn_factory


class EarlyStopper:
    max_patience: int = 10
    best_loss: float = math.inf
    patience: int = 0
    early_stopped: bool = False

    def __init__(self, max_patience: int, use_early_stopper: bool = False):
        self.max_patience = max_patience
        self.use_early_stopper = use_early_stopper

    def new_epoch(self, loss: float) -> bool:
        if self.use_early_stopper:
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience = 0
            else:
                self.patience += 1
                if self.patience == self.max_patience:
                    self.early_stopped = True
                    return True
            return False
        else:
            return False

    def reset_patience(self):
        self.patience = 0
        self.best_loss = math.inf
