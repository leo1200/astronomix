"""
Configuration classes for multi-GPU training with flexible problem sampling.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiGPUTrainingConfig:
    """Configuration for multi-GPU training with flexible problem sampling.

    This config extends the base training paradigm with:
    - Per-problem sampling control (e.g., 30 blasts, 10 turbulence per epoch)
    - Multi-GPU parallelization settings
    - Checkpoint and best-model tracking configuration

    Attributes:
        problem_counts: Dict mapping problem names to number of samples per epoch.
                       Example: {"mhd_blast": 30, "turbulence": 10, "ot_vortex": 5}
        num_gpus: Number of GPUs to use for training. If None, uses all available.
        threads_per_gpu: Number of threads to spawn per GPU for gradient computation.
        fixed_problems: If True, train on fixed equally-spaced subset each epoch.
                       If False, randomly sample problems each epoch.
        save_every_epoch: If True, save checkpoint after every epoch. If False, save
                         every N epochs (see checkpoint_interval).
        checkpoint_interval: Save checkpoint every N epochs (only if save_every_epoch=False).
        checkpoint_override: If True, override previous checkpoints with consistent naming.
                           If False, keep all checkpoints with epoch number.
        track_best_model: If True, separately track and save best model parameters.
        early_stopping_enabled: If True, use early stopping based on validation loss.
        early_stopping_patience: Number of epochs to wait before early stopping.
        noise_perturbation: If True, apply noise to initial states during training.
        use_config_gradient: If True, use conFIG for gradient averaging instead of simple mean.
        normalize_per_problem: If True, normalize gradients per problem by their L2 norm before aggregation.
                              Incompatible with clip_per_problem.
        clip_per_problem: If True, clip gradients per problem to L2 norm of 1.0 before aggregation.
                         Incompatible with normalize_per_problem.
    """

    problem_counts: Dict[str, int] = field(default_factory=dict)
    num_gpus: Optional[int] = 1
    threads_per_gpu: int = 1
    fixed_problems: bool = False
    save_every_epoch: bool = True
    checkpoint_interval: int = 50
    checkpoint_override: bool = True
    track_best_model: bool = True
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 35
    noise_perturbation: bool = True
    use_config_gradient: bool = True
    problems_per_batch: int = 8  # Max problems to load into GPU memory at once
    grads_debug: bool = False  # Enable gradient analysis debug mode
    subset_size: Optional[int] = None  # Per-epoch sample size (requires fixed_problems=True)
    normalize_per_problem: bool = False  # Normalize gradients per problem by L2 norm
    clip_per_problem: bool = False  # Clip gradients per problem to L2 norm of 1.0

    def validate(self, available_problems: List[str]) -> None:
        """Validate configuration against available problems.

        Args:
            available_problems: List of problem names available in the dataset.

        Raises:
            ValueError: If validation fails.
        """
        if not self.problem_counts:
            raise ValueError("problem_counts cannot be empty")

        # Check all specified problems exist
        invalid_problems = set(self.problem_counts.keys()) - set(available_problems)
        if invalid_problems:
            raise ValueError(
                f"Invalid problem names in problem_counts: {invalid_problems}. "
                f"Available problems: {available_problems}"
            )

        # Check all counts are positive
        for problem, count in self.problem_counts.items():
            if count <= 0:
                raise ValueError(
                    f"Problem '{problem}' has invalid count {count}. "
                    f"All counts must be positive integers."
                )

        # Check GPU count
        if self.num_gpus is not None and self.num_gpus <= 0:
            raise ValueError(f"num_gpus must be positive, got {self.num_gpus}")

        # Check threads per GPU
        if self.threads_per_gpu <= 0:
            raise ValueError(
                f"threads_per_gpu must be positive, got {self.threads_per_gpu}"
            )

        # Check checkpoint interval
        if not self.save_every_epoch and self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {self.checkpoint_interval}"
            )

        # Check subset_size
        if self.subset_size is not None:
            if not self.fixed_problems:
                raise ValueError(
                    "subset_size can only be used with fixed_problems=True. "
                    "Use --fixed-problems when specifying --subset."
                )
            if self.subset_size <= 0:
                raise ValueError(
                    f"subset_size must be positive, got {self.subset_size}"
                )
            # Check that subset_size doesn't exceed pool size for any problem
            for problem, pool_size in self.problem_counts.items():
                if self.subset_size > pool_size:
                    raise ValueError(
                        f"subset_size ({self.subset_size}) cannot exceed pool size "
                        f"for problem '{problem}' (pool_size={pool_size}). "
                        f"Either increase the pool size or decrease --subset."
                    )

        logger.info(
            f"✓ Config validated successfully with problems: {list(self.problem_counts.keys())}"
        )

    def get_total_samples_per_epoch(self) -> int:
        """Get total number of samples per epoch across all problems."""
        return sum(self.problem_counts.values())

    def get_problem_weights(self) -> Dict[str, float]:
        """Get normalized weights for each problem based on sample counts.

        Returns:
            Dictionary mapping problem names to their normalized weights (sum=1.0).
        """
        total = self.get_total_samples_per_epoch()
        return {
            problem: count / total for problem, count in self.problem_counts.items()
        }

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return asdict(self)

    def __repr__(self) -> str:
        """Human-readable representation of the config."""
        total_samples = self.get_total_samples_per_epoch()
        problem_str = ", ".join(
            f"{name}={count}" for name, count in self.problem_counts.items()
        )
        
        # Add subset info if applicable
        if self.subset_size is not None:
            subset_info = f", subset={self.subset_size}/epoch"
        else:
            subset_info = ""
        
        return (
            f"MultiGPUTrainingConfig("
            f"problems=[{problem_str}] (total={total_samples}/epoch{subset_info}), "
            f"gpus={self.num_gpus}, "
            f"threads={self.threads_per_gpu}, "
            f"checkpoint_override={self.checkpoint_override}, "
            f"early_stop={self.early_stopping_enabled})"
        )
