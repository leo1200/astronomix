"""
H5-based Problem Descriptor for tracking problems loaded from h5 files.

Maps problems to (h5_file, problem_index) pairs with their hyperparameters.
Provides reproducibility by storing exact problem instances used in training.
"""

from dataclasses import dataclass, field, replace as _dc_replace
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class H5ProblemDescriptor:
    """Descriptor for a problem instance loaded from h5 file.

    Identifies a specific problem by h5 file and index, along with
    the hyperparameters that define that problem instance.

    Attributes:
        problem_name: Problem type (mhd_blast, turbulence, ot_vortex)
        h5_file_path: Path to h5 file containing this problem
        problem_index: Index in h5 file (0-based)
        hyperparams: Dict of hyperparameters for this problem
        nickname: Display name for logging (auto-computed)
    """

    problem_name: str
    h5_file_path: str
    problem_index: int
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    nickname: str = field(default="")

    def __post_init__(self):
        """Auto-generate nickname if not provided."""
        if not self.nickname:
            # Create readable nickname from problem type and index
            object.__setattr__(
                self, "nickname", f"{self.problem_name}_{self.problem_index:05d}"
            )

    def _asdict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "problem_name": self.problem_name,
            "h5_file_path": self.h5_file_path,
            "problem_index": self.problem_index,
            "hyperparams": self.hyperparams,
            "nickname": self.nickname,
        }

    def _replace(self, **kwargs) -> "H5ProblemDescriptor":
        """Create new descriptor with replaced fields."""
        return _dc_replace(self, **kwargs)

    def __repr__(self) -> str:
        hyperparams_str = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(self.hyperparams.items())
        )
        return (
            f"H5ProblemDescriptor("
            f"name={self.problem_name}, "
            f"idx={self.problem_index}, "
            f"hyperparams=[{hyperparams_str}])"
        )

    @staticmethod
    def validate(descriptor: "H5ProblemDescriptor") -> None:
        """Validate descriptor has required fields.

        Args:
            descriptor: Descriptor to validate

        Raises:
            ValueError: If descriptor invalid
        """
        if not descriptor.problem_name:
            raise ValueError("problem_name cannot be empty")

        if not descriptor.h5_file_path:
            raise ValueError("h5_file_path cannot be empty")

        if descriptor.problem_index < 0:
            raise ValueError(
                f"problem_index must be >= 0, got {descriptor.problem_index}"
            )

        logger.debug(f"✓ H5ProblemDescriptor validated: {descriptor}")
