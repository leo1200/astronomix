"""
H5 Dataset Loader for pre-computed problem data.

This module provides utilities to load problems from h5 files created by
dataset_creation_*.py scripts. Each h5 file contains:
- Initial high-res states (4800 simulations)
- Final high-res states (ground truth)
- Problem-specific hyperparameters (e.g., B0, B_direction, r0)
- Config and params (stored as JSON in attributes)
"""

import h5py
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence, Tuple, Union
import jax.numpy as jnp
import numpy as np
import logging

logger = logging.getLogger(__name__)


class H5DatasetLoader:
    """Loads problems from pre-computed h5 dataset files.

    Each h5 file contains data for multiple problem instances with the same
    problem type (e.g., 4800 different MHD blast instances).

    Attributes:
        h5_file_path: Path to h5 file
        group_name: Name of group in h5 file (e.g., 'mhd_blast')
        problem_name: Problem type (mhd_blast, turbulence, ot_vortex)
    """

    # Hyperparameter keys for each problem type
    HYPERPARAMETER_KEYS = {
        "mhd_blast": ["B0", "B_direction", "r0"],
        "turbulence": [
            "B0",
            "B_direction",
            "rng_seed",
        ],
        "ot_vortex": ["vortex_axis", "parity", "epsilon_p"],
    }

    def __init__(self, h5_file_path: str, problem_name: str):
        """Initialize H5 dataset loader.

        Args:
            h5_file_path: Path to h5 file
            problem_name: Problem type (mhd_blast, turbulence, ot_vortex)

        Raises:
            FileNotFoundError: If h5 file doesn't exist
            ValueError: If h5 file doesn't have required structure
        """
        self.h5_file_path = Path(h5_file_path)
        self.problem_name = problem_name
        self.group_name = problem_name

        if not self.h5_file_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_file_path}")

        # Validate file structure
        with h5py.File(self.h5_file_path, "r") as f:
            if self.group_name not in f:
                raise ValueError(
                    f"Group '{self.group_name}' not found in {h5_file_path}. "
                    f"Available groups: {list(f.keys())}"
                )

            group = f[self.group_name]
            self._validate_group_structure(group)

            # Cache config and params
            self._config_json = group.attrs.get("config")
            self._params_json = group.attrs.get("params")
            self._num_problems = group["initial_state"].shape[0]

        logger.info(
            f"Loaded H5 dataset: {self.h5_file_path} | "
            f"group={self.group_name} | problems={self._num_problems}"
        )

    def _validate_group_structure(self, group: h5py.Group) -> None:
        """Validate that group has required datasets and attributes.

        Args:
            group: H5 group to validate

        Raises:
            ValueError: If required datasets or attributes missing
        """
        required_datasets = {"initial_state", "final_state"}
        missing = required_datasets - set(group.keys())
        if missing:
            raise ValueError(f"Missing datasets in h5 group: {missing}")

        # Validate shapes match
        initial_shape = group["initial_state"].shape
        final_shape = group["final_state"].shape
        if initial_shape != final_shape:
            raise ValueError(
                f"initial_state and final_state shapes don't match: "
                f"{initial_shape} vs {final_shape}"
            )

        # Validate config and params present
        if "config" not in group.attrs:
            logger.warning(f"Config not found in attributes for {self.problem_name}")
        if "params" not in group.attrs:
            logger.warning(f"Params not found in attributes for {self.problem_name}")

    def get_initial_state(self, index: int) -> jnp.ndarray:
        """Load initial state for a specific problem instance.

        Args:
            index: Problem index (0-based)

        Returns:
            Initial state array with shape (11, 32, 32, 32)

        Raises:
            IndexError: If index out of range
        """
        self._validate_index(index)

        with h5py.File(self.h5_file_path, "r") as f:
            state = f[self.group_name]["initial_state"][index]
            return jnp.array(state, dtype=jnp.float32)

    def get_final_state(self, index: int) -> jnp.ndarray:
        """Load final state for a specific problem instance.

        Args:
            index: Problem index (0-based)

        Returns:
            Final state array with shape (11, 32, 32, 32)

        Raises:
            IndexError: If index out of range
        """
        self._validate_index(index)

        with h5py.File(self.h5_file_path, "r") as f:
            state = f[self.group_name]["final_state"][index]
            return jnp.array(state, dtype=jnp.float32)

    def get_hyperparams(self, index: int) -> Dict[str, Any]:
        """Load hyperparameters for a specific problem instance.

        Args:
            index: Problem index (0-based)

        Returns:
            Dictionary of hyperparameters (e.g., {'B0': 5.2, 'B_direction': [...], 'r0': 0.125})
        """
        self._validate_index(index)

        with h5py.File(self.h5_file_path, "r") as f:
            group = f[self.group_name]
            return self._extract_hyperparams(group, index)

    def get_problem(
        self, index: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Load a single full problem entry from h5.

        Args:
            index: Problem index (0-based)

        Returns:
            Tuple of (initial_state, final_state, hyperparams)
        """
        self._validate_index(index)

        with h5py.File(self.h5_file_path, "r") as f:
            group = f[self.group_name]
            initial_state = jnp.array(group["initial_state"][index], dtype=jnp.float32)
            final_state = jnp.array(group["final_state"][index], dtype=jnp.float32)
            hyperparams = self._extract_hyperparams(group, index)
            return initial_state, final_state, hyperparams

    def get_problem_batch(
        self, indices: Sequence[int]
    ) -> List[Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]]:
        """Load a batch of problem entries lazily in one file-open pass.

        Args:
            indices: Iterable of problem indices

        Returns:
            List of tuples (initial_state, final_state, hyperparams), preserving
            input order (including duplicates).
        """
        index_list = self._normalize_indices(indices)
        if not index_list:
            return []

        batch = []
        with h5py.File(self.h5_file_path, "r") as f:
            group = f[self.group_name]
            initial_ds = group["initial_state"]
            final_ds = group["final_state"]

            for index in index_list:
                initial_state = jnp.array(initial_ds[index], dtype=jnp.float32)
                final_state = jnp.array(final_ds[index], dtype=jnp.float32)
                hyperparams = self._extract_hyperparams(group, index)
                batch.append((initial_state, final_state, hyperparams))

        return batch

    def get_filtered_indices(
        self, param_filters: Dict[str, Tuple[float, float]]
    ) -> List[int]:
        """Return indices where all scalar param values fall within their ranges.

        Only scalar hyperparameters (stored as 1-D datasets) can be filtered.
        Non-scalar keys (e.g. B_direction) are silently ignored.

        Args:
            param_filters: Mapping of param_name -> (min_val, max_val), inclusive.

        Returns:
            Sorted list of matching indices.

        Raises:
            KeyError: If a filter key is not present in the h5 file.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            group = f[self.group_name]
            mask = np.ones(self._num_problems, dtype=bool)
            for param_name, (lo, hi) in param_filters.items():
                if param_name not in group:
                    raise KeyError(
                        f"Parameter '{param_name}' not found in h5 group "
                        f"'{self.group_name}'. Available: {list(group.keys())}"
                    )
                values = group[param_name][:]
                if values.ndim != 1:
                    logger.warning(
                        f"Skipping non-scalar parameter '{param_name}' "
                        f"(shape {values.shape}) in param filter."
                    )
                    continue
                mask &= (values >= lo) & (values <= hi)

        indices = np.where(mask)[0].tolist()
        logger.info(
            f"param_filter on {self.problem_name} {param_filters}: "
            f"{len(indices)}/{self._num_problems} problems match"
        )
        return indices

    def _validate_index(self, index: int) -> None:
        """Validate a single index against dataset bounds."""
        if not 0 <= index < self._num_problems:
            raise IndexError(
                f"Problem index {index} out of range [0, {self._num_problems})"
            )

    def _normalize_indices(self, indices: Sequence[int]) -> List[int]:
        """Validate and normalize a sequence of indices."""
        normalized = [int(index) for index in indices]
        for index in normalized:
            self._validate_index(index)
        return normalized

    def _extract_hyperparams(self, group: h5py.Group, index: int) -> Dict[str, Any]:
        """Extract hyperparameters for an index from an already-open h5 group."""
        hyperparams = {}
        for key in self.HYPERPARAMETER_KEYS.get(self.problem_name, []):
            if key not in group:
                logger.warning(
                    f"Hyperparameter '{key}' not found in h5 for {self.problem_name}"
                )
                continue

            value = group[key][index]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            hyperparams[key] = value

        return hyperparams

    def get_config_json(self) -> Optional[str]:
        """Get cached config JSON string from h5 attributes.

        Returns:
            JSON string of config, or None if not available
        """
        return self._config_json

    def get_params_json(self) -> Optional[str]:
        """Get cached params JSON string from h5 attributes.

        Returns:
            JSON string of params, or None if not available
        """
        return self._params_json

    @property
    def num_problems(self) -> int:
        """Total number of problem instances in this dataset."""
        return self._num_problems

    def __repr__(self) -> str:
        return (
            f"H5DatasetLoader("
            f"file={self.h5_file_path.name}, "
            f"problem={self.problem_name}, "
            f"num_problems={self._num_problems})"
        )
