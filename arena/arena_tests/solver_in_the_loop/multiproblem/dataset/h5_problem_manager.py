"""
H5-based Problem Manager for loading training data from pre-computed h5 files.

Manages multiple h5 files, loads problems by index, and integrates with
the existing ProblemManager to get config/params for each problem type.
"""

import logging
import json
from typing import Dict, Iterator, List, Optional, Tuple

from arena.arena_tests.solver_in_the_loop.multiproblem.dataset.h5_dataset_loader import (
    H5DatasetLoader,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.dataset.h5_problem_descriptor import (
    H5ProblemDescriptor,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.problem_manager import (
    _build_hr_config_and_params,
    TrainingPair,
    SimulationBundle,
)
from arena.arena_tests.solver_in_the_loop.model_manager import TrainingConfig
import jax.numpy as jnp
from astronomix import finalize_config, get_registered_variables

logger = logging.getLogger(__name__)


class H5ProblemManager:
    """Manages problems loaded from h5 dataset files.

    Integrates with existing ProblemManager to:
    1. Get default config/params for each problem type
    2. Load states from h5 files
    3. Create training pairs from h5 data
    Supports multiple h5 files (splits) and flexible problem ranges.
    """

    def __init__(
        self,
        h5_file_paths: Dict[str, str],
        training_config: TrainingConfig,
        problem_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        param_filters: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    ):
        """Initialize H5 Problem Manager.

        Args:
            h5_file_paths: Mapping of {problem_name: h5_file_path}
            training_config: Training configuration
            problem_ranges: Optional mapping of {problem_name: (start_idx, end_idx)}
                           If None, uses all available problems.
                           Ignored for any problem that has an entry in param_filters.
            param_filters: Optional mapping of
                           {problem_name: {param_name: (min_val, max_val)}}.
                           When provided for a problem, the index range is derived
                           from the H5 hyperparameter arrays rather than problem_ranges.
                           Only scalar hyperparameters (e.g. B0, r0) are supported.

        Raises:
            FileNotFoundError: If any h5 file not found
            ValueError: If problem_ranges invalid
        """
        self.training_config = training_config
        self.problem_ranges = problem_ranges or {}
        self.param_filters: Dict[str, Dict[str, Tuple[float, float]]] = param_filters or {}

        # Load H5 loaders for each problem type
        self.h5_loaders: Dict[str, H5DatasetLoader] = {}
        for problem_name, h5_path in h5_file_paths.items():
            try:
                loader = H5DatasetLoader(h5_path, problem_name)
                self.h5_loaders[problem_name] = loader
                logger.info(f"Loaded H5 dataset for {problem_name}: {loader}")
            except Exception as e:
                logger.error(f"Failed to load H5 for {problem_name}: {e}")
                raise

        # Get config/params for each problem type from ProblemManager
        self.problem_configs: Dict[str, Dict] = {}
        self._setup_problem_configs()

        # Caches
        self._problem_descriptors: Optional[List[H5ProblemDescriptor]] = None
        self._training_pairs: Optional[List[TrainingPair]] = None

        logger.info(
            f"H5ProblemManager initialized with {len(self.h5_loaders)} problem types"
        )

    def _setup_problem_configs(self) -> None:
        """Setup config and params for each problem type.

        Uses H5-serialized config/params and hydrates them into SimulationConfig
        and SimulationParams instances using training defaults as templates.
        """
        config_template, params_template = _build_hr_config_and_params(self.training_config)

        for problem_name, loader in self.h5_loaders.items():
            logger.info(f"Setting up config/params for {problem_name}")

            config_json = loader.get_config_json()
            params_json = loader.get_params_json()

            if config_json and params_json:
                config_data = json.loads(config_json)
                params_data = json.loads(params_json)
                config = self._hydrate_like(config_template, config_data)
                params = self._hydrate_like(params_template, params_data)
                self.problem_configs[problem_name] = {
                    "config": config,
                    "params": params,
                }
                logger.info(f"  ✓ Loaded config/params for {problem_name} from h5")
            else:
                logger.warning(
                    f"  ✗ Could not load config/params for {problem_name}, "
                    f"config={config_json is not None}, params={params_json is not None}"
                )

    def _hydrate_like(self, template, raw_value):
        """Recursively hydrate JSON data into template-compatible structures."""
        if hasattr(template, "_fields") and isinstance(raw_value, dict):
            updates = {}
            for field in template._fields:
                if field in raw_value:
                    updates[field] = self._hydrate_like(
                        getattr(template, field), raw_value[field]
                    )
            return template._replace(**updates)

        if isinstance(template, jnp.ndarray):
            return jnp.array(raw_value, dtype=template.dtype)

        if isinstance(raw_value, list):
            return jnp.array(raw_value)

        return raw_value

    def get_problem_index_range(self, problem_name: str) -> Tuple[int, int]:
        """Get the problem index range for a problem type.

        Args:
            problem_name: Problem type (mhd_blast, turbulence, ot_vortex)

        Returns:
            (start_idx, end_idx) tuple for valid problem indices
        """
        if problem_name not in self.h5_loaders:
            raise ValueError(f"Unknown problem type: {problem_name}")

        loader = self.h5_loaders[problem_name]
        total = loader.num_problems

        if problem_name in self.problem_ranges:
            start, end = self.problem_ranges[problem_name]
            if end > total:
                logger.warning(
                    f"Problem range for {problem_name} exceeds available: "
                    f"({start}, {end}) > {total}, capping to {total}"
                )
                end = total
            return (start, min(end, total))

        return (0, total)

    def get_problem_descriptors(self) -> List[H5ProblemDescriptor]:
        """Get lightweight descriptors without materializing state tensors."""
        if self._problem_descriptors is not None:
            return self._problem_descriptors

        descriptors: List[H5ProblemDescriptor] = []
        for problem_name, loader in self.h5_loaders.items():
            if problem_name in self.param_filters:
                indices = loader.get_filtered_indices(self.param_filters[problem_name])
            else:
                start, end = self.get_problem_index_range(problem_name)
                indices = range(start, end)

            descriptors.extend(
                H5ProblemDescriptor(
                    problem_name=problem_name,
                    h5_file_path=str(loader.h5_file_path),
                    problem_index=index,
                )
                for index in indices
            )

        self._problem_descriptors = descriptors
        logger.info(f"Prepared {len(descriptors)} lazy H5 descriptors")
        return self._problem_descriptors

    def get_training_pairs_for_descriptors(
        self,
        descriptors: List[H5ProblemDescriptor],
    ) -> List[TrainingPair]:
        """Materialize training pairs for a selected descriptor batch."""
        if not descriptors:
            return []

        grouped_indices: Dict[str, List[int]] = {}
        for descriptor in descriptors:
            grouped_indices.setdefault(descriptor.problem_name, []).append(
                descriptor.problem_index
            )

        grouped_data: Dict[str, List[Tuple[jnp.ndarray, jnp.ndarray, Dict]]] = {}
        for problem_name, indices in grouped_indices.items():
            loader = self.h5_loaders[problem_name]
            grouped_data[problem_name] = loader.get_problem_batch(indices)

        grouped_positions = {problem_name: 0 for problem_name in grouped_data}
        training_pairs: List[TrainingPair] = []

        for descriptor in descriptors:
            problem_name = descriptor.problem_name
            position = grouped_positions[problem_name]
            grouped_positions[problem_name] += 1

            initial_state, final_state, hyperparams = grouped_data[problem_name][position]
            descriptor_with_hyperparams = descriptor._replace(hyperparams=hyperparams)

            pair = TrainingPair(
                problem_descriptor=descriptor_with_hyperparams,
                target_state=final_state,
                lr_bundle=self._create_lr_bundle(problem_name, initial_state),
            )
            training_pairs.append(pair)

        return training_pairs

    def iter_training_pair_batches(
        self,
        descriptors: List[H5ProblemDescriptor],
        batch_size: int,
    ) -> Iterator[List[TrainingPair]]:
        """Yield materialized training pairs in descriptor batches."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        for start in range(0, len(descriptors), batch_size):
            yield self.get_training_pairs_for_descriptors(
                descriptors[start : start + batch_size]
            )

    def get_training_pairs(self) -> List[TrainingPair]:
        """Load and cache all configured training pairs from h5 files.

        Note:
            This eagerly materializes all states and is memory-heavy for large
            datasets. Prefer `get_problem_descriptors` + `get_training_pairs_for_descriptors`
            for lazy usage.

        Returns:
            List of TrainingPair objects with states loaded from h5
        """
        if self._training_pairs is not None:
            return self._training_pairs

        self._training_pairs = self.get_training_pairs_for_descriptors(
            self.get_problem_descriptors()
        )
        logger.info(f"Loaded {len(self._training_pairs)} training pairs from h5")
        return self._training_pairs

    def _create_lr_bundle(
        self,
        problem_name: str,
        initial_state_lr,
    ) -> SimulationBundle:
        """Create a low-resolution bundle from h5-loaded state.

        Args:
            problem_name: Problem type
            initial_state_lr: Already downsampled initial state from h5

        Returns:
            SimulationBundle with loaded state and config/params
        """
        config = self.problem_configs.get(problem_name, {}).get("config")
        params = self.problem_configs.get(problem_name, {}).get("params")
        if config is None or params is None:
            raise ValueError(
                f"Missing hydrated config/params for problem '{problem_name}'"
            )

        config_lr = finalize_config(
            config._replace(num_cells=int(initial_state_lr.shape[-1])),
            initial_state_lr.shape,
        )
        reg_vars = get_registered_variables(config_lr)

        config_lr = config_lr._replace(
            return_snapshots=True,
            progress_bar=False,
            use_specific_snapshot_timepoints=True,
            num_snapshots=1,
        )
        params = params._replace(snapshot_timepoints=jnp.array([params.t_end]))

        logger.debug(
            f"Creating LR bundle for {problem_name}: "
            f"state_shape={initial_state_lr.shape}, "
            f"cells={config_lr.num_cells}, "
            f"t_end={params.t_end}"
        )

        bundle = SimulationBundle(
            initial_state=initial_state_lr,
            config=config_lr,
            params=params,
            reg_vars=reg_vars,
        )

        return bundle

    def get_problem_hyperparams(self, problem_name: str, index: int) -> Dict:
        """Get hyperparameters for a specific problem instance.

        Args:
            problem_name: Problem type
            index: Problem index

        Returns:
            Dictionary of hyperparameters
        """
        if problem_name not in self.h5_loaders:
            raise ValueError(f"Unknown problem type: {problem_name}")

        return self.h5_loaders[problem_name].get_hyperparams(index)
