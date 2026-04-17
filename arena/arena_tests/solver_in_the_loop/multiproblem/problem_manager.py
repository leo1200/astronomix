from typing import Callable, List, NamedTuple, Optional, Tuple
import os
from pathlib import Path
import random

from arena.arena_tests.solver_in_the_loop.multiproblem.problems.baseproblem import (
    BaseProblem,
)
from astronomix import time_integration
from astronomix.data_classes.simulation_snapshot_data import SnapshotData
from astronomix.variable_registry.registered_variables import RegisteredVariables

from astronomix import (
    SimulationConfig,
    SimulationParams,
    finalize_config,
    get_registered_variables,
)

from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    FINITE_DIFFERENCE,
    STATE_TYPE,
)

from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)

from arena.arena_tests.solver_in_the_loop.model_manager import (
    TrainingConfig,
)
from arena.arena_tests.solver_in_the_loop.utils import downaverage

import jax.numpy as jnp
from jaxtyping import Array
import logging
from dataclasses import dataclass, field, replace as _dc_replace

import json

# Problem IC functions — each new problem file adds one to this dict
from arena.arena_tests.solver_in_the_loop.multiproblem.problems.mhd_blast import (
    MhdBlast,
)

from arena.arena_tests.solver_in_the_loop.multiproblem.problems.ot_vortex_3d import (
    OtVortex,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.problems.turbulence import (
    Turbulence,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.problems.turbulent_blast import (
    TurbulentBlast,
)
import numpy as _np

MULTIPROBLEM_BASE_DIR = "arena/data/models/multiproblem_w_dataset"


def _serialize_value(v):
    """Recursively convert JAX/numpy arrays (and nested containers) to JSON-safe types."""
    if isinstance(v, (jnp.ndarray, _np.ndarray)):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    return v


def _deserialize_value(v):
    """Recursively convert lists back to jnp arrays inside nested containers."""
    if isinstance(v, list):
        return jnp.array(v)
    if isinstance(v, dict):
        return {k: _deserialize_value(val) for k, val in v.items()}
    return v


logger = logging.getLogger(__name__)

# Type alias for an IC function: (config, params) -> (state, config, params, reg_vars)
ICFunction = Callable[
    [SimulationConfig, SimulationParams],
    Tuple[STATE_TYPE, SimulationConfig, SimulationParams, RegisteredVariables],
]

# Add new problems here
PROBLEM_CATALOG: dict[str, type[BaseProblem]] = {
    "mhd_blast": MhdBlast,
    "ot_vortex": OtVortex,
    "turbulence": Turbulence,
    "turbulent_blast": TurbulentBlast,
}

PROBLEM_DICTIONARY: dict[str, dict[str, float]] = {
    problem: {"t_end": PROBLEM_CATALOG[problem]().t_end} for problem in PROBLEM_CATALOG
}


@dataclass
class SimulationBundle:
    initial_state: STATE_TYPE
    config: SimulationConfig
    params: SimulationParams
    reg_vars: RegisteredVariables

    def override_solver_in_the_loop(
        self,
        corrector_config: CNNMHDconfig,
        corrector_params: CNNMHDParams,
    ):
        self.config = self.config._replace(cnn_mhd_corrector_config=corrector_config)
        self.params = self.params._replace(cnn_mhd_corrector_params=corrector_params)
        return self

    def override_config(self, strict=False, **overrides):
        valid_fields = self.config._fields
        valid, invalid = {}, []

        for k, v in overrides.items():
            if k in valid_fields:
                valid[k] = v
            else:
                invalid.append(k)

        if invalid:
            msg = f"Invalid config keys: {invalid}"
            if strict:
                raise KeyError(msg)
            else:
                print(f"[override_config] Warning: {msg}")

        self.config = self.config._replace(**valid)
        return self

    def override_params(self, strict=False, **overrides):
        valid_fields = self.params._fields
        valid, invalid = {}, []

        for k, v in overrides.items():
            if k in valid_fields:
                valid[k] = v
            else:
                invalid.append(k)

        if invalid:
            msg = f"Invalid config keys: {invalid}"
            if strict:
                raise KeyError(msg)
            else:
                print(f"[override_config] Warning: {msg}")

        self.params = self.params._replace(**valid)
        return self

    def unpack_integrate(self):
        kwargs = {
            "primitive_state": self.initial_state,
            "config": self.config,
            "params": self.params,
            "registered_variables": self.reg_vars,
        }
        return kwargs

    def copy(self):
        """Create a shallow copy of the bundle."""
        return SimulationBundle(
            self.initial_state.copy(),
            self.config,
            self.params,
            self.reg_vars,
        )

    def convert_to_lr(self, downaverage_factor: int):
        initial_state_lr = downaverage(
            state=self.initial_state, downaverage_factor=downaverage_factor
        )
        config_lr = finalize_config(
            self.config._replace(num_cells=self.config.num_cells // downaverage_factor),
            initial_state_lr.shape,
        )
        reg_vars_lr = get_registered_variables(config_lr)
        params_lr = self.params
        return SimulationBundle(
            initial_state_lr,
            config_lr,
            params_lr,
            reg_vars_lr,
        )


# NOTE: thinking about it a bit more in depth i think we should get rid ofthis
# and add everyhing to the problem class
@dataclass(frozen=True)
class ProblemDescriptor:
    name: str
    params: dict = field(default_factory=dict)
    config_overrides: dict = field(default_factory=dict)
    # Auto-computed from problem.filename when not provided.
    nickname: str = field(default="")

    def __post_init__(self):
        if not self.nickname:
            object.__setattr__(
                self, "nickname", PROBLEM_CATALOG[self.name](**self.params).filename
            )

    def _asdict(self) -> dict:
        return {
            "name": self.name,
            "params": self.params,
            "config_overrides": self.config_overrides,
            "nickname": self.nickname,
        }

    def _replace(self, **kwargs) -> "ProblemDescriptor":
        return _dc_replace(self, **kwargs)


class TrainingPair(NamedTuple):
    """A precomputed (target, lr_bundle) pair ready for training."""

    problem_descriptor: ProblemDescriptor
    target_state: Array  # downsampled HR snapshots
    lr_bundle: SimulationBundle


class EvaluationResults(NamedTuple):
    problem_descriptor: ProblemDescriptor
    hr_snapshot_data: SnapshotData
    lr_snapshot_data: Optional[SnapshotData]
    lr_sol_snapshot_data: Optional[SnapshotData]


class HREvaluationError(Exception):
    pass


def _build_hr_config_and_params(
    training_config: TrainingConfig,
) -> Tuple[SimulationConfig, SimulationParams]:
    """Build the HR simulation config and params from training configs."""
    # TODO: integrating this with the multiproblem setup has several complications
    # so far we are going to just use one timepoint to train which corresponds with the t_end
    # the main problem comes from the fact that the t_end depends on the problem setup
    # for some the interesting physics appear till 0.2 and for others its 6.2
    #
    # snapshot_timepoints = jnp.array(training_config.snapshot_timepoints_train)
    # if training_config.direction == BACK_TO_FRONT:
    #     snapshot_timepoints = jnp.sort(snapshot_timepoints)
    #     if not jnp.any(snapshot_timepoints == sim_config_training.t_end):
    #         snapshot_timepoints = jnp.append(
    #             snapshot_timepoints, sim_config_training.t_end
    #         )
    # elif training_config.direction == FRONT_TO_BACK:
    #     assert isinstance(sim_config_training.t_end, float)
    #     snapshot_timepoints = jnp.array([sim_config_training.t_end])
    # else:
    #     raise ValueError("The direction given doesnt exist")
    #
    # logger.debug(f"snapshot timepoints {snapshot_timepoints}")
    simulation_params = SimulationParams(
        C_cfl=training_config.c_cfl,
        # t_end=sim_config_training.t_end, handled by the problems
        # snapshot_timepoints=snapshot_timepoints,
    )

    simulation_config = SimulationConfig(
        num_cells=training_config.num_cells_high_res,
        solver_mode=FINITE_DIFFERENCE,
        differentiation_mode=BACKWARDS,
        mhd=True,
        dimensionality=3,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ),
        return_snapshots=False,
        # num_snapshots=len(snapshot_timepoints),
        num_checkpoints=100,
        progress_bar=True,
        runtime_debugging=False,
        limiter=training_config.limiter,
    )

    return simulation_config, simulation_params


def load_problem_descriptors(
    model_name: str, base_dir: str = "arena/data/models/multiproblem_w_dataset"
):
    path = Path(base_dir) / model_name / "problem_descriptors.json"

    with open(path, "r") as f:
        problem_descriptor_list = json.load(f)

    problem_descriptors = []
    for p_descriptor in problem_descriptor_list:
        deserialized = {k: _deserialize_value(v) for k, v in p_descriptor.items()}
        problem_descriptors.append(ProblemDescriptor(**deserialized))

    logger.info(f"problem descriptors loaded from {path}")
    return problem_descriptors


class ProblemManager:
    """Manages multiple IC problems and produces training pairs.

    Usage:
        pm = ProblemManager(
            names=["mhd_blast"],
            sim_config_training=sim_config_training,
            training_config=training_config,
        )
        training_pairs = pm.get_training_pairs()
        # training_pairs is a list of TrainingPair(name, target_states, lr_bundle)

        # pick one at random during training:
        pair = pm.sample_training_pair()
    """

    def __init__(
        self,
        problem_descriptors: List[ProblemDescriptor],
        training_config: TrainingConfig,
    ):
        for p_descriptor in problem_descriptors:
            if p_descriptor.name not in PROBLEM_CATALOG:
                raise ValueError(
                    f"Unknown problem '{p_descriptor.name}'. Available: {list(PROBLEM_CATALOG.keys())}"
                )

        self.problem_descriptors = problem_descriptors
        self.training_config = training_config

        self.hr_config, self.hr_params = _build_hr_config_and_params(training_config)

        self._training_pairs: Optional[List[TrainingPair]] = None

    def save_problem_descriptors(
        self, model_name: str, base_dir: Path = Path("arena/data/models/multiproblem_w_dataset")
    ):
        problem_descriptor_list = []
        for p_descriptor in self.problem_descriptors:
            raw = p_descriptor._asdict()
            problem_descriptor_list.append(
                {k: _serialize_value(v) for k, v in raw.items()}
            )

        path = base_dir / model_name / "problem_descriptors.json"

        with open(path, "w") as f:
            json.dump(problem_descriptor_list, f, indent=2)

        logger.info(f"problem descriptors saved to {path}")

    def load_problem_descriptors(
        self, model_name: str, base_dir: str = "arena/data/models/multiproblem_w_dataset"
    ):
        path = Path(base_dir) / model_name / "problem_descriptors.json"

        with open(path, "r") as f:
            problem_descriptor_list = json.load(f)

        self.problem_descriptors = []
        for p_descriptor in problem_descriptor_list:
            deserialized = {k: _deserialize_value(v) for k, v in p_descriptor.items()}
            self.problem_descriptors.append(ProblemDescriptor(**deserialized))

        logger.info(f"problem descriptors loaded from {path}")

    @staticmethod
    def available_problems() -> List[str]:
        return list(PROBLEM_CATALOG.keys())

    # TODO: bring the saving and loading logic from the utils.py file
    def _build_training_pair(
        self, problem_descriptor: ProblemDescriptor
    ) -> TrainingPair:
        """Build one (target, lr_bundle) pair for a given problem."""
        problem = PROBLEM_CATALOG[problem_descriptor.name](**problem_descriptor.params)
        if problem_descriptor.config_overrides == {}:
            state_tuple = problem.generate_initial_state(
                config=self.hr_config, params=self.hr_params
            )
        else:
            state_tuple = problem.generate_initial_state_with_config_overrides(
                config=self.hr_config,
                params=self.hr_params,
                config_overrides=problem_descriptor.config_overrides,
            )
        hr_bundle = SimulationBundle(*state_tuple)

        filename = (
            problem.filename
            + f"_{self.hr_config.num_cells}_{self.training_config.downaverage_factor}"
            + ".npy"
        )
        filepath = f"arena/data/{filename}"
        # Run HR simulation to get target snapshots
        if not os.path.exists(filepath):
            logger.info(f"Running HR simulation for '{problem.name}'...")
            final_target_state = time_integration(**hr_bundle.unpack_integrate())
            assert isinstance(final_target_state, Array)
            jnp.save(filepath, final_target_state)
            logger.info(f"Saved HR simulation in '{filepath}'")
        else:
            final_target_state = jnp.load(filepath)
            logger.info(f"Loaded HR simulation from '{filepath}'")

        # Downsample targets
        final_target_state = downaverage(
            final_target_state,
            downaverage_factor=self.training_config.downaverage_factor,
        )

        # Build LR bundle (no corrector yet — that's attached during training)
        lr_bundle = hr_bundle.convert_to_lr(
            downaverage_factor=self.training_config.downaverage_factor
        )

        # TODO: Same as with the multisnapshot training changing the t_corr is not
        # straightforward for the multiproblem setup
        #
        # Advance LR initial state to start_correction_time if not correcting from the beginning
        # if not self.sim_config_training.correct_from_beggining:
        #     t_corr = self.sim_config_training.start_correction_time
        #     logger.info(f"Advancing '{problem.name}' LR initial state to t={t_corr}")
        #     initial_state_lr = time_integration(
        #         primitive_state=lr_bundle.initial_state,
        #         config=lr_bundle.config._replace(
        #             return_snapshots=False,
        #             progress_bar=True,
        #             exact_end_time=True,
        #         ),
        #         params=lr_bundle.params._replace(t_end=t_corr),
        #         registered_variables=lr_bundle.reg_vars,
        #     )
        #     assert isinstance(initial_state_lr, STATE_TYPE)
        #     lr_bundle.initial_state = initial_state_lr
        # else:
        #     logger.info(f"Using {problem.name} LR state from beginning of simulation")
        #
        #
        lr_bundle.override_config(
            return_snapshots=True,
            progress_bar=False,
            use_specific_snapshot_timepoints=True,
            num_snapshots=1,
        )
        lr_bundle.override_params(
            snapshot_timepoints=jnp.array([hr_bundle.params.t_end])
        )

        logger.info(f"Training pair ready for '{problem.name}'")
        return TrainingPair(
            problem_descriptor=problem_descriptor,
            target_state=final_target_state,
            lr_bundle=lr_bundle,
        )

    def get_training_pairs(self) -> List[TrainingPair]:
        """Compute and cache all training pairs."""
        if self._training_pairs is None:
            self._training_pairs = []
            for p_descriptor in self.problem_descriptors:
                self._training_pairs.append(
                    self._build_training_pair(problem_descriptor=p_descriptor)
                )
        return self._training_pairs

    def _build_evaluation_data(
        self,
        problem: BaseProblem,
        corrector_config: CNNMHDconfig,
        corrector_params: CNNMHDParams,
        snapshot_timepoints: Optional[Array],
        step_callbacks: Optional[dict[str, Callable[..., None]]] = None,
    ) -> Tuple[SnapshotData, Optional[SnapshotData], Optional[SnapshotData]]:
        """Compute the hr, lr and sol_lr target data for a given problem with default config override"""
        # NOTE: Right now the config overrides are pretty hardcoded, i dont think ill need more than this
        # but it could expanded to depending on the problem and more
        if snapshot_timepoints is None:
            snapshot_timepoints = jnp.array([problem.t_end])
        config_overrides = problem.get_config_overrides_evaluation(snapshot_timepoints)

        # NOTE: This should be handled outside of this function, this is plot depending
        #
        # snapshot_tps = config_overrides["snapshot_timepoints"]
        # if not jnp.any(jnp.isclose(snapshot_tps, problem.t_end)):
        #     snapshot_tps = jnp.sort(
        #         jnp.append(
        #             snapshot_tps, jnp.array(problem.t_end, dtype=snapshot_tps.dtype)
        #         )
        #     )
        #     config_overrides = {
        #         **config_overrides,
        #         "snapshot_timepoints": snapshot_tps,
        #         "num_snapshots": int(len(snapshot_tps)),
        #     }
        #

        state_tuple = problem.generate_initial_state_with_config_overrides(
            config=self.hr_config,
            params=self.hr_params,
            config_overrides=config_overrides,
        )
        hr_bundle = SimulationBundle(*state_tuple)
        step_callbacks = step_callbacks or {}
        hr_step_callable = step_callbacks.get("hr")
        lr_step_callable = step_callbacks.get("lr")
        lr_sol_step_callable = step_callbacks.get("lr_sol")

        # Run HR simulation to get target snapshots
        logger.info(f"Running HR simulation for '{problem.name}'...")
        logger.info(
            f"Return snapshots {hr_bundle.config.return_snapshots} / Specific snapshots {hr_bundle.config.use_specific_snapshot_timepoints}"
        )
        try:
            hr_bundle.override_config(
                activate_snapshot_callback=hr_step_callable is not None
            )
            hr_snapshot_data = time_integration(
                **hr_bundle.unpack_integrate(), step_callable=hr_step_callable
            )
        except Exception as exc:
            raise HREvaluationError(
                f"HR simulation failed for '{problem.name}' with params {problem.get_hyperparams().get('params', {})}"
            ) from exc

        # Build LR bundle (no corrector yet — that's attached during training)
        lr_bundle = hr_bundle.convert_to_lr(
            downaverage_factor=self.training_config.downaverage_factor
        )

        logger.info(f"Running LR simulation for '{problem.name}'...")
        try:
            lr_bundle.override_config(
                progress_bar=False,
                activate_snapshot_callback=lr_step_callable is not None,
            )
            lr_snapshot_data = time_integration(
                **lr_bundle.unpack_integrate(), step_callable=lr_step_callable
            )
        except Exception:
            logger.warning(
                "LR simulation failed for '%s' with params %s; storing sentinel for plotting.",
                problem.name,
                problem.get_hyperparams().get("params", {}),
            )
            lr_snapshot_data = None

        logger.info(f"Running LR-SOL simulation for '{problem.name}'...")
        lr_bundle.override_solver_in_the_loop(
            corrector_config=corrector_config, corrector_params=corrector_params
        )
        try:
            lr_bundle.override_config(
                activate_snapshot_callback=lr_sol_step_callable is not None
            )
            lr_sol_snapshot_data = time_integration(
                **lr_bundle.unpack_integrate(), step_callable=lr_sol_step_callable
            )
            print(
                "lr_sol_snapshot_data final time ", lr_sol_snapshot_data.time_points[-1]
            )
        except Exception:
            lr_sol_snapshot_data = None

        assert isinstance(hr_snapshot_data, SnapshotData)
        assert isinstance(lr_snapshot_data, SnapshotData | None)
        assert isinstance(lr_sol_snapshot_data, SnapshotData | None)
        return (hr_snapshot_data, lr_snapshot_data, lr_sol_snapshot_data)

    # TODO: The snapshot_timepoints should be a per problem basis, could be a dict to begin with
    def get_evaluation_data(
        self,
        corrector_config: CNNMHDconfig,
        corrector_params: CNNMHDParams,
        snapshot_timepoints: Optional[Array],
        step_callbacks: Optional[dict[str, Callable[..., None]]] = None,
    ) -> list[EvaluationResults]:
        """Compute and cache all training pairs."""
        evaluation_data = []
        for p_descriptor in self.problem_descriptors:
            logger.info(
                f"computing problem {p_descriptor.name} with params {p_descriptor.params}"
            )
            problem = PROBLEM_CATALOG[p_descriptor.name](**p_descriptor.params)
            try:
                snapshot_data = self._build_evaluation_data(
                    problem,
                    corrector_config=corrector_config,
                    corrector_params=corrector_params,
                    snapshot_timepoints=snapshot_timepoints,
                    step_callbacks=step_callbacks,
                )
            except HREvaluationError:
                logger.warning(
                    "Skipping descriptor for '%s' with params %s due to HR failure.",
                    p_descriptor.name,
                    p_descriptor.params,
                )
                continue

            evaluation_data.append(
                EvaluationResults(
                    problem_descriptor=p_descriptor,
                    hr_snapshot_data=snapshot_data[0],
                    lr_snapshot_data=snapshot_data[1],
                    lr_sol_snapshot_data=snapshot_data[2],
                )
            )
        return evaluation_data

    def sample_training_pair(self) -> TrainingPair:
        """Return a random training pair."""
        pairs = self.get_training_pairs()
        return random.choice(pairs)
