from typing import Callable, List, NamedTuple, Optional, Tuple
import random

from jf1uids import time_integration
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData
from jf1uids.variable_registry.registered_variables import RegisteredVariables

from jf1uids import (
    SimulationConfig,
    SimulationParams,
    finalize_config,
    get_helper_data,
    get_registered_variables,
)

from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    FINITE_DIFFERENCE,
    STATE_TYPE,
)

from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)

from arena.arena_tests.solver_in_the_loop.model_manager import (
    SimulationConfigTraining,
    TrainingConfig,
)
from arena.arena_tests.solver_in_the_loop.utils import downaverage

import jax.numpy as jnp
from jaxtyping import Array
import logging
from dataclasses import dataclass

from arena.arena_tests.solver_in_the_loop.timepoint_updater import (
    BACK_TO_FRONT,
    FRONT_TO_BACK,
)

# Problem IC functions — each new problem file adds one to this dict
from arena.arena_tests.solver_in_the_loop.multiproblem.problems.mhd_blast import (
    initial_state_mhd_blast,
)

logger = logging.getLogger(__name__)

# Type alias for an IC function: (config, params) -> (state, config, params, helper, reg_vars)
ICFunction = Callable[
    [SimulationConfig, SimulationParams],
    Tuple[
        STATE_TYPE, SimulationConfig, SimulationParams, HelperData, RegisteredVariables
    ],
]

# Add new problems here
PROBLEM_CATALOG: dict[str, ICFunction] = {
    "mhd_blast": initial_state_mhd_blast,
}


@dataclass
class SimulationBundle:
    initial_state: Array
    config: SimulationConfig
    params: SimulationParams
    helper: HelperData
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

    def unpack_integrate(self):
        kwargs = {
            "primitive_state": self.initial_state,
            "config": self.config,
            "params": self.params,
            "helper_data": self.helper,
            "registered_variables": self.reg_vars,
        }
        return kwargs

    def copy(self):
        """Create a shallow copy of the bundle."""
        return SimulationBundle(
            self.initial_state.copy(),
            self.config,
            self.params,
            self.helper,
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
        helper_lr = get_helper_data(config_lr)
        reg_vars_lr = get_registered_variables(config_lr)
        params_lr = self.params
        return SimulationBundle(
            initial_state_lr,
            config_lr,
            params_lr,
            helper_lr,
            reg_vars_lr,
        )


class TrainingPair(NamedTuple):
    """A precomputed (target, lr_bundle) pair ready for training."""

    problem_name: str
    target_states: Array  # downsampled HR snapshots
    lr_bundle: SimulationBundle


def _build_hr_config_and_params(
    sim_config_training: SimulationConfigTraining,
    training_config: TrainingConfig,
) -> Tuple[SimulationConfig, SimulationParams]:
    """Build the HR simulation config and params from training configs."""
    snapshot_timepoints = jnp.array(training_config.snapshot_timepoints_train)
    if training_config.direction == BACK_TO_FRONT:
        snapshot_timepoints = jnp.sort(snapshot_timepoints)
        if not jnp.any(snapshot_timepoints == sim_config_training.t_end):
            snapshot_timepoints = jnp.append(
                snapshot_timepoints, sim_config_training.t_end
            )
    elif training_config.direction == FRONT_TO_BACK:
        assert isinstance(sim_config_training.t_end, float)
        snapshot_timepoints = jnp.array([sim_config_training.t_end])
    else:
        raise ValueError("The direction given doesnt exist")

    logger.debug(f"snapshot timepoints {snapshot_timepoints}")
    simulation_params = SimulationParams(
        C_cfl=sim_config_training.c_cfl,
        t_end=sim_config_training.t_end,
        snapshot_timepoints=snapshot_timepoints,
    )

    simulation_config = SimulationConfig(
        num_cells=sim_config_training.num_cells_high_res,
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
        use_specific_snapshot_timepoints=True,
        return_snapshots=True,
        num_snapshots=len(snapshot_timepoints),
        num_checkpoints=100,
        progress_bar=True,
        runtime_debugging=False,
        limiter=sim_config_training.limiter,
    )

    return simulation_config, simulation_params


class ProblemManager:
    """Manages multiple IC problems and produces training pairs.

    Usage:
        pm = ProblemManager(
            problem_names=["mhd_blast"],
            sim_config_training=sim_config_training,
            training_config=training_config,
        )
        training_pairs = pm.get_training_pairs()
        # training_pairs is a list of TrainingPair(problem_name, target_states, lr_bundle)

        # pick one at random during training:
        pair = pm.sample_training_pair()
    """

    def __init__(
        self,
        problem_names: List[str],
        sim_config_training: SimulationConfigTraining,
        training_config: TrainingConfig,
    ):
        for name in problem_names:
            if name not in PROBLEM_CATALOG:
                raise ValueError(
                    f"Unknown problem '{name}'. Available: {list(PROBLEM_CATALOG.keys())}"
                )

        self.problem_names = problem_names
        self.sim_config_training = sim_config_training
        self.training_config = training_config

        self.hr_config, self.hr_params = _build_hr_config_and_params(
            sim_config_training, training_config
        )

        self._training_pairs: Optional[List[TrainingPair]] = None

    @staticmethod
    def available_problems() -> List[str]:
        return list(PROBLEM_CATALOG.keys())

    def _build_training_pair(self, problem_name: str) -> TrainingPair:
        """Build one (target, lr_bundle) pair for a given problem."""
        ic_fn = PROBLEM_CATALOG[problem_name]
        state_tuple = ic_fn(self.hr_config, self.hr_params)
        hr_bundle = SimulationBundle(*state_tuple)

        # Run HR simulation to get target snapshots
        logger.info(f"Running HR simulation for '{problem_name}'...")
        target_snapshot_data = time_integration(**hr_bundle.unpack_integrate())
        assert isinstance(target_snapshot_data, SnapshotData)

        # Downsample targets
        target_states_lr = downaverage(
            target_snapshot_data.states,
            downaverage_factor=self.sim_config_training.downaverage_factor,
        )

        # Build LR bundle (no corrector yet — that's attached during training)
        lr_bundle = hr_bundle.convert_to_lr(
            downaverage_factor=self.sim_config_training.downaverage_factor
        )

        # Advance LR initial state to start_correction_time if not correcting from the beginning
        if not self.sim_config_training.correct_from_beggining:
            t_corr = self.sim_config_training.start_correction_time
            logger.info(
                f"Advancing '{problem_name}' LR initial state to t={t_corr}"
            )
            lr_bundle.initial_state = time_integration(
                primitive_state=lr_bundle.initial_state,
                config=lr_bundle.config._replace(
                    return_snapshots=False,
                    progress_bar=True,
                    exact_end_time=True,
                ),
                params=lr_bundle.params._replace(t_end=t_corr),
                helper_data=lr_bundle.helper,
                registered_variables=lr_bundle.reg_vars,
            )
        else:
            logger.info(f"Using '{problem_name}' LR state from beginning of simulation")

        lr_bundle.override_config(
            return_snapshots=True,
            use_specific_snapshot_timepoints=True,
            progress_bar=False,
        )

        logger.info(f"Training pair ready for '{problem_name}'")
        return TrainingPair(
            problem_name=problem_name,
            target_states=target_states_lr,
            lr_bundle=lr_bundle,
        )

    def get_training_pairs(self) -> List[TrainingPair]:
        """Compute and cache all training pairs."""
        if self._training_pairs is None:
            self._training_pairs = [
                self._build_training_pair(name) for name in self.problem_names
            ]
        return self._training_pairs

    def sample_training_pair(self) -> TrainingPair:
        """Return a random training pair."""
        pairs = self.get_training_pairs()
        return random.choice(pairs)
