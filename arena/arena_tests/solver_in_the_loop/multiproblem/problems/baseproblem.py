from abc import ABC, abstractmethod
import numpy as np
from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import STATE_TYPE
from astronomix.variable_registry.registered_variables import RegisteredVariables
import jax.numpy as jnp
from jaxtyping import Array
from typing import Tuple
import logging

logger = logging.getLogger("BaseProblem")


class BaseProblem(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def t_end(self) -> float:
        pass

    @abstractmethod
    def generate_initial_state(
        self, config: SimulationConfig, params: SimulationParams
    ) -> Tuple[STATE_TYPE, SimulationConfig, SimulationParams, RegisteredVariables]:
        pass

    def generate_initial_state_with_config_overrides(
        self, config: SimulationConfig, params: SimulationParams, config_overrides: dict
    ) -> Tuple[STATE_TYPE, SimulationConfig, SimulationParams, RegisteredVariables]:
        initial_state, config, params, registered_variables = (
            self.generate_initial_state(config, params)
        )
        # NOTE: maybe should split this two, i think there are no overlapping names
        # but still this is pretty bad implementation
        config_update = {
            k: v for k, v in config_overrides.items() if k in config._fields
        }
        params_update = {
            k: v for k, v in config_overrides.items() if k in params._fields
        }
        config = config._replace(**config_update)
        params = params._replace(**params_update)
        return (initial_state, config, params, registered_variables)

    @abstractmethod
    def get_hyperparams(self) -> dict:
        return {"problem_name": "base_problem", "params": {}}

    def get_config_overrides_evaluation(self, snapshot_timepoints: Array) -> dict:
        config_overrides = {
            "progress_bar": True,
            "return_snapshots": True,
            "use_specific_snapshot_timepoints": True,
            "snapshot_timepoints": snapshot_timepoints,
            "num_snapshots": len(snapshot_timepoints),
            "t_end": snapshot_timepoints[-1],
        }

        return config_overrides

    @property
    def filename(self) -> str:
        def _fmt(v) -> str:
            # Collapse arrays/lists to element-separated string
            if isinstance(v, (list, tuple)):
                return "x".join(_fmt(x) for x in v)
            try:
                import jax.numpy as jnp

                if isinstance(v, jnp.ndarray):
                    v = np.asarray(v)
            except ImportError:
                pass
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    return _fmt(v.item())
                return "x".join(_fmt(x) for x in v.tolist())
            return str(v).replace(".", "p")

        hyperparams = self.get_hyperparams()
        parts = [self.name]
        for k, v in hyperparams.get("params", {}).items():
            parts.append(f"{k}_{_fmt(v)}")
        return "_".join(parts)
