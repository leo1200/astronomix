"""
astronomix: a differentiable finite-volume / finite-difference (magneto)hydrodynamics code.

This top-level package re-exports the most commonly used entry points — the
configuration and parameter containers, the variable registry, the initial
condition helpers and the ``time_integration`` driver — so that user scripts can
import them directly from ``astronomix``.
"""

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FORWARDS,
    BACKWARDS,
    MINMOD,
    OSHER,
    HLL,
    HLLC,
    HLLC_LM,
    OPEN_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    PERIODIC_BOUNDARY,
    CARTESIAN,
    CYLINDRICAL,
    SPHERICAL,
    ON_DEVICE,
    TO_DISK,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix._modules._stellar_wind.stellar_wind_options import WindParams
from astronomix.units import CodeUnits

# astronomix functions
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix.time_stepping.time_integration import time_integration
from astronomix.setup_helpers import restart_from_latest_checkpoint
