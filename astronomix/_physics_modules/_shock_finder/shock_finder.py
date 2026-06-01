from functools import partial

import jax
import jax.numpy as jnp

from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)

from astronomix._physics_modules._shock_finder._data_structures import ShockFinderResult
from astronomix._physics_modules._shock_finder._shock_direction import _calculate_shock_direction
from astronomix._physics_modules._shock_finder._shock_zones import identify_shock_zones
from astronomix._physics_modules._shock_finder._shock_surface import identify_shock_surface
from astronomix._physics_modules._shock_finder._mach import _calculate_mach_at_surface

##### public entry point ######
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def find_shocks_pfrommer(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    mach_min: float = 1.3
) -> ShockFinderResult:
    """
    Main entry point: Identify shocks using Pfrommer et al. 2017 methodology.
    
    This orchestrates all phases:
    1. Calculate shock direction
    2. Identify shock zones (3-4 cells thick)
    3. Refine to shock surface (single layer)
    4. Calculate Mach numbers
    5. Return structured result
    
    Args:
        primitive_state: Primitive state variables [num_variables, num_cells]
        config: Simulation configuration
        registered_variables: Registry of variable indices
        helper_data: Helper data (geometric centers, etc.)
        mach_min: Minimum Mach number threshold (default 1.3)
    
    Returns:
        ShockFinderResult with shock metrics
    """
    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    r = helper_data.geometric_centers if config.geometry == SPHERICAL else None
    
    # Phase 1: Calculate shock direction
    shock_direction = _calculate_shock_direction(pressure, density, config, r)
    
    # Phase 2: Identify shock zones
    shock_zones = identify_shock_zones(
        primitive_state, config, registered_variables,
        helper_data, shock_direction, mach_min
    )
    
    # Phase 3: Identify shock surface
    shock_surface = identify_shock_surface(
        primitive_state, shock_zones, shock_direction,
        config, registered_variables
    )
    
    # Phase 4: Calculate Mach numbers at surface
    mach_numbers = _calculate_mach_at_surface(
        primitive_state, shock_surface, shock_direction,
        config, registered_variables
    )
    
    # num_shocks counts distinct surface cells
    num_shocks = jnp.sum(shock_surface, dtype=jnp.int32)
    
    # Shock IDs: for now, single label per cell on surface
    shock_ids = jnp.where(shock_surface, 1, 0)
    shock_zone_ids = jnp.where(shock_zones, 1, 0)
    
    return ShockFinderResult(
        shock_surface_cells=shock_surface,
        shock_direction=shock_direction,
        mach_numbers=mach_numbers,
        shock_zones=shock_zones,
        num_shocks=num_shocks,
        shock_ids=shock_ids,
        shock_zone_ids=shock_zone_ids
    )