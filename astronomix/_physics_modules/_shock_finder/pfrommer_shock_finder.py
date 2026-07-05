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
from astronomix._physics_modules._shock_finder._gradients import _calculate_shock_direction
from astronomix._physics_modules._shock_finder._shock_zones import identify_shock_zones
from astronomix._physics_modules._shock_finder._shock_surface import identify_shock_surface
from astronomix._physics_modules._shock_finder._shock_mach import _calculate_mach_at_surface
from astronomix._physics_modules._shock_finder._energy_dissipation import calculate_thermal_energy_flux

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def find_shocks_pfrommer(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    mach_min: float = 1.3,
) -> ShockFinderResult:
    """
    Main entry point: Identify shocks using Pfrommer et al. 2017 methodology.

    Phases:
        1. Shock direction:  d_s = -∇T / |∇T|,  shape (ndim, *spatial_shape)
        2. Shock zones:      cells satisfying all three criteria (~3-4 cells thick)
        3. Shock surface:    single cell of max compression per zone
        4. Mach numbers:     Rankine-Hugoniot M at surface cells

    Works for 1D, 2D, and 3D without modification.

    Args:
        primitive_state:      (num_vars, *spatial_shape)
        config:               simulation configuration
        registered_variables: registry of variable indices
        helper_data:          geometric centers etc.
        mach_min:             minimum Mach threshold (default 1.3)

    Returns:
        ShockFinderResult
    """
    pressure = primitive_state[registered_variables.pressure_index]
    density  = primitive_state[registered_variables.density_index]
    r = helper_data.geometric_centers if config.geometry == SPHERICAL else None

    # Phase 1: shock direction (ndim, *spatial_shape)
    shock_direction = _calculate_shock_direction(pressure, density, config, r)

    # Phase 2: shock zones (*spatial_shape)
    shock_zones = identify_shock_zones(
        primitive_state, config, registered_variables,
        helper_data, shock_direction, mach_min,
    )

    # Phase 3: shock surface (*spatial_shape)
    shock_surface = identify_shock_surface(
        primitive_state, shock_zones, shock_direction,
        config, registered_variables,
    )

    # Phase 4: Mach numbers (*spatial_shape)
    mach_numbers = _calculate_mach_at_surface(
        primitive_state, shock_surface, shock_direction,
        config, registered_variables,
    )
    
    # Phase 5: thermal-energy flux at shock-surface cells
    thermal_energy_flux = calculate_thermal_energy_flux(
        primitive_state=primitive_state,
        shock_surface=shock_surface,
        shock_direction=shock_direction,
        mach_numbers=mach_numbers,
        config=config,
        registered_variables=registered_variables,
    )

    num_shocks     = jnp.sum(shock_surface, dtype=jnp.int32)
    shock_ids      = jnp.where(shock_surface, 1, 0)
    shock_zone_ids = jnp.where(shock_zones,   1, 0)

    return ShockFinderResult(
        shock_surface_cells=shock_surface,
        shock_direction=shock_direction,
        mach_numbers=mach_numbers,
        thermal_energy_flux=thermal_energy_flux,
        shock_zones=shock_zones,
        num_shocks=num_shocks,
        shock_ids=shock_ids,
        shock_zone_ids=shock_zone_ids,
    )