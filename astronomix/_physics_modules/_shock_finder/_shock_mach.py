from functools import partial

import jax
import jax.numpy as jnp

from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    BOOL_FIELD_TYPE,
    FIELD_TYPE,
    STATE_TYPE,
    SimulationConfig,
)
from astronomix._physics_modules._shock_finder._shock_zones import get_post_pre_shock_values

"""
Calculate Mach number for all cells, 
but only keep it at the shock surface (where shock_surface is True) via filter
* To do this to shoot follow shock direction through all cells -> get post/pre values at all cells
* Then apply some equation to get Mach number for all cells from p_post/p_pre
* then filter to keep only Mach for surface cells
"""
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _calculate_mach_at_surface(
    primitive_state: STATE_TYPE,
    shock_surface: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,        # ← needed for direction-aware p_post/p_pre
    config: SimulationConfig,
    registered_variables: RegisteredVariables
) -> FIELD_TYPE:
    gamma_gas = 5 / 3

    pressure = primitive_state[registered_variables.pressure_index]
    density  = primitive_state[registered_variables.density_index]
    temperature = pressure / density

    # direction-aware post/pre selection — same helper as criterion 3
    p_post, p_pre, _, _ = get_post_pre_shock_values(
        shock_direction, pressure, temperature,
        max_steps=8
    )
    
    # calculate Mach number for all cells
    # p₂/p₁ = p_post/p_pre, but clamp to 1 to avoid numerical issues with very weak shocks
    p_ratio = jnp.maximum(p_post / jnp.maximum(p_pre, 1e-30), 1.0)
    # as p₂/p₁ = (2γM² − (γ−1)) / (γ+1) so M = √[ (p₂/p₁ · (γ+1) + (γ−1)) / (2γ) ]
    M = jnp.sqrt((p_ratio * (gamma_gas + 1) + (gamma_gas - 1)) / (2 * gamma_gas))

    # write Mach only at surface cells, zero elsewhere
    mach_array = jnp.where(shock_surface, M, 0.0)

    return mach_array