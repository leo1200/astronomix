import jax
from functools import partial
from astronomix._physics_modules._shock_finder._helpers import (
    _calculate_temperature_gradient,
    _normalize_vector,
)
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    SimulationConfig
)



@partial(jax.jit, static_argnames=["config"])
def _calculate_shock_direction(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None
) -> FIELD_TYPE:
    """
    Calculate the shock direction vector at each cell.
    Shock direction points from the hot (shocked) gas toward the cold (pre-shocked) gas.
    
    The shock direction is defined as:
        d_s = -∇T / |∇T|
    ∇T from _calculate_temperature_gradient with given pressure and density fields.

    step: compute ∇T, negate it, normalize it
    
    Physical interpretation:
        - Negative gradient means temperature decreases in that direction
        - The negative sign points in the direction of decreasing temperature
    
    Args:
        pressure: Gas pressure field
        density: Density field
        config: Simulation configuration
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Normalized shock direction field (unit vector at each cell)
    """
    grad_T = _calculate_temperature_gradient(pressure, density, config, r)
    
    # Shock direction is -∇T (points toward cold gas)
    shock_dir = -grad_T
    
    shock_dir_normalized = _normalize_vector(shock_dir)
    
    return shock_dir_normalized