from functools import partial
import jax.numpy as jnp
import jax
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FIELD_TYPE,
    SPHERICAL,
    SimulationConfig,
)


@partial(jax.jit, static_argnames=["config"])
def _calculate_gradient(
    field: FIELD_TYPE, 
    config: SimulationConfig, 
    r: FIELD_TYPE = None
) -> FIELD_TYPE:
    """
    Calculate the spatial gradient of a scalar field using central differences.
    
    For Cartesian geometry:
        grad[i] = (field[i+1] - field[i-1]) / (2 * dx)
    
    For Spherical geometry (1D):
        grad[i] = (r[i+1]^2 * field[i+1] - r[i-1]^2 * field[i-1]) / (2 * dx * r[i]^2)
    
    Args:
        field: Scalar field to differentiate
        config: Simulation configuration, only needed for geometry type and grid spacing
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Gradient field (boundaries set to zero)
    """
    grad_field = jnp.zeros_like(field)
    
    if config.geometry == CARTESIAN:
        # grad_field[i] is immutable, so we use .at[].set() to update the interior points
        # grad_field[1:-1] means consider all points except the first and last (boundary points)
        grad_field = grad_field.at[1:-1].set(
            (field[2:] - field[:-2]) / (2 * config.grid_spacing)
        )
    elif config.geometry == SPHERICAL:
        if r is None:
            raise ValueError("Radial coordinates r required for spherical geometry")
        grad_field = grad_field.at[1:-1].set(
            (r[2:] ** 2 * field[2:] - r[:-2] ** 2 * field[:-2])
            / (2 * config.grid_spacing * r[1:-1] ** 2)
        )
    else:
        raise NotImplementedError(
            "Only Cartesian and Spherical geometry supported for shock finder."
        )
    
    return grad_field

@partial(jax.jit, static_argnames=["config"])
def _calculate_temperature_gradient(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None
) -> FIELD_TYPE:
    """
    Computes ∇T across the grid
    
    T is not the true thermodynamic temperature but a pseudo-temperature
    defined as T_eff[i] = P[i] / ρ[i]
    
    then gradient is computed via _calculate_gradient for T_eff field
    
    Args:
        pressure: Gas pressure field
        density: Density field
        config: Simulation configuration
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Temperature gradient field
    """
    pseudo_temperature = pressure / density
    grad_T = _calculate_gradient(pseudo_temperature, config, r)
    return grad_T


@partial(jax.jit, static_argnames=["config"])
def _calculate_density_gradient(
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None
) -> FIELD_TYPE:
    """
    Calculate the density gradient ∇ρ across the grid
    by using _calculate_gradient on the density field.
    
    Args:
        density: Density field
        config: Simulation configuration
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Density gradient field
    """
    grad_rho = _calculate_gradient(density, config, r)
    return grad_rho


@partial(jax.jit)
def _normalize_vector(
    vector: FIELD_TYPE,
    epsilon: float = 1e-12
) -> FIELD_TYPE:
    """
    Converts a vector field to unit magnitude everywhere:
    
    normalized[i] = vector[i] / (|vector[i]| + epsilon)
    
    The epsilon prevents division by zero when the vector is near zero.
    
    Args:
        vector: Vector field to normalize
        epsilon: Small constant for numerical stability
    
    Returns:
        Normalized vector field (magnitude 1 except where input is near-zero)
    """
    ## TODO: do this for 2D and 3D also, now only works for 1D scalar fields (shock direction in 1D is just a sign)
    magnitude = jnp.abs(vector) + epsilon
    normalized = vector / magnitude
    return normalized