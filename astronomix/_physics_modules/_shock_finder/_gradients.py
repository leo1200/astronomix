from functools import partial
import jax.numpy as jnp
import jax
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FIELD_TYPE,
    SPHERICAL,
    SimulationConfig,
)


"""
calculate gradients helper
only support Cartesian geometries
"""
@partial(jax.jit, static_argnames=["config", "axis"])
def _calculate_gradient(
    field: FIELD_TYPE,
    config: SimulationConfig,
    axis: int = 0,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    ndim = field.ndim

    # build index tuples for interior / forward / backward slices along axis
    interior = [slice(None)] * ndim
    forward  = [slice(None)] * ndim
    backward = [slice(None)] * ndim

    interior[axis] = slice(1, -1)
    forward[axis]  = slice(2, None)
    backward[axis] = slice(None, -2)

    interior = tuple(interior)
    forward  = tuple(forward)
    backward = tuple(backward)

    grad_field = jnp.zeros_like(field)

    if config.geometry == CARTESIAN:
        grad_field = grad_field.at[interior].set(
            (field[forward] - field[backward]) / (2 * config.grid_spacing)
        )

    else:
        raise NotImplementedError(
            "Only Cartesian and Spherical geometry supported for shock finder."
        )

    return grad_field

"""
Calculate the full spatial gradient ∇f of a scalar field.
Calls _calculate_gradient once per spatial axis and stacks the results,
"""
@partial(jax.jit, static_argnames=["config"])
def _calculate_scalar_gradient(
    field: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    ndim = field.ndim
    return jnp.stack(
        [
            _calculate_gradient(
                field,
                config,
                axis=ax,
                r=r if ax == 0 else None,  # r only used on axis=0 for spherical
            )
            for ax in range(ndim)
        ],
        axis=0,
    )

"""
Compute ∇T_eff where T_eff = P / ρ (pseudo-temperature).
"""
@partial(jax.jit, static_argnames=["config"])
def _calculate_temperature_gradient(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    pseudo_temperature = pressure / density
    return _calculate_scalar_gradient(pseudo_temperature, config, r)

"""
Compute ∇ρ
"""
@partial(jax.jit, static_argnames=["config"])
def _calculate_density_gradient(
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    return _calculate_scalar_gradient(density, config, r)

"""
Normalize a vector field to unit magnitude everywhere
"""
@partial(jax.jit)
def _normalize_vector(
    vector: FIELD_TYPE,
    epsilon: float = 1e-12,
) -> FIELD_TYPE:
    magnitude = jnp.sqrt(jnp.sum(vector ** 2, axis=0, keepdims=True)) + epsilon
    return vector / magnitude

"""
Compute the shock direction unit vector d_s = -∇T / |∇T|
for all cells
"""
@partial(jax.jit, static_argnames=["config"])
def _calculate_shock_direction(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
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
    return _normalize_vector(-grad_T)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _calculate_velocity_divergence(
    primitive_state,
    config: SimulationConfig,
    registered_variables,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    """
    Compute ∇·v = sum_i ∂v_i/∂x_i over all spatial axes.

    In 1D: ∂vx/∂x
    In 2D: ∂vx/∂x + ∂vy/∂y
    In 3D: ∂vx/∂x + ∂vy/∂y + ∂vz/∂z

    velocity_index is either an int (1D) or StaticIntVector (2D/3D).

    Returns:
        Scalar field, shape (*spatial_shape)
    """
    vel_idx = registered_variables.velocity_index

    if isinstance(vel_idx, int):
        # 1D: single velocity component
        vx = primitive_state[vel_idx]
        return _calculate_gradient(vx, config, axis=0, r=r)

    else:
        # 2D/3D: StaticIntVector with .x, .y, .z
        # sum partial derivatives along each active axis
        div_v = None

        for ax, idx in enumerate([vel_idx.x, vel_idx.y, vel_idx.z]):
            if idx == -1:
                continue
            v_component = primitive_state[idx]
            dv = _calculate_gradient(v_component, config, axis=ax, r=r if ax == 0 else None)
            div_v = dv if div_v is None else div_v + dv

        return div_v