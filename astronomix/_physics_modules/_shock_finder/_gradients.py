from functools import partial
import jax.numpy as jnp
import jax
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FIELD_TYPE,
    SPHERICAL,
    SimulationConfig,
)


@partial(jax.jit, static_argnames=["config", "axis"])
def _calculate_gradient(
    field: FIELD_TYPE,
    config: SimulationConfig,
    axis: int = 0,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    """
    Calculate the spatial gradient of a scalar field along a single axis
    using central differences. Boundaries along that axis are set to zero.

    For Cartesian geometry:
        grad[..., i, ...] = (field[..., i+1, ...] - field[..., i-1, ...]) / (2 * dx)

    For Spherical geometry (axis=0 only, 1D):
        grad[i] = (r[i+1]^2 * field[i+1] - r[i-1]^2 * field[i-1])
                  / (2 * dx * r[i]^2)

    Args:
        field: Scalar field, shape (*spatial_shape)
        config: Simulation configuration (geometry, grid_spacing)
        axis: Spatial axis along which to differentiate (0=x, 1=y, 2=z)
        r: Radial coordinates, required for spherical geometry (1D only)

    Returns:
        Gradient field, same shape as field, zero at axis boundaries.
    """
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

    elif config.geometry == SPHERICAL:
        # spherical correction only defined for the radial axis (axis=0) in 1D
        if r is None:
            raise ValueError("Radial coordinates r required for spherical geometry.")
        if axis != 0:
            raise ValueError("Spherical geometry gradient only supported on axis=0.")
        grad_field = grad_field.at[interior].set(
            (r[2:] ** 2 * field[forward] - r[:-2] ** 2 * field[backward])
            / (2 * config.grid_spacing * r[1:-1] ** 2)
        )

    else:
        raise NotImplementedError(
            "Only Cartesian and Spherical geometry supported for shock finder."
        )

    return grad_field


@partial(jax.jit, static_argnames=["config"])
def _calculate_scalar_gradient(
    field: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    """
    Calculate the full spatial gradient ∇f of a scalar field.

    Calls _calculate_gradient once per spatial axis and stacks the results,
    producing a vector field where axis 0 indexes the gradient component.

    Returns:
        Shape (ndim, *spatial_shape)
            1D: (1, nx)
            2D: (2, nx, ny)
            3D: (3, nx, ny, nz)
    """
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


@partial(jax.jit, static_argnames=["config"])
def _calculate_temperature_gradient(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    """
    Compute ∇T_eff where T_eff = P / ρ (pseudo-temperature).

    Returns:
        Vector field, shape (ndim, *spatial_shape)
    """
    pseudo_temperature = pressure / density
    return _calculate_scalar_gradient(pseudo_temperature, config, r)


@partial(jax.jit, static_argnames=["config"])
def _calculate_density_gradient(
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    """
    Compute ∇ρ.

    Returns:
        Vector field, shape (ndim, *spatial_shape)
    """
    return _calculate_scalar_gradient(density, config, r)


@partial(jax.jit)
def _normalize_vector(
    vector: FIELD_TYPE,
    epsilon: float = 1e-12,
) -> FIELD_TYPE:
    """
    Normalize a vector field to unit magnitude everywhere.

    Args:
        vector: Shape (ndim, *spatial_shape)
            1D: (1, nx)
            2D: (2, nx, ny)
            3D: (3, nx, ny, nz)
        epsilon: Small constant to avoid division by zero.

    Returns:
        Unit vector field, same shape as input.
        magnitude = sqrt(sum(v**2, axis=0)), keepdims for broadcasting.
    """
    magnitude = jnp.sqrt(jnp.sum(vector ** 2, axis=0, keepdims=True)) + epsilon
    return vector / magnitude


@partial(jax.jit, static_argnames=["config"])
def _calculate_shock_direction(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
) -> FIELD_TYPE:
    """
    Compute the shock direction unit vector d_s = -∇T / |∇T|.

    Points from hot (post-shock) gas toward cold (pre-shock) gas.

    Args:
        pressure: Shape (*spatial_shape)
        density:  Shape (*spatial_shape)
        config:   Simulation configuration
        r:        Radial coordinates for spherical geometry (1D only)

    Returns:
        Unit vector field, shape (ndim, *spatial_shape)
            1D: (1, nx)
            2D: (2, nx, ny)
            3D: (3, nx, ny, nz)
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