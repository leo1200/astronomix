"""
The relaxation zone: a region where the primitive state is pulled towards a
prescribed target state, once per time step.

Every cell relaxes as the linear ordinary differential equation

    dq/dt = -rate (q - q_target),

integrated exactly over the time step:

    q <- q + (q_target - q) (1 - exp(-rate dt)).

The update is unconditionally stable and, as a convex combination of two
physical states, keeps the density and the pressure positive. The relaxed
fraction 1 - exp(-rate dt) is evaluated as -expm1(-rate dt): accurate when
rate dt is small, and exactly zero where the rate is zero, so the state
outside the zone is left bit for bit as it is.

Typical uses are an absorbing layer at the edge of the domain (outgoing waves
and outflows are damped instead of reflected), a reservoir that holds the gas
entering the domain at a background state (an atmosphere in hydrostatic
equilibrium, a Bondi inflow), and the boundary region of a zoom-in simulation,
whose target comes from a larger simulation and changes between calls of the
time integration (``params`` can be replaced without recompiling).

The zone does not conserve mass, momentum or energy: it exchanges them with the
target state. It is not available for MHD, where relaxing the magnetic field
cell by cell would break div B = 0.
"""

# typing
from jaxtyping import Array, Float

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams


def _with_ghost_cells(field, reference_shape, config: SimulationConfig, fill_value: float):
    """
    Pad the spatial axes of a field defined without ghost cells to the shape of
    a state with ghost cells.

    Args:
        field: The field, with leading non-spatial axes (variables) allowed.
        reference_shape: The shape of the state with ghost cells.
        config: The simulation configuration.
        fill_value: The value of the ghost cells.

    Returns:
        The padded field (unchanged when it already has the spatial shape of
        the state).
    """
    spatial_shape = reference_shape[1:]
    if field.shape[-config.dimensionality:] == spatial_shape:
        return field
    ghost = config.num_ghost_cells
    padding = [(0, 0)] * (field.ndim - config.dimensionality) + [(ghost, ghost)] * config.dimensionality
    return jnp.pad(field, padding, constant_values=fill_value)


def apply_relaxation_zone(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams,
) -> STATE_TYPE:
    """
    Relax the primitive state towards the target state of the zone over one
    time step.

    Args:
        primitive_state: The primitive state, with or without ghost cells.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters (``relaxation_zone_params``).

    Returns:
        The relaxed primitive state.
    """
    zone = params.relaxation_zone_params
    # The ghost cells get a rate of zero: they are set by the boundary
    # conditions before the next flux computation, and relaxing them here
    # would only hide what the boundaries do.
    target_state = _with_ghost_cells(zone.target_state, primitive_state.shape, config, fill_value=0.0)
    rate = _with_ghost_cells(zone.rate, primitive_state.shape, config, fill_value=0.0)
    if rate.ndim == primitive_state.ndim - 1:
        rate = rate[jnp.newaxis, ...]

    relaxed_fraction = -jnp.expm1(-rate * dt)
    return primitive_state + (target_state - primitive_state) * relaxed_fraction
