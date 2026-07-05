"""
Physical fluxes for the (hydrodynamic) Euler equations.

Provides the x-direction Euler flux vector; fluxes in the other directions are
obtained by permuting the state arrays accordingly.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE, FOURTH_ORDER_CONSERVATIVE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import AxisInfo, RegisteredVariables

# astronomix functions
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    total_energy_from_primitives_with_crs,
)
from astronomix._fluid_equations._equations import (
    get_absolute_velocity,
    total_energy_from_primitives,
)
from astronomix._modules._gravity._poisson_solver import _compute_gravitational_potential


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["config", "registered_variables", "flux_direction_index"]
)
def _euler_flux(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int,
) -> STATE_TYPE:
    """Compute the Euler fluxes for the given primitive states.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        gamma: The adiabatic index of the fluid.
        config: The simulation configuration.
        registered_variables: The registered variables.
        flux_direction_index: The index of the velocity component in the flux direction of interest.

    Returns:
        The Euler fluxes for the given primitive states.
    """
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]

    # Start the flux vector from the primitive state. No explicit copy is needed:
    # the subsequent ``.at[...].set/add`` operations are functional and produce a
    # new array rather than mutating ``primitive_state`` in place.
    flux_vector = primitive_state

    # Compute the total energy that enters the energy flux.
    utotal = get_absolute_velocity(primitive_state, config, registered_variables)

    if registered_variables.cosmic_ray_n_active:
        E = total_energy_from_primitives_with_crs(primitive_state, registered_variables)
    else:
        E = total_energy_from_primitives(rho, utotal, p, gamma)

    # Add the total energy onto the pressure slot of the flux vector (the
    # pressure index doubles as the energy slot in the conserved layout).
    flux_vector = flux_vector.at[registered_variables.pressure_index].add(E)

    # Scale the velocity components by the density to form the momentum fluxes.
    if config.dimensionality == 1:
        flux_vector = flux_vector.at[registered_variables.velocity_index].set(
            primitive_state[registered_variables.velocity_index] * rho
        )
    elif config.dimensionality == 2:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(
            primitive_state[registered_variables.velocity_index.x] * rho
        )
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(
            primitive_state[registered_variables.velocity_index.y] * rho
        )
    elif config.dimensionality == 3:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(
            primitive_state[registered_variables.velocity_index.x] * rho
        )
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(
            primitive_state[registered_variables.velocity_index.y] * rho
        )
        flux_vector = flux_vector.at[registered_variables.velocity_index.z].set(
            primitive_state[registered_variables.velocity_index.z] * rho
        )

    # Multiply the whole vector by the velocity component in the flux direction.
    # Indexing ``primitive_state[flux_direction_index]`` works only because of the
    # current registry ordering. TODO: generalize this to be registry-agnostic.
    flux_vector = primitive_state[flux_direction_index] * flux_vector

    # Add the pressure to the momentum component in the flux direction.
    flux_vector = flux_vector.at[flux_direction_index].add(p)

    return flux_vector