# general
import jax
import jax.numpy as jnp
from functools import partial

# type checking
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from typing import Union

# astronomix containers
from astronomix._physics_modules._cosmic_rays.cr_fluid_equations import (
    total_energy_from_primitives_with_crs,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from astronomix.data_classes.simulation_helper_data import HelperData

# astronomix functions
from astronomix._physics_modules._self_gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._fluid_equations._equations import (
    get_absolute_velocity,
    total_energy_from_primitives,
)


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_internal_energy(state, helper_data, gamma, config, registered_variables):
    p = state[registered_variables.pressure_index]

    if config.cosmic_ray_config.cosmic_rays:
        gamma_cr = 4 / 3
        p = p - state[registered_variables.cosmic_ray_n_index] ** gamma_cr

    internal_energy = p / (gamma - 1)

    if config.dimensionality == 1:
        return jnp.sum(internal_energy * helper_data.cell_volumes)
    else:
        return jnp.sum(internal_energy * config.grid_spacing**config.dimensionality)


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_radial_momentum(state, helper_data, config, registered_variables):
    rho = state[registered_variables.density_index]
    box_center = jnp.zeros(config.dimensionality) + config.box_size / 2
    geometric_centers = helper_data.geometric_centers
    r_hat = (geometric_centers - box_center) / jnp.linalg.norm(
        geometric_centers - box_center, axis=-1, keepdims=True
    )

    if config.dimensionality == 1:
        u = state[registered_variables.velocity_index]
    else:
        u = state[
            registered_variables.velocity_index.x : registered_variables.velocity_index.x
            + config.dimensionality
        ]

    u_radial = jnp.sum(jnp.moveaxis(u, 0, -1) * r_hat, axis=-1)

    radial_momentum = rho * u_radial

    if config.dimensionality == 1:
        return jnp.sum(radial_momentum * helper_data.cell_volumes)
    else:
        return jnp.sum(radial_momentum * config.grid_spacing**config.dimensionality)


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_kinetic_energy(state, helper_data, config, registered_variables):
    rho = state[registered_variables.density_index]
    u = get_absolute_velocity(state, config, registered_variables)

    kinetic_energy = 0.5 * rho * u**2

    if config.dimensionality == 1:
        return jnp.sum(kinetic_energy * helper_data.cell_volumes)
    else:
        return jnp.sum(kinetic_energy * config.grid_spacing**config.dimensionality)


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_gravitational_energy(
    state, helper_data, gravitational_constant, config, registered_variables
):
    rho = state[registered_variables.density_index]

    potential = _compute_gravitational_potential(
        rho, config.grid_spacing, config, gravitational_constant
    )
    gravitational_energy = 0.5 * rho * potential
    if config.dimensionality == 1:
        return jnp.sum(gravitational_energy * helper_data.cell_volumes)
    else:
        return jnp.sum(
            gravitational_energy * config.grid_spacing**config.dimensionality
        )


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_total_energy(
    primitive_state: STATE_TYPE,
    helper_data: HelperData,
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> Float[Array, ""]:
    """
    Calculate the total energy in the domain.

    Args:
        primitive_state: The primitive state array.
        helper_data: The helper data.
        gamma: The adiabatic index.
        num_ghost_cells: The number of ghost cells.

    Returns:
        The total energy.
    """

    rho = primitive_state[registered_variables.density_index]
    u = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]

    if config.cosmic_ray_config.cosmic_rays:
        energy = total_energy_from_primitives_with_crs(
            primitive_state, registered_variables
        )
    else:
        energy = total_energy_from_primitives(rho, u, p, gamma)

    if config.self_gravity:
        potential = _compute_gravitational_potential(
            rho, config.grid_spacing, config, gravitational_constant
        )
        energy += 0.5 * rho * potential

    if config.dimensionality == 1:
        return jnp.sum(energy * helper_data.cell_volumes)
    else:
        return jnp.sum(energy * config.grid_spacing**config.dimensionality)


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config"])
def calculate_total_mass(
    primitive_state: STATE_TYPE,
    helper_data: HelperData,
    config: SimulationConfig,
) -> Float[Array, ""]:
    """
    Calculate the total mass in the domain.

    Args:
        primitive_state: The primitive state array.
        helper_data: The helper data.
        config: The simulation configuration.

    Returns:
        The total mass.
    """
    num_ghost_cells = config.num_ghost_cells

    if config.dimensionality == 1:
        return jnp.sum(primitive_state[0] * helper_data.cell_volumes)
    else:
        return jnp.sum(primitive_state[0]) * config.box_size**config.dimensionality
