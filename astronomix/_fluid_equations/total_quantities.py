"""
Domain-integrated diagnostic quantities.

Computes globally integrated quantities of the fluid state (internal, kinetic,
gravitational and total energy, total mass and radial momentum) used for the
simulation diagnostics.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._fluid_equations._equation_of_state import is_barotropic
from astronomix._modules._gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._modules._gravity._utils import _pad_external_potential
from astronomix._fluid_equations._equations import (
    get_absolute_velocity,
    total_energy_from_primitives,
)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_internal_energy(state, helper_data, gamma, config, registered_variables):
    """The total internal energy of the domain: p / (gamma - 1) of the ideal gas, over the cell volumes."""
    p = state[registered_variables.pressure_index]

    internal_energy = p / (gamma - 1)

    if config.dimensionality == 1:
        return jnp.sum(internal_energy * helper_data.cell_volumes)
    else:
        return jnp.sum(internal_energy * config.grid_spacing**config.dimensionality)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_radial_momentum(state, helper_data, config, registered_variables):
    """The total radial momentum of the domain: rho v . r_hat, r measured from the box centre, over the cells."""
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


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_kinetic_energy(state, helper_data, config, registered_variables):
    """The total kinetic energy of the domain, 0.5 rho v^2 integrated over the cell volumes."""
    rho = state[registered_variables.density_index]
    u = get_absolute_velocity(state, config, registered_variables)

    kinetic_energy = 0.5 * rho * u**2

    if config.dimensionality == 1:
        return jnp.sum(kinetic_energy * helper_data.cell_volumes)
    else:
        return jnp.sum(kinetic_energy * config.grid_spacing**config.dimensionality)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_gravitational_energy(
    state, helper_data, gravitational_constant, params, config, registered_variables
):
    """The total gravitational energy: 0.5 rho phi of self-gravity plus rho phi of an external potential."""
    rho = state[registered_variables.density_index]

    gravitational_energy = jnp.zeros_like(rho)

    # self-gravity uses the factor 1/2 to avoid double counting the mutual
    # interaction, while a fixed external potential contributes its full
    # potential energy rho * phi_ext.
    if config.gravity_config.self_gravity:
        self_potential = _compute_gravitational_potential(
            rho, config.grid_spacing, config, gravitational_constant
        )
        gravitational_energy = gravitational_energy + 0.5 * rho * self_potential

    if config.gravity_config.external_potential:
        external_potential = _pad_external_potential(
            params.gravitational_potential, rho, config, registered_variables, params
        )
        gravitational_energy = gravitational_energy + rho * external_potential

    if config.dimensionality == 1:
        return jnp.sum(gravitational_energy * helper_data.cell_volumes)
    else:
        return jnp.sum(
            gravitational_energy * config.grid_spacing**config.dimensionality
        )


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_total_energy(
    primitive_state: STATE_TYPE,
    helper_data: HelperData,
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> Float[Array, ""]:
    """
    Calculate the total energy in the domain (for a barotropic gas, which has
    no energy equation, the kinetic and the potential energy).

    Args:
        primitive_state: The primitive state array.
        helper_data: The helper data.
        gamma: The adiabatic index.
        gravitational_constant: The gravitational constant.
        params: The simulation parameters (provides the external potential).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The total energy.
    """

    rho = primitive_state[registered_variables.density_index]
    u = get_absolute_velocity(primitive_state, config, registered_variables)
    if is_barotropic(config):
        # No energy equation: the kinetic energy (and the potential energy below).
        energy = 0.5 * rho * u**2
    else:
        p = primitive_state[registered_variables.pressure_index]
        energy = total_energy_from_primitives(rho, u, p, gamma)

    # self-gravity carries the factor 1/2 (mutual interaction); a fixed
    # external potential contributes its full potential energy rho * phi_ext.
    if config.gravity_config.self_gravity:
        self_potential = _compute_gravitational_potential(
            rho, config.grid_spacing, config, gravitational_constant
        )
        energy += 0.5 * rho * self_potential

    if config.gravity_config.external_potential:
        external_potential = _pad_external_potential(
            params.gravitational_potential, rho, config, registered_variables, params
        )
        energy += rho * external_potential

    if config.dimensionality == 1:
        return jnp.sum(energy * helper_data.cell_volumes)
    else:
        return jnp.sum(energy * config.grid_spacing**config.dimensionality)


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
        # cell-volume-weighted sum (matches the energy/momentum diagnostics):
        # grid_spacing is a scalar, unlike box_size which is a 3-vector in 3D.
        return jnp.sum(primitive_state[0] * config.grid_spacing**config.dimensionality)
