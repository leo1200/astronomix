# general
import jax.numpy as jnp
import jax
from functools import partial

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# astronomix classes
from astronomix.variable_registry.registered_variables import RegisteredVariables

# NOTE: currently only supports 1d setups, TODO: generalize

# HERE SET MANUALLY, SHOULD COME FROM
# THE SIMULATION PARAMS
gamma_gas = 5 / 3
gamma_cr = 4 / 3


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["registered_variables"])
def total_energy_from_primitives_with_crs(
    primitive_state: Float[Array, "num_vars num_cells"],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_cells"]:
    """
    Calculates the total energy density from primitive variables in a system with cosmic rays.

    Args:
        primitive_state: Array of primitive variables
        registered_variables: Object containing indices for accessing different physical quantities

    Returns:
        Total energy density array
    """

    # get the cosmic ray pressure
    cosmic_ray_pressure = (
        primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )

    # get the cosmic ray energy (density)
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # get the gas pressure
    gas_pressure = (
        primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure
    )

    # get the gas energy
    rho_gas = primitive_state[registered_variables.density_index]
    velocity = primitive_state[registered_variables.velocity_index]
    gas_energy = gas_pressure / (gamma_gas - 1) + 0.5 * rho_gas * velocity**2

    # total energy
    E_tot = gas_energy + cosmic_ray_energy

    return E_tot


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["registered_variables"])
def gas_pressure_from_primitives_with_crs(
    primitive_state: Float[Array, "num_vars num_cells"],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_cells"]:
    """
    Calculates the gas pressure from the primitive state when cosmic rays
    are considered in the simulation.

    Args:
        primitive_state: Array of primitive variables
        registered_variables: Object containing indices for accessing different physical quantities

    Returns:
        gas pressure
    """

    # get the cosmic ray pressure
    cosmic_ray_pressure = (
        primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )

    # return the gas pressure
    return primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure


# TODO: make 2D and 3D ready
# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["registered_variables"])
def total_pressure_from_conserved_with_crs(
    conserved_state: Float[Array, "num_vars num_cells"],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_cells"]:
    """
    Calculates the total pressure from the conserved state when cosmic rays
    are considered in the simulation.

    Args:
        primitive_state: Array of primitive variables
        registered_variables: Object containing indices for accessing different physical quantities

    Returns:
        total pressure
    """

    # get the cosmic ray pressure
    cosmic_ray_pressure = (
        conserved_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )

    # get the cosmic ray energy (density)
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # get the gas energy
    gas_energy = (
        conserved_state[registered_variables.pressure_index] - cosmic_ray_energy
    )

    # get the gas pressure
    rho_gas = conserved_state[registered_variables.density_index]
    velocity = conserved_state[registered_variables.velocity_index] / rho_gas
    gas_pressure = (gas_energy - 0.5 * rho_gas * velocity**2) * (gamma_gas - 1)

    # get the total pressure
    total_pressure = cosmic_ray_pressure + gas_pressure

    return total_pressure


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["registered_variables"])
def speed_of_sound_crs(
    primitive_state: Float[Array, "num_vars num_cells"],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_cells"]:
    """
    Calculates the speed of sound from the primitive state
    when cosmic rays are considered in the simulation, where
    c_s = sqrt((gamma_gas * P_gas + gamma_cr * P_CR) / rho)

    Args:
        primitive_state: Array of primitive variables
        registered_variables: Object containing indices for accessing different physical quantities

    Returns:
        sound speed
    """

    # get the cosmic ray pressure
    cosmic_ray_pressure = (
        primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )

    # get the gas pressure
    gas_pressure = (
        primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure
    )

    return jnp.sqrt(
        (gamma_gas * gas_pressure + gamma_cr * cosmic_ray_pressure)
        / primitive_state[registered_variables.density_index]
    )
