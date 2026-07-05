"""
Two-fluid (gas + cosmic-ray) equation-of-state helpers.

These routines convert between primitive and conserved quantities for a fluid
that carries a cosmic-ray component alongside the thermal gas. The cosmic rays
are tracked through ``n_cr = P_cr ** (1 / gamma_cr)`` (an advected scalar), so
the cosmic-ray pressure is recovered as ``P_cr = n_cr ** gamma_cr``. The total
pressure stored in ``pressure_index`` is the sum of the gas and cosmic-ray
pressures.
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

# astronomix containers
from astronomix.variable_registry.registered_variables import RegisteredVariables

# NOTE: these routines currently only support 1D setups; generalising them to
# 2D / 3D is still outstanding.

# WARNING: the adiabatic indices are fixed here rather than read from
# ``SimulationParams``. They should eventually be sourced from the simulation
# parameters so a run can override them consistently.
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

    # Recover the cosmic-ray pressure from the advected scalar n_cr.
    cosmic_ray_pressure = (
        primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )

    # Cosmic-ray energy density from its (relativistic) equation of state.
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # The stored pressure is the total; the gas pressure is what remains after
    # removing the cosmic-ray contribution.
    gas_pressure = (
        primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure
    )

    # Gas energy density: internal (thermal) plus kinetic.
    rho_gas = primitive_state[registered_variables.density_index]
    velocity = primitive_state[registered_variables.velocity_index]
    gas_energy = gas_pressure / (gamma_gas - 1) + 0.5 * rho_gas * velocity**2

    # Total energy density is the sum of the two components.
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

    # Recover the cosmic-ray pressure from the advected scalar n_cr.
    cosmic_ray_pressure = (
        primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )

    # The stored pressure is the total, so subtract the cosmic-ray part.
    return primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure


# NOTE: this still needs to be generalised to 2D and 3D.
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

    # Recover the cosmic-ray pressure from the advected scalar n_cr.
    cosmic_ray_pressure = (
        conserved_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )

    # Cosmic-ray energy density from its equation of state.
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # In the conserved state the energy slot holds the total energy density;
    # the gas energy is what remains after removing the cosmic-ray part.
    gas_energy = (
        conserved_state[registered_variables.pressure_index] - cosmic_ray_energy
    )

    # Back out the gas pressure from the gas energy by removing the kinetic part.
    rho_gas = conserved_state[registered_variables.density_index]
    velocity = conserved_state[registered_variables.velocity_index] / rho_gas
    gas_pressure = (gas_energy - 0.5 * rho_gas * velocity**2) * (gamma_gas - 1)

    # The total pressure is the sum of both pressure components.
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

    # Recover the cosmic-ray pressure from the advected scalar n_cr.
    cosmic_ray_pressure = (
        primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )

    # The stored pressure is the total, so subtract the cosmic-ray part.
    gas_pressure = (
        primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure
    )

    # Both components stiffen the fluid, so the effective sound speed mixes the
    # gas and cosmic-ray pressures weighted by their adiabatic indices.
    return jnp.sqrt(
        (gamma_gas * gas_pressure + gamma_cr * cosmic_ray_pressure)
        / primitive_state[registered_variables.density_index]
    )
