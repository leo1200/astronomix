"""
Equations for 3D adiabatic ideal magnetohydrodynamics (MHD).
"""

import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float

from typing import Union

from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
    SimulationConfig,
)


@partial(jax.jit, static_argnames=["registered_variables"])
def _u_squared3D(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    return (
        primitive_state[registered_variables.velocity_index.x] ** 2
        + primitive_state[registered_variables.velocity_index.y] ** 2
        + primitive_state[registered_variables.velocity_index.z] ** 2
    )


def _b_squared3D(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    return (
        primitive_state[registered_variables.magnetic_index.x] ** 2
        + primitive_state[registered_variables.magnetic_index.y] ** 2
        + primitive_state[registered_variables.magnetic_index.z] ** 2
    )


@jax.jit
def thermal_pressure_from_energy_mhd(E, rho, u_squared, b_squared, gamma):
    """Calculate the pressure from the total energy in MHD.

    Args:
        E: The total energy.
        rho: The density.
        u_squared: The squared velocity.
        b_squared: The squared magnetic field.
        gamma: The adiabatic index.
    Returns:
        The pressure.
    """
    return (gamma - 1) * (E - 0.5 * rho * u_squared - 0.5 * b_squared)


@jax.jit
def total_energy_from_primitives_mhd(rho, u_squared, p, b_squared, gamma):
    return p / (gamma - 1) + 0.5 * rho * u_squared + 0.5 * b_squared


@partial(jax.jit, static_argnames=["registered_variables"])
def conserved_state_from_primitive_mhd(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    For now only 3D.
    """

    rho = primitive_state[registered_variables.density_index]

    u_squared = _u_squared3D(primitive_state, registered_variables)

    p = primitive_state[registered_variables.pressure_index]  # thermal pressure

    b_squared = _b_squared3D(primitive_state, registered_variables)

    # calculate total energy
    E = total_energy_from_primitives_mhd(rho, u_squared, p, b_squared, gamma)

    # create conserved state
    conserved_state = primitive_state.at[registered_variables.pressure_index].set(E)

    # set momentum density
    conserved_state = conserved_state.at[
        registered_variables.velocity_index.x : registered_variables.velocity_index.z
        + 1
    ].set(
        rho
        * primitive_state[
            registered_variables.velocity_index.x : registered_variables.velocity_index.z
            + 1
        ]
    )

    return conserved_state


@partial(jax.jit, static_argnames=["registered_variables", 'config'])
def primitive_state_from_conserved_mhd(
    conserved_state: STATE_TYPE,
    rhomin: Union[float, Float[Array, ""]],
    pgmin: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    For now only 3D.
    """

    rho = conserved_state[registered_variables.density_index]
    E = conserved_state[registered_variables.pressure_index]

    ux = conserved_state[registered_variables.velocity_index.x] / rho
    uy = conserved_state[registered_variables.velocity_index.y] / rho
    uz = conserved_state[registered_variables.velocity_index.z] / rho

    u_squared = ux**2 + uy**2 + uz**2

    b_squared = _b_squared3D(conserved_state, registered_variables)

    p = thermal_pressure_from_energy_mhd(E, rho, u_squared, b_squared, gamma)

    # set the primitive state

    # pressure
    primitive_state = conserved_state.at[registered_variables.pressure_index].set(p)

    # velocities
    primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(ux)
    primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(uy)
    primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(uz)

    if config.enforce_positivity:
        # enforce positivity of density and pressure
        primitive_state = primitive_state.at[registered_variables.density_index].set(
            jnp.maximum(
                primitive_state[registered_variables.density_index], rhomin
            )
        )
        primitive_state = primitive_state.at[registered_variables.pressure_index].set(
            jnp.maximum(
                primitive_state[registered_variables.pressure_index], pgmin
            )
        )

    return primitive_state


@partial(jax.jit, static_argnames=["registered_variables"])
def total_pressure_from_conserved_mhd(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    """
    For now only 3D.
    """

    rho = conserved_state[registered_variables.density_index]
    E = conserved_state[registered_variables.pressure_index]

    ux = conserved_state[registered_variables.velocity_index.x] / rho
    uy = conserved_state[registered_variables.velocity_index.y] / rho
    uz = conserved_state[registered_variables.velocity_index.z] / rho

    u_squared = ux**2 + uy**2 + uz**2

    b_squared = _b_squared3D(conserved_state, registered_variables)

    p_thermal = thermal_pressure_from_energy_mhd(E, rho, u_squared, b_squared, gamma)

    return p_thermal + 0.5 * b_squared
