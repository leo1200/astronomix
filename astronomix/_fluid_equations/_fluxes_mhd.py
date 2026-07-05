"""
Physical fluxes for the magnetohydrodynamic (MHD) equations.

Only the x-direction fluxes are defined here; fluxes in the other directions are
obtained by permuting the state arrays accordingly. Variants are provided for the
ideal-gas MHD, isothermal MHD and isothermal hydro equations of state.
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
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import AxisInfo, RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equations_mhd import (
    primitive_state_from_conserved_isothermal,
    primitive_state_from_conserved_mhd,
    total_energy_from_primitives_mhd,
    total_pressure_from_conserved_mhd,
)


# We only define the flux in the x-direction here; the other directions are
# obtained by permuting the arrays accordingly. For the IDEAL_GAS variant see
# ``_mhd_flux_x`` below, and for the ISOTHERMAL variants see the functions that
# follow it.
@partial(
    jax.jit, static_argnames=["registered_variables", "config"]
)
def _mhd_flux_x(
    conserved_state: STATE_TYPE,
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Compute the ideal-gas MHD x-direction flux for a conserved state.

    Args:
        conserved_state: The conserved MHD state.
        minimum_density: The density floor used when recovering primitives.
        minimum_pressure: The pressure floor used when recovering primitives.
        gamma: The adiabatic index of the fluid.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The MHD flux vector in the x-direction.
    """

    primitive_state = primitive_state_from_conserved_mhd(
        conserved_state, minimum_density, minimum_pressure, gamma, config, registered_variables
    )

    # Retrieve the primitive quantities entering the flux.
    rho = primitive_state[registered_variables.density_index]
    v_x = primitive_state[registered_variables.velocity_index.x]
    v_y = primitive_state[registered_variables.velocity_index.y]
    v_z = primitive_state[registered_variables.velocity_index.z]
    B_x = primitive_state[registered_variables.magnetic_index.x]
    B_y = primitive_state[registered_variables.magnetic_index.y]
    B_z = primitive_state[registered_variables.magnetic_index.z]
    p_gas = primitive_state[registered_variables.pressure_index]

    # Compute the derived quantities (total pressure including magnetic pressure,
    # the v.B work term and the total energy) that enter the MHD fluxes.
    b_squared = B_x**2 + B_y**2 + B_z**2
    v_squared = v_x**2 + v_y**2 + v_z**2
    total_pressure = p_gas + 0.5 * b_squared
    v_dot_B = v_x * B_x + v_y * B_y + v_z * B_z
    E = total_energy_from_primitives_mhd(
        rho,
        v_squared,
        p_gas,
        b_squared,
        gamma,
    )

    # Assemble the MHD flux vector.
    flux = jnp.zeros_like(primitive_state)
    flux = flux.at[registered_variables.density_index].set(rho * v_x)
    flux = flux.at[registered_variables.velocity_index.x].set(rho * v_x**2 + total_pressure - B_x**2)
    flux = flux.at[registered_variables.velocity_index.y].set(rho * v_x * v_y - B_x * B_y)
    flux = flux.at[registered_variables.velocity_index.z].set(rho * v_x * v_z - B_x * B_z)
    flux = flux.at[registered_variables.pressure_index].set((E + total_pressure) * v_x - v_dot_B * B_x)
    flux = flux.at[registered_variables.magnetic_index.x].set(0.0)
    flux = flux.at[registered_variables.magnetic_index.y].set(B_y * v_x - B_x * v_y)
    flux = flux.at[registered_variables.magnetic_index.z].set(B_z * v_x - B_x * v_z)

    return flux

def _mhd_flux_isothermal_x(
    conserved_state: STATE_TYPE,
    minimum_density: Union[float, Float[Array, ""]],
    isothermal_sound_speed: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Compute the isothermal MHD x-direction flux for a conserved state.

    There is no energy equation in the isothermal case; the gas pressure follows
    directly from the fixed isothermal sound speed via ``p = c_s^2 rho``.

    Args:
        conserved_state: The conserved isothermal MHD state.
        minimum_density: The density floor used when recovering primitives.
        isothermal_sound_speed: The fixed isothermal sound speed.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The isothermal MHD flux vector in the x-direction.
    """

    primitive_state = primitive_state_from_conserved_isothermal(
        conserved_state, minimum_density, config, registered_variables
    )

    # Retrieve the primitive quantities entering the flux.
    rho = primitive_state[registered_variables.density_index]
    v_x = primitive_state[registered_variables.velocity_index.x]
    v_y = primitive_state[registered_variables.velocity_index.y]
    v_z = primitive_state[registered_variables.velocity_index.z]
    B_x = primitive_state[registered_variables.magnetic_index.x]
    B_y = primitive_state[registered_variables.magnetic_index.y]
    B_z = primitive_state[registered_variables.magnetic_index.z]

    # Compute the derived quantities. The isothermal gas pressure is fixed by the
    # sound speed, and the total pressure adds the magnetic pressure.
    b_squared = B_x**2 + B_y**2 + B_z**2
    p_gas = isothermal_sound_speed**2 * rho
    total_pressure = p_gas + 0.5 * b_squared

    # Assemble the isothermal MHD flux vector.
    flux = jnp.zeros_like(primitive_state)
    flux = flux.at[registered_variables.density_index].set(rho * v_x)
    flux = flux.at[registered_variables.velocity_index.x].set(rho * v_x**2 + total_pressure - B_x**2)
    flux = flux.at[registered_variables.velocity_index.y].set(rho * v_x * v_y - B_x * B_y)
    flux = flux.at[registered_variables.velocity_index.z].set(rho * v_x * v_z - B_x * B_z)
    flux = flux.at[registered_variables.magnetic_index.x].set(0.0)
    flux = flux.at[registered_variables.magnetic_index.y].set(B_y * v_x - B_x * v_y)
    flux = flux.at[registered_variables.magnetic_index.z].set(B_z * v_x - B_x * v_z)

    return flux

@partial(
    jax.jit, static_argnames=["config", "registered_variables"]
)
def _euler_flux_isothermal_x(
    conserved_state: STATE_TYPE,
    minimum_density: Union[float, Float[Array, ""]],
    isothermal_sound_speed: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Compute the Euler fluxes for the given conserved state.

    F = [
        ρ v_x,                         // density_index
        ρ v_x^2 + p,                   // velocity_index.x
        ρ v_x v_y,                     // velocity_index.y
        ρ v_x v_z,                     // velocity_index.z
        # no energy equation in the isothermal case
    ]
    """

    flux_vector = jnp.zeros_like(conserved_state)

    if config.dimensionality == 1:
        m_x = conserved_state[registered_variables.velocity_index]
    else:
        m_x = conserved_state[registered_variables.velocity_index.x]

    rho = conserved_state[registered_variables.density_index]

    if config.positivity_config.clamp_in_estimates:
        rho = jnp.maximum(rho, minimum_density)

    p = isothermal_sound_speed**2 * rho

    # Assemble the isothermal Euler flux vector.
    flux_vector = flux_vector.at[registered_variables.density_index].set(m_x)

    if config.dimensionality == 1:
        flux_vector = flux_vector.at[registered_variables.velocity_index].set(m_x**2 / rho + p)
    elif config.dimensionality == 2:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(m_x**2 / rho + p)
        v_y = conserved_state[registered_variables.velocity_index.y] / rho
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(m_x * v_y)
    elif config.dimensionality == 3:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(m_x**2 / rho + p)
        v_y = conserved_state[registered_variables.velocity_index.y] / rho
        v_z = conserved_state[registered_variables.velocity_index.z] / rho
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(m_x * v_y)
        flux_vector = flux_vector.at[registered_variables.velocity_index.z].set(m_x * v_z)

    return flux_vector