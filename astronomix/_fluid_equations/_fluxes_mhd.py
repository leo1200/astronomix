"""
Physical fluxes for the magnetohydrodynamic (MHD) equations.

Only the x-direction fluxes are defined here; fluxes in the other directions are
obtained by permuting the state arrays accordingly. Variants are provided for the
ideal-gas MHD and for barotropic (isothermal or polytropic) MHD and hydrodynamics.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import (
    Array,
    Float,
)

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equation_of_state import law_pressure
from astronomix._fluid_equations._equations_mhd import (
    primitive_state_from_conserved_isothermal,
    primitive_state_from_conserved_mhd,
    total_energy_from_primitives_mhd,
)


# We only define the flux in the x-direction here; the other directions are
# obtained by permuting the arrays accordingly. For the IDEAL_GAS variant see
# ``_mhd_flux_x`` below, and for the barotropic variants see the functions that
# follow it.
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _mhd_flux_x(
    conserved_state: STATE_TYPE,
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
) -> STATE_TYPE:
    """Compute the ideal-gas MHD x-direction flux for a conserved state.

    Args:
        conserved_state: The conserved MHD state.
        minimum_density: The density floor used when recovering primitives.
        minimum_pressure: The pressure floor used when recovering primitives.
        gamma: The adiabatic index of the fluid.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The separately advected dual-energy ``g``, or
            None; when given, the pressure recovery is dual-energy switched.

    Returns:
        The MHD flux vector in the x-direction.
    """

    primitive_state = primitive_state_from_conserved_mhd(
        conserved_state,
        minimum_density,
        minimum_pressure,
        gamma,
        config,
        registered_variables,
        internal_energy_density=internal_energy_density,
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
    flux = flux.at[registered_variables.velocity_index.x].set(
        rho * v_x**2 + total_pressure - B_x**2
    )
    flux = flux.at[registered_variables.velocity_index.y].set(rho * v_x * v_y - B_x * B_y)
    flux = flux.at[registered_variables.velocity_index.z].set(rho * v_x * v_z - B_x * B_z)
    flux = flux.at[registered_variables.pressure_index].set(
        (E + total_pressure) * v_x - v_dot_B * B_x
    )
    flux = flux.at[registered_variables.magnetic_index.x].set(0.0)
    flux = flux.at[registered_variables.magnetic_index.y].set(B_y * v_x - B_x * v_y)
    flux = flux.at[registered_variables.magnetic_index.z].set(B_z * v_x - B_x * v_z)

    return flux


def _mhd_flux_barotropic_x(
    conserved_state: STATE_TYPE,
    minimum_density: Union[float, Float[Array, ""]],
    barotropic_law,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Compute the barotropic (isothermal or polytropic) MHD x-direction flux for a conserved state.

    There is no energy equation for a barotropic gas; the gas pressure follows
    from the density, ``p = c_s^2 rho`` or ``p = K rho^Gamma``.

    Args:
        conserved_state: The conserved barotropic MHD state.
        minimum_density: The density floor used when recovering primitives.
        barotropic_law: The isothermal sound speed or the ``PolytropicLaw``
            (``barotropic_law`` of ``_equation_of_state.py``).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The barotropic MHD flux vector in the x-direction.
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

    # Compute the derived quantities. The gas pressure follows from the density,
    # and the total pressure adds the magnetic pressure.
    b_squared = B_x**2 + B_y**2 + B_z**2
    p_gas = law_pressure(barotropic_law, rho, minimum_density)
    total_pressure = p_gas + 0.5 * b_squared

    # Assemble the barotropic MHD flux vector.
    flux = jnp.zeros_like(primitive_state)
    flux = flux.at[registered_variables.density_index].set(rho * v_x)
    flux = flux.at[registered_variables.velocity_index.x].set(
        rho * v_x**2 + total_pressure - B_x**2
    )
    flux = flux.at[registered_variables.velocity_index.y].set(rho * v_x * v_y - B_x * B_y)
    flux = flux.at[registered_variables.velocity_index.z].set(rho * v_x * v_z - B_x * B_z)
    flux = flux.at[registered_variables.magnetic_index.x].set(0.0)
    flux = flux.at[registered_variables.magnetic_index.y].set(B_y * v_x - B_x * v_y)
    flux = flux.at[registered_variables.magnetic_index.z].set(B_z * v_x - B_x * v_z)

    return flux


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _euler_flux_barotropic_x(
    conserved_state: STATE_TYPE,
    minimum_density: Union[float, Float[Array, ""]],
    barotropic_law,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Compute the barotropic (isothermal or polytropic) Euler x-direction flux for a conserved state.

    F = [
        ρ v_x,                         // density_index
        ρ v_x^2 + p,                   // velocity_index.x
        ρ v_x v_y,                     // velocity_index.y
        ρ v_x v_z,                     // velocity_index.z
    ]

    with ``p = c_s^2 ρ`` or ``p = K ρ^Γ``; there is no energy equation for a barotropic gas.

    Args:
        conserved_state: The conserved barotropic hydro state.
        minimum_density: The density floor (applied when
            ``clamp_in_estimates`` is set, and in the power of the polytropic pressure).
        barotropic_law: The isothermal sound speed or the ``PolytropicLaw``
            (``barotropic_law`` of ``_equation_of_state.py``).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The barotropic Euler flux vector in the x-direction.
    """

    flux_vector = jnp.zeros_like(conserved_state)

    if config.dimensionality == 1:
        m_x = conserved_state[registered_variables.velocity_index]
    else:
        m_x = conserved_state[registered_variables.velocity_index.x]

    rho = conserved_state[registered_variables.density_index]

    if config.positivity_config.clamp_in_estimates:
        rho = jnp.maximum(rho, minimum_density)

    p = law_pressure(barotropic_law, rho, minimum_density)

    # Assemble the barotropic Euler flux vector.
    flux_vector = flux_vector.at[registered_variables.density_index].set(m_x)

    if config.dimensionality == 1:
        flux_vector = flux_vector.at[registered_variables.velocity_index].set(
            m_x**2 / rho + p
        )
    elif config.dimensionality == 2:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(
            m_x**2 / rho + p
        )
        v_y = conserved_state[registered_variables.velocity_index.y] / rho
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(m_x * v_y)
    elif config.dimensionality == 3:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(
            m_x**2 / rho + p
        )
        v_y = conserved_state[registered_variables.velocity_index.y] / rho
        v_z = conserved_state[registered_variables.velocity_index.z] / rho
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(m_x * v_y)
        flux_vector = flux_vector.at[registered_variables.velocity_index.z].set(m_x * v_z)

    return flux_vector