"""
Equations for 3D adiabatic ideal magnetohydrodynamics (MHD).

Provides conversions between primitive and conserved states (ideal-gas and
isothermal variants) and the MHD thermodynamic relations (thermal pressure,
total energy and total pressure).
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
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._dual_energy_switch import (
    DUAL_ENERGY_ETA,
    dual_energy_internal_energy,
)


@partial(jax.jit, static_argnames=["registered_variables"])
def _u_squared3D(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    """Return the squared velocity magnitude from a 3D primitive state."""
    return (
        primitive_state[registered_variables.velocity_index.x] ** 2
        + primitive_state[registered_variables.velocity_index.y] ** 2
        + primitive_state[registered_variables.velocity_index.z] ** 2
    )


def _b_squared3D(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    """Return the squared magnetic field magnitude from a 3D state."""
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


def dual_switched_pressure_mhd(
    E,
    rho,
    u_squared,
    b_squared,
    gamma,
    internal_energy_density,
    eta,
):
    """
    Gas pressure with the dual-energy switch (Bryan et al. 1995), MHD.

    The internal energy recovered from the total energy,
    ``e_E = E - KE - ME``, is trustworthy only when it is a non-negligible
    fraction of ``E``; at high Mach numbers or low plasma beta floating-point
    cancellation destroys it and the separately advected internal energy
    density ``g`` is used instead (see ``dual_energy_internal_energy``). The
    recovery is coupled into the WENO flux and eigenstructure, so the scheme
    never sees the corrupted pressure.

    Args:
        E: The total energy density.
        rho: The density.
        u_squared: The squared velocity.
        b_squared: The squared magnetic field.
        gamma: The adiabatic index.
        internal_energy_density: The separately advected internal energy
            density ``g``.
        eta: The switch threshold (``DUAL_ENERGY_ETA``).

    Returns:
        The switched thermal pressure ``(gamma - 1) e_int``.
    """
    internal_energy_from_total = E - 0.5 * (rho * u_squared + b_squared)
    internal_energy = dual_energy_internal_energy(
        internal_energy_from_total,
        E,
        internal_energy_density,
        eta,
    )
    return (gamma - 1.0) * internal_energy


@jax.jit
def total_energy_from_primitives_mhd(rho, u_squared, p, b_squared, gamma):
    """Calculate the total energy from the primitive variables in MHD.

    Args:
        rho: The density.
        u_squared: The squared velocity.
        p: The thermal pressure.
        b_squared: The squared magnetic field.
        gamma: The adiabatic index.

    Returns:
        The total energy (internal + kinetic + magnetic).
    """
    return p / (gamma - 1) + 0.5 * rho * u_squared + 0.5 * b_squared


@partial(jax.jit, static_argnames=["registered_variables"])
def conserved_state_from_primitive_mhd(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Convert the primitive state to the conserved state for ideal-gas MHD.

    Currently only the 3D case is supported.

    Args:
        primitive_state: The primitive MHD state.
        gamma: The adiabatic index of the fluid.
        registered_variables: The registered variables.

    Returns:
        The conserved MHD state.
    """

    rho = primitive_state[registered_variables.density_index]

    u_squared = _u_squared3D(primitive_state, registered_variables)

    # The pressure slot holds the thermal pressure in the primitive layout.
    p = primitive_state[registered_variables.pressure_index]

    b_squared = _b_squared3D(primitive_state, registered_variables)

    # Compute the total energy and store it in the pressure/energy slot.
    E = total_energy_from_primitives_mhd(rho, u_squared, p, b_squared, gamma)

    conserved_state = primitive_state.at[registered_variables.pressure_index].set(E)

    # Convert the velocities into momentum densities.
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


@partial(jax.jit, static_argnames=["clamp", "registered_variables", "config"])
def primitive_state_from_conserved_mhd(
    conserved_state: STATE_TYPE,
    rhomin: Union[float, Float[Array, ""]],
    pgmin: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
    clamp: bool = True,
) -> STATE_TYPE:
    """Convert the conserved state to the primitive state for ideal-gas MHD.

    Currently only the 3D case is supported.

    Args:
        conserved_state: The conserved MHD state.
        rhomin: The density floor (applied when ``clamp_in_estimates`` is set).
        pgmin: The pressure floor (applied when ``clamp_in_estimates`` is set).
        gamma: The adiabatic index of the fluid.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The separately advected dual-energy ``g``, or
            None. When given, the pressure recovery is dual-energy switched, so
            the total-energy value is not used where it suffers catastrophic
            cancellation.
        clamp: Whether the estimate clamps (``clamp_in_estimates``) may act.
            Pass False for the state that is carried to the next step: those
            clamps must never modify the solution.

    Returns:
        The primitive MHD state.
    """

    rho = conserved_state[registered_variables.density_index]
    E = conserved_state[registered_variables.pressure_index]

    ux = conserved_state[registered_variables.velocity_index.x] / rho
    uy = conserved_state[registered_variables.velocity_index.y] / rho
    uz = conserved_state[registered_variables.velocity_index.z] / rho

    u_squared = ux**2 + uy**2 + uz**2

    b_squared = _b_squared3D(conserved_state, registered_variables)

    if internal_energy_density is None:
        p = thermal_pressure_from_energy_mhd(E, rho, u_squared, b_squared, gamma)
    else:
        p = dual_switched_pressure_mhd(
            E,
            rho,
            u_squared,
            b_squared,
            gamma,
            internal_energy_density,
            DUAL_ENERGY_ETA,
        )

    # Write the recovered thermal pressure and velocities into the primitive state.
    primitive_state = conserved_state.at[registered_variables.pressure_index].set(p)

    primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(ux)
    primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(uy)
    primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(uz)

    if clamp and config.positivity_config.clamp_in_estimates:
        # Keep density and pressure positive in primitives that only feed
        # estimates (time step, wave speeds, source terms).
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


@partial(jax.jit, static_argnames=["clamp", "registered_variables", "config"])
def primitive_state_from_conserved_isothermal(
    conserved_state: STATE_TYPE,
    minimum_density: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    clamp: bool = True,
) -> STATE_TYPE:
    """Convert the conserved state to the primitive state for the isothermal case.

    Args:
        conserved_state: The conserved state (hydro or MHD).
        minimum_density: The density floor (applied when ``clamp_in_estimates``
            is set).
        config: The simulation configuration.
        registered_variables: The registered variables.
        clamp: Whether the estimate clamp (``clamp_in_estimates``) may act.
            Pass False for the state that is carried to the next step, as in
            ``primitive_state_from_conserved_mhd``.

    Returns:
        The primitive state.
    """

    rho = conserved_state[registered_variables.density_index]

    if clamp and config.positivity_config.clamp_in_estimates:
        rho = jnp.maximum(rho, minimum_density)

    if config.dimensionality == 1 and not config.mhd:
        primitive_state = conserved_state.at[registered_variables.velocity_index].set(
            conserved_state[registered_variables.velocity_index] / rho
        )
    elif config.dimensionality == 2 and not config.mhd:
        primitive_state = conserved_state.at[registered_variables.velocity_index.x].set(
            conserved_state[registered_variables.velocity_index.x] / rho
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(
            conserved_state[registered_variables.velocity_index.y] / rho
        )
    # In the FD MHD case there are always 3 velocity components, even in 2D.
    elif config.dimensionality == 3 or config.mhd:
        primitive_state = conserved_state.at[registered_variables.velocity_index.x].set(
            conserved_state[registered_variables.velocity_index.x] / rho
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(
            conserved_state[registered_variables.velocity_index.y] / rho
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(
            conserved_state[registered_variables.velocity_index.z] / rho
        )

    # There is no pressure variable in the isothermal case, so nothing to set there.
    return primitive_state


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def conserved_state_from_primitive_isothermal(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Convert the primitive state to the conserved state for the isothermal case.

    Args:
        primitive_state: The primitive state (hydro or MHD).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The conserved state.
    """

    rho = primitive_state[registered_variables.density_index]

    if config.dimensionality == 1 and not config.mhd:
        primitive_state = primitive_state.at[registered_variables.velocity_index].set(
            primitive_state[registered_variables.velocity_index] * rho
        )
    elif config.dimensionality == 2 and not config.mhd:
        primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(
            primitive_state[registered_variables.velocity_index.x] * rho
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(
            primitive_state[registered_variables.velocity_index.y] * rho
        )
    # In the FD MHD case there are always 3 velocity components, even in 2D.
    elif config.dimensionality == 3 or config.mhd:
        primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(
            primitive_state[registered_variables.velocity_index.x] * rho
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(
            primitive_state[registered_variables.velocity_index.y] * rho
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(
            primitive_state[registered_variables.velocity_index.z] * rho
        )

    # There is no pressure variable in the isothermal case, so nothing to set there.
    return primitive_state


