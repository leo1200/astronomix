"""
The local Lax-Friedrichs Riemann solvers for GLM-MHD and for hydrodynamics, as in AthenaPK. The gas is an
ideal gas or, with the sound-speed function of a barotropic law, a gas without energy equation.

Conventions (interface frame, elementwise evaluation) and shared helpers: ``_face_state.py``.
"""

# jax
import jax.numpy as jnp

# astronomix functions
from astronomix._finite_volume._riemann._face_state import (
    HydroFaceState,
    HydroFlux,
    MHDFaceState,
    MHDFlux,
    _glm_interface_field_and_psi,
    _hydro_energy,
    _square,
    face_fast_speed,
    face_sound_speed,
)


def llf_mhd_flux(
    left: MHDFaceState,
    right: MHDFaceState,
    gamma,
    cleaning_speed,
    sound_speed_squared_of=None,
) -> MHDFlux:
    """
    The local Lax-Friedrichs (Rusanov) flux for GLM-MHD, as AthenaPK's
    ``glmmhd_dc_llf.hpp`` (used by the first-order flux correction). Under a
    barotropic law the face states carry the pressure of the law, and the
    energy value of the result has no meaning (the caller drops it).

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index (ideal gas only).
        cleaning_speed: The hyperbolic divergence-cleaning speed ``c_h``.
        sound_speed_squared_of: ``None`` for the ideal gas, or the squared
            sound speed of a barotropic law as a function of the density.

    Returns:
        The GLM-MHD interface flux.
    """
    # The fluxes use the normal field of the exact GLM interface solution; the
    # reconstructed ``left.field_normal`` / ``right.field_normal`` only enter
    # the fast speeds below.
    interface_field_normal, interface_psi = _glm_interface_field_and_psi(
        left,
        right,
        cleaning_speed,
    )

    # --------------- ↓ Sum of the left and right fluxes ↓ ----------------

    mass_flux_left = left.density * left.velocity_normal
    mass_flux_right = right.density * right.velocity_normal
    magnetic_term_left = 0.5 * (
        _square(left.field_transverse_1)
        + _square(left.field_transverse_2)
        - _square(interface_field_normal)
    )
    magnetic_term_right = 0.5 * (
        _square(right.field_transverse_1)
        + _square(right.field_transverse_2)
        - _square(interface_field_normal)
    )

    sum_mass = mass_flux_left + mass_flux_right
    sum_momentum_normal = (
        mass_flux_left * left.velocity_normal
        + mass_flux_right * right.velocity_normal
        + magnetic_term_left
        + magnetic_term_right
        + (left.pressure + right.pressure)
    )
    sum_momentum_transverse_1 = (
        mass_flux_left * left.velocity_transverse_1
        + mass_flux_right * right.velocity_transverse_1
        - interface_field_normal * (left.field_transverse_1 + right.field_transverse_1)
    )
    sum_momentum_transverse_2 = (
        mass_flux_left * left.velocity_transverse_2
        + mass_flux_right * right.velocity_transverse_2
        - interface_field_normal * (left.field_transverse_2 + right.field_transverse_2)
    )
    sum_field_transverse_1 = (
        left.field_transverse_1 * left.velocity_normal
        + right.field_transverse_1 * right.velocity_normal
        - interface_field_normal * (left.velocity_transverse_1 + right.velocity_transverse_1)
    )
    sum_field_transverse_2 = (
        left.field_transverse_2 * left.velocity_normal
        + right.field_transverse_2 * right.velocity_normal
        - interface_field_normal * (left.velocity_transverse_2 + right.velocity_transverse_2)
    )

    if sound_speed_squared_of is None:
        inverse_gamma_minus_one = 1.0 / (gamma - 1.0)
        energy_left = (
            left.pressure * inverse_gamma_minus_one
            + 0.5
            * left.density
            * (
                _square(left.velocity_normal)
                + _square(left.velocity_transverse_1)
                + _square(left.velocity_transverse_2)
            )
            + magnetic_term_left
            + _square(interface_field_normal)
        )
        energy_right = (
            right.pressure * inverse_gamma_minus_one
            + 0.5
            * right.density
            * (
                _square(right.velocity_normal)
                + _square(right.velocity_transverse_1)
                + _square(right.velocity_transverse_2)
            )
            + magnetic_term_right
            + _square(interface_field_normal)
        )
    else:
        # No energy equation: the energy value of the flux is formed from zeros.
        energy_left = energy_right = 0.0
    sum_energy = (
        (energy_left + left.pressure + magnetic_term_left) * left.velocity_normal
        + (energy_right + right.pressure + magnetic_term_right) * right.velocity_normal
        - interface_field_normal
        * (
            left.field_transverse_1 * left.velocity_transverse_1
            + left.field_transverse_2 * left.velocity_transverse_2
        )
        - interface_field_normal
        * (
            right.field_transverse_1 * right.velocity_transverse_1
            + right.field_transverse_2 * right.velocity_transverse_2
        )
    )

    # --------------- ↑ Sum of the left and right fluxes ↑ ----------------

    # --------------- ↓ Dissipation ↓ ----------------

    fast_speed_left = face_fast_speed(left, gamma, sound_speed_squared_of)
    fast_speed_right = face_fast_speed(right, gamma, sound_speed_squared_of)
    maximum_speed = jnp.maximum(
        jnp.abs(left.velocity_normal) + fast_speed_left,
        jnp.abs(right.velocity_normal) + fast_speed_right,
    )

    jump_mass = maximum_speed * (right.density - left.density)
    jump_momentum_normal = maximum_speed * (
        right.density * right.velocity_normal - left.density * left.velocity_normal
    )
    jump_momentum_transverse_1 = maximum_speed * (
        right.density * right.velocity_transverse_1 - left.density * left.velocity_transverse_1
    )
    jump_momentum_transverse_2 = maximum_speed * (
        right.density * right.velocity_transverse_2 - left.density * left.velocity_transverse_2
    )
    jump_energy = maximum_speed * (energy_right - energy_left)
    jump_field_transverse_1 = maximum_speed * (right.field_transverse_1 - left.field_transverse_1)
    jump_field_transverse_2 = maximum_speed * (right.field_transverse_2 - left.field_transverse_2)

    # --------------- ↑ Dissipation ↑ ----------------

    # The decoupled GLM subsystem contributes F(B_normal) = psi* and
    # F(psi) = c_h^2 B_normal* (Dedner et al. 2002); the MHD solver does not
    # touch these components.
    return MHDFlux(
        mass=0.5 * (sum_mass - jump_mass),
        momentum_normal=0.5 * (sum_momentum_normal - jump_momentum_normal),
        momentum_transverse_1=0.5 * (sum_momentum_transverse_1 - jump_momentum_transverse_1),
        momentum_transverse_2=0.5 * (sum_momentum_transverse_2 - jump_momentum_transverse_2),
        energy=0.5 * (sum_energy - jump_energy),
        field_normal=interface_psi,
        field_transverse_1=0.5 * (sum_field_transverse_1 - jump_field_transverse_1),
        field_transverse_2=0.5 * (sum_field_transverse_2 - jump_field_transverse_2),
        psi=cleaning_speed * cleaning_speed * interface_field_normal,
    )


def llf_hydro_flux(
    left: HydroFaceState,
    right: HydroFaceState,
    gamma,
    sound_speed_squared_of=None,
) -> HydroFlux:
    """
    The local Lax-Friedrichs (Rusanov) flux for hydrodynamics, as AthenaPK's
    ``hydro_dc_llf.hpp``. Under a barotropic law the face states carry the
    pressure of the law, and the energy value of the result has no meaning
    (the caller drops it).

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index (ideal gas only).
        sound_speed_squared_of: ``None`` for the ideal gas, or the squared
            sound speed of a barotropic law as a function of the density.

    Returns:
        The hydrodynamic interface flux.
    """
    # --------------- ↓ Sum of the left and right fluxes ↓ ----------------

    mass_flux_left = left.density * left.velocity_normal
    mass_flux_right = right.density * right.velocity_normal

    sum_mass = mass_flux_left + mass_flux_right
    sum_momentum_normal = (
        mass_flux_left * left.velocity_normal
        + mass_flux_right * right.velocity_normal
        + (left.pressure + right.pressure)
    )
    sum_momentum_transverse_1 = (
        mass_flux_left * left.velocity_transverse_1
        + mass_flux_right * right.velocity_transverse_1
    )
    sum_momentum_transverse_2 = (
        mass_flux_left * left.velocity_transverse_2
        + mass_flux_right * right.velocity_transverse_2
    )

    if sound_speed_squared_of is None:
        inverse_gamma_minus_one = 1.0 / (gamma - 1.0)
        energy_left = _hydro_energy(left, inverse_gamma_minus_one)
        energy_right = _hydro_energy(right, inverse_gamma_minus_one)
    else:
        # No energy equation: the energy value of the flux is formed from zeros.
        energy_left = energy_right = 0.0
    sum_energy = (energy_left + left.pressure) * left.velocity_normal + (
        energy_right + right.pressure
    ) * right.velocity_normal

    # --------------- ↑ Sum of the left and right fluxes ↑ ----------------

    # --------------- ↓ Dissipation ↓ ----------------

    sound_speed_left = face_sound_speed(left, gamma, sound_speed_squared_of)
    sound_speed_right = face_sound_speed(right, gamma, sound_speed_squared_of)
    maximum_speed = jnp.maximum(
        jnp.abs(left.velocity_normal) + sound_speed_left,
        jnp.abs(right.velocity_normal) + sound_speed_right,
    )

    jump_mass = maximum_speed * (right.density - left.density)
    jump_momentum_normal = maximum_speed * (
        right.density * right.velocity_normal - left.density * left.velocity_normal
    )
    jump_momentum_transverse_1 = maximum_speed * (
        right.density * right.velocity_transverse_1 - left.density * left.velocity_transverse_1
    )
    jump_momentum_transverse_2 = maximum_speed * (
        right.density * right.velocity_transverse_2 - left.density * left.velocity_transverse_2
    )
    jump_energy = maximum_speed * (energy_right - energy_left)

    # --------------- ↑ Dissipation ↑ ----------------

    return HydroFlux(
        mass=0.5 * (sum_mass - jump_mass),
        momentum_normal=0.5 * (sum_momentum_normal - jump_momentum_normal),
        momentum_transverse_1=0.5 * (sum_momentum_transverse_1 - jump_momentum_transverse_1),
        momentum_transverse_2=0.5 * (sum_momentum_transverse_2 - jump_momentum_transverse_2),
        energy=0.5 * (sum_energy - jump_energy),
    )
