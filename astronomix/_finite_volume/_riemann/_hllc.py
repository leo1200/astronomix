"""
The HLLC Riemann solver for adiabatic hydrodynamics, as in AthenaPK.

Conventions (interface frame, elementwise evaluation) and shared helpers: ``_face_state.py``.
"""

# jax
import jax.numpy as jnp

# astronomix functions
from astronomix._finite_volume._riemann._face_state import (
    HydroFaceState,
    HydroFlux,
    _TINY_NUMBER,
    _hydro_energy,
    _hydro_flux_minus_speed_times_state,
    adiabatic_sound_speed,
)


def hllc_hydro_flux(left: HydroFaceState, right: HydroFaceState, gamma) -> HydroFlux:
    """
    The HLLC flux for adiabatic hydrodynamics with PVRS wave-speed estimates
    (Toro 10.5.2), as AthenaPK's ``hydro_hllc.hpp``.

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index.

    Returns:
        The hydrodynamic interface flux.
    """
    inverse_gamma_minus_one = 1.0 / (gamma - 1.0)

    # --------------- ↓ Wave speeds ↓ ----------------

    sound_speed_left = adiabatic_sound_speed(gamma, left.density, left.pressure)
    sound_speed_right = adiabatic_sound_speed(gamma, right.density, right.pressure)
    energy_left = _hydro_energy(left, inverse_gamma_minus_one)
    energy_right = _hydro_energy(right, inverse_gamma_minus_one)

    # The middle-state pressure estimate of the primitive-variable solver.
    average_density = 0.5 * (left.density + right.density)
    average_sound_speed = 0.5 * (sound_speed_left + sound_speed_right)
    middle_pressure = 0.5 * (
        left.pressure
        + right.pressure
        + (left.velocity_normal - right.velocity_normal) * average_density * average_sound_speed
    )

    # A shock (middle pressure above the outer one) moves faster than sound;
    # these factors correct the outer wave speeds for its strength. The square
    # root's argument is at least one wherever its branch is selected; the
    # guard only keeps the discarded branch real in a strong rarefaction, whose
    # NaN derivative would otherwise leak into reverse-mode gradients.
    shock_factor_left = jnp.where(
        middle_pressure <= left.pressure,
        1.0,
        jnp.sqrt(
            jnp.maximum(
                1.0 + (gamma + 1) / (2 * gamma) * (middle_pressure / left.pressure - 1.0),
                1.0,
            )
        ),
    )
    shock_factor_right = jnp.where(
        middle_pressure <= right.pressure,
        1.0,
        jnp.sqrt(
            jnp.maximum(
                1.0 + (gamma + 1) / (2 * gamma) * (middle_pressure / right.pressure - 1.0),
                1.0,
            )
        ),
    )

    speed_left = left.velocity_normal - sound_speed_left * shock_factor_left
    speed_right = right.velocity_normal + sound_speed_right * shock_factor_right

    speed_plus = jnp.where(speed_right > 0.0, speed_right, _TINY_NUMBER)
    speed_minus = jnp.where(speed_left < 0.0, speed_left, -_TINY_NUMBER)

    # --------------- ↑ Wave speeds ↑ ----------------

    # --------------- ↓ Contact wave ↓ ----------------

    # The contact speed and pressure use the unclamped outer wave speeds.
    velocity_minus_speed_left = left.velocity_normal - speed_left
    velocity_minus_speed_right = right.velocity_normal - speed_right

    momentum_term_left = (
        left.pressure + velocity_minus_speed_left * left.density * left.velocity_normal
    )
    momentum_term_right = (
        right.pressure + velocity_minus_speed_right * right.density * right.velocity_normal
    )

    mass_term_left = left.density * velocity_minus_speed_left
    mass_term_right = -(right.density * velocity_minus_speed_right)

    contact_speed = (momentum_term_left - momentum_term_right) / (mass_term_left + mass_term_right)
    contact_pressure = (
        mass_term_left * momentum_term_right + mass_term_right * momentum_term_left
    ) / (mass_term_left + mass_term_right)
    contact_pressure = jnp.where(contact_pressure > 0.0, contact_pressure, 0.0)

    # --------------- ↑ Contact wave ↑ ----------------

    # --------------- ↓ HLLC flux ↓ ----------------

    # The outer fluxes use the clamped wave speeds.
    relative_velocity_left = left.velocity_normal - speed_minus
    relative_velocity_right = right.velocity_normal - speed_plus

    flux_left = _hydro_flux_minus_speed_times_state(left, relative_velocity_left, energy_left)
    flux_right = _hydro_flux_minus_speed_times_state(right, relative_velocity_right, energy_right)

    # The weights of the left, right and contact contributions.
    contact_moves_right = contact_speed >= 0.0
    safe_left_denominator = jnp.where(contact_moves_right, contact_speed - speed_minus, 1.0)
    safe_right_denominator = jnp.where(contact_moves_right, 1.0, speed_plus - contact_speed)
    weight_left = jnp.where(contact_moves_right, contact_speed / safe_left_denominator, 0.0)
    weight_right = jnp.where(contact_moves_right, 0.0, -contact_speed / safe_right_denominator)
    weight_contact = jnp.where(
        contact_moves_right,
        -speed_minus / safe_left_denominator,
        speed_plus / safe_right_denominator,
    )

    # --------------- ↑ HLLC flux ↑ ----------------

    return HydroFlux(
        mass=weight_left * flux_left.mass + weight_right * flux_right.mass,
        momentum_normal=(
            weight_left * flux_left.momentum_normal
            + weight_right * flux_right.momentum_normal
            + weight_contact * contact_pressure
        ),
        momentum_transverse_1=(
            weight_left * flux_left.momentum_transverse_1
            + weight_right * flux_right.momentum_transverse_1
        ),
        momentum_transverse_2=(
            weight_left * flux_left.momentum_transverse_2
            + weight_right * flux_right.momentum_transverse_2
        ),
        energy=(
            weight_left * flux_left.energy
            + weight_right * flux_right.energy
            + weight_contact * contact_pressure * contact_speed
        ),
    )
