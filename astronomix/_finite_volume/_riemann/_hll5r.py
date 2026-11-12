"""
HLL5R: the 5-wave relaxation solver for MHD of Bouchut, Klingenberg and Waagan (2007; 2010, Numer. Math. 115,
647), with a normal field that may jump at the interface (Waagan 2009, J. Comput. Phys. 228, 8609, section 4;
Waagan, Federrath and Klingenberg 2011): the solver of the FLASH scheme of Beattie et al. (2025).

The relaxation system replaces the total pressure of the normal direction and the magnetic tension by the
relaxation pressures ``pi = p + |B_t|^2 / 2 - B_n^2 / 2`` and ``pi_t = -B_n B_t`` (index t: the two transverse
components), which travel with two Lagrangian speeds per side, ``c_b`` (the normal waves) and ``c_a`` (the
transverse waves). All its waves are linearly degenerate: a contact with the speed ``u*`` and, on each side
of it, a normal and a transverse wave, the faster of the two outside. The speeds are chosen from the two
states (Bouchut et al. 2010, eq. 3.50 to 3.55) so that densities and internal energies stay positive.

The pressure law enters through the square of the sound speed of each state and the exponent of the law in
``alpha = (exponent + 1) / 2``: the adiabatic index of the ideal gas, the exponent of a polytrope, one for the
isothermal gas. Without energy equation (a barotropic law) the energy value of the flux has no meaning.

A normal field that differs between the two states jumps at the contact: each side uses its own normal field,
and the two neighbours of the interface get different magnetic fluxes (the upwind form of the Powell term of
the induction equation). ``hll5r_flux`` returns both; they differ by ``-u* (B_n,right - B_n,left)`` with the
velocity vector of the contact, and are equal for a continuous normal field.

Three quotients can lose their denominator, all by a transverse Lagrangian speed ``c_a`` that vanishes with
the normal field of its side; ``_positive_or_one`` replaces a zero by one there. Where the denominator is
positive the replacement is the identity, so the flux is unchanged. Where it is zero the numerators are zero
as well (they carry the normal field, or a jump across the transverse wave, which then coincides with the
contact and is never the region of the interface). A contact at rest needs no guard: no quotient contains
``u*``, and the two states next to the contact have the same flux at ``u* = 0``.

Conventions (interface frame, elementwise evaluation): ``_face_state.py``.
"""

# typing
from typing import Tuple

# jax
import jax.numpy as jnp

# astronomix functions
from astronomix._finite_volume._riemann._face_state import (
    MHDFaceState,
    MHDFlux,
    _square,
)


def _positive_or_one(value):
    """Return ``value`` where it is positive and one elsewhere (a denominator whose numerator vanishes with it)."""
    return jnp.where(value > 0.0, value, 1.0)


def _square_root(value):
    """The square root with the derivative zero (not infinite) at zero."""
    return jnp.where(value > 0.0, jnp.sqrt(_positive_or_one(value)), 0.0)


def _relaxation_pressures(state: MHDFaceState):
    """The relaxation pressures ``(pi, pi_t1, pi_t2)`` of a state at equilibrium."""
    transverse_field_squared = _square(state.field_transverse_1) + _square(state.field_transverse_2)
    normal_pressure = state.pressure + 0.5 * transverse_field_squared - 0.5 * _square(state.field_normal)
    return (
        normal_pressure,
        -state.field_normal * state.field_transverse_1,
        -state.field_normal * state.field_transverse_2,
    )


def _sound_speeds_squared(left: MHDFaceState, right: MHDFaceState, exponent, sound_speed_squared_of):
    """The squared sound speeds of the two states: the ideal gas, or the given barotropic law."""
    if sound_speed_squared_of is None:
        return exponent * left.pressure / left.density, exponent * right.pressure / right.density
    return sound_speed_squared_of(left.density), sound_speed_squared_of(right.density)


def _magnetic_parts(state: MHDFaceState):
    """The magnetic parts of the squared speeds: ``(|B_t|^2 + |B_n B_t|) / rho`` and ``B_n^2 + |B_n B_t|``."""
    transverse_field_squared = _square(state.field_transverse_1) + _square(state.field_transverse_2)
    field_product = jnp.abs(state.field_normal) * _square_root(transverse_field_squared)
    return (transverse_field_squared + field_product) / state.density, _square(state.field_normal) + field_product


def _side_lagrangian_speeds(density, sound_speed_squared, magnetic_parts, speed, compression, alpha):
    """
    The Lagrangian speeds ``(c_a, c_b)`` of one side (Bouchut et al. 2010, eq. 3.50 to 3.55).

    Args:
        density: The density of the side's state.
        sound_speed_squared: Its squared sound speed.
        magnetic_parts: Its magnetic parts (``_magnetic_parts``).
        speed: The speed ``a`` of the state, ``a^2 = c_s^2 + (|B_t|^2 + |B_n B_t|) / rho``.
        compression: The measure ``X >= 0`` of the compression the side can undergo, in units of ``a``.
        alpha: Half the exponent of the pressure law plus one half.

    Returns:
        ``(c_a, c_b)``.
    """
    density_ratio = 1.0 - compression / (1.0 + alpha * compression)
    relaxed_speed = jnp.sqrt(sound_speed_squared + magnetic_parts[0] / density_ratio)
    normal_speed = density * (relaxed_speed + alpha * speed * compression)
    transverse_speed = _square_root(density / density_ratio * magnetic_parts[1])
    return transverse_speed, normal_speed


def hll5r_lagrangian_speeds(left: MHDFaceState, right: MHDFaceState, exponent, sound_speed_squared_of=None):
    """
    The Lagrangian speeds of the relaxation system at an interface.

    Args:
        left: The left state.
        right: The right state.
        exponent: The adiabatic index of the ideal gas, or the exponent of the barotropic law.
        sound_speed_squared_of: ``None`` for the ideal gas, or the squared sound speed of a barotropic law as
            a function of the density.

    Returns:
        ``((c_a, c_b) of the left side, (c_a, c_b) of the right side)``.
    """
    alpha = 0.5 * (exponent + 1.0)
    states = (left, right)
    sounds = _sound_speeds_squared(left, right, exponent, sound_speed_squared_of)
    parts = tuple(_magnetic_parts(state) for state in states)
    speeds = tuple(jnp.sqrt(sound + part[0]) for sound, part in zip(sounds, parts))

    # A side is compressed by the closing speed of the two states and by an excess of pi on the other side.
    closing_speed = jnp.maximum(left.velocity_normal - right.velocity_normal, 0.0)
    pressure_jump = (_relaxation_pressures(right)[0] - _relaxation_pressures(left)[0]) / (
        left.density * speeds[0] + right.density * speeds[1]
    )
    excesses = (jnp.maximum(pressure_jump, 0.0), jnp.maximum(-pressure_jump, 0.0))
    return tuple(
        _side_lagrangian_speeds(state.density, sound, part, speed, (closing_speed + excess) / speed, alpha)
        for state, sound, part, speed, excess in zip(states, sounds, parts, speeds, excesses)
    )


def hll5r_signal_speed(left: MHDFaceState, right: MHDFaceState, exponent, sound_speed_squared_of=None):
    """
    The largest signal speed of the solver at an interface, the speed its time-step condition bounds (Bouchut
    et al. 2010, eq. 2.15). For equal states it exceeds the fast speed by a factor of up to 1.099; with a
    compression or a pressure jump at the interface it grows beyond that.

    Args: as ``hll5r_lagrangian_speeds``.

    Returns:
        The larger magnitude of the two outermost wave speeds.
    """
    left_speeds, right_speeds = hll5r_lagrangian_speeds(left, right, exponent, sound_speed_squared_of)
    leftmost = left.velocity_normal - jnp.maximum(*left_speeds) / left.density
    rightmost = right.velocity_normal + jnp.maximum(*right_speeds) / right.density
    return jnp.maximum(jnp.abs(leftmost), jnp.abs(rightmost))


def _contact_values(left: MHDFaceState, right: MHDFaceState, left_speeds, right_speeds):
    """
    The values next to the contact: normal velocity and ``pi`` between the two normal waves, transverse
    velocities and ``pi_t`` between the two transverse waves.

    Returns:
        ``(u*, pi*, (v_t1*, v_t2*), (pi_t1*, pi_t2*))``.
    """
    left_pressures = _relaxation_pressures(left)
    right_pressures = _relaxation_pressures(right)
    (left_transverse, left_normal), (right_transverse, right_normal) = left_speeds, right_speeds

    normal_sum = left_normal + right_normal
    velocity = (
        left_normal * left.velocity_normal + right_normal * right.velocity_normal
        + left_pressures[0] - right_pressures[0]
    ) / normal_sum
    pressure = (
        right_normal * left_pressures[0] + left_normal * right_pressures[0]
        - left_normal * right_normal * (right.velocity_normal - left.velocity_normal)
    ) / normal_sum

    # Both transverse speeds vanish only with both normal fields: the numerators are then zero.
    transverse_sum = _positive_or_one(left_transverse + right_transverse)
    transverse_velocities = []
    transverse_pressures = []
    for component in (1, 2):
        left_velocity, right_velocity = left[1 + component], right[1 + component]
        left_pressure, right_pressure = left_pressures[component], right_pressures[component]
        transverse_velocities.append(
            (left_transverse * left_velocity + right_transverse * right_velocity + left_pressure - right_pressure)
            / transverse_sum
        )
        transverse_pressures.append(
            (
                right_transverse * left_pressure + left_transverse * right_pressure
                - left_transverse * right_transverse * (right_velocity - left_velocity)
            )
            / transverse_sum
        )
    return velocity, pressure, tuple(transverse_velocities), tuple(transverse_pressures)


def _flux_of_side(side: MHDFaceState, side_speeds, direction, contact, specific_internal_energy) -> MHDFlux:
    """
    The relaxation flux at the interface from the wave fan of one side of the contact.

    The side has three regions: its own state, the state behind its normal wave (density, normal velocity
    and ``pi`` take their contact values) and behind its transverse wave (transverse velocities and ``pi_t``
    take theirs). The other quantities follow from what the waves of a side carry unchanged:
    ``1 / rho + pi / c_b^2``, ``B_t / rho + B_n pi_t / c_a^2`` and
    ``e + |B|^2 / (2 rho) - pi^2 / (2 c_b^2) - |pi_t|^2 / (2 c_a^2)``, written with the jump relations
    ``pi* - pi = direction c_b (u* - u)`` and ``pi_t* - pi_t = direction c_a (v_t* - v_t)``.

    Args:
        side: The state of the side (with its own normal field).
        side_speeds: Its Lagrangian speeds ``(c_a, c_b)``.
        direction: -1 for the left side, +1 for the right side.
        contact: The values next to the contact (``_contact_values``).
        specific_internal_energy: The internal energy per mass of the side's state.

    Returns:
        The flux, with a zero for the normal field.
    """
    transverse_speed, normal_speed = side_speeds
    contact_velocity, contact_pressure, contact_transverse_velocities, contact_transverse_pressures = contact
    side_pressures = _relaxation_pressures(side)
    safe_transverse_speed = _positive_or_one(transverse_speed)

    # Which waves of the side lie between its state and the interface. The transverse wave is the outer one
    # where c_a >= c_b; with c_a = 0 it coincides with the contact and is never passed.
    contact_density = 1.0 / (1.0 / side.density - direction * (contact_velocity - side.velocity_normal) / normal_speed)
    normal_wave = side.velocity_normal + direction * normal_speed / side.density
    transverse_wave = jnp.where(
        transverse_speed >= normal_speed,
        side.velocity_normal + direction * transverse_speed / side.density,
        contact_velocity + direction * transverse_speed / contact_density,
    )
    behind_normal_wave = direction * normal_wave > 0.0
    behind_transverse_wave = direction * transverse_wave > 0.0

    density = jnp.where(behind_normal_wave, contact_density, side.density)
    velocity = jnp.where(behind_normal_wave, contact_velocity, side.velocity_normal)
    pressure = jnp.where(behind_normal_wave, contact_pressure, side_pressures[0])
    specific_energy = (
        specific_internal_energy
        + 0.5 * (_square(side.field_normal) + _square(side.field_transverse_1) + _square(side.field_transverse_2))
        / side.density
        + direction * (velocity - side.velocity_normal) * (pressure + side_pressures[0]) / (2.0 * normal_speed)
        + 0.5 * _square(velocity)
    )

    transverse_momentum_fluxes = []
    field_fluxes = []
    energy_flux_of_tension = 0.0
    for component, contact_transverse_velocity, contact_transverse_pressure in zip(
        (1, 2), contact_transverse_velocities, contact_transverse_pressures
    ):
        side_velocity, side_field, side_pressure = side[1 + component], side[5 + component], side_pressures[component]
        transverse_velocity = jnp.where(behind_transverse_wave, contact_transverse_velocity, side_velocity)
        transverse_pressure = jnp.where(behind_transverse_wave, contact_transverse_pressure, side_pressure)
        velocity_jump = transverse_velocity - side_velocity
        field = density * (
            side_field / side.density - direction * side.field_normal * velocity_jump / safe_transverse_speed
        )
        specific_energy = (
            specific_energy
            + direction * velocity_jump * (transverse_pressure + side_pressure) / (2.0 * safe_transverse_speed)
            + 0.5 * _square(transverse_velocity)
        )
        transverse_momentum_fluxes.append(density * velocity * transverse_velocity + transverse_pressure)
        field_fluxes.append(field * velocity - side.field_normal * transverse_velocity)
        energy_flux_of_tension = energy_flux_of_tension + transverse_pressure * transverse_velocity

    zero = jnp.zeros_like(density)
    return MHDFlux(
        density * velocity,
        density * _square(velocity) + pressure,
        transverse_momentum_fluxes[0],
        transverse_momentum_fluxes[1],
        (density * specific_energy + pressure) * velocity + energy_flux_of_tension,
        zero,
        field_fluxes[0],
        field_fluxes[1],
        zero,
    )


def hll5r_flux(
    left: MHDFaceState,
    right: MHDFaceState,
    exponent,
    sound_speed_squared_of=None,
) -> Tuple[MHDFlux, MHDFlux]:
    """
    The HLL5R flux for a cell-centred field without cleaning variable (``psi`` is ignored).

    Args:
        left: The left state.
        right: The right state.
        exponent: The adiabatic index of the ideal gas, or the exponent of the barotropic law.
        sound_speed_squared_of: ``None`` for the ideal gas, or the squared sound speed of a barotropic law as
            a function of the density (the energy value of the flux then has no meaning).

    Returns:
        The flux for the left neighbour of the interface and the flux for its right neighbour. They differ in
        the magnetic field only, by the jump of the normal field times the velocity vector of the contact.
    """
    left_speeds, right_speeds = hll5r_lagrangian_speeds(left, right, exponent, sound_speed_squared_of)
    contact = _contact_values(left, right, left_speeds, right_speeds)
    contact_velocity, _, contact_transverse_velocities, _ = contact

    # The interface lies on the left of the contact for u* >= 0: evaluate the fan of that side only.
    from_left = contact_velocity >= 0.0
    side = MHDFaceState(
        *(jnp.where(from_left, left_value, right_value) for left_value, right_value in zip(left, right))
    )
    side_speeds = tuple(
        jnp.where(from_left, left_speed, right_speed) for left_speed, right_speed in zip(left_speeds, right_speeds)
    )
    direction = jnp.where(from_left, -1.0, 1.0)
    if sound_speed_squared_of is None:
        specific_internal_energy = side.pressure / ((exponent - 1.0) * side.density)
    else:
        specific_internal_energy = jnp.zeros_like(side.density)
    flux = _flux_of_side(side, side_speeds, direction, contact, specific_internal_energy)

    # The Powell term in upwind form: the neighbour on the far side of the contact sees the jump of the
    # normal field, carried with the velocity vector of the contact.
    field_jump = right.field_normal - left.field_normal
    contact_velocities = (contact_velocity,) + contact_transverse_velocities
    field_fluxes = (flux.field_normal, flux.field_transverse_1, flux.field_transverse_2)
    for_left = [
        field_flux + jnp.where(from_left, 0.0, velocity * field_jump)
        for field_flux, velocity in zip(field_fluxes, contact_velocities)
    ]
    for_right = [
        field_flux - jnp.where(from_left, velocity * field_jump, 0.0)
        for field_flux, velocity in zip(field_fluxes, contact_velocities)
    ]
    return (
        flux._replace(field_normal=for_left[0], field_transverse_1=for_left[1], field_transverse_2=for_left[2]),
        flux._replace(field_normal=for_right[0], field_transverse_1=for_right[1], field_transverse_2=for_right[2]),
    )
