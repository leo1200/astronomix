"""
The HLLE Riemann solvers for GLM-MHD and for hydrodynamics, as in AthenaPK. The gas is an ideal gas or, with
the sound-speed function of a barotropic law, a gas without energy equation (Athena++'s isothermal branch).

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
    _MHDConserved,
    _TINY_NUMBER,
    _glm_interface_field_and_psi,
    _hydro_energy,
    _hydro_flux_minus_speed_times_state,
    _square,
    barotropic_roe_sound_speed_squared,
    face_fast_speed,
    face_sound_speed,
)


def hlle_mhd_flux(
    left: MHDFaceState,
    right: MHDFaceState,
    gamma,
    cleaning_speed,
    sound_speed_squared_of=None,
) -> MHDFlux:
    """
    The HLLE flux for GLM-MHD with Roe-averaged (Einfeldt) wave speeds,
    as AthenaPK's ``glmmhd_hlle.hpp``.

    Under a barotropic law the Roe-averaged fast speed is the one of Athena++'s
    isothermal solver (Stone et al. 2008, eq. B18 without the energy), with
    the averaged sound speed of ``barotropic_roe_sound_speed_squared``. The face
    states then carry the pressure of the law, and the energy value of the
    result has no meaning (the caller drops it).

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

    # --------------- ↓ Roe-averaged state ↓ ----------------

    sqrt_density_left = jnp.sqrt(left.density)
    sqrt_density_right = jnp.sqrt(right.density)
    inverse_sum_sqrt_density = 1.0 / (sqrt_density_left + sqrt_density_right)

    roe_density = sqrt_density_left * sqrt_density_right
    roe_velocity_normal = (
        sqrt_density_left * left.velocity_normal + sqrt_density_right * right.velocity_normal
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_1 = (
        sqrt_density_left * left.velocity_transverse_1
        + sqrt_density_right * right.velocity_transverse_1
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_2 = (
        sqrt_density_left * left.velocity_transverse_2
        + sqrt_density_right * right.velocity_transverse_2
    ) * inverse_sum_sqrt_density
    # The Roe average of the field weights the sides the other way round.
    roe_field_transverse_1 = (
        sqrt_density_right * left.field_transverse_1
        + sqrt_density_left * right.field_transverse_1
    ) * inverse_sum_sqrt_density
    roe_field_transverse_2 = (
        sqrt_density_right * left.field_transverse_2
        + sqrt_density_left * right.field_transverse_2
    ) * inverse_sum_sqrt_density

    # The terms X (transverse field jump) and Y (density ratio) of the
    # Roe-averaged MHD eigensystem (Stone et al. 2008, appendix B).
    transverse_field_jump_term = (
        0.5
        * (
            _square(left.field_transverse_1 - right.field_transverse_1)
            + _square(left.field_transverse_2 - right.field_transverse_2)
        )
        / _square(sqrt_density_left + sqrt_density_right)
    )
    density_ratio_term = 0.5 * (left.density + right.density) / roe_density

    magnetic_pressure_left = 0.5 * (
        _square(interface_field_normal)
        + _square(left.field_transverse_1)
        + _square(left.field_transverse_2)
    )
    magnetic_pressure_right = 0.5 * (
        _square(interface_field_normal)
        + _square(right.field_transverse_1)
        + _square(right.field_transverse_2)
    )
    roe_transverse_field_squared = _square(roe_field_transverse_1) + _square(roe_field_transverse_2)
    normal_alfven_speed_squared = _square(interface_field_normal) / roe_density

    # --------------- ↑ Roe-averaged state ↑ ----------------

    # --------------- ↓ Wave speeds ↓ ----------------

    fast_speed_left = face_fast_speed(left, gamma, sound_speed_squared_of)
    fast_speed_right = face_fast_speed(right, gamma, sound_speed_squared_of)

    # The energies of the sides and the two terms of the Roe-averaged fast
    # speed that depend on the equation of state (Stone et al. 2008, eq. B18).
    if sound_speed_squared_of is None:
        gamma_minus_one = gamma - 1.0
        gamma_minus_two = gamma_minus_one - 1.0

        # Roe (1981): average the enthalpy H = (E + P) / rho rather than E or P.
        energy_left = (
            left.pressure / gamma_minus_one
            + 0.5
            * left.density
            * (
                _square(left.velocity_normal)
                + _square(left.velocity_transverse_1)
                + _square(left.velocity_transverse_2)
            )
            + magnetic_pressure_left
        )
        energy_right = (
            right.pressure / gamma_minus_one
            + 0.5
            * right.density
            * (
                _square(right.velocity_normal)
                + _square(right.velocity_transverse_1)
                + _square(right.velocity_transverse_2)
            )
            + magnetic_pressure_right
        )
        roe_enthalpy = (
            (energy_left + left.pressure + magnetic_pressure_left) / sqrt_density_left
            + (energy_right + right.pressure + magnetic_pressure_right) / sqrt_density_right
        ) * inverse_sum_sqrt_density

        transverse_field_star_squared = (
            gamma_minus_one - gamma_minus_two * density_ratio_term
        ) * roe_transverse_field_squared
        enthalpy_minus_magnetic = roe_enthalpy - (
            normal_alfven_speed_squared + roe_transverse_field_squared / roe_density
        )
        roe_velocity_squared = (
            _square(roe_velocity_normal)
            + _square(roe_velocity_transverse_1)
            + _square(roe_velocity_transverse_2)
        )
        sound_speed_squared_tilde = jnp.maximum(
            gamma_minus_one * (enthalpy_minus_magnetic - 0.5 * roe_velocity_squared)
            - gamma_minus_two * transverse_field_jump_term,
            0.0,
        )
    else:
        # No energy equation: the energy value of the flux is formed from zeros.
        energy_left = energy_right = 0.0
        transverse_field_star_squared = density_ratio_term * roe_transverse_field_squared
        sound_speed_squared_tilde = (
            barotropic_roe_sound_speed_squared(left, right, sound_speed_squared_of) + transverse_field_jump_term
        )

    transverse_alfven_speed_squared = transverse_field_star_squared / roe_density
    squared_speed_sum = (
        normal_alfven_speed_squared + transverse_alfven_speed_squared + sound_speed_squared_tilde
    )
    squared_speed_difference = (
        normal_alfven_speed_squared + transverse_alfven_speed_squared - sound_speed_squared_tilde
    )
    discriminant_root = jnp.sqrt(
        squared_speed_difference * squared_speed_difference
        + 4.0 * sound_speed_squared_tilde * transverse_alfven_speed_squared
    )
    roe_fast_speed = jnp.sqrt(0.5 * (squared_speed_sum + discriminant_root))

    speed_left = jnp.minimum(
        roe_velocity_normal - roe_fast_speed,
        left.velocity_normal - fast_speed_left,
    )
    speed_right = jnp.maximum(
        roe_velocity_normal + roe_fast_speed,
        right.velocity_normal + fast_speed_right,
    )

    speed_plus = jnp.where(speed_right > 0.0, speed_right, 0.0)
    speed_minus = jnp.where(speed_left < 0.0, speed_left, 0.0)

    # --------------- ↑ Wave speeds ↑ ----------------

    # --------------- ↓ HLLE flux ↓ ----------------

    # The left/right fluxes along the lines speed_minus / speed_plus:
    # F_L - S_L U_L and F_R - S_R U_R.
    relative_velocity_left = left.velocity_normal - speed_minus
    relative_velocity_right = right.velocity_normal - speed_plus

    def side_flux(state: MHDFaceState, relative_velocity, magnetic_pressure, energy):
        """``F - S U`` of one outer state, with ``relative_velocity = v_normal - S``."""
        return _MHDConserved(
            density=state.density * relative_velocity,
            momentum_normal=(
                state.density * state.velocity_normal * relative_velocity
                + magnetic_pressure
                - _square(interface_field_normal)
                + state.pressure
            ),
            momentum_transverse_1=(
                state.density * state.velocity_transverse_1 * relative_velocity
                - interface_field_normal * state.field_transverse_1
            ),
            momentum_transverse_2=(
                state.density * state.velocity_transverse_2 * relative_velocity
                - interface_field_normal * state.field_transverse_2
            ),
            energy=(
                energy * relative_velocity
                + state.velocity_normal
                * (state.pressure + magnetic_pressure - _square(interface_field_normal))
                - interface_field_normal
                * (
                    state.field_transverse_1 * state.velocity_transverse_1
                    + state.field_transverse_2 * state.velocity_transverse_2
                )
            ),
            field_transverse_1=(
                state.field_transverse_1 * relative_velocity
                - interface_field_normal * state.velocity_transverse_1
            ),
            field_transverse_2=(
                state.field_transverse_2 * relative_velocity
                - interface_field_normal * state.velocity_transverse_2
            ),
        )

    flux_left = side_flux(left, relative_velocity_left, magnetic_pressure_left, energy_left)
    flux_right = side_flux(right, relative_velocity_right, magnetic_pressure_right, energy_right)

    speeds_differ = speed_plus != speed_minus
    safe_speed_difference = jnp.where(speeds_differ, speed_plus - speed_minus, 1.0)
    weight = jnp.where(
        speeds_differ,
        0.5 * (speed_plus + speed_minus) / safe_speed_difference,
        0.0,
    )

    hlle_flux = _MHDConserved(
        *(
            0.5 * (value_left + value_right) + (value_left - value_right) * weight
            for value_left, value_right in zip(flux_left, flux_right)
        )
    )

    # --------------- ↑ HLLE flux ↑ ----------------

    # The decoupled GLM subsystem contributes F(B_normal) = psi* and
    # F(psi) = c_h^2 B_normal* (Dedner et al. 2002); the MHD solver does not
    # touch these components.
    return MHDFlux(
        mass=hlle_flux.density,
        momentum_normal=hlle_flux.momentum_normal,
        momentum_transverse_1=hlle_flux.momentum_transverse_1,
        momentum_transverse_2=hlle_flux.momentum_transverse_2,
        energy=hlle_flux.energy,
        field_normal=interface_psi,
        field_transverse_1=hlle_flux.field_transverse_1,
        field_transverse_2=hlle_flux.field_transverse_2,
        psi=cleaning_speed * cleaning_speed * interface_field_normal,
    )


def hlle_hydro_flux(
    left: HydroFaceState,
    right: HydroFaceState,
    gamma,
    sound_speed_squared_of=None,
) -> HydroFlux:
    """
    The HLLE flux for hydrodynamics with Roe-averaged (Einfeldt) wave speeds,
    as AthenaPK's ``hydro_hlle.hpp``.

    Under a barotropic law the averaged sound speed of the bounds is the one
    of ``barotropic_roe_sound_speed_squared``. The face states then carry the
    pressure of the law, and the energy value of the result has no meaning
    (the caller drops it).

    NOTE: AthenaPK clamps a non-negative left speed to ``+TINY_NUMBER`` (HLLC
    uses ``-TINY_NUMBER``); the clamp only affects the result at the level of
    1e-20, and AthenaPK's sign is kept.

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index (ideal gas only).
        sound_speed_squared_of: ``None`` for the ideal gas, or the squared
            sound speed of a barotropic law as a function of the density.

    Returns:
        The hydrodynamic interface flux.
    """
    # --------------- ↓ Roe-averaged state and sound speeds ↓ ----------------

    sqrt_density_left = jnp.sqrt(left.density)
    sqrt_density_right = jnp.sqrt(right.density)
    inverse_sum_sqrt_density = 1.0 / (sqrt_density_left + sqrt_density_right)

    roe_velocity_normal = (
        sqrt_density_left * left.velocity_normal + sqrt_density_right * right.velocity_normal
    ) * inverse_sum_sqrt_density
    sound_speed_left = face_sound_speed(left, gamma, sound_speed_squared_of)
    sound_speed_right = face_sound_speed(right, gamma, sound_speed_squared_of)

    if sound_speed_squared_of is None:
        gamma_minus_one = gamma - 1.0
        inverse_gamma_minus_one = 1.0 / gamma_minus_one

        roe_velocity_transverse_1 = (
            sqrt_density_left * left.velocity_transverse_1
            + sqrt_density_right * right.velocity_transverse_1
        ) * inverse_sum_sqrt_density
        roe_velocity_transverse_2 = (
            sqrt_density_left * left.velocity_transverse_2
            + sqrt_density_right * right.velocity_transverse_2
        ) * inverse_sum_sqrt_density

        # Roe (1981): average the enthalpy H = (E + p) / rho rather than E or p.
        energy_left = _hydro_energy(left, inverse_gamma_minus_one)
        energy_right = _hydro_energy(right, inverse_gamma_minus_one)
        roe_enthalpy = (
            (energy_left + left.pressure) / sqrt_density_left
            + (energy_right + right.pressure) / sqrt_density_right
        ) * inverse_sum_sqrt_density

        enthalpy_term = roe_enthalpy - 0.5 * (
            _square(roe_velocity_normal)
            + _square(roe_velocity_transverse_1)
            + _square(roe_velocity_transverse_2)
        )
        # The double where keeps the Roe sound speed and its derivatives finite
        # where the enthalpy term is negative: the inner where keeps the square
        # root's argument non-negative, the outer one discards the infinite
        # derivative of the square root at zero.
        roe_sound_speed = jnp.where(
            enthalpy_term < 0.0,
            0.0,
            jnp.sqrt(gamma_minus_one * jnp.where(enthalpy_term < 0.0, 0.0, enthalpy_term)),
        )
    else:
        # No energy equation: the energy value of the flux is formed from zeros.
        energy_left = energy_right = 0.0
        roe_sound_speed = jnp.sqrt(barotropic_roe_sound_speed_squared(left, right, sound_speed_squared_of))

    # --------------- ↑ Roe-averaged state and sound speeds ↑ ----------------

    # --------------- ↓ Wave speeds ↓ ----------------

    speed_left = jnp.minimum(
        roe_velocity_normal - roe_sound_speed,
        left.velocity_normal - sound_speed_left,
    )
    speed_right = jnp.maximum(
        roe_velocity_normal + roe_sound_speed,
        right.velocity_normal + sound_speed_right,
    )

    speed_plus = jnp.where(speed_right > 0.0, speed_right, _TINY_NUMBER)
    speed_minus = jnp.where(speed_left < 0.0, speed_left, _TINY_NUMBER)

    # --------------- ↑ Wave speeds ↑ ----------------

    # --------------- ↓ HLLE flux ↓ ----------------

    relative_velocity_left = left.velocity_normal - speed_minus
    relative_velocity_right = right.velocity_normal - speed_plus

    flux_left = _hydro_flux_minus_speed_times_state(left, relative_velocity_left, energy_left)
    flux_right = _hydro_flux_minus_speed_times_state(right, relative_velocity_right, energy_right)

    speeds_differ = speed_plus != speed_minus
    safe_speed_difference = jnp.where(speeds_differ, speed_plus - speed_minus, 1.0)
    weight = jnp.where(
        speeds_differ,
        0.5 * (speed_plus + speed_minus) / safe_speed_difference,
        0.0,
    )

    # --------------- ↑ HLLE flux ↑ ----------------

    return HydroFlux(
        *(
            0.5 * (value_left + value_right) + (value_left - value_right) * weight
            for value_left, value_right in zip(flux_left, flux_right)
        )
    )
