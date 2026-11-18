"""
The HLLD Riemann solver for GLM-MHD (Miyoshi & Kusano 2005), as in AthenaPK.

Conventions (interface frame, elementwise evaluation) and shared helpers: ``_face_state.py``.
"""

# jax
import jax.numpy as jnp

# astronomix functions
from astronomix._finite_volume._riemann._face_state import (
    MHDFaceState,
    MHDFlux,
    _MHDConserved,
    _glm_interface_field_and_psi,
    _square,
    fast_magnetosonic_speed,
)


#: Relative threshold of the HLLD degeneracy check (AthenaPK's ``SMALL_NUMBER``).
_HLLD_SMALL_NUMBER = 1.0e-8


def _conserved_sum(first: _MHDConserved, second: _MHDConserved) -> _MHDConserved:
    """Return the component-wise sum ``first + second``."""
    return _MHDConserved(
        *(first_value + second_value for first_value, second_value in zip(first, second))
    )


def _scaled_jump(speed, upper: _MHDConserved, lower: _MHDConserved) -> _MHDConserved:
    """
    Return the wave jump ``S_k * (U_k - U_{k-1})`` that turns the flux of one
    region of the wave fan into the flux of the next (``F*_L = F_L + S_L
    (U*_L - U_L)``).
    """
    return _MHDConserved(
        *(speed * (upper_value - lower_value) for upper_value, lower_value in zip(upper, lower))
    )


def hlld_flux(left: MHDFaceState, right: MHDFaceState, gamma, cleaning_speed) -> MHDFlux:
    """
    The HLLD flux of Miyoshi & Kusano (2005) in AthenaPK's GLM form
    (``glmmhd_hlld.hpp``, including the Athena++ degeneracy check).

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index.
        cleaning_speed: The hyperbolic divergence-cleaning speed ``c_h``.

    Returns:
        The GLM-MHD interface flux.
    """

    # --------------- ↓ GLM subsystem and left/right states ↓ ----------------

    gamma_minus_one = gamma - 1.0
    inverse_gamma_minus_one = 1.0 / gamma_minus_one

    # The MHD fan uses the normal field of the exact GLM interface solution;
    # the reconstructed ``left.field_normal`` / ``right.field_normal`` only
    # enter the fast speeds below.
    interface_field_normal, interface_psi = _glm_interface_field_and_psi(
        left,
        right,
        cleaning_speed,
    )
    interface_field_normal_squared = interface_field_normal * interface_field_normal

    # The transverse components are grouped first for floating-point
    # associativity symmetry between the left and right states.
    magnetic_pressure_left = 0.5 * (
        interface_field_normal_squared
        + (_square(left.field_transverse_1) + _square(left.field_transverse_2))
    )
    magnetic_pressure_right = 0.5 * (
        interface_field_normal_squared
        + (_square(right.field_transverse_1) + _square(right.field_transverse_2))
    )
    kinetic_energy_left = 0.5 * left.density * (
        _square(left.velocity_normal)
        + (_square(left.velocity_transverse_1) + _square(left.velocity_transverse_2))
    )
    kinetic_energy_right = 0.5 * right.density * (
        _square(right.velocity_normal)
        + (_square(right.velocity_transverse_1) + _square(right.velocity_transverse_2))
    )

    conserved_left = _MHDConserved(
        density=left.density,
        momentum_normal=left.velocity_normal * left.density,
        momentum_transverse_1=left.velocity_transverse_1 * left.density,
        momentum_transverse_2=left.velocity_transverse_2 * left.density,
        energy=(
            left.pressure * inverse_gamma_minus_one
            + kinetic_energy_left
            + magnetic_pressure_left
        ),
        field_transverse_1=left.field_transverse_1,
        field_transverse_2=left.field_transverse_2,
    )
    conserved_right = _MHDConserved(
        density=right.density,
        momentum_normal=right.velocity_normal * right.density,
        momentum_transverse_1=right.velocity_transverse_1 * right.density,
        momentum_transverse_2=right.velocity_transverse_2 * right.density,
        energy=(
            right.pressure * inverse_gamma_minus_one
            + kinetic_energy_right
            + magnetic_pressure_right
        ),
        field_transverse_1=right.field_transverse_1,
        field_transverse_2=right.field_transverse_2,
    )

    # --------------- ↑ GLM subsystem and left/right states ↑ ----------------

    # --------------- ↓ Outer wave speeds and left/right fluxes ↓ ----------------

    # The fast speeds use each side's own reconstructed normal field, as in
    # AthenaPK (Miyoshi & Kusano eq. 67).
    fast_speed_left = fast_magnetosonic_speed(
        gamma,
        left.density,
        left.pressure,
        left.field_normal,
        left.field_transverse_1,
        left.field_transverse_2,
    )
    fast_speed_right = fast_magnetosonic_speed(
        gamma,
        right.density,
        right.pressure,
        right.field_normal,
        right.field_transverse_1,
        right.field_transverse_2,
    )

    speed_left_outer = jnp.minimum(
        left.velocity_normal - fast_speed_left,
        right.velocity_normal - fast_speed_right,
    )
    speed_right_outer = jnp.maximum(
        left.velocity_normal + fast_speed_left,
        right.velocity_normal + fast_speed_right,
    )

    total_pressure_left = left.pressure + magnetic_pressure_left
    total_pressure_right = right.pressure + magnetic_pressure_right

    def side_flux(state: MHDFaceState, conserved: _MHDConserved, total_pressure) -> _MHDConserved:
        """The physical flux ``F(U)`` of one outer state."""
        return _MHDConserved(
            density=conserved.momentum_normal,
            momentum_normal=(
                conserved.momentum_normal * state.velocity_normal
                + total_pressure
                - interface_field_normal_squared
            ),
            momentum_transverse_1=(
                conserved.momentum_transverse_1 * state.velocity_normal
                - interface_field_normal * conserved.field_transverse_1
            ),
            momentum_transverse_2=(
                conserved.momentum_transverse_2 * state.velocity_normal
                - interface_field_normal * conserved.field_transverse_2
            ),
            energy=(
                state.velocity_normal
                * (conserved.energy + total_pressure - interface_field_normal_squared)
                - interface_field_normal
                * (
                    state.velocity_transverse_1 * conserved.field_transverse_1
                    + state.velocity_transverse_2 * conserved.field_transverse_2
                )
            ),
            field_transverse_1=(
                conserved.field_transverse_1 * state.velocity_normal
                - interface_field_normal * state.velocity_transverse_1
            ),
            field_transverse_2=(
                conserved.field_transverse_2 * state.velocity_normal
                - interface_field_normal * state.velocity_transverse_2
            ),
        )

    flux_left = side_flux(left, conserved_left, total_pressure_left)
    flux_right = side_flux(right, conserved_right, total_pressure_right)

    # --------------- ↑ Outer wave speeds and left/right fluxes ↑ ----------------

    # --------------- ↓ Contact and Alfvén speeds ↓ ----------------

    speed_minus_velocity_left = speed_left_outer - left.velocity_normal
    speed_minus_velocity_right = speed_right_outer - right.velocity_normal

    # The contact speed S_M (Miyoshi & Kusano 2005, eq. 38); the pressure terms
    # are grouped for floating-point associativity symmetry.
    contact_speed = (
        speed_minus_velocity_right * conserved_right.momentum_normal
        - speed_minus_velocity_left * conserved_left.momentum_normal
        + (total_pressure_left - total_pressure_right)
    ) / (
        speed_minus_velocity_right * conserved_right.density
        - speed_minus_velocity_left * conserved_left.density
    )

    speed_minus_contact_left = speed_left_outer - contact_speed
    speed_minus_contact_right = speed_right_outer - contact_speed
    inverse_speed_minus_contact_left = 1.0 / speed_minus_contact_left
    inverse_speed_minus_contact_right = 1.0 / speed_minus_contact_right

    # The star-state densities follow from mass conservation across the outer
    # waves (eq. 43).
    star_density_left = (
        conserved_left.density * speed_minus_velocity_left * inverse_speed_minus_contact_left
    )
    star_density_right = (
        conserved_right.density * speed_minus_velocity_right * inverse_speed_minus_contact_right
    )
    inverse_star_density_left = 1.0 / star_density_left
    inverse_star_density_right = 1.0 / star_density_right
    sqrt_star_density_left = jnp.sqrt(star_density_left)
    sqrt_star_density_right = jnp.sqrt(star_density_right)

    # The rotational (Alfvén) waves move at S_M -+ |B_n| / sqrt(rho*) (eq. 51).
    speed_left_alfven = contact_speed - jnp.abs(interface_field_normal) / sqrt_star_density_left
    speed_right_alfven = contact_speed + jnp.abs(interface_field_normal) / sqrt_star_density_right

    # --------------- ↑ Contact and Alfvén speeds ↑ ----------------

    # --------------- ↓ Star (*) states ↓ ----------------

    # The total pressure is constant across the star region (eq. 41); its two
    # one-sided estimates agree analytically and are averaged.
    star_total_pressure_left = total_pressure_left + (
        conserved_left.density
        * speed_minus_velocity_left
        * (contact_speed - left.velocity_normal)
    )
    star_total_pressure_right = total_pressure_right + (
        conserved_right.density
        * speed_minus_velocity_right
        * (contact_speed - right.velocity_normal)
    )
    star_total_pressure = 0.5 * (star_total_pressure_right + star_total_pressure_left)

    def star_state(
        state: MHDFaceState,
        conserved: _MHDConserved,
        total_pressure,
        star_density,
        inverse_star_density,
        speed_minus_velocity,
        speed_minus_contact,
        inverse_speed_minus_contact,
    ):
        """
        The star state between an outer fast wave and the adjacent Alfvén wave
        (Miyoshi & Kusano 2005, eqs. 43-48).

        Args:
            state: The primitive outer state.
            conserved: The conserved outer state.
            total_pressure: The total (gas + magnetic) pressure of the outer state.
            star_density: The star-state density.
            inverse_star_density: ``1 / star_density``.
            speed_minus_velocity: ``S_outer - v_normal`` of the outer state.
            speed_minus_contact: ``S_outer - S_M``.
            inverse_speed_minus_contact: ``1 / (S_outer - S_M)``.

        Returns:
            The conserved star state and its ``v* . B*``.
        """
        star_momentum_normal = star_density * contact_speed

        # Eqs. (44)-(47), with the Athena++ guard against the degenerate case
        # in which the denominator vanishes (the star state then keeps the
        # transverse velocity and field of the outer state).
        denominator = (
            conserved.density * speed_minus_velocity * speed_minus_contact
            - interface_field_normal_squared
        )
        degenerate = jnp.abs(denominator) < _HLLD_SMALL_NUMBER * star_total_pressure
        safe_denominator = jnp.where(degenerate, 1.0, denominator)

        velocity_factor = (
            interface_field_normal * (speed_minus_velocity - speed_minus_contact) / safe_denominator
        )
        field_factor = (
            conserved.density * _square(speed_minus_velocity) - interface_field_normal_squared
        ) / safe_denominator

        star_momentum_transverse_1 = jnp.where(
            degenerate,
            star_density * state.velocity_transverse_1,
            star_density
            * (state.velocity_transverse_1 - conserved.field_transverse_1 * velocity_factor),
        )
        star_momentum_transverse_2 = jnp.where(
            degenerate,
            star_density * state.velocity_transverse_2,
            star_density
            * (state.velocity_transverse_2 - conserved.field_transverse_2 * velocity_factor),
        )
        star_field_transverse_1 = jnp.where(
            degenerate,
            conserved.field_transverse_1,
            conserved.field_transverse_1 * field_factor,
        )
        star_field_transverse_2 = jnp.where(
            degenerate,
            conserved.field_transverse_2,
            conserved.field_transverse_2 * field_factor,
        )

        # v* . B*, with the transverse terms grouped for associativity symmetry.
        star_velocity_dot_field = (
            star_momentum_normal * interface_field_normal
            + (
                star_momentum_transverse_1 * star_field_transverse_1
                + star_momentum_transverse_2 * star_field_transverse_2
            )
        ) * inverse_star_density

        # The star energy follows from energy conservation across the outer
        # wave (eq. 48).
        star_energy = (
            speed_minus_velocity * conserved.energy
            - total_pressure * state.velocity_normal
            + star_total_pressure * contact_speed
            + interface_field_normal
            * (
                state.velocity_normal * interface_field_normal
                + (
                    state.velocity_transverse_1 * conserved.field_transverse_1
                    + state.velocity_transverse_2 * conserved.field_transverse_2
                )
                - star_velocity_dot_field
            )
        ) * inverse_speed_minus_contact

        star = _MHDConserved(
            density=star_density,
            momentum_normal=star_momentum_normal,
            momentum_transverse_1=star_momentum_transverse_1,
            momentum_transverse_2=star_momentum_transverse_2,
            energy=star_energy,
            field_transverse_1=star_field_transverse_1,
            field_transverse_2=star_field_transverse_2,
        )
        return star, star_velocity_dot_field

    star_left, star_velocity_dot_field_left = star_state(
        left,
        conserved_left,
        total_pressure_left,
        star_density_left,
        inverse_star_density_left,
        speed_minus_velocity_left,
        speed_minus_contact_left,
        inverse_speed_minus_contact_left,
    )
    star_right, star_velocity_dot_field_right = star_state(
        right,
        conserved_right,
        total_pressure_right,
        star_density_right,
        inverse_star_density_right,
        speed_minus_velocity_right,
        speed_minus_contact_right,
        inverse_speed_minus_contact_right,
    )

    # --------------- ↑ Star (*) states ↑ ----------------

    # --------------- ↓ Double-star (**) states ↓ ----------------

    # If B_normal is near zero these coincide with the star states.
    inverse_sum_sqrt_density = 1.0 / (sqrt_star_density_left + sqrt_star_density_right)
    # The sign is built from a typed operand: a select between two bare
    # literals would enter the Pallas/Triton lowering as the default float type.
    ones = jnp.ones_like(interface_field_normal)
    field_normal_sign = jnp.where(interface_field_normal > 0.0, ones, -ones)

    # The two double-star states share their transverse velocity (eqs. 59, 60)
    # and transverse field (eqs. 61, 62).
    double_star_velocity_transverse_1 = inverse_sum_sqrt_density * (
        sqrt_star_density_left * (star_left.momentum_transverse_1 * inverse_star_density_left)
        + sqrt_star_density_right * (star_right.momentum_transverse_1 * inverse_star_density_right)
        + field_normal_sign * (star_right.field_transverse_1 - star_left.field_transverse_1)
    )
    double_star_velocity_transverse_2 = inverse_sum_sqrt_density * (
        sqrt_star_density_left * (star_left.momentum_transverse_2 * inverse_star_density_left)
        + sqrt_star_density_right * (star_right.momentum_transverse_2 * inverse_star_density_right)
        + field_normal_sign * (star_right.field_transverse_2 - star_left.field_transverse_2)
    )

    double_star_field_transverse_1 = inverse_sum_sqrt_density * (
        sqrt_star_density_left * star_right.field_transverse_1
        + sqrt_star_density_right * star_left.field_transverse_1
        + field_normal_sign
        * sqrt_star_density_left
        * sqrt_star_density_right
        * (
            (star_right.momentum_transverse_1 * inverse_star_density_right)
            - (star_left.momentum_transverse_1 * inverse_star_density_left)
        )
    )
    double_star_field_transverse_2 = inverse_sum_sqrt_density * (
        sqrt_star_density_left * star_right.field_transverse_2
        + sqrt_star_density_right * star_left.field_transverse_2
        + field_normal_sign
        * sqrt_star_density_left
        * sqrt_star_density_right
        * (
            (star_right.momentum_transverse_2 * inverse_star_density_right)
            - (star_left.momentum_transverse_2 * inverse_star_density_left)
        )
    )

    double_star_left_momentum_transverse_1 = star_left.density * double_star_velocity_transverse_1
    double_star_left_momentum_transverse_2 = star_left.density * double_star_velocity_transverse_2

    # The double-star energies follow from energy conservation across the
    # Alfvén waves (eq. 63).
    double_star_velocity_dot_field = contact_speed * interface_field_normal + (
        double_star_left_momentum_transverse_1 * double_star_field_transverse_1
        + double_star_left_momentum_transverse_2 * double_star_field_transverse_2
    ) / star_left.density

    double_star_left = _MHDConserved(
        density=star_left.density,
        momentum_normal=star_left.momentum_normal,
        momentum_transverse_1=double_star_left_momentum_transverse_1,
        momentum_transverse_2=double_star_left_momentum_transverse_2,
        energy=star_left.energy
        - sqrt_star_density_left
        * field_normal_sign
        * (star_velocity_dot_field_left - double_star_velocity_dot_field),
        field_transverse_1=double_star_field_transverse_1,
        field_transverse_2=double_star_field_transverse_2,
    )
    double_star_right = _MHDConserved(
        density=star_right.density,
        momentum_normal=star_right.momentum_normal,
        momentum_transverse_1=star_right.density * double_star_velocity_transverse_1,
        momentum_transverse_2=star_right.density * double_star_velocity_transverse_2,
        energy=star_right.energy
        + sqrt_star_density_right
        * field_normal_sign
        * (star_velocity_dot_field_right - double_star_velocity_dot_field),
        field_transverse_1=double_star_field_transverse_1,
        field_transverse_2=double_star_field_transverse_2,
    )

    # --------------- ↑ Double-star (**) states ↑ ----------------

    # --------------- ↓ Flux selection ↓ ----------------

    # The flux of each region of the wave fan is the flux of its outer
    # neighbour plus the jump S_k * (U_k - U_{k-1}) across the separating wave.
    jump_left_alfven = _scaled_jump(speed_left_alfven, double_star_left, star_left)
    jump_left_outer = _scaled_jump(speed_left_outer, star_left, conserved_left)
    jump_right_alfven = _scaled_jump(speed_right_alfven, double_star_right, star_right)
    jump_right_outer = _scaled_jump(speed_right_outer, star_right, conserved_right)

    flux_left_star = _conserved_sum(flux_left, jump_left_outer)
    flux_right_star = _conserved_sum(flux_right, jump_right_outer)
    flux_left_double_star = _conserved_sum(flux_left_star, jump_left_alfven)
    flux_right_double_star = _conserved_sum(flux_right_star, jump_right_alfven)

    def select_upwind_flux(
        left_flux,
        left_star_flux,
        left_double_star_flux,
        right_double_star_flux,
        right_star_flux,
        right_flux,
    ):
        """
        Pick (for one component) the flux of the wave-fan region that contains
        the interface. Later selections take precedence, so the inner waves are
        tested first and the outer waves last.
        """
        selected = jnp.where(contact_speed >= 0.0, left_double_star_flux, right_double_star_flux)
        selected = jnp.where(speed_right_alfven <= 0.0, right_star_flux, selected)
        selected = jnp.where(speed_left_alfven >= 0.0, left_star_flux, selected)
        selected = jnp.where(speed_right_outer <= 0.0, right_flux, selected)
        selected = jnp.where(speed_left_outer >= 0.0, left_flux, selected)
        return selected

    selected_flux = _MHDConserved(
        *(
            select_upwind_flux(*region_fluxes)
            for region_fluxes in zip(
                flux_left,
                flux_left_star,
                flux_left_double_star,
                flux_right_double_star,
                flux_right_star,
                flux_right,
            )
        )
    )

    # --------------- ↑ Flux selection ↑ ----------------

    # The decoupled GLM subsystem contributes F(B_normal) = psi* and
    # F(psi) = c_h^2 B_normal* (Dedner et al. 2002); the MHD solver does not
    # touch these components.
    return MHDFlux(
        mass=selected_flux.density,
        momentum_normal=selected_flux.momentum_normal,
        momentum_transverse_1=selected_flux.momentum_transverse_1,
        momentum_transverse_2=selected_flux.momentum_transverse_2,
        energy=selected_flux.energy,
        field_normal=interface_psi,
        field_transverse_1=selected_flux.field_transverse_1,
        field_transverse_2=selected_flux.field_transverse_2,
        psi=cleaning_speed * cleaning_speed * interface_field_normal,
    )
