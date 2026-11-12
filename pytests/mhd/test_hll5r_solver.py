"""
The HLL5R relaxation solver alone (``_finite_volume/_riemann/_hll5r.py``), in double precision on the CPU.

Checked on random states for the ideal gas, the isothermal gas and a polytrope:

* consistency: the flux of two equal states is the physical flux, and the two fluxes of the pair are equal;
* mirror symmetry;
* for equal states the Lagrangian speeds are those of Bouchut et al. (2010), eq. 3.38;
* the wave fan conserves: the flux is continuous in the velocity of the frame, so every wave of the fan obeys
  its jump condition for mass, momentum, energy and field;
* waves the solver resolves exactly: a contact, a tangential discontinuity at rest (no normal field), and a
  rotational wave at rest up to an error of second order in the transverse field;
* finite values and derivatives without normal field, without field and with a contact at rest;
* the two magnetic fluxes differ by the jump of the normal field times the velocity vector of the contact.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/mhd/test_hll5r_solver.py
"""

# numerics
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# testing
import pytest

# astronomix functions
from astronomix._finite_volume._riemann._face_state import MHDFaceState
from astronomix._finite_volume._riemann._hll5r import (
    _contact_values,
    hll5r_flux,
    hll5r_lagrangian_speeds,
)

GAMMA = 5.0 / 3.0
ISOTHERMAL_SOUND_SPEED_SQUARED = 0.7
POLYTROPIC_CONSTANT = 0.8
POLYTROPIC_EXPONENT = 1.4

#: The pressure laws: name -> (exponent, squared sound speed of the density or None, pressure of the density or None).
LAWS = {
    "ideal_gas": (GAMMA, None, None),
    "isothermal": (
        1.0,
        lambda density: ISOTHERMAL_SOUND_SPEED_SQUARED + 0.0 * density,
        lambda density: ISOTHERMAL_SOUND_SPEED_SQUARED * density,
    ),
    "polytropic": (
        POLYTROPIC_EXPONENT,
        lambda density: POLYTROPIC_EXPONENT * POLYTROPIC_CONSTANT * density ** (POLYTROPIC_EXPONENT - 1.0),
        lambda density: POLYTROPIC_CONSTANT * density**POLYTROPIC_EXPONENT,
    ),
}
FIELD_SLOTS = (5, 6, 7)


def _random_states(law: str, number: int, seed: int) -> MHDFaceState:
    """Random states (density and pressure positive; the pressure of the law for a barotropic gas)."""
    generator = np.random.default_rng(seed)
    density = generator.uniform(0.2, 3.0, number)
    velocities = generator.normal(0.0, 1.5, (3, number))
    pressure = generator.uniform(0.1, 4.0, number)
    fields = generator.normal(0.0, 1.2, (3, number))
    pressure_of = LAWS[law][2]
    if pressure_of is not None:
        pressure = pressure_of(density)
    return MHDFaceState(*(jnp.asarray(v) for v in (density, *velocities, pressure, *fields, 0.0 * density)))


def _flux_pair(law: str, left: MHDFaceState, right: MHDFaceState):
    exponent, sound_speed_squared_of, _ = LAWS[law]
    return hll5r_flux(left, right, exponent, sound_speed_squared_of)


def _physical_flux(law: str, state: MHDFaceState):
    """The flux of ideal MHD in frame order (mass, three momenta, energy, three field components)."""
    density, normal, first, second, pressure, field_normal, field_first, field_second, _ = state
    field_squared = field_normal**2 + field_first**2 + field_second**2
    total_pressure = pressure + 0.5 * field_squared
    field_dot_velocity = field_normal * normal + field_first * first + field_second * second
    internal_energy = pressure / (GAMMA - 1.0) if law == "ideal_gas" else 0.0 * pressure
    energy = internal_energy + 0.5 * density * (normal**2 + first**2 + second**2) + 0.5 * field_squared
    return (
        density * normal,
        density * normal**2 + total_pressure - field_normal**2,
        density * normal * first - field_normal * field_first,
        density * normal * second - field_normal * field_second,
        (energy + total_pressure) * normal - field_normal * field_dot_velocity,
        0.0 * density,
        field_first * normal - field_normal * first,
        field_second * normal - field_normal * second,
    )


def _assert_fluxes_close(flux, reference, tolerance):
    for slot, (value, expected) in enumerate(zip(flux, reference)):
        scale = 1.0 + np.max(np.abs(expected))
        np.testing.assert_allclose(value, expected, rtol=0.0, atol=tolerance * scale, err_msg=f"slot {slot}")


@pytest.mark.parametrize("law", LAWS)
def test_equal_states_give_the_physical_flux(law):
    state = _random_states(law, 2000, seed=1)
    for_left, for_right = _flux_pair(law, state, state)
    _assert_fluxes_close(for_left[:8], _physical_flux(law, state), 1e-13)
    for slot in FIELD_SLOTS:
        np.testing.assert_array_equal(for_left[slot], for_right[slot])


@pytest.mark.parametrize("law", LAWS)
def test_mirror_symmetry(law):
    left = _random_states(law, 2000, seed=2)
    right = _random_states(law, 2000, seed=3)

    def mirrored(state):
        return state._replace(velocity_normal=-state.velocity_normal, field_normal=-state.field_normal)

    flux = _flux_pair(law, left, right)
    mirrored_flux = _flux_pair(law, mirrored(right), mirrored(left))
    # Mass, transverse momentum, energy and transverse field change sign; the normal momentum does not. The
    # normal field flux keeps its sign (the flux is odd and the field is mirrored).
    signs = (-1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0)
    for neighbour in (0, 1):
        expected = [sign * value for sign, value in zip(signs, mirrored_flux[1 - neighbour][:8])]
        _assert_fluxes_close(flux[neighbour][:8], expected, 1e-12)


@pytest.mark.parametrize("law", LAWS)
def test_speeds_of_equal_states(law):
    state = _random_states(law, 2000, seed=4)
    exponent, sound_speed_squared_of, _ = LAWS[law]
    (transverse, normal), (other_transverse, other_normal) = hll5r_lagrangian_speeds(
        state, state, exponent, sound_speed_squared_of
    )
    density = state.density
    sound_speed_squared = GAMMA * state.pressure / density if law == "ideal_gas" else sound_speed_squared_of(density)
    transverse_field_squared = state.field_transverse_1**2 + state.field_transverse_2**2
    field_product = jnp.abs(state.field_normal) * jnp.sqrt(transverse_field_squared)
    np.testing.assert_allclose(transverse**2, density * (state.field_normal**2 + field_product), rtol=1e-13)
    np.testing.assert_allclose(
        normal**2, density**2 * sound_speed_squared + density * (transverse_field_squared + field_product), rtol=1e-13
    )
    np.testing.assert_array_equal(transverse, other_transverse)
    np.testing.assert_array_equal(normal, other_normal)


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("jump_of_normal_field", (False, True))
def test_the_wave_fan_conserves(law, jump_of_normal_field):
    """
    In a frame moving with the velocity ``-shift`` the interface samples the fan at ``x / t = -shift``. The flux
    of a conservative fan is continuous in the shift (a wave at rest in the frame has equal fluxes on its two
    sides), so its largest change over one step of the shift falls with the step; a wave that violated its
    jump condition would leave a change that does not.
    """
    left = _random_states(law, 40, seed=5)
    right = _random_states(law, 40, seed=6)
    if not jump_of_normal_field:
        right = right._replace(field_normal=left.field_normal)

    def largest_change(number_of_shifts):
        shifts = jnp.linspace(-12.0, 12.0, number_of_shifts)[:, None]
        shifted_left = left._replace(velocity_normal=left.velocity_normal[None, :] + shifts)
        shifted_right = right._replace(velocity_normal=right.velocity_normal[None, :] + shifts)
        for_left, _ = _flux_pair(law, shifted_left, shifted_right)
        slots = range(8) if law == "ideal_gas" else (0, 1, 2, 3, 5, 6, 7)
        return max(float(jnp.max(jnp.abs(jnp.diff(for_left[slot], axis=0)))) for slot in slots)

    coarse = largest_change(2401)
    fine = largest_change(24001)
    assert fine < 0.15 * coarse, (coarse, fine)


@pytest.mark.parametrize("velocity", (0.8, -0.8, 0.0))
def test_contact_is_exact(velocity):
    left = _random_states("ideal_gas", 500, seed=7)
    left = left._replace(velocity_normal=velocity + 0.0 * left.density)
    right = left._replace(density=_random_states("ideal_gas", 500, seed=8).density)
    for_left, for_right = _flux_pair("ideal_gas", left, right)
    upwind = left if velocity >= 0.0 else right
    _assert_fluxes_close(for_left[:8], _physical_flux("ideal_gas", upwind), 1e-13)
    _assert_fluxes_close(for_right[:8], _physical_flux("ideal_gas", upwind), 1e-13)
    if velocity == 0.0:
        np.testing.assert_array_equal(for_left.mass, 0.0)


def test_tangential_discontinuity_at_rest_is_exact():
    """Without normal field: jumps of density, transverse velocity and transverse field at equal total pressure."""
    left = _random_states("ideal_gas", 500, seed=9)
    right = _random_states("ideal_gas", 500, seed=10)
    zero = 0.0 * left.density
    left = left._replace(velocity_normal=zero, field_normal=zero)
    total_pressure = 6.0 + zero
    right = right._replace(velocity_normal=zero, field_normal=zero)
    left = left._replace(pressure=total_pressure - 0.5 * (left.field_transverse_1**2 + left.field_transverse_2**2))
    right = right._replace(pressure=total_pressure - 0.5 * (right.field_transverse_1**2 + right.field_transverse_2**2))
    for_left, for_right = _flux_pair("ideal_gas", left, right)
    expected = (zero, total_pressure, zero, zero, zero, zero, zero, zero)
    _assert_fluxes_close(for_left[:8], expected, 1e-13)
    _assert_fluxes_close(for_right[:8], expected, 1e-13)


def test_rotational_wave_at_rest_is_sharp_for_a_small_transverse_field():
    """
    A rotational wave at rest (gas streaming through it at the Alfven speed) has equal physical fluxes on its
    two sides. The solver's flux differs from it by a term of second order in the transverse field, while the
    jump of the state is of first order: the smearing vanishes relative to the wave as the transverse field
    does. (For a finite transverse field the 5-wave solver does not resolve the wave exactly.)
    """
    density, pressure, field_normal, angle = 1.3, 0.9, 0.8, 2.1
    errors = []
    for magnitude in (1e-2, 1e-3):
        velocity_normal = field_normal / np.sqrt(density)
        left_field = np.array([magnitude, 0.0])
        right_field = magnitude * np.array([np.cos(angle), np.sin(angle)])
        left_velocity = np.array([0.3, -0.2])
        right_velocity = left_velocity + (right_field - left_field) / np.sqrt(density)
        values = [density, velocity_normal, 0.0, 0.0, pressure, field_normal, 0.0, 0.0, 0.0]
        left = MHDFaceState(*(jnp.asarray([v]) for v in values))._replace(
            velocity_transverse_1=jnp.asarray([left_velocity[0]]),
            velocity_transverse_2=jnp.asarray([left_velocity[1]]),
            field_transverse_1=jnp.asarray([left_field[0]]),
            field_transverse_2=jnp.asarray([left_field[1]]),
        )
        right = left._replace(
            velocity_transverse_1=jnp.asarray([right_velocity[0]]),
            velocity_transverse_2=jnp.asarray([right_velocity[1]]),
            field_transverse_1=jnp.asarray([right_field[0]]),
            field_transverse_2=jnp.asarray([right_field[1]]),
        )
        exact = _physical_flux("ideal_gas", left)
        _assert_fluxes_close(_physical_flux("ideal_gas", right), exact, 1e-13)
        for_left, _ = _flux_pair("ideal_gas", left, right)
        errors.append(max(float(jnp.max(jnp.abs(value - expected))) for value, expected in zip(for_left[:8], exact)))
    assert errors[0] < 0.5 * 1e-2, errors
    assert errors[1] < 0.02 * errors[0], errors


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("case", ("no_normal_field", "no_field", "contact_at_rest", "one_normal_field"))
def test_finite_values_and_derivatives(law, case):
    left = _random_states(law, 300, seed=11)
    right = _random_states(law, 300, seed=12)
    zero = 0.0 * left.density
    if case == "no_normal_field":
        left, right = left._replace(field_normal=zero), right._replace(field_normal=zero)
    if case == "one_normal_field":
        left = left._replace(field_normal=zero)
    if case == "no_field":
        no_field = dict(field_normal=zero, field_transverse_1=zero, field_transverse_2=zero)
        left, right = left._replace(**no_field), right._replace(**no_field)
    if case == "contact_at_rest":
        # Mirrored states: the contact is at rest by symmetry.
        right = left._replace(velocity_normal=-left.velocity_normal, field_normal=-left.field_normal)

    def summed_flux(left_values, right_values):
        pair = _flux_pair(law, MHDFaceState(*left_values), MHDFaceState(*right_values))
        return sum(jnp.sum(value) for flux in pair for value in flux[:8])

    for flux in _flux_pair(law, left, right):
        for slot, value in enumerate(flux[:8]):
            assert bool(jnp.all(jnp.isfinite(value))), (slot, case)
    for derivatives in jax.grad(summed_flux, argnums=(0, 1))(tuple(left), tuple(right)):
        for slot, derivative in enumerate(derivatives[:8]):
            assert bool(jnp.all(jnp.isfinite(derivative))), (slot, case)
    if case == "contact_at_rest":
        np.testing.assert_allclose(_flux_pair(law, left, right)[0].mass, 0.0, atol=1e-13)


@pytest.mark.parametrize("law", LAWS)
def test_the_two_magnetic_fluxes(law):
    left = _random_states(law, 2000, seed=13)
    right = _random_states(law, 2000, seed=14)
    exponent, sound_speed_squared_of, _ = LAWS[law]
    speeds = hll5r_lagrangian_speeds(left, right, exponent, sound_speed_squared_of)
    contact_velocity, _, contact_transverse_velocities, _ = _contact_values(left, right, *speeds)
    for_left, for_right = _flux_pair(law, left, right)
    field_jump = right.field_normal - left.field_normal
    for slot, velocity in zip(FIELD_SLOTS, (contact_velocity, *contact_transverse_velocities)):
        np.testing.assert_allclose(for_right[slot] - for_left[slot], -velocity * field_jump, rtol=0.0, atol=1e-13)
    for slot in (0, 1, 2, 3, 4):
        np.testing.assert_array_equal(for_left[slot], for_right[slot])
    # The normal field has no flux on the upwind side of the contact.
    upwind_normal_flux = jnp.where(contact_velocity >= 0.0, for_left.field_normal, for_right.field_normal)
    np.testing.assert_array_equal(upwind_normal_flux, 0.0)

    # A continuous normal field: one flux.
    right = right._replace(field_normal=left.field_normal)
    for_left, for_right = _flux_pair(law, left, right)
    for slot in FIELD_SLOTS:
        np.testing.assert_array_equal(for_left[slot], for_right[slot])
