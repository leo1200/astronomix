"""
The sweep of the reference scheme (dimensionally split MUSCL-Hancock with the HLL5R solver, Powell's term in the
induction equation only, parabolic cleaning; ``divergence_control=INDUCTION_SOURCE``), piece by piece:

* the predictor rows are the primitive form of the system the sweep solves (conserved mass, momentum and energy,
  the induction equation with the source ``-u dB_n/dn``), derived here by automatic differentiation from the
  conserved variables, the flux and the source; the sign printed in Waagan (2009), eq. 4.17, is not;
* the slope clips bound the differences of density, normal velocity and pressure over a cell;
* the update of the field is the published one: the flux for the left neighbour at the right face, the flux for
  the right neighbour at the left face and the cell term ``-dt u dB_n / h``;
* a divergence error is carried with the gas (the conservative form would leave it in place);
* mass, momentum and energy are conserved to rounding in 3D;
* the sweeps run x, y in even steps and y, x in odd steps;
* the parabolic cleaning damps a monopole mode by the factor of its difference stencil, close to the rate of the
  heat equation, and leaves a field without divergence untouched.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/mhd/test_reference_sweep.py
"""

# numerics
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    IDEAL_GAS,
    ISOTHERMAL,
    PERIODIC_BOUNDARY,
    PIECEWISE_LINEAR,
)

# astronomix containers
from astronomix.option_classes.presets import BEATTIE_REFERENCE
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
import astronomix._finite_volume._reconstruction as reconstruction_module
from astronomix._finite_volume._divergence._induction_source import (
    _parabolic_cleaning,
    _sweep_fluxes_and_induction_source,
)
from astronomix._finite_volume._interface_frame import _barotropic_law, _interface_frame_indices, _mhd_face_state
from astronomix._finite_volume._reconstruction import (
    _clipped_gradients,
    _hancock_predictor,
    _reconstructed_face_states,
)
from astronomix._finite_volume._riemann._hll5r import hll5r_flux
from astronomix._finite_volume._stage import _hancock_terms
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

pytestmark = pytest.mark.x64

GAMMA = 5.0 / 3.0
SOUND_SPEED = 0.9
EQUATIONS_OF_STATE = (("ideal gas", IDEAL_GAS), ("isothermal", ISOTHERMAL))
NO_CLEANING = BEATTIE_REFERENCE._replace(parabolic_cleaning=False)


def _setup(equation_of_state, shape, solver_config=BEATTIE_REFERENCE, **settings):
    """Finalized configuration, parameters and variable registry of a periodic box of ``shape`` cells."""
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    dimensionality = len(shape)
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        equation_of_state=equation_of_state,
        solver_config=solver_config,
        mhd=True,
        dimensionality=dimensionality,
        box_size=1.0 if dimensionality == 1 else StaticFloatVector(1.0, 1.0, 1.0),
        boundary_settings=periodic if dimensionality == 1 else BoundarySettings(periodic, periodic, periodic),
        progress_bar=False,
        **settings,
    )
    registered_variables = get_registered_variables(config)
    config = finalize_config(config, (registered_variables.num_vars, *shape))
    params = SimulationParams(C_cfl=0.8, gamma=GAMMA, isothermal_sound_speed=SOUND_SPEED)
    return config, params, registered_variables


def _smooth_state(equation_of_state, shape, seed):
    """A periodic state of a few long waves: (rho, v, p, B) for the ideal gas, (rho, v, B) for the isothermal."""
    random = np.random.default_rng(seed)
    position = np.meshgrid(*[(np.arange(cells) + 0.5) / cells for cells in shape], indexing="ij")

    def waves(mean, amplitude):
        value = mean
        for _ in range(3):
            wave_numbers = random.integers(-2, 3, size=len(shape))
            phase = 2.0 * np.pi * (sum(k * x for k, x in zip(wave_numbers, position)) + random.random())
            value = value + amplitude * random.uniform(0.3, 1.0) * np.sin(phase)
        return value

    rows = [waves(1.0, 0.1)] + [waves(0.0, 0.4) for _ in range(3)]
    if equation_of_state == IDEAL_GAS:
        rows.append(waves(1.0, 0.1))
    rows += [waves(mean, 0.3) for mean in (0.6, -0.4, 0.2)]
    return jnp.asarray(np.stack(rows))


# -------------------------------------------------------------
# ==================== ↓ Predictor rows ↓ =====================
# -------------------------------------------------------------


def _conserved_variables(primitive, sound_speed_squared):
    """(rho, rho v, E, B) of one cell, or (rho, rho v, B) with a sound speed (isothermal gas)."""
    density, velocity = primitive[0], primitive[1:4]
    if sound_speed_squared is not None:
        return jnp.concatenate([density[None], density * velocity, primitive[4:7]])
    pressure, field = primitive[4], primitive[5:8]
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity @ velocity + 0.5 * field @ field
    return jnp.concatenate([density[None], density * velocity, energy[None], field])


def _flux_along_the_first_axis(primitive, sound_speed_squared):
    """The flux of ideal MHD along the first component of velocity and field, laid out like the conserved variables."""
    density, velocity = primitive[0], primitive[1:4]
    isothermal = sound_speed_squared is not None
    field = primitive[4:7] if isothermal else primitive[5:8]
    pressure = sound_speed_squared * density if isothermal else primitive[4]
    total_pressure = pressure + 0.5 * field @ field
    momentum_flux = density * velocity[0] * velocity - field[0] * field + total_pressure * jnp.array([1.0, 0.0, 0.0])
    field_flux = velocity[0] * field - field[0] * velocity
    mass_flux = (density * velocity[0])[None]
    if isothermal:
        return jnp.concatenate([mass_flux, momentum_flux, field_flux])
    energy = _conserved_variables(primitive, None)[4]
    energy_flux = (energy + total_pressure) * velocity[0] - field[0] * (velocity @ field)
    return jnp.concatenate([mass_flux, momentum_flux, energy_flux[None], field_flux])


def _rate_of_the_system(primitive, gradient, sound_speed_squared):
    """
    ``A(W) dW/dn`` of one cell from ``dU/dt + dF/dn = S`` with the source ``S = -u dB_n/dn`` in the induction
    equation only: ``A dW/dn = (dU/dW)^-1 (dF/dW dW/dn - S)``.
    """
    conserved_jacobian = jax.jacfwd(_conserved_variables)(primitive, sound_speed_squared)
    _, flux_gradient = jax.jvp(lambda w: _flux_along_the_first_axis(w, sound_speed_squared), (primitive,), (gradient,))
    field_rows = slice(4, 7) if sound_speed_squared is not None else slice(5, 8)
    source = jnp.zeros_like(primitive).at[field_rows].set(-primitive[1:4] * gradient[field_rows][0])
    return jnp.linalg.solve(conserved_jacobian, flux_gradient - source)


def _rows_of_the_sweep_frame(axis, has_pressure):
    """State rows in the order (rho, v_n, v_t1, v_t2, [p], B_n, B_t1, B_t2) of a sweep along ``axis``."""
    order = [axis - 1] + [component for component in range(3) if component != axis - 1]
    field_start = 5 if has_pressure else 4
    return [0] + [1 + k for k in order] + ([4] if has_pressure else []) + [field_start + k for k in order]


def _predictor_and_system_rates(equation_of_state, axis):
    shape = (4, 4, 4)
    config, params, registered_variables = _setup(equation_of_state, shape)
    state = _smooth_state(equation_of_state, shape, seed=axis)
    gradients = _smooth_state(equation_of_state, shape, seed=10 + axis) - 0.3
    barotropic_law = _barotropic_law(params, config)
    hancock = _hancock_terms(axis, 0.01, GAMMA, config, None, registered_variables, barotropic_law)
    rate = np.asarray(_hancock_predictor(state, gradients, hancock))

    isothermal = equation_of_state == ISOTHERMAL
    rows = _rows_of_the_sweep_frame(axis, has_pressure=not isothermal)
    per_cell = lambda values: values[jnp.array(rows)].reshape(len(rows), -1).T
    frame_rate = jax.vmap(lambda w, g: _rate_of_the_system(w, g, SOUND_SPEED**2 if isothermal else None))(
        per_cell(state), per_cell(gradients)
    )
    system_rate = np.empty_like(rate)
    system_rate[rows] = np.asarray(frame_rate).T.reshape(len(rows), *shape)
    return rate, system_rate, hancock, np.asarray(state), np.asarray(gradients)


@pytest.mark.parametrize("name, equation_of_state", EQUATIONS_OF_STATE)
@pytest.mark.parametrize("axis", (1, 2, 3))
def test_predictor_rows_are_the_primitive_form_of_the_system(name, equation_of_state, axis):
    rate, system_rate, _, _, _ = _predictor_and_system_rates(equation_of_state, axis)
    assert np.abs(rate - system_rate).max() < 1e-13 * np.abs(system_rate).max()


def test_the_printed_sign_is_not_the_primitive_form(monkeypatch):
    """With the sign of Waagan (2009), eq. 4.17, the normal-velocity row is off by ``2 B_n dB_n/dn / rho``."""
    monkeypatch.setattr(reconstruction_module, "_NORMAL_FIELD_FORCE_SIGN", 1.0)
    rate, system_rate, hancock, state, gradients = _predictor_and_system_rates(IDEAL_GAS, axis=2)
    normal_field = hancock.field_rows.normal_field_index
    excess = 2.0 * state[normal_field] * gradients[normal_field] / state[0]
    difference = rate - system_rate
    assert np.abs(difference[hancock.normal_velocity_index] - excess).max() < 1e-13
    assert np.abs(np.delete(difference, hancock.normal_velocity_index, axis=0)).max() < 1e-13
    assert np.abs(excess).max() > 0.1


# -------------------------------------------------------------
# ===================== ↓ Slope clips ↓ =======================
# -------------------------------------------------------------


@pytest.mark.parametrize("name, equation_of_state", EQUATIONS_OF_STATE)
def test_slope_clips(name, equation_of_state):
    cells, half_step = 8, 0.02
    config, params, registered_variables = _setup(equation_of_state, (cells,))
    hancock = _hancock_terms(1, half_step, GAMMA, config, None, registered_variables, _barotropic_law(params, config))
    state = _smooth_state(equation_of_state, (cells,), seed=3)
    spacing = 1.0 / cells
    exponent = 1.0 if equation_of_state == ISOTHERMAL else GAMMA
    margin = reconstruction_module._SLOPE_CLIP_MARGIN
    bounds = {0: margin * 0.5 * np.asarray(state[0]) / spacing, 1: margin / ((1.0 + exponent) * 2.0 * half_step)}
    if equation_of_state == IDEAL_GAS:
        bounds[4] = margin * np.asarray(state[4]) / ((1.0 + exponent) * spacing)

    steep = jnp.full(state.shape, 1.0e3) * jnp.where(jnp.arange(cells) % 2 == 0, 1.0, -1.0)
    clipped = np.asarray(_clipped_gradients(state, steep, hancock))
    gentle = 1.0e-3 * jnp.ones(state.shape)
    for row in range(state.shape[0]):
        if row in bounds:
            assert np.allclose(np.abs(clipped[row]), bounds[row], rtol=1e-14)
            assert np.array_equal(np.sign(clipped[row]), np.sign(np.asarray(steep[row])))
        else:
            assert np.array_equal(clipped[row], np.asarray(steep[row]))
    assert np.array_equal(np.asarray(_clipped_gradients(state, gentle, hancock)), np.asarray(gentle))


# -------------------------------------------------------------
# ================= ↓ Update of the field ↓ ===================
# -------------------------------------------------------------


@pytest.mark.parametrize("name, equation_of_state", EQUATIONS_OF_STATE)
def test_update_is_the_published_form_with_two_magnetic_fluxes(name, equation_of_state):
    """Waagan, Federrath & Klingenberg (2011), eq. 9: ``F_l`` at the right face, ``F_r`` at the left, the cell term."""
    cells = 16
    config, params, registered_variables = _setup(equation_of_state, (cells,))
    barotropic_law = _barotropic_law(params, config)
    state = _smooth_state(equation_of_state, (cells,), seed=5)
    spacing = config.grid_spacing

    fluxes, source = _sweep_fluxes_and_induction_source(
        state, 1, PIECEWISE_LINEAR, GAMMA, config, registered_variables, barotropic_law=barotropic_law
    )
    rate = -(jnp.roll(fluxes, -1, axis=1) - fluxes) / spacing - source

    left_states, right_states = _reconstructed_face_states(state, 1, PIECEWISE_LINEAR, config.solver_config.limiter)
    frame_indices = _interface_frame_indices(1, config, registered_variables)
    left = _mhd_face_state(left_states, frame_indices, barotropic_law)
    right = _mhd_face_state(right_states, frame_indices, barotropic_law)
    if barotropic_law is None:
        flux_for_left, flux_for_right = hll5r_flux(left, right, GAMMA)
    else:
        flux_for_left, flux_for_right = hll5r_flux(
            left, right, barotropic_law.exponent, barotropic_law.sound_speed_squared_of
        )
    # the edge values of cell i: its right edge is the left state of face i + 1/2, its left edge the right state
    # of face i - 1/2
    right_edge = lambda values: jnp.roll(values, -1)
    normal_field_difference = right_edge(left.field_normal) - right.field_normal
    assert float(jnp.abs(normal_field_difference).max()) > 1e-3
    field_rows = tuple(registered_variables.magnetic_index)
    for row, flux_name, velocity_name in zip(
        field_rows,
        ("field_normal", "field_transverse_1", "field_transverse_2"),
        ("velocity_normal", "velocity_transverse_1", "velocity_transverse_2"),
    ):
        velocity = 0.5 * (right_edge(getattr(left, velocity_name)) + getattr(right, velocity_name))
        published = (
            -(right_edge(getattr(flux_for_left, flux_name)) - getattr(flux_for_right, flux_name)) / spacing
            - velocity * normal_field_difference / spacing
        )
        assert float(jnp.abs(rate[row] - published).max()) < 1e-12
    # all other rows: the flux is single-valued, no source
    other_rows = [row for row in range(state.shape[0]) if row not in field_rows]
    assert float(jnp.abs(source[jnp.array(other_rows)]).max()) == 0.0


def _field_after_half_a_crossing(solver_config):
    """A weak monopole wave ``B_x = a sin(2 pi x)`` in a uniform stream ``u = 1`` along x at ``t = 0.5``."""
    cells, amplitude = 64, 1.0e-4
    config, params, registered_variables = _setup(ISOTHERMAL, (cells,), solver_config)
    x = (jnp.arange(cells) + 0.5) / cells
    zero, one = jnp.zeros(cells), jnp.ones(cells)
    initial_field = amplitude * jnp.sin(2.0 * jnp.pi * x)
    state = jnp.stack([one, one, zero, zero, initial_field, zero, zero])
    final_state = time_integration(state, config, params._replace(t_end=0.5), registered_variables)
    return np.asarray(final_state[4]) / amplitude, np.asarray(initial_field) / amplitude


def test_divergence_error_moves_with_the_gas():
    """The source ``-u dB_n/dn`` turns the frozen normal field of the conservative form into an advected one."""
    field, initial_field = _field_after_half_a_crossing(NO_CLEANING)
    assert np.abs(field + initial_field).mean() < 0.02
    assert np.abs(field - initial_field).mean() > 1.0


# -------------------------------------------------------------
# ============ ↓ Conservation and sweep order ↓ ===============
# -------------------------------------------------------------


@pytest.mark.parametrize("name, equation_of_state", EQUATIONS_OF_STATE)
@pytest.mark.parametrize("solver_config", (BEATTIE_REFERENCE, NO_CLEANING))
def test_mass_momentum_and_energy_are_conserved(name, equation_of_state, solver_config):
    """
    The sweeps conserve mass, momentum and energy. The cleaning changes the field at fixed pressure, so with it
    the energy is not conserved. The induction source does not conserve the mean field.
    """
    shape = (8, 8, 8)
    config, params, registered_variables = _setup(equation_of_state, shape, solver_config)
    state = _smooth_state(equation_of_state, shape, seed=7)
    final_state = time_integration(state, config, params._replace(t_end=0.05), registered_variables)

    sound_speed_squared = SOUND_SPEED**2 if equation_of_state == ISOTHERMAL else None
    conserved = jax.vmap(lambda w: _conserved_variables(w, sound_speed_squared), in_axes=1, out_axes=1)
    totals = lambda values: np.asarray(conserved(values.reshape(values.shape[0], -1)).mean(axis=1))
    before, after = totals(state), totals(final_state)
    assert np.abs(np.asarray(final_state) - np.asarray(state)).max() > 1e-2
    has_energy = equation_of_state == IDEAL_GAS
    conserved_rows = 5 if has_energy and not solver_config.parabolic_cleaning else 4
    assert np.abs(after - before)[:conserved_rows].max() < 1e-14
    # the state has a divergence error of order one, and the mean field drifts with it
    assert np.abs(after - before)[-3:].max() > 1e-6


def _swap_x_and_y(state):
    """The mirror image of a 2D isothermal state (rho, v, B) at the diagonal."""
    return jnp.swapaxes(state, 1, 2)[jnp.array([0, 2, 1, 3, 5, 4, 6])]


def test_sweep_order_is_reversed_in_odd_steps():
    """Two steps x, y and y, x: the second equals the first on the mirror image at the diagonal."""
    shape, time_step = (16, 16), 4.0e-3
    state = _smooth_state(ISOTHERMAL, shape, seed=11)

    def steps(initial_state, number):
        config, params, registered_variables = _setup(
            ISOTHERMAL, shape, fixed_timestep=True, num_timesteps=number
        )
        return time_integration(initial_state, config, params._replace(t_end=number * time_step), registered_variables)

    two_steps = steps(state, 2)
    mirrored_second_step = _swap_x_and_y(steps(_swap_x_and_y(steps(state, 1)), 1))
    same_order_twice = steps(steps(state, 1), 1)
    assert float(jnp.abs(two_steps - mirrored_second_step).max()) < 1e-13
    assert float(jnp.abs(two_steps - same_order_twice).max()) > 1e-8


# -------------------------------------------------------------
# =================== ↓ Parabolic cleaning ↓ ==================
# -------------------------------------------------------------


def test_cleaning_damps_a_monopole_at_the_rate_of_its_stencil():
    """
    ``B += dt eta grad(div B)`` with ``eta = lambda h / (2 d)`` and central differences: a mode of wave vector
    ``k`` along ``s = (sin(k_d h))_d``, the longitudinal direction of the stencil, decays per step by
    ``1 - dt eta |s|^2 / h^2``, for ``k h -> 0`` the heat equation's ``exp(-eta k^2 dt)``. At a Courant number
    of 0.8 every mode is damped.
    """
    cells, signal_speed, courant = 32, 2.0, 0.8
    config, _, registered_variables = _setup(ISOTHERMAL, (cells, cells))
    spacing = 1.0 / cells
    time_step = courant * spacing / signal_speed
    diffusivity = signal_speed * spacing / 4.0
    x, y = np.meshgrid(*[(np.arange(cells) + 0.5) * spacing] * 2, indexing="ij")
    for wave_numbers in ((1, 2), (3, 0), (cells // 4, cells // 4)):
        angular = 2.0 * np.pi * np.array(wave_numbers)
        mode = np.sin(angular[0] * x + angular[1] * y)
        direction = np.sin(angular * spacing)
        rows = [1.0 + 0 * x, 0 * x, 0 * x, 0 * x, direction[0] * mode, direction[1] * mode, mode]
        state = jnp.asarray(np.stack(rows))
        cleaned = _parabolic_cleaning(state, signal_speed, time_step, config, registered_variables)
        factor = 1.0 - time_step * diffusivity * np.sum(direction**2) / spacing**2
        assert 0.0 < factor < 1.0
        assert np.abs(np.asarray(cleaned[4:6]) - factor * np.asarray(state[4:6])).max() < 1e-13
        assert np.array_equal(np.asarray(cleaned[:4]), np.asarray(state[:4]))
        assert np.array_equal(np.asarray(cleaned[6]), np.asarray(state[6]))
        if max(wave_numbers) <= 3:
            heat_equation = np.exp(-diffusivity * np.sum(angular**2) * time_step)
            assert abs(factor - heat_equation) < 0.1 * (1.0 - heat_equation)


def test_cleaning_leaves_a_field_without_divergence():
    cells = 16
    config, _, registered_variables = _setup(ISOTHERMAL, (cells, cells))
    x, y = np.meshgrid(*[(np.arange(cells) + 0.5) / cells] * 2, indexing="ij")
    wave = np.sin(2.0 * np.pi * (x + 2.0 * y))
    # transverse to (sin(k_x h), sin(k_y h)): no divergence with central differences
    transverse = (np.sin(4.0 * np.pi / cells), -np.sin(2.0 * np.pi / cells))
    state = jnp.asarray(np.stack([1.0 + 0 * x, 0 * x, 0 * x, 0 * x, transverse[0] * wave, transverse[1] * wave, wave]))
    cleaned = _parabolic_cleaning(state, 2.0, 0.01, config, registered_variables)
    assert float(jnp.abs(cleaned - state).max()) < 1e-13
