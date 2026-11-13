"""
The finite-difference scheme with a Riemann flux (``FDSolverConfig.scheme = RIEMANN_FLUX``),
in the two configurations of the presets ``FD_RIEMANN_ISOTHERMAL`` and ``FD_RIEMANN_IDEAL_GAS``.

Checks the convergence order on the circularly polarised Alfven wave, that steps on a rough
state keep div B at rounding, that a strong blast (ideal gas) and a Mach-10 collision
(isothermal gas) stay positive, that more ghost cells than the rule of ``finalize_config``
give the same result with open boundaries, that the derivative of the magnetic half of a
stage (constrained transport, its energy flux, rebuild of the cell-centred field) agrees with
a central difference and that its JVP and VJP satisfy the dot-product identity, that the
second derivative through the two quotients with their own derivative rule is finite in
float32, that the rule of the floored quotient forms no infinite value at the floor, and that
``finalize_config`` rejects what the scheme does not serve.

Most of the time of a test is the compilation of the step on the CPU (about a minute per
configuration and array shape). The two tests that compare two runs therefore share the
first run with the test before them through a cache; run alone they compute both.

    JAX_PLATFORMS=cpu python -m pytest pytests/mhd/test_riemann_flux_scheme.py
"""

# general
import functools

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# testing
import pytest

# jax
import jax

# The divergence, identity and derivative checks compare to rounding.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    HLL,
    HLLD,
    HLLD_BOUNDED,
    HLLEM_T,
    IDEAL_GAS,
    ISOTHERMAL,
    LSRK4,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    RIEMANN_FLUX,
    SSPRK33,
)

# astronomix containers
from astronomix.option_classes.presets import FD_RIEMANN_IDEAL_GAS, FD_RIEMANN_ISOTHERMAL
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    SimulationConfig,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_difference._magnetic_update._constrained_transport import update_cell_center_fields
from astronomix._finite_difference._magnetic_update._flux_form_ct import (
    flux_form_ct_rhs,
    flux_form_ct_rhs_with_energy,
)
from astronomix._finite_difference._ratio_rules import DIVISION_FLOOR, _floored_ratio, _ratio
from astronomix._spatial_operators._differencing import (
    _interface_field_divergence,
    finite_difference_int6,
)
from astronomix._spatial_operators._interpolate import interp_face_to_center
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.test_setups.mhd.alfven_wave3D import (
    CPAlfvenWave3DSettings,
    cp_alfven_wave_solution,
    setup_cp_alfven_wave,
)
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

NUM_CELLS = 16
SOUND_SPEED = 0.1
GAMMA = 5.0 / 3.0
BOTH_GASES = pytest.mark.parametrize("equation_of_state", [ISOTHERMAL, IDEAL_GAS], ids=["isothermal", "ideal_gas"])


def _config(equation_of_state, num_cells=NUM_CELLS, boundary=PERIODIC_BOUNDARY, **fields):
    """A 3D finite-difference MHD configuration with the Riemann-flux preset of the gas (float64)."""
    side = BoundarySettings1D(boundary, boundary)
    settings = dict(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        mhd=True,
        equation_of_state=equation_of_state,
        num_cells=num_cells,
        box_size=1.0,
        boundary_settings=BoundarySettings(side, side, side),
        backend_config=BackendConfig(backend=NATIVE_JAX),
        solver_config=FD_RIEMANN_IDEAL_GAS if equation_of_state == IDEAL_GAS else FD_RIEMANN_ISOTHERMAL,
        progress_bar=False,
    )
    settings.update(fields)
    return SimulationConfig(**settings)


def _params(equation_of_state, t_end):
    """The parameters of a run to ``t_end`` at a CFL number of 0.5."""
    if equation_of_state == IDEAL_GAS:
        return SimulationParams(C_cfl=0.5, gamma=GAMMA, minimum_density=1e-10, minimum_pressure=1e-12, t_end=t_end)
    return SimulationParams(C_cfl=0.5, minimum_density=1e-10, isothermal_sound_speed=SOUND_SPEED, t_end=t_end)


def _cell_coordinates(num_cells=NUM_CELLS):
    """The cell-centre coordinates of the unit box."""
    x = (np.arange(num_cells) + 0.5) / num_cells
    return np.meshgrid(x, x, x, indexing="ij")


def _divergence_free_faces(seed, amplitude, mean_field, num_cells=NUM_CELLS):
    """
    Interface fields: a uniform field plus the discrete curl of a random edge vector potential,
    taken with the staggered derivative of the constrained transport, so that the interface
    divergence vanishes to rounding.

    Args:
        seed: The random seed.
        amplitude: The amplitude of the random part of the field.
        mean_field: The three components of the uniform field.
        num_cells: The cells per axis.

    Returns:
        The interface fields (bx, by, bz).
    """
    keys = jax.random.split(jax.random.key(seed), 3)
    spacing = 1.0 / num_cells
    a_x, a_y, a_z = (amplitude * spacing * jax.random.normal(key, (num_cells,) * 3) for key in keys)
    bx = (finite_difference_int6(a_z, 1) - finite_difference_int6(a_y, 2)) / spacing
    by = (finite_difference_int6(a_x, 2) - finite_difference_int6(a_z, 0)) / spacing
    bz = (finite_difference_int6(a_y, 0) - finite_difference_int6(a_x, 1)) / spacing
    return bx + mean_field[0], by + mean_field[1], bz + mean_field[2]


def _state(config, registered_variables, density, velocity, faces, pressure=None):
    """The primitive state with the cell-centred field interpolated from the interface fields."""
    fields = dict(
        density=jnp.asarray(density),
        velocity_x=jnp.asarray(velocity[0]),
        velocity_y=jnp.asarray(velocity[1]),
        velocity_z=jnp.asarray(velocity[2]),
        magnetic_field_x=interp_face_to_center(faces[0], 0),
        magnetic_field_y=interp_face_to_center(faces[1], 1),
        magnetic_field_z=interp_face_to_center(faces[2], 2),
        interface_magnetic_field_x=faces[0],
        interface_magnetic_field_y=faces[1],
        interface_magnetic_field_z=faces[2],
    )
    if config.equation_of_state == IDEAL_GAS:
        fields["gas_pressure"] = jnp.asarray(pressure)
    return construct_primitive_state(config=config, registered_variables=registered_variables, **fields)


def _run(config, state, params):
    """(final primitive state, registered variables, finalized configuration) of a run."""
    registered_variables = get_registered_variables(config)
    config = finalize_config(config, state.shape)
    return time_integration(state, config, params, registered_variables), registered_variables, config


def _rough_state(config, registered_variables, seed):
    """Density from 0.01 to 1, velocities of rms 1 (Mach 10 for the isothermal gas), a rough solenoidal field."""
    keys = jax.random.split(jax.random.key(seed), 5)
    shape = (NUM_CELLS,) * 3
    density = 10.0 ** jax.random.uniform(keys[0], shape, minval=-2.0, maxval=0.0)
    velocity = [jax.random.normal(key, shape) for key in keys[1:4]]
    pressure = density * SOUND_SPEED**2 * 10.0 ** jax.random.uniform(keys[4], shape, minval=-1.0, maxval=1.0)
    faces = _divergence_free_faces(seed + 1, 0.4, (0.45, 0.0, 0.0))
    return _state(config, registered_variables, density, velocity, faces, pressure)


def _smooth_fields(equation_of_state):
    """Density, velocity, interface fields and pressure of a smooth state without a grid symmetry."""
    X, Y, Z = _cell_coordinates()

    def s(a):
        return np.sin(2.0 * np.pi * a)

    def c(a):
        return np.cos(2.0 * np.pi * a)

    half = 0.5 / NUM_CELLS
    density = 1.0 + 0.2 * s(X + 0.07) * c(Y + 0.13) + 0.1 * s(Z + 0.11)
    velocity = [
        0.3 * s(Y + 0.21) * c(Z + 0.03) + 0.08 * s(X + 0.17),
        0.25 * s(Z + 0.41) * c(X + 0.09) + 0.08 * s(Y + 0.57),
        0.2 * s(X + 0.29) * c(Y + 0.19) + 0.08 * c(Z + 0.23),
    ]
    # Each interface component does not depend on its own coordinate, plus a pair whose
    # differences cancel: the interface divergence is zero.
    faces = [
        0.45 + 0.15 * s(Y + 0.12) * c(Z + 0.04) + 0.06 * s(X + half + 0.37) * c(Y + 0.27),
        0.2 + 0.15 * s(Z + 0.33) * c(X + 0.06) - 0.06 * c(X + 0.37) * s(Y + half + 0.27),
        0.1 + 0.15 * s(X + 0.71) * c(Y + 0.08),
    ]
    pressure = 0.6 * (1.0 + 0.2 * c(X + Y + 0.05) + 0.1 * s(Z + 0.15))
    return density, velocity, [jnp.asarray(face) for face in faces], pressure


# ======================= convergence =======================


#: Lower bounds of the measured order and upper bounds of the errors (float64, CFL 0.5, half a period).
#: Measured on the CPU: isothermal 1.489e-03 at 8 cells and 1.589e-05 at 16 (order 6.55), ideal gas
#: 1.602e-03 and 1.282e-05 (order 6.97). The error bounds are about 1.5 times the measured values.
ALFVEN_WAVE_ORDER = {ISOTHERMAL: 6.0, IDEAL_GAS: 6.5}
ALFVEN_WAVE_ERROR = {(ISOTHERMAL, 8): 2.3e-3, (ISOTHERMAL, 16): 2.4e-5, (IDEAL_GAS, 8): 2.4e-3, (IDEAL_GAS, 16): 2.0e-5}


@functools.lru_cache(maxsize=None)
def _alfven_wave_error(equation_of_state, cells):
    """The mean absolute error of velocity and field after half a period of the wave at (2 cells, cells, cells)."""
    settings = CPAlfvenWave3DSettings(t_end=0.5)
    config = _config(equation_of_state, num_cells=StaticIntVector(2 * cells, cells, cells))
    params = _params(equation_of_state, settings.t_end)._replace(isothermal_sound_speed=np.sqrt(settings.p_0))
    state, config, params = setup_cp_alfven_wave(config, params, settings)
    registered_variables = get_registered_variables(config)
    final = time_integration(state, config, params, registered_variables)
    exact = cp_alfven_wave_solution(config, registered_variables, params, get_helper_data(config), settings)
    velocity, field = registered_variables.velocity_index, registered_variables.magnetic_index
    rows = np.array([velocity.x, velocity.y, velocity.z, field.x, field.y, field.z])
    assert bool(jnp.all(jnp.isfinite(final)))
    return float(jnp.mean(jnp.abs(final[rows] - exact[rows])))


@BOTH_GASES
def test_alfven_wave_error_at_eight_cells(equation_of_state):
    """The circularly polarised Alfven wave, oblique to the grid, at 8 cells per transverse wavelength."""
    error = _alfven_wave_error(equation_of_state, 8)
    print(f"alfven wave eos {equation_of_state}: error at 8 cells {error:.3e}")
    assert error < ALFVEN_WAVE_ERROR[equation_of_state, 8]


@BOTH_GASES
def test_alfven_wave_convergence_order(equation_of_state):
    """From 8 to 16 cells the error of the wave falls with the order of the scheme."""
    coarse = _alfven_wave_error(equation_of_state, 8)
    fine = _alfven_wave_error(equation_of_state, 16)
    order = np.log2(coarse / fine)
    print(f"alfven wave eos {equation_of_state}: errors {coarse:.3e} {fine:.3e}, order {order:.2f}")
    assert fine < ALFVEN_WAVE_ERROR[equation_of_state, 16]
    assert order > ALFVEN_WAVE_ORDER[equation_of_state]


# ======================= divergence, positivity =======================


@BOTH_GASES
def test_steps_keep_divergence_at_rounding(equation_of_state):
    """About ten steps on a rough state leave the interface divergence at rounding."""
    config = _config(equation_of_state)
    state = _rough_state(config, get_registered_variables(config), seed=3)
    final, registered_variables, _ = _run(config, state, _params(equation_of_state, 0.02))
    faces = registered_variables.interface_magnetic_field_index
    spacing = 1.0 / NUM_CELLS
    divergence = _interface_field_divergence(final[faces.x], final[faces.y], final[faces.z], spacing)
    largest_field = float(jnp.max(jnp.abs(final[jnp.array([faces.x, faces.y, faces.z])])))
    print(f"div B eos {equation_of_state}: {float(jnp.max(jnp.abs(divergence))) * spacing / largest_field:.2e}")
    assert bool(jnp.all(jnp.isfinite(final)))
    assert float(jnp.min(final[registered_variables.density_index])) > 0.0
    assert float(jnp.max(jnp.abs(divergence))) * spacing / largest_field < 1e-12


def test_strong_blast_stays_positive():
    """
    The early blast of Seo and Ryu (pressure ratio 100, plasma beta 0.02 outside) at 24 cells, ideal gas: the
    initial condition of ``examples/scripts/forward/mhd/mhd_blast.py``.
    """
    num_cells = 24
    inner_radius, outer_radius = 0.125, 1.1 * 0.125
    X, Y, Z = _cell_coordinates(num_cells)
    radius = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    ramp = np.clip((outer_radius - radius) / (outer_radius - inner_radius), 0.0, 1.0)
    pressure = 1.0 + 99.0 * ramp
    strength = 10.0 / np.sqrt(2.0)
    faces = [jnp.full((num_cells,) * 3, value) for value in (strength, strength, 0.0)]
    config = _config(IDEAL_GAS, num_cells=num_cells)
    registered_variables = get_registered_variables(config)
    state = _state(config, registered_variables, np.ones_like(X), [np.zeros_like(X)] * 3, faces, pressure)
    final, _, _ = _run(config, state, _params(IDEAL_GAS, 0.01))
    density = final[registered_variables.density_index]
    pressure = final[registered_variables.pressure_index]
    print(f"blast: density {float(density.min()):.3e} to {float(density.max()):.3e}, "
          f"pressure {float(pressure.min()):.3e} to {float(pressure.max()):.3e}")
    assert bool(jnp.all(jnp.isfinite(final)))
    assert float(density.min()) > 0.0
    assert float(pressure.min()) > 0.0
    assert float(density.max()) > 1.2


def test_mach_ten_collision_stays_positive():
    """Two isothermal streams of Mach 10 collide at x = 0.5 and part at x = 0 (periodic box)."""
    X, Y, Z = _cell_coordinates()
    key = jax.random.key(7)
    velocity = [np.where(X < 0.5, 1.0, -1.0) * 10.0 * SOUND_SPEED, 0.0 * X, 0.0 * X]
    velocity[1] = 0.01 * jax.random.normal(key, X.shape)
    faces = [jnp.full(X.shape, value) for value in (0.05, 0.05, 0.02)]
    config = _config(ISOTHERMAL)
    registered_variables = get_registered_variables(config)
    state = _state(config, registered_variables, np.ones_like(X), velocity, faces)
    final, _, _ = _run(config, state, _params(ISOTHERMAL, 0.3))
    density = final[registered_variables.density_index]
    print(f"collision: density {float(density.min()):.3e} to {float(density.max()):.3e}")
    assert bool(jnp.all(jnp.isfinite(final)))
    assert float(density.min()) > 0.0
    assert float(density.max()) > 5.0


# ======================= ghost cells =======================


#: The ghost layers that ``finalize_config`` derives from the stencils of the scheme and from the boundary
#: fills of the time integrator: 7 with SSPRK(3,3) (isothermal preset) and SSPRK(5,4) (ideal-gas preset), which
#: both fill the boundary before every right-hand side, rounded up to 8 so that a kernel block tiles the grid.
DERIVED_GHOST_CELLS = {ISOTHERMAL: 8, IDEAL_GAS: 8}


@functools.lru_cache(maxsize=None)
def _open_boundary_run(equation_of_state, extra_ghost_cells):
    """(initial state, final state) of a few steps of the smooth state with open boundaries."""
    density, velocity, faces, pressure = _smooth_fields(equation_of_state)
    config = _config(equation_of_state, boundary=OPEN_BOUNDARY)
    registered_variables = get_registered_variables(config)
    state = _state(config, registered_variables, density, velocity, faces, pressure)
    config = finalize_config(config, state.shape)
    assert config.num_ghost_cells == DERIVED_GHOST_CELLS[equation_of_state]
    config = config._replace(num_ghost_cells=config.num_ghost_cells + extra_ghost_cells)
    final = time_integration(jnp.array(state), config, _params(equation_of_state, 0.03), registered_variables)
    return np.asarray(state), np.asarray(final)


@BOTH_GASES
def test_open_boundaries_with_the_derived_ghost_cells(equation_of_state):
    """A few steps with open boundaries and the ghost layers of ``finalize_config``."""
    state, final = _open_boundary_run(equation_of_state, 0)
    assert np.all(np.isfinite(final))
    assert np.max(np.abs(final - state)) > 1e-3


@BOTH_GASES
def test_more_ghost_cells_give_the_same_result(equation_of_state):
    """
    With open boundaries the result does not change when the ghost layers exceed the derived count. The
    comparison is to rounding (1e-12 for values of order one), not bitwise: on the CPU the compiled step
    depends on the padded array shape at the level of 1e-15, and a missing layer changes this state by 6e-6.
    """
    _, derived = _open_boundary_run(equation_of_state, 0)
    _, wider = _open_boundary_run(equation_of_state, 2)
    print(f"ghost cells eos {equation_of_state}: {int(np.sum(derived != wider))} of {derived.size} entries "
          f"differ, largest difference {np.max(np.abs(derived - wider)):.2e}")
    np.testing.assert_allclose(derived, wider, rtol=0.0, atol=1e-12)


# ======================= derivatives =======================


def _conserved_smooth_state(equation_of_state, registered_variables):
    """The conserved state and the interface fields of the smooth state."""
    density, velocity, faces, pressure = _smooth_fields(equation_of_state)
    field = [interp_face_to_center(faces[axis], axis) for axis in range(3)]
    state = jnp.zeros((registered_variables.num_vars,) + density.shape)
    state = state.at[registered_variables.density_index].set(density)
    momentum, magnetic = registered_variables.momentum_index, registered_variables.magnetic_index
    for axis, (row, field_row) in enumerate(zip(momentum, magnetic, strict=True)):
        state = state.at[row].set(density * velocity[axis])
        state = state.at[field_row].set(field[axis])
    if equation_of_state == IDEAL_GAS:
        kinetic = 0.5 * density * sum(component**2 for component in velocity)
        energy = pressure / (GAMMA - 1.0) + kinetic + 0.5 * sum(component**2 for component in field)
        state = state.at[registered_variables.energy_index].set(energy)
    return state, tuple(faces)


def _inner(a, b):
    return sum(float(jnp.vdot(x, y)) for x, y in zip(a, b, strict=True))


#: Bounds of the dot-product identity and of the JVP against the central difference (relative, float64).
#: Measured on the CPU: identity 2.1e-16 (isothermal) and 2.0e-14 (ideal gas), central difference with the
#: step 1e-6 7.9e-10 and 7.8e-10. The bounds are about 50 and 13 times the larger measured value.
IDENTITY_BOUND = 1e-12
CENTRAL_DIFFERENCE_BOUND = 1e-8


@BOTH_GASES
def test_derivative_of_the_field_update(equation_of_state):
    """
    The magnetic half of a stage (constrained transport, its energy flux for the ideal gas, rebuild of the
    cell-centred field): the JVP agrees with a central difference in a smooth direction, and JVP and VJP
    satisfy the dot-product identity. The interface flux is left out: its derivative alone compiles for
    more than ten minutes on four CPU cores.
    """
    config = _config(equation_of_state)
    registered_variables = get_registered_variables(config)
    state, faces = _conserved_smooth_state(equation_of_state, registered_variables)
    config = finalize_config(config, state.shape)
    params = _params(equation_of_state, 1.0)
    spacing = 1.0 / NUM_CELLS
    dt = 0.1 * spacing

    def field_update(conserved, bx, by, bz):
        arguments = (conserved, bx, by, bz, dt, spacing, params, config, registered_variables)
        if equation_of_state == IDEAL_GAS:
            *increments, energy = flux_form_ct_rhs_with_energy(*arguments)
            conserved = conserved.at[registered_variables.energy_index].add(energy)
        else:
            increments = flux_form_ct_rhs(*arguments)
        new_faces = tuple(face + increment for face, increment in zip((bx, by, bz), increments, strict=True))
        return (update_cell_center_fields(conserved, *new_faces, config, registered_variables, params), *new_faces)

    X, Y, Z = _cell_coordinates()

    def wave(phase):
        return jnp.asarray(np.sin(2.0 * np.pi * (X + 2.0 * Y - Z + phase)))

    arrays = (state, *faces)
    fluid_direction = jnp.stack([wave(0.07 + 0.11 * row) for row in range(state.shape[0])])
    direction = (fluid_direction, wave(0.37), wave(0.59), wave(0.83))
    keys = jax.random.split(jax.random.key(1), 4)
    weight = tuple(jax.random.normal(key, a.shape) for key, a in zip(keys, arrays, strict=True))

    # One compilation holds the JVP and the VJP; its primal output also serves the central difference.
    @jax.jit
    def derivatives(point):
        result, tangent = jax.jvp(field_update, point, direction)
        return result, tangent, jax.vjp(field_update, *point)[1](weight)

    result, tangent, cotangent = derivatives(arrays)
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in (*result, *tangent, *cotangent))

    forward, backward = _inner(weight, tangent), _inner(cotangent, direction)
    print(f"derivative eos {equation_of_state}: identity {abs(forward - backward) / abs(forward):.2e}")
    assert abs(forward - backward) < IDENTITY_BOUND * abs(forward)

    epsilon = 1e-6
    plus = derivatives(tuple(a + epsilon * d for a, d in zip(arrays, direction, strict=True)))[0]
    minus = derivatives(tuple(a - epsilon * d for a, d in zip(arrays, direction, strict=True)))[0]
    difference = tuple((p - m) / (2.0 * epsilon) for p, m in zip(plus, minus, strict=True))
    error = np.sqrt(sum(float(jnp.sum((d - t) ** 2)) for d, t in zip(difference, tangent, strict=True)))
    size = np.sqrt(sum(float(jnp.sum(t**2)) for t in tangent))
    print(f"derivative eos {equation_of_state}: central difference {error / size:.2e}")
    assert error < CENTRAL_DIFFERENCE_BOUND * size


def _second_derivative(function, x, y):
    """The Hessian-vector product of ``function(x, y)`` with the direction (1, 1), in float32."""
    point = (jnp.float32(x), jnp.float32(y))
    ones = (jnp.float32(1.0), jnp.float32(1.0))
    return jax.jvp(jax.grad(function, argnums=(0, 1)), point, ones)[1]


@pytest.mark.parametrize("numerator", [0.0, 1.0])
@pytest.mark.parametrize("denominator", [0.0, 1e-35, DIVISION_FLOOR, 1e-18, 1e-10, 1.0])
def test_second_derivative_of_the_ratio_rules_is_finite_in_float32(numerator, denominator):
    """
    A Hessian-vector product through ``_floored_ratio`` (at, below and above the floor) and through
    ``_ratio`` (above it) is finite in float32 where the second derivatives are representable:
    at a vanishing numerator, and at any numerator for a denominator of 1e-10 or more.
    """
    if numerator != 0.0 and DIVISION_FLOOR < denominator < 1e-10:
        pytest.skip("the second derivative 2 x / y^3 is not representable in float32")
    hessian_vector = _second_derivative(_floored_ratio, numerator, denominator)
    assert all(part.dtype == jnp.float32 for part in hessian_vector)
    assert all(bool(jnp.isfinite(part)) for part in hessian_vector)
    if denominator > DIVISION_FLOOR:
        hessian_vector = _second_derivative(_ratio, numerator, denominator)
        assert all(bool(jnp.isfinite(part)) for part in hessian_vector)
    if denominator == 1.0:
        # d2(x / y) (1, 1) = (-1 / y^2, -1 / y^2 + 2 x / y^3)
        np.testing.assert_allclose(np.array(hessian_vector), [-1.0, -1.0 + 2.0 * numerator], rtol=1e-6)


@pytest.mark.parametrize("denominator", [0.0, 1e-35, DIVISION_FLOOR])
def test_floored_ratio_rule_forms_no_infinite_value_in_float32(denominator):
    """
    At and below the floor the derivative rule of ``_floored_ratio`` forms no infinite value in float32,
    not even one that a selection discards (``jax.debug_infs`` checks every operation of the eager
    evaluation). With the discarded product x times 1e60 the compiled reverse pass of the interface flux
    was not finite on the CPU in float32; that pass compiles for minutes and is not a test here.
    """
    point = (jnp.float32(0.5), jnp.float32(denominator))
    with jax.debug_infs(True):
        value, tangent = jax.jvp(_floored_ratio, point, (jnp.float32(1.0), jnp.float32(1.0)))
        cotangent = jax.vjp(_floored_ratio, *point)[1](jnp.float32(1.0))
    assert all(bool(jnp.isfinite(part)) for part in (value, tangent, *cotangent))
    # the slope with respect to the denominator is zero there; the one of the numerator is 1 / floor
    np.testing.assert_allclose(np.array(cotangent), [1.0 / DIVISION_FLOOR, 0.0], rtol=1e-6)


# ======================= validity rules =======================

_REJECTED = {
    "MHD in two dimensions": dict(dimensionality=2, num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS)),
    "reflective boundaries": dict(
        boundary_settings=BoundarySettings(*(BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),) * 3)
    ),
    "HLLD": dict(solver_config=FD_RIEMANN_IDEAL_GAS._replace(riemann_solver=HLLD)),
    "the solver of the other gas": dict(solver_config=FD_RIEMANN_IDEAL_GAS._replace(riemann_solver=HLLEM_T)),
    "LSRK4": dict(solver_config=FD_RIEMANN_IDEAL_GAS._replace(time_integrator=LSRK4)),
    "the positivity limiter of the production scheme": dict(
        solver_config=FD_RIEMANN_IDEAL_GAS._replace(weno_positivity_preserving=True)
    ),
    "the upwind EMF dissipation": dict(solver_config=FD_RIEMANN_IDEAL_GAS._replace(ct_upwind_dissipation=True)),
    "SSPRK33 with the production scheme": dict(solver_config=FDSolverConfig(time_integrator=SSPRK33)),
    "a Riemann solver with the production scheme": dict(solver_config=FDSolverConfig(riemann_solver=HLL)),
}


@pytest.mark.parametrize("name", list(_REJECTED))
def test_finalize_config_rejects_what_the_scheme_does_not_serve(name):
    config = _config(IDEAL_GAS, **_REJECTED[name])
    with pytest.raises(ValueError):
        finalize_config(config, (get_registered_variables(config).num_vars,) + (NUM_CELLS,) * 3)


def test_finalize_config_serves_the_presets():
    """The presets, and HLL with either gas, pass; the Riemann solver of the gas is chosen in the flux module."""
    for equation_of_state, preset, own in (
        (ISOTHERMAL, FD_RIEMANN_ISOTHERMAL, HLLEM_T),
        (IDEAL_GAS, FD_RIEMANN_IDEAL_GAS, HLLD_BOUNDED),
    ):
        for solver in (preset, preset._replace(riemann_solver=HLL), preset._replace(riemann_solver=own)):
            config = _config(equation_of_state, solver_config=solver)
            config = finalize_config(config, (get_registered_variables(config).num_vars,) + (NUM_CELLS,) * 3)
            assert config.solver_config.scheme == RIEMANN_FLUX
            assert config.num_ghost_cells == 0
