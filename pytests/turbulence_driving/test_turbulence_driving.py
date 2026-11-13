"""
Turbulence driving: the JAX port of TurbGen.

What is checked:

* against the C++ code (``data/turbgen_reference.npz``, see ``data/README.md``;
  band, parabola and power-law spectra in one, two and three dimensions, a
  non-cubic box, non-cubic grids, the automatic amplitude adjustment): the
  list of modes, the scales, and, with the Gaussian deviates of the C++ run
  fed in, the Ornstein-Uhlenbeck state, the projected mode vectors and the
  acceleration field, all to round-off in float64; the pattern schedule and
  the automatic amplitude adjustment of ``advance_turbulence_driving``;
* the statistics of a long sequence with ``jax.random``: variance per mode,
  correlation time, several patterns due at once; the solenoidal fraction and
  the rms of the field for solenoidal weights 0, 0.5 and 1;
* the field of a part of the grid (four slabs with index offsets, a padded
  grid, an array sharded over four devices) equals that of the whole grid;
* ``jax.grad`` with respect to the parameters, single precision, the kick
  with the bulk-motion correction of the FLASH plugin and the measured
  velocity dispersion (the wiring into the time loop is checked in
  ``test_driving_in_the_time_loop.py``).

The four devices need ``JAX_PLATFORMS=cpu`` and jax not started by another
test file before this one::

    JAX_PLATFORMS=cpu python -m pytest pytests/turbulence_driving
"""

# ==== device setup ====
import os
if os.environ.get("JAX_PLATFORMS") == "cpu":
    # first, so that a malformed entry of an existing XLA_FLAGS cannot hide it
    os.environ["XLA_FLAGS"] = ("--xla_force_host_platform_device_count=4 " + os.environ.get("XLA_FLAGS", "")).strip()
# ruff: noqa: E402
# ======================

# general
import json
from types import SimpleNamespace

# jax
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

# numerics
import numpy as np

# testing
import pytest

# astronomix functions
from astronomix._modules._turbulence_driving import (
    BAND,
    PARABOLA,
    TurbulenceDrivingConfig,
    TurbulenceDrivingParams,
    TurbulenceDrivingState,
    add_velocity_kick,
    advance_turbulence_driving,
    apply_turbulence_driving,
    build_driving_modes,
    cell_center_coordinates,
    driving_acceleration,
    init_turbulence_driving,
    ornstein_uhlenbeck_update,
    turnover_time,
    update_interval,
    velocity_dispersion,
)
from astronomix._modules._turbulence_driving._modes import driving_wavenumber_range
from astronomix._modules._turbulence_driving._turbulence_driving import (
    ou_standard_deviation,
    project_modes,
    solenoidal_weight_normalisation,
)

jax.config.update("jax_enable_x64", True)

REFERENCE = np.load(os.path.join(os.path.dirname(__file__), "data", "turbgen_reference.npz"))
CASES = sorted({key.split("/")[0] for key in REFERENCE.files})
ROUND_OFF = 1e-13

# the driving of Beattie et al. (2025): the defaults
BEATTIE = TurbulenceDrivingConfig(turbulence_driving=True)


def relative_difference(value, reference):
    """Largest difference in units of the largest reference value."""
    return float(np.max(np.abs(np.asarray(value) - reference)) / np.max(np.abs(reference)))


def reference_case(case):
    """Configuration, parameters and data of a case of the C++ reference."""
    data = {key.split("/")[1]: REFERENCE[key] for key in REFERENCE.files if key.startswith(case + "/")}
    parameters = json.loads(str(data["parameters"]))
    ndim = int(parameters["ndim"])
    per_axis = lambda text: ([float(v) for v in str(text).split(",")] * ndim)[:ndim]
    config = TurbulenceDrivingConfig(
        turbulence_driving=True,
        dimensionality=ndim,
        box_size=tuple(per_axis(parameters["length"])),
        spectral_form=int(parameters["spect_form"]),
        k_min=float(parameters["k_min"]),
        k_max=float(parameters["k_max"]),
        power_law_exponent=float(parameters.get("power_law_exp", 0.0)),
        angles_exponent=float(parameters.get("angles_exp", 0.0)),
        num_steps_per_turnover=int(parameters["nsteps_per_t_turb"]),
        random_seed=int(parameters["random_seed"]),
        amplitude_auto_adjust=bool(int(parameters["ampl_auto_adjust"])),
    )
    params = TurbulenceDrivingParams(
        velocity=float(parameters["target_vdisp"]),
        k_driving=float(parameters["k_driv"]),
        solenoidal_weight=float(parameters["sol_weight"]),
        amplitude_factor=jnp.asarray(per_axis(parameters["ampl_factor"])),
    )
    data["grid"] = [int(n) for n in parameters["grid"].split(",")][:ndim]
    data["velocity_dispersion"] = per_axis(parameters["vturb"]) if "vturb" in parameters else None
    return config, params, data


def grid_cell_centers(config, grid, dtype=None):
    return tuple(cell_center_coordinates(n, length / n, dtype=dtype) for n, length in zip(grid, config.box_size))


def simulation_stand_in(config, params, grid, velocity_index=(1, 2, 3)):
    """The fields of the simulation configuration, parameters and registry that the kick reads."""
    simulation_config = SimpleNamespace(
        dimensionality=config.dimensionality, num_cells=tuple(grid), turbulence_driving_config=config
    )
    simulation_params = SimpleNamespace(turbulence_driving_params=params)
    registered_variables = SimpleNamespace(density_index=0, velocity_index=velocity_index)
    return simulation_config, simulation_params, registered_variables


def stationary_state(config, seed=0, batch=()):
    """A driving state with mode vectors from the stationary distribution."""
    state = init_turbulence_driving(config)
    ou_phases = jax.random.normal(jax.random.key(seed), (*batch, *state.ou_phases.shape))
    return state._replace(ou_phases=ou_phases, step=jnp.asarray(0, jnp.int32))


# -------------------------------------------------------------
# ============== ↓ against the C++ reference ↓ ================
# -------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_modes_and_scales_match_turbgen(case):
    config, params, data = reference_case(case)
    num_modes, _, t_decay, dt, ou_var, weight_norm, kmin, kmax = data["scalars"]
    modes = build_driving_modes(config)
    assert modes.amplitudes.size == int(num_modes)
    assert relative_difference(modes.wave_vectors, data["modes"]) < ROUND_OFF

    # the parabola vanishes on the band edges: those amplitudes are the root of a round-off residual
    largest = data["amplitudes"].max()
    resolved = data["amplitudes"] > 1e-6 * largest
    np.testing.assert_allclose(modes.amplitudes[resolved], data["amplitudes"][resolved], rtol=ROUND_OFF)
    np.testing.assert_allclose(modes.amplitudes[~resolved], data["amplitudes"][~resolved], atol=1e-7 * largest)

    np.testing.assert_allclose(driving_wavenumber_range(config), (kmin, kmax), rtol=ROUND_OFF)
    np.testing.assert_allclose(turnover_time(config, params), t_decay, rtol=ROUND_OFF)
    np.testing.assert_allclose(update_interval(config, params), dt, rtol=ROUND_OFF)
    np.testing.assert_allclose(ou_standard_deviation(config, params), ou_var, rtol=ROUND_OFF)
    normalisation = solenoidal_weight_normalisation(params.solenoidal_weight, config.dimensionality)
    np.testing.assert_allclose(normalisation, weight_norm, rtol=ROUND_OFF)


def test_beattie_mode_set():
    """Parabola over 1 <= k <= 3 in 3D: 122 modes, 36 of them on the band edges with vanishing amplitude."""
    modes = build_driving_modes(BEATTIE)
    numbers = np.rint(modes.wave_vectors / (2 * np.pi)).astype(int)
    assert modes.amplitudes.size == 122
    assert np.all(np.isin(-numbers, numbers).all(axis=1))  # the full k-space: k and -k are separate modes
    assert np.sum(modes.amplitudes < 1e-6) == 36
    peak = np.sum(numbers**2, axis=1) == 4
    np.testing.assert_allclose(modes.amplitudes[peak], 1.0, rtol=ROUND_OFF)


def test_unusable_configurations_are_refused():
    with pytest.raises(ValueError, match="no mode"):
        build_driving_modes(BEATTIE._replace(k_min=0.2, k_max=0.4))
    with pytest.raises(ValueError, match="spectral form"):
        build_driving_modes(BEATTIE._replace(spectral_form=7))
    with pytest.raises(ValueError, match="dimensionality"):
        build_driving_modes(BEATTIE._replace(dimensionality=4))


@pytest.mark.parametrize("case", CASES)
def test_mode_state_and_field_match_turbgen(case):
    """The sequence of the C++ run, driven with its Gaussian deviates."""
    config, params, data = reference_case(case)
    wave_vectors = jnp.asarray(build_driving_modes(config).wave_vectors)
    sigma = float(ou_standard_deviation(config, params))
    damping_factor = np.exp(-1.0 / config.num_steps_per_turnover)
    cell_centers = grid_cell_centers(config, data["grid"])
    deviates = jnp.asarray(data["deviates"])

    ou_phases, num_drawn, step = deviates[0], 1, -1  # the initialisation: the stationary distribution
    difference = {"state": relative_difference(sigma * ou_phases, data["ou_phases"][0]), "projection": 0.0, "field": 0.0}
    for call, (_, _, step_after) in enumerate(data["times"]):
        for _ in range(int(step_after) - step):
            ou_phases = ornstein_uhlenbeck_update(ou_phases, deviates[num_drawn], damping_factor)
            num_drawn += 1
        step = int(step_after)
        projected = sigma * project_modes(ou_phases, wave_vectors, params.solenoidal_weight)
        projected = np.stack([projected.real, projected.imag], axis=-1)
        adjustment = data["ampl_factor"][call + 1] / np.asarray(params.amplitude_factor) ** 1.5
        state = TurbulenceDrivingState(ou_phases, jnp.asarray(step), jax.random.key(0), jnp.asarray(adjustment))
        field = driving_acceleration(state, config, params, cell_centers)
        assert field.shape == data["field"][call].shape
        difference = {
            "state": max(difference["state"], relative_difference(sigma * ou_phases, data["ou_phases"][call + 1])),
            "projection": max(difference["projection"], relative_difference(projected, data["coefficients"][call + 1])),
            "field": max(difference["field"], relative_difference(field, data["field"][call])),
        }
    print(f"\n{case}: largest relative difference to TurbGen", {k: f"{v:.1e}" for k, v in difference.items()})
    assert num_drawn == deviates.shape[0]
    assert max(difference.values()) < ROUND_OFF


@pytest.mark.parametrize("case", CASES)
def test_pattern_schedule_and_amplitude_adjustment_match_turbgen(case):
    config, params, data = reference_case(case)
    state = init_turbulence_driving(config)
    assert int(state.step) == -1
    for call, (time, updated, step_after) in enumerate(data["times"]):
        new_state = advance_turbulence_driving(state, time, config, params, data["velocity_dispersion"])
        assert int(new_state.step) == int(step_after)
        assert bool(jnp.any(new_state.ou_phases != state.ou_phases)) == bool(updated)
        internal_factor = np.asarray(params.amplitude_factor) ** 1.5 * new_state.amplitude_adjustment
        np.testing.assert_allclose(internal_factor, data["ampl_factor"][call + 1], rtol=ROUND_OFF)
        state = new_state
    if case == "auto_adjust":
        assert not np.allclose(state.amplitude_adjustment, 1.0, atol=1e-2)


def test_advance_draws_the_pattern_from_its_number():
    """Pattern n uses the deviates of fold_in(key, n): the sequence is independent of the calls in between."""
    params = TurbulenceDrivingParams()
    interval = update_interval(BEATTIE, params)
    state = init_turbulence_driving(BEATTIE)
    for due, step in ((1, 0), (3, 3)):
        advanced = advance_turbulence_driving(state, (step + 0.4) * interval, BEATTIE, params)
        deviates = jax.random.normal(jax.random.fold_in(state.key, step), state.ou_phases.shape)
        damping_factor = np.exp(-due / BEATTIE.num_steps_per_turnover)
        expected = ornstein_uhlenbeck_update(state.ou_phases, deviates, damping_factor)
        np.testing.assert_allclose(advanced.ou_phases, expected, rtol=ROUND_OFF, atol=ROUND_OFF)
        assert int(advanced.step) == step
        again = advance_turbulence_driving(advanced, (step + 0.9) * interval, BEATTIE, params)
        np.testing.assert_array_equal(again.ou_phases, advanced.ou_phases)
        state = advanced


# -------------------------------------------------------------
# ============== ↑ against the C++ reference ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ statistics ↓ ========================
# -------------------------------------------------------------


@pytest.mark.parametrize("stride", [1, 3])
def test_ornstein_uhlenbeck_statistics(stride):
    """
    A sequence of 4000 patterns (400 turnover times with stride 1): every
    mode component has variance 1 in units of the OU standard deviation and
    the correlation exp(-lag / T). With stride 3, three patterns are due at
    every call and applied at once.
    """
    params = TurbulenceDrivingParams()
    interval = update_interval(BEATTIE, params)
    num_patterns = 4000

    @jax.jit
    def sequence(state):
        def next_pattern(state, pattern):
            state = advance_turbulence_driving(state, (stride * pattern + 0.5) * interval, BEATTIE, params)
            return state, state.ou_phases

        return jax.lax.scan(next_pattern, state, jnp.arange(num_patterns))

    state, ou_phases = sequence(init_turbulence_driving(BEATTIE))
    ou_phases = np.asarray(ou_phases)
    assert int(state.step) == stride * (num_patterns - 1)

    variance = np.mean(ou_phases**2)
    variance_per_mode = np.mean(ou_phases**2, axis=(0, 2, 3))
    correlation = lambda lag: np.mean(ou_phases[:-lag] * ou_phases[lag:]) / variance
    turnover_times_per_call = stride / BEATTIE.num_steps_per_turnover
    correlation_time = -turnover_times_per_call / np.log(correlation(1))  # in turnover times
    print(f"\nstride {stride}: mean {np.mean(ou_phases):.4f}, variance {variance:.4f}, per mode "
          f"{variance_per_mode.min():.3f} to {variance_per_mode.max():.3f}, correlation time {correlation_time:.4f} T, "
          f"correlation at 3 calls {correlation(3):.4f} (expected {np.exp(-3 * turnover_times_per_call):.4f})")
    assert abs(np.mean(ou_phases)) < 0.02
    assert abs(variance - 1.0) < 0.02
    assert np.max(np.abs(variance_per_mode - 1.0)) < 0.25
    assert abs(correlation_time - 1.0) < 0.03
    assert abs(correlation(3) - np.exp(-3 * turnover_times_per_call)) < 0.015


def solenoidal_power_fraction(fields):
    """Helmholtz decomposition of periodic fields (batch, component, nx[, ny[, nz]]) in Fourier space."""
    ndim = fields.shape[1]
    transformed = np.fft.fftn(fields, axes=tuple(range(2, 2 + ndim)))
    numbers = np.meshgrid(*[np.fft.fftfreq(n) * n for n in fields.shape[2:]], indexing="ij")
    k_squared = sum(n**2 for n in numbers)
    k_squared[(0,) * ndim] = 1.0
    compressive = np.abs(sum(n * transformed[:, d] for d, n in enumerate(numbers))) ** 2 / k_squared
    return 1.0 - np.sum(compressive) / np.sum(np.abs(transformed) ** 2)


@pytest.mark.parametrize(
    "dimensionality, solenoidal_weight, spectral_form",
    [(3, 0.0, PARABOLA), (3, 0.5, PARABOLA), (3, 1.0, PARABOLA), (2, 0.5, PARABOLA), (2, 0.3, BAND), (3, 0.8, BAND)],
)
def test_solenoidal_fraction_and_rms_of_the_field(dimensionality, solenoidal_weight, spectral_form):
    """
    For n components and the solenoidal weight w, the solenoidal part carries
    the fraction (n - 1) w^2 / (1 - 2 w + n w^2) of the power (the weights of
    TurbGen's normalisation; eq. 9 of Federrath et al. 2010 is the compressive
    counterpart), and the normalisation makes the rms acceleration
    independent of w: a_rms = 6 sigma sqrt(sum_m A_m^2 / n).
    """
    config = BEATTIE._replace(dimensionality=dimensionality, spectral_form=spectral_form)
    params = TurbulenceDrivingParams(velocity=1.7, solenoidal_weight=solenoidal_weight)
    cell_centers = grid_cell_centers(config, (12,) * dimensionality)
    states = stationary_state(config, seed=1, batch=(256,))
    field_of = lambda ou_phases: driving_acceleration(states._replace(ou_phases=ou_phases), config, params, cell_centers)
    fields = np.asarray(jax.jit(jax.vmap(field_of))(states.ou_phases))

    weight_sum = 1.0 - 2.0 * solenoidal_weight + dimensionality * solenoidal_weight**2
    expected_fraction = (dimensionality - 1) * solenoidal_weight**2 / weight_sum
    fraction = solenoidal_power_fraction(fields)
    amplitudes = build_driving_modes(config).amplitudes
    expected_rms = 6.0 * float(ou_standard_deviation(config, params)) * np.sqrt(np.sum(amplitudes**2) / dimensionality)
    rms = np.sqrt(np.mean(np.sum(fields**2, axis=1)))
    print(f"\n{dimensionality}D, weight {solenoidal_weight}: solenoidal fraction {fraction:.4f} (expected "
          f"{expected_fraction:.4f}), rms acceleration {rms:.4f} (expected {expected_rms:.4f})")
    if solenoidal_weight in (0.0, 1.0):
        assert abs(fraction - expected_fraction) < 1e-12
    assert abs(fraction - expected_fraction) < 0.02
    assert abs(rms / expected_rms - 1.0) < 0.03


# -------------------------------------------------------------
# ===================== ↑ statistics ↑ ========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ parts of the grid ↓ ======================
# -------------------------------------------------------------

NON_CUBIC = BEATTIE._replace(box_size=(1.0, 2.0, 0.5), k_max=2.0)
NON_CUBIC_GRID = (16, 12, 8)


def test_slabs_with_offsets_equal_the_whole_grid():
    params = TurbulenceDrivingParams(solenoidal_weight=0.7)
    state = stationary_state(NON_CUBIC)
    cell_widths = [length / n for length, n in zip(NON_CUBIC.box_size, NON_CUBIC_GRID)]
    x, y, z = grid_cell_centers(NON_CUBIC, NON_CUBIC_GRID)
    whole = driving_acceleration(state, NON_CUBIC, params, (x, y, z))

    slabs = []
    for shard in range(4):
        x_slab = cell_center_coordinates(4, cell_widths[0], index_offset=4 * shard)
        slabs.append(driving_acceleration(state, NON_CUBIC, params, (x_slab, y, z)))
    np.testing.assert_allclose(jnp.concatenate(slabs, axis=1), whole, rtol=0, atol=ROUND_OFF * float(jnp.max(whole)))

    # a block of the grid with two ghost cells on every side; the field is periodic
    block = [(4, 8), (0, 12), (2, 6)]
    padded_centers = [
        cell_center_coordinates(end - start + 4, width, index_offset=start - 2)
        for (start, end), width in zip(block, cell_widths)
    ]
    padded = driving_acceleration(state, NON_CUBIC, params, padded_centers)
    periodic = jnp.pad(whole, ((0, 0), (2, 2), (2, 2), (2, 2)), mode="wrap")
    expected = periodic[:, 4:12, 0:16, 2:10]
    np.testing.assert_allclose(padded, expected, rtol=0, atol=ROUND_OFF * float(jnp.max(whole)))


@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 devices (JAX_PLATFORMS=cpu, this file on its own)")
def test_sharded_evaluation_equals_single_device():
    params = TurbulenceDrivingParams(solenoidal_weight=0.7)
    state = stationary_state(NON_CUBIC)
    x, y, z = grid_cell_centers(NON_CUBIC, NON_CUBIC_GRID)
    whole = driving_acceleration(state, NON_CUBIC, params, (x, y, z))

    mesh = jax.make_mesh((4,), ("x",), axis_types=(AxisType.Auto,), devices=jax.devices()[:4])
    field_sharding = NamedSharding(mesh, P(None, "x"))
    x_sharded = jax.device_put(x, NamedSharding(mesh, P("x")))
    evaluate = jax.jit(lambda state, x: driving_acceleration(state, NON_CUBIC, params, (x, y, z)),
                       out_shardings=field_sharding)
    sharded = evaluate(state, x_sharded)
    assert sharded.sharding.is_equivalent_to(field_sharding, sharded.ndim)
    assert len(sharded.sharding.device_set) == 4
    np.testing.assert_allclose(sharded, whole, rtol=0, atol=ROUND_OFF * float(jnp.max(whole)))

    state_sharding = NamedSharding(mesh, P(None, "x"))
    primitive_state = jax.device_put(jnp.ones((5, *NON_CUBIC_GRID)), state_sharding)
    stand_in = simulation_stand_in(NON_CUBIC, params, NON_CUBIC_GRID)
    kick = jax.jit(lambda primitive_state, state: apply_turbulence_driving(primitive_state, state, 0.1, *stand_in))
    kicked = kick(primitive_state, state)
    assert kicked.sharding.is_equivalent_to(state_sharding, kicked.ndim)
    # the uniform velocity 1 is bulk motion and goes; the mean of the field is zero but for round-off
    bulk_motion = np.mean(np.asarray(whole), axis=(1, 2, 3), keepdims=True)
    np.testing.assert_allclose(kicked[1:4], 0.1 * (whole - bulk_motion), rtol=0, atol=ROUND_OFF)
    np.testing.assert_array_equal(kicked[0], 1.0)


# -------------------------------------------------------------
# ================ ↑ parts of the grid ↑ ======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======== ↓ gradients, single precision, the kick ↓ ==========
# -------------------------------------------------------------


def test_gradients_with_respect_to_the_parameters():
    """
    The field scales as velocity^2, k_driving^(1/2) and amplitude_factor^(3/2),
    which fixes the gradient of sum(a^2); the one with respect to the
    solenoidal weight is compared with a finite difference.
    """
    state = stationary_state(BEATTIE, seed=2)
    cell_centers = grid_cell_centers(BEATTIE, (8, 8, 8))
    interval = update_interval(BEATTIE, TurbulenceDrivingParams())

    def power_per_component(params):
        advanced = advance_turbulence_driving(state, 2.5 * interval, BEATTIE, params)
        return jnp.sum(driving_acceleration(advanced, BEATTIE, params, cell_centers) ** 2, axis=(1, 2, 3))

    power = lambda params: jnp.sum(power_per_component(params))
    params = TurbulenceDrivingParams(
        velocity=jnp.asarray(1.3), k_driving=jnp.asarray(2.0), solenoidal_weight=jnp.asarray(0.4),
        amplitude_factor=jnp.asarray([0.8, 1.0, 1.2]),
    )
    gradient = jax.jit(jax.grad(power))(params)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in gradient)
    np.testing.assert_allclose(gradient.velocity, 4.0 * power(params) / params.velocity, rtol=1e-10)
    np.testing.assert_allclose(gradient.k_driving, power(params) / params.k_driving, rtol=1e-10)
    expected = 3.0 * power_per_component(params) / params.amplitude_factor
    np.testing.assert_allclose(gradient.amplitude_factor, expected, rtol=1e-10)
    shifted = lambda h: power(params._replace(solenoidal_weight=params.solenoidal_weight + h))
    np.testing.assert_allclose(gradient.solenoidal_weight, (shifted(1e-5) - shifted(-1e-5)) / 2e-5, rtol=1e-7)


def test_single_precision():
    """The precision of the state and the coordinates is the precision of the field."""
    params = TurbulenceDrivingParams()
    interval = float(update_interval(BEATTIE, params))
    grid = (8, 12, 16)
    state = advance_turbulence_driving(init_turbulence_driving(BEATTIE, jnp.float32), 3.2 * interval, BEATTIE, params)
    assert state.ou_phases.dtype == jnp.float32 and state.amplitude_adjustment.dtype == jnp.float32
    field = driving_acceleration(state, BEATTIE, params, grid_cell_centers(BEATTIE, grid, jnp.float32))
    assert field.dtype == jnp.float32

    double = state._replace(ou_phases=state.ou_phases.astype(jnp.float64),
                            amplitude_adjustment=state.amplitude_adjustment.astype(jnp.float64))
    reference = driving_acceleration(double, BEATTIE, params, grid_cell_centers(BEATTIE, grid))
    assert reference.dtype == jnp.float64
    assert relative_difference(field, np.asarray(reference)) < 2e-5

    with jax.enable_x64(False):  # the default of a single-precision run
        state = advance_turbulence_driving(init_turbulence_driving(BEATTIE), 3.2 * interval, BEATTIE, params)
        field = jax.jit(driving_acceleration, static_argnums=1)(state, BEATTIE, params, grid_cell_centers(BEATTIE, grid))
        assert field.dtype == jnp.float32 and state.ou_phases.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(field)))


@pytest.mark.parametrize("dimensionality, velocity_index", [(1, 1), (2, (1, 2, -1)), (3, (1, 2, 3)), (2, (4, 2))])
@pytest.mark.parametrize("ghost_layers", [0, 2])
def test_velocity_kick(dimensionality, velocity_index, ghost_layers):
    """
    v <- v + dt a - (dt F + P) / M in the physical cells, F = sum rho a and
    P = sum rho v (FLASH: no net momentum after the kick); everything else,
    the ghost cells included, untouched.
    """
    config = BEATTIE._replace(dimensionality=dimensionality, box_size=(1.0, 0.75, 0.5)[:dimensionality])
    params = TurbulenceDrivingParams()
    state = stationary_state(config, seed=3)
    grid = (16, 12, 8)[:dimensionality]
    acceleration = np.asarray(driving_acceleration(state, config, params, grid_cell_centers(config, grid)))
    assert acceleration.shape == (dimensionality, *grid)

    padded = tuple(n + 2 * ghost_layers for n in grid)
    primitive_state = jax.random.uniform(jax.random.key(4), (6, *padded), minval=0.5, maxval=1.5)
    stand_in = simulation_stand_in(config, params, grid, velocity_index=velocity_index)
    dt = 0.01
    kicked = jax.jit(lambda primitive_state, state: apply_turbulence_driving(primitive_state, state, dt, *stand_in))(
        primitive_state, state
    )

    cells = (slice(None),) + tuple(slice(ghost_layers, ghost_layers + n) for n in grid)
    inner = np.asarray(primitive_state)[cells]
    indices = [velocity_index] if isinstance(velocity_index, int) else list(velocity_index[:dimensionality])
    density = inner[0]
    momentum = np.array([np.sum(density * inner[index]) for index in indices])
    force = np.sum(density * acceleration, axis=tuple(range(1, dimensionality + 1)))
    correction = (dt * force + momentum) / np.sum(density)
    expected = np.array(primitive_state)
    for component, index in enumerate(indices):
        expected[(index,) + cells[1:]] += dt * acceleration[component] - correction[component]
    np.testing.assert_allclose(kicked, expected, rtol=0, atol=ROUND_OFF)
    kicked_inner = np.asarray(kicked)[cells]
    net_momentum = [np.sum(kicked_inner[0] * kicked_inner[index]) for index in indices]
    np.testing.assert_allclose(net_momentum, 0.0, atol=ROUND_OFF * np.sum(density))
    stored_kick = jnp.asarray(expected[indices] - np.asarray(primitive_state)[indices])  # the kick as a stored field
    stored = add_velocity_kick(primitive_state, stored_kick, velocity_index)
    np.testing.assert_allclose(stored, expected, rtol=0, atol=ROUND_OFF)


def test_velocity_dispersion_is_that_of_the_flash_plugin():
    """s_d = sqrt(<v_d^2>_V - (P_d / M)^2) over the physical cells (Stir.F90)."""
    config = BEATTIE
    grid = (8, 6, 4)
    stand_in = simulation_stand_in(config, TurbulenceDrivingParams(), grid)
    primitive_state = jax.random.uniform(jax.random.key(5), (5, 12, 10, 8), minval=0.5, maxval=1.5)
    measured = velocity_dispersion(primitive_state, stand_in[0], stand_in[2])
    inner = np.asarray(primitive_state)[:, 2:10, 2:8, 2:6]
    mean_velocity = [np.sum(inner[0] * inner[d]) / np.sum(inner[0]) for d in (1, 2, 3)]
    expected = [np.sqrt(np.mean(inner[d] ** 2) - mean_velocity[d - 1] ** 2) for d in (1, 2, 3)]
    np.testing.assert_allclose(measured, expected, rtol=1e-12)


# -------------------------------------------------------------
# ======== ↑ gradients, single precision, the kick ↑ ==========
# -------------------------------------------------------------
