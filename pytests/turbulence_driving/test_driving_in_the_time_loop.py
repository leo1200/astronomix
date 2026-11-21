"""
Turbulence driving in the time loop of every scheme.

What is checked:

* every scheme family and state layout kicks with the driving field: a
  uniform gas at rest carries v = dt a after one step, with the acceleration
  of the pattern at the end of the step (finite difference with flux
  splitting and with a Riemann flux, VL2 with GLM and with constrained
  transport, RK2_SSP on a ghost-cell layout, MUSCL-Hancock and
  BEATTIE_REFERENCE; isothermal, polytropic and ideal gas; one, two and three
  dimensions; periodic roll and ghost cells); the pressure of the ideal gas
  is unchanged, so the energy grows by the kinetic energy of the kick;
* ``finalize_config`` passes the box to the driving;
* the time step is limited to half the update interval;
* the patterns follow the time, not the steps; a restart from a checkpoint
  continues the run bit for bit; the automatic amplitude adjustment acts;
* derivatives with respect to ``velocity`` and ``solenoidal_weight``: the VJP
  of a short driven run is finite and passes the dot-product identity with
  the JVP; through the checkpointed adaptive loop it has the scaling of the
  kinetic energy with velocity^4;
* four interpreted CPU devices give the result of one device;
* on a GPU, the Pallas kernels give the result of the array form.

The four devices: the conftest splits the CPU into four when JAX runs on the
CPU::

    JAX_PLATFORMS=cpu python -m pytest pytests/turbulence_driving/test_driving_in_the_time_loop.py
"""

# jax
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec

# numerics
import numpy as np

# testing
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    CONSTRAINED_TRANSPORT,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLL,
    HLLC,
    HLLD,
    IDEAL_GAS,
    ISOTHERMAL,
    MUSCL_HANCOCK,
    NATIVE_JAX,
    ON_DEVICE,
    OPEN_BOUNDARY,
    PALLAS,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    POLYTROPIC,
    RK2_SSP,
    SPHERICAL,
    TO_DISK,
    VL2,
)

# astronomix containers
from astronomix.option_classes.presets import (
    ATHENAPK_VL2,
    BEATTIE_REFERENCE,
    FD_PAPER,
    FD_RIEMANN_IDEAL_GAS,
    FD_RIEMANN_ISOTHERMAL,
)
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    FVSolverConfig,
    SimulationConfig,
    SnapshotSettings,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix._modules._turbulence_driving import (
    TurbulenceDrivingConfig,
    TurbulenceDrivingParams,
    advance_turbulence_driving,
    cell_center_coordinates,
    driving_acceleration,
    init_turbulence_driving,
    update_interval,
)

# astronomix functions
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix._snapshotting._orbax_storage import load_loop_checkpoint
from astronomix.setup_helpers import restart_from_latest_checkpoint
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

pytestmark = pytest.mark.x64

DRIVING = TurbulenceDrivingConfig(turbulence_driving=True)
TIME_STEP = 0.01
ROUND_OFF = 1e-12

# name: (solver mode, solver settings, equation of state, mhd, dimensionality, cells, boundary)
SCHEMES = {
    "fd_flux_splitting_hydro_ideal_gas": (FINITE_DIFFERENCE, FDSolverConfig(), IDEAL_GAS, False, 3, 8, "periodic"),
    "fd_flux_splitting_hydro_ghost_cells": (FINITE_DIFFERENCE, FDSolverConfig(), IDEAL_GAS, False, 3, 8, "open"),
    "fd_flux_splitting_hydro_2d": (FINITE_DIFFERENCE, FDSolverConfig(), IDEAL_GAS, False, 2, 16, "periodic"),
    "fd_paper_mhd_ideal_gas": (FINITE_DIFFERENCE, FD_PAPER, IDEAL_GAS, True, 3, 8, "periodic"),
    "fd_flux_splitting_mhd_isothermal": (FINITE_DIFFERENCE, FDSolverConfig(), ISOTHERMAL, True, 3, 8, "periodic"),
    "fd_riemann_isothermal": (FINITE_DIFFERENCE, FD_RIEMANN_ISOTHERMAL, ISOTHERMAL, True, 3, 8, "periodic"),
    "fd_riemann_ideal_gas": (FINITE_DIFFERENCE, FD_RIEMANN_IDEAL_GAS, IDEAL_GAS, True, 3, 8, "periodic"),
    "fv_vl2_glm_mhd_ideal_gas": (FINITE_VOLUME, ATHENAPK_VL2, IDEAL_GAS, True, 3, 8, "periodic"),
    "fv_vl2_ct_mhd_ideal_gas": (
        FINITE_VOLUME,
        FVSolverConfig(time_integrator=VL2, riemann_solver=HLLD, divergence_control=CONSTRAINED_TRANSPORT),
        IDEAL_GAS, True, 3, 8, "periodic",
    ),
    "fv_vl2_hydro_polytropic": (
        FINITE_VOLUME, FVSolverConfig(time_integrator=VL2, riemann_solver=HLL), POLYTROPIC, False, 3, 8, "periodic"
    ),
    "fv_rk2_hydro_ghost_cells": (
        FINITE_VOLUME, FVSolverConfig(time_integrator=RK2_SSP, riemann_solver=HLLC), IDEAL_GAS, False, 3, 8, "periodic"
    ),
    "fv_muscl_hancock_hydro_isothermal_1d": (
        FINITE_VOLUME, FVSolverConfig(time_integrator=MUSCL_HANCOCK, riemann_solver=HLL), ISOTHERMAL, False, 1, 32,
        "periodic",
    ),
    "beattie_reference_isothermal": (FINITE_VOLUME, BEATTIE_REFERENCE, ISOTHERMAL, True, 3, 8, "periodic"),
}


def scheme_config(name, backend=NATIVE_JAX, **settings):
    """The configuration of a scheme of ``SCHEMES`` with driving (not finalised)."""
    solver_mode, solver_config, equation_of_state, mhd, dimensionality, _, boundary = SCHEMES[name]
    boundary_type = PERIODIC_BOUNDARY if boundary == "periodic" else OPEN_BOUNDARY
    sides = BoundarySettings1D(boundary_type, boundary_type)
    defaults = dict(
        solver_mode=solver_mode,
        solver_config=solver_config,
        equation_of_state=equation_of_state,
        mhd=mhd,
        dimensionality=dimensionality,
        box_size=1.0,
        boundary_settings=sides if dimensionality == 1 else BoundarySettings(*([sides] * dimensionality)),
        backend_config=BackendConfig(backend=backend),
        turbulence_driving_config=DRIVING,
        progress_bar=False,
    )
    return SimulationConfig(**{**defaults, **settings})


def uniform_state(config, registered_variables, cells, velocity=None):
    """A uniform gas at rest (or with the given velocity components), a weak uniform field for MHD."""
    shape = (cells,) * config.dimensionality
    ones = jnp.ones(shape)
    velocity = velocity if velocity is not None else (0 * ones,) * 3
    fields = dict(density=ones, velocity_x=velocity[0])
    if config.dimensionality > 1 or config.mhd:
        fields.update(velocity_y=velocity[1], velocity_z=velocity[2])
    if config.equation_of_state == IDEAL_GAS:
        fields["gas_pressure"] = ones
    if config.mhd:
        field = dict(magnetic_field_x=0.1 * ones, magnetic_field_y=0 * ones, magnetic_field_z=0 * ones)
        fields.update(field)
        if config.solver_mode == FINITE_DIFFERENCE:
            fields.update({"interface_" + name: value for name, value in field.items()})
    return construct_primitive_state(config, registered_variables, **fields)


def expected_acceleration(config, params, time, dtype=jnp.float64):
    """The acceleration of the pattern that holds at ``time`` on the physical grid of a finalised configuration."""
    driving_config = config.turbulence_driving_config
    driving_params = params.turbulence_driving_params
    state = advance_turbulence_driving(init_turbulence_driving(driving_config, dtype), time, driving_config,
                                       driving_params)
    num_cells = tuple(config.num_cells)[: config.dimensionality]
    lengths = driving_config.box_size
    centers = [cell_center_coordinates(n, length / n, 0, dtype) for n, length in zip(num_cells, lengths)]
    return np.asarray(driving_acceleration(state, driving_config, driving_params, centers))


def velocity_rows(state, registered_variables, num_components):
    index = registered_variables.velocity_index
    indices = (index,) if isinstance(index, int) else tuple(index)[:num_components]
    return np.stack([np.asarray(state[i]) for i in indices])


def one_step(name, backend=NATIVE_JAX, **settings):
    """Initial state, final state, finalised configuration and registry of one fixed step of a scheme."""
    config = scheme_config(name, backend, fixed_timestep=True, num_timesteps=1, **settings)
    registered_variables = get_registered_variables(config)
    initial_state = uniform_state(config, registered_variables, SCHEMES[name][5])
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=TIME_STEP, C_cfl=0.4)
    final_state = time_integration(initial_state, config, params, registered_variables)
    return np.asarray(initial_state), np.asarray(final_state), config, params, registered_variables


# -------------------------------------------------------------
# ================ ↓ every scheme and layout ↓ ================
# -------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SCHEMES))
def test_every_scheme_kicks_with_the_driving_field(name):
    initial_state, final_state, config, params, registered_variables = one_step(name)
    if name.endswith("ghost_cells"):
        assert config.boundary_handling != PERIODIC_ROLL and config.num_ghost_cells > 0
    acceleration = expected_acceleration(config, params, TIME_STEP)
    bulk_motion = np.mean(acceleration, axis=tuple(range(1, acceleration.ndim)), keepdims=True)
    velocity = velocity_rows(final_state, registered_variables, config.dimensionality)
    scale = TIME_STEP * np.max(np.abs(acceleration))
    np.testing.assert_allclose(velocity, TIME_STEP * (acceleration - bulk_motion), rtol=0, atol=ROUND_OFF * scale)
    assert scale > 1e-3
    np.testing.assert_allclose(final_state[registered_variables.density_index], 1.0, rtol=ROUND_OFF)
    if config.equation_of_state == IDEAL_GAS:
        pressure = final_state[registered_variables.pressure_index]
        np.testing.assert_allclose(pressure, initial_state[registered_variables.pressure_index], rtol=ROUND_OFF)


def test_finalize_config_passes_the_box_to_the_driving():
    config = scheme_config("fd_flux_splitting_hydro_2d", box_size=2.0)
    config = finalize_config(config, (5, 16, 16))
    assert config.turbulence_driving_config.dimensionality == 2
    assert config.turbulence_driving_config.box_size == (2.0, 2.0)
    with pytest.raises(ValueError, match="requires geometry=CARTESIAN"):
        spherical = scheme_config("fv_muscl_hancock_hydro_isothermal_1d", geometry=SPHERICAL)
        finalize_config(spherical, (3, 32))


# -------------------------------------------------------------
# ================ ↑ every scheme and layout ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ============ ↓ time step, sequence, restart ↓ ===============
# -------------------------------------------------------------


def driven_run(directory, t_end, num_snapshots=1, courant_number=0.4, driving=DRIVING, velocity=1.0, **settings):
    """
    An adaptive driven run of the 3D VL2 hydro scheme on 8^3 cells,
    checkpointed to ``directory`` unless the settings say otherwise.
    """
    storage = dict(snapshot_storage_mode=TO_DISK, snapshot_storage_path=str(directory), num_snapshots=num_snapshots)
    config = scheme_config("fv_vl2_hydro_polytropic", turbulence_driving_config=driving, **{**storage, **settings})
    registered_variables = get_registered_variables(config)
    initial_state = uniform_state(config, registered_variables, 8)
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_end=t_end, C_cfl=courant_number, turbulence_driving_params=TurbulenceDrivingParams(velocity=velocity)
    )
    return config, params, registered_variables, initial_state


def test_time_step_is_limited_to_half_the_update_interval():
    on_device = dict(
        snapshot_storage_mode=ON_DEVICE,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    config, params, registered_variables, initial_state = driven_run(None, 0.0, velocity=50.0, **on_device)
    half_interval = 0.5 * float(update_interval(config.turbulence_driving_config, params.turbulence_driving_params))
    params = params._replace(t_end=10 * half_interval)
    result = time_integration(initial_state, config, params, registered_variables)
    assert int(result.num_iterations) in (10, 11)


def test_patterns_follow_the_time_not_the_steps(tmp_path):
    t_end = 2.5 * float(update_interval(DRIVING._replace(dimensionality=3), TurbulenceDrivingParams()))
    states = []
    for courant_number in (0.4, 0.15):
        directory = tmp_path / str(courant_number)
        run = driven_run(directory, t_end, courant_number=courant_number)
        config, params, registered_variables, initial_state = run
        time_integration(initial_state, config, params, registered_variables)
        states.append(load_loop_checkpoint(directory).driving_state)
    assert int(states[0].step) == int(states[1].step) == 2
    np.testing.assert_array_equal(states[0].ou_phases, states[1].ou_phases)
    np.testing.assert_array_equal(jax.random.key_data(states[0].key), jax.random.key_data(states[1].key))


def test_restart_continues_the_run_bit_for_bit(tmp_path):
    t_end = 3.5 * float(update_interval(DRIVING._replace(dimensionality=3), TurbulenceDrivingParams()))
    config, params, registered_variables, initial_state = driven_run(tmp_path / "whole", t_end, num_snapshots=2)
    uninterrupted = np.asarray(time_integration(initial_state, config, params, registered_variables))

    state, restart_params, restart_state = restart_from_latest_checkpoint(tmp_path / "whole", params, step=1)
    assert restart_state.driving_state is not None and int(restart_state.driving_state.step) >= 1
    restarted_config = config._replace(snapshot_storage_path=str(tmp_path / "restarted"), num_snapshots=1)
    restarted = np.asarray(
        time_integration(state, restarted_config, restart_params, registered_variables, restart_state=restart_state)
    )
    np.testing.assert_array_equal(restarted, uninterrupted)
    whole = load_loop_checkpoint(tmp_path / "whole", 2).driving_state
    resumed = load_loop_checkpoint(tmp_path / "restarted").driving_state
    np.testing.assert_array_equal(resumed.ou_phases, whole.ou_phases)
    assert int(resumed.step) == int(whole.step) == 3


@pytest.mark.parametrize("auto_adjust", [False, True], ids=["fixed amplitude", "auto adjust"])
def test_in_memory_chunks_continue_the_run_bit_for_bit(auto_adjust):
    # Integrating in memory in chunks, each call resuming from the loop state
    # the previous call returned, reproduces one uninterrupted call bit for
    # bit: the driving sequence (and the amplitude adjustment) continues
    # across the calls. Fixed time steps make both take the same steps: a
    # power of two, so that every chunk's start, end and step are exact, and
    # not a divisor of the update interval, so that no pattern switch falls
    # on a step end.
    driving = DRIVING._replace(amplitude_auto_adjust=auto_adjust)
    interval = float(update_interval(driving._replace(dimensionality=3), TurbulenceDrivingParams()))
    time_step = 2.0**-7
    assert 2.0 < interval / time_step < 10.0 and interval / time_step % 1.0 > 0.1
    chunk_steps = (9, 11, 8)
    fixed_steps = dict(snapshot_storage_mode=ON_DEVICE, fixed_timestep=True, num_timesteps=sum(chunk_steps))
    config, params, registered_variables, initial_state = driven_run(
        None, sum(chunk_steps) * time_step, driving=driving, **fixed_steps
    )
    uninterrupted, whole = time_integration(
        initial_state, config, params, registered_variables, return_loop_state=True
    )

    state, loop_state, steps_done = initial_state, None, 0
    for num_steps in chunk_steps:
        chunk_config = config._replace(num_timesteps=num_steps)
        chunk_params = params._replace(t_start=steps_done * time_step, t_end=(steps_done + num_steps) * time_step)
        state, loop_state = time_integration(
            state, chunk_config, chunk_params, registered_variables, restart_state=loop_state, return_loop_state=True
        )
        steps_done += num_steps

    assert int(loop_state.driving_state.step) == int(whole.driving_state.step) >= 3
    np.testing.assert_array_equal(loop_state.driving_state.ou_phases, whole.driving_state.ou_phases)
    np.testing.assert_array_equal(
        loop_state.driving_state.amplitude_adjustment, whole.driving_state.amplitude_adjustment
    )
    np.testing.assert_array_equal(np.asarray(loop_state.primitive_state), np.asarray(state))
    np.testing.assert_array_equal(np.asarray(state), np.asarray(uninterrupted))


def test_restarting_from_a_fresh_loop_state_repeats_the_sequence():
    # Without the loop state, a second call starts the driving sequence again
    # from its seed: the reason to carry the loop state between chunks.
    interval = float(update_interval(DRIVING._replace(dimensionality=3), TurbulenceDrivingParams()))
    config, params, registered_variables, initial_state = driven_run(
        None, 1.5 * interval, snapshot_storage_mode=ON_DEVICE
    )
    _, first = time_integration(initial_state, config, params, registered_variables, return_loop_state=True)
    second_params = params._replace(t_start=1.5 * interval, t_end=3.0 * interval)
    _, carried = time_integration(
        first.primitive_state, config, second_params, registered_variables, restart_state=first, return_loop_state=True
    )
    _, fresh = time_integration(
        first.primitive_state, config, second_params, registered_variables, return_loop_state=True
    )
    assert int(carried.driving_state.step) == 3
    assert not np.array_equal(np.asarray(carried.driving_state.ou_phases), np.asarray(fresh.driving_state.ou_phases))


def test_automatic_amplitude_adjustment_acts_in_the_loop(tmp_path):
    driving = DRIVING._replace(amplitude_auto_adjust=True)
    t_end = 4.5 * float(update_interval(driving._replace(dimensionality=3), TurbulenceDrivingParams()))
    config, params, registered_variables, initial_state = driven_run(tmp_path, t_end, driving=driving)
    time_integration(initial_state, config, params, registered_variables)
    adjustment = np.asarray(load_loop_checkpoint(tmp_path).driving_state.amplitude_adjustment)
    assert adjustment.shape == (3,) and np.all(adjustment > 1.0)  # far below the target velocity, it grows


# -------------------------------------------------------------
# ============ ↑ time step, sequence, restart ↑ ===============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ derivatives ↓ =======================
# -------------------------------------------------------------


def final_state_of(velocity, solenoidal_weight, config, registered_variables, initial_state, t_end):
    params = SimulationParams(
        t_end=t_end,
        C_cfl=0.4,
        turbulence_driving_params=TurbulenceDrivingParams(velocity=velocity, solenoidal_weight=solenoidal_weight),
    )
    return time_integration(initial_state, config, params, registered_variables)


def test_vjp_passes_the_dot_product_identity():
    config = scheme_config("fv_vl2_hydro_polytropic", fixed_timestep=True, num_timesteps=3)
    registered_variables = get_registered_variables(config)
    initial_state = uniform_state(config, registered_variables, 8)
    config = finalize_config(config, initial_state.shape)

    def run(velocity, solenoidal_weight):
        return final_state_of(velocity, solenoidal_weight, config, registered_variables, initial_state, 3 * TIME_STEP)

    primal = (jnp.asarray(1.2), jnp.asarray(0.6))
    tangent = (jnp.asarray(0.7), jnp.asarray(-0.4))
    final_state, jvp_result = jax.jvp(run, primal, tangent)
    cotangent = jax.random.normal(jax.random.key(1), final_state.shape)
    _, pullback = jax.vjp(run, *primal)
    vjp_result = pullback(cotangent)
    assert all(bool(jnp.isfinite(g)) for g in vjp_result)
    forward = float(jnp.sum(cotangent * jvp_result))
    backward = float(sum(g * t for g, t in zip(vjp_result, tangent)))
    np.testing.assert_allclose(forward, backward, rtol=1e-10)
    assert abs(forward) > 0


def test_vjp_through_the_checkpointed_adaptive_loop():
    """At early times the kinetic energy grows as velocity^4 (the acceleration as velocity^2)."""
    config = scheme_config("fv_vl2_hydro_polytropic", differentiation_mode=BACKWARDS, num_checkpoints=4)
    registered_variables = get_registered_variables(config)
    initial_state = uniform_state(config, registered_variables, 8)
    config = finalize_config(config, initial_state.shape)

    def kinetic_energy(velocity, solenoidal_weight):
        state = final_state_of(velocity, solenoidal_weight, config, registered_variables, initial_state, 0.02)
        return 0.5 * jnp.sum(state[0] * jnp.sum(state[1:4] ** 2, axis=0))

    velocity = jnp.asarray(0.3)
    gradient = jax.grad(kinetic_energy, argnums=(0, 1))(velocity, jnp.asarray(0.5))
    assert all(bool(jnp.isfinite(g)) for g in gradient)
    np.testing.assert_allclose(float(gradient[0]) * velocity / kinetic_energy(velocity, 0.5), 4.0, rtol=0.02)


# -------------------------------------------------------------
# ===================== ↑ derivatives ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================= ↓ devices and kernels ↓ ===================
# -------------------------------------------------------------


def driven_turbulent_run(name, backend, sharding=None, cells=16, num_timesteps=4, smooth=True, **settings):
    """
    A few fixed steps of a scheme from a random velocity field: smooth (Fourier modes |n_i| < 4, rms 0.5) or
    white noise (rms 0.5).
    """
    config = scheme_config(name, backend, fixed_timestep=True, num_timesteps=num_timesteps, **settings)
    registered_variables = get_registered_variables(config)
    velocity = jax.random.normal(jax.random.key(7), (3, cells, cells, cells))
    if smooth:
        low = jnp.abs(jnp.fft.fftfreq(cells) * cells) < 4
        modes = low[:, None, None] & low[None, :, None] & low[None, None, :]
        velocity = jnp.fft.ifftn(jnp.fft.fftn(velocity, axes=(1, 2, 3)) * modes, axes=(1, 2, 3)).real
    velocity = 0.5 * velocity / jnp.sqrt(jnp.mean(jnp.sum(velocity**2, axis=0)))
    initial_state = uniform_state(config, registered_variables, cells, velocity=tuple(velocity))
    config = finalize_config(config, initial_state.shape)
    if sharding is not None:
        initial_state = jax.device_put(initial_state, sharding)
    params = SimulationParams(t_end=num_timesteps * 2e-3, C_cfl=0.4)
    return np.asarray(time_integration(initial_state, config, params, registered_variables, sharding=sharding))


@pytest.mark.slow
@pytest.mark.devices(4)
@pytest.mark.parametrize("name", ["fd_riemann_isothermal", "beattie_reference_isothermal", "fv_vl2_glm_mhd_ideal_gas"])
def test_four_devices_equal_one_device(name):
    single = driven_turbulent_run(name, NATIVE_JAX)
    axes = ("variable", "x", "y", "z")
    mesh = jax.make_mesh((1, 4, 1, 1), axes, axis_types=(AxisType.Auto,) * 4, devices=jax.devices()[:4])
    sharded = driven_turbulent_run(name, NATIVE_JAX, NamedSharding(mesh, PartitionSpec(*axes)))
    np.testing.assert_allclose(sharded, single, rtol=0, atol=ROUND_OFF * np.max(np.abs(single)))


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["fd_riemann_isothermal", "beattie_reference_isothermal", "fd_paper_mhd_ideal_gas"])
def test_pallas_kernels_equal_the_array_form(name):
    """
    The compiled kernels run in single precision on grids their blocks tile. Two steps: the second step's kernels
    take the kicked state; over more steps the flow amplifies the rounding of the kernels in single cells, driven
    or not.
    """
    with jax.enable_x64(False):
        array_form = driven_turbulent_run(name, NATIVE_JAX, cells=32, num_timesteps=2)
        kernels = driven_turbulent_run(name, PALLAS, cells=32, num_timesteps=2)
    assert array_form.dtype == np.float32
    np.testing.assert_allclose(kernels, array_form, rtol=0, atol=2e-5 * np.max(np.abs(array_form)))


# -------------------------------------------------------------
# ================= ↑ devices and kernels ↑ ===================
# -------------------------------------------------------------
