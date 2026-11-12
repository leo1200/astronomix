"""
The kernels of the reference scheme (``_finite_volume/_reference_pallas.py``) against its array form:

* one sweep per axis, the cleaning step and the largest signal speed agree to rounding, in two and three
  dimensions, for the isothermal and the ideal gas, without ghost cells and with them;
* a run of a few steps through ``time_integration`` agrees to rounding in both layouts;
* the derivative of a step with the kernels selected is the derivative of the array form;
* the predicates refuse what the kernels do not cover;
* where a GPU is present: the compiled kernels in single precision against the array form.

All comparisons but the last run the kernels through the interpreted path, in double precision. Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/mhd/test_reference_kernel.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from types import SimpleNamespace

# numerics
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSTRAINED_TRANSPORT,
    DONOR_CELL,
    FINITE_VOLUME,
    GHOST_CELLS,
    IDEAL_GAS,
    ISOTHERMAL,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    POLYTROPIC,
)

# astronomix containers
from astronomix.option_classes.presets import BEATTIE_REFERENCE
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SnapshotSettings,
    StaticFloatVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
import astronomix._finite_volume._reference_pallas as reference_pallas
from astronomix._finite_volume._divergence._induction_source import _parabolic_cleaning
from astronomix._finite_volume._integrators import muscl_hancock_step
from astronomix._finite_volume._reference_pallas import (
    _maximum_hll5r_speed_pallas,
    _parabolic_cleaning_pallas,
    _periodic_kernels_supported,
    _reference_kernels_supported,
    _reference_sweep_pallas,
)
from astronomix._finite_volume._stage import _sweep_stage
from astronomix._finite_volume._timestep_estimation._timestep_estimator import (
    _cfl_time_step,
    _maximum_hll5r_speed,
    _maximum_signal_speed_vl2,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping._utils import _unpad
from astronomix.time_stepping.time_integration import _prepare_padded_state, time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

GAMMA = 5.0 / 3.0
ROUNDING_DOUBLE = 1e-12
ROUNDING_SINGLE = 2e-5
INTERPRETED = BackendConfig(backend=PALLAS, pallas_interpret=True)
COMPILED = BackendConfig(backend=PALLAS)
ARRAY_FORM = BackendConfig(backend=NATIVE_JAX)
PARAMS = SimulationParams(C_cfl=0.8, t_end=0.05, gamma=GAMMA, isothermal_sound_speed=1.0)
LAYOUTS = pytest.mark.parametrize(
    "shape, equation_of_state, periodic_layout",
    [
        ((16, 8), ISOTHERMAL, True),
        ((8, 16), IDEAL_GAS, False),
        ((8, 8, 8), ISOTHERMAL, False),
        ((8, 8, 8), IDEAL_GAS, True),
    ],
)


def _state(shape, equation_of_state):
    """A state oblique to the grid with supersonic velocities, a density contrast of nine and a field in all rows."""
    centres = np.meshgrid(*[(np.arange(cells) + 0.5) / cells for cells in shape], indexing="ij")
    phase = 2.0 * np.pi * sum(centres)
    density = 1.0 + 0.8 * np.sin(phase)
    velocity = [2.0 * np.cos(phase), -1.5 * np.sin(2.0 * phase), 0.5 * np.cos(phase + 1.0)]
    pressure = [0.6 + 0.3 * np.cos(2.0 * phase)] * (equation_of_state == IDEAL_GAS)
    field = [0.7 + 0.2 * np.sin(phase), 0.5 * np.cos(phase), -0.4 * np.sin(phase + 2.0)]
    return jnp.asarray(np.stack([density, *velocity, *pressure, *field]))


def _configuration(shape, equation_of_state, backend_config, periodic_layout, **settings):
    """The preset in a periodic box of cubic cells.

    Finalised, the box runs on the periodic-roll layout. Without ``periodic_layout`` the layout of other boxes
    (two ghost cells) is set afterwards; its ghost cells are refreshed periodically.
    """
    sides = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    box_extents = [cells / max(shape) for cells in shape] + [1.0]
    config = SimulationConfig(
        **settings,
        solver_mode=FINITE_VOLUME,
        equation_of_state=equation_of_state,
        solver_config=BEATTIE_REFERENCE,
        mhd=True,
        dimensionality=len(shape),
        box_size=StaticFloatVector(*box_extents[:3]),
        boundary_settings=BoundarySettings(sides, sides, sides),
        backend_config=backend_config,
        progress_bar=False,
    )
    num_variables = 7 + (equation_of_state == IDEAL_GAS)
    config = finalize_config(config, (num_variables, *shape))
    assert config.boundary_handling == PERIODIC_ROLL and config.num_ghost_cells == 0
    return config if periodic_layout else config._replace(boundary_handling=GHOST_CELLS, num_ghost_cells=2)


def _setup(shape, equation_of_state, periodic_layout, kernel_backend=INTERPRETED):
    """Configurations with the kernels and with the array form, the padded state and the variable registry."""
    kernel_config = _configuration(shape, equation_of_state, kernel_backend, periodic_layout)
    array_config = _configuration(shape, equation_of_state, ARRAY_FORM, periodic_layout)
    registered_variables = get_registered_variables(array_config)
    state = _prepare_padded_state(_state(shape, equation_of_state), array_config, PARAMS, registered_variables)
    return kernel_config, array_config, state, registered_variables


def _assert_close(kernel, array, config, rounding=ROUNDING_DOUBLE):
    """The updated cells agree within ``rounding`` times the largest magnitude of the array form's result."""
    kernel, array = np.asarray(_unpad(kernel, config)), np.asarray(_unpad(array, config))
    assert np.all(np.isfinite(kernel))
    assert np.abs(kernel - array).max() <= rounding * np.abs(array).max()


@LAYOUTS
def test_the_sweep_kernel_is_the_array_sweep(shape, equation_of_state, periodic_layout):
    kernel_config, array_config, state, registered_variables = _setup(shape, equation_of_state, periodic_layout)
    assert _reference_kernels_supported(state, kernel_config)
    time_step = _cfl_time_step(state, array_config, PARAMS, registered_variables)
    for axis in range(1, len(shape) + 1):
        kernel = _reference_sweep_pallas(state, axis, time_step, GAMMA, kernel_config, PARAMS, registered_variables)
        array = _sweep_stage(state, axis, time_step, GAMMA, array_config, PARAMS, registered_variables)
        assert kernel.shape == state.shape
        _assert_close(kernel, array, array_config)


@pytest.mark.parametrize("shape, equation_of_state", [((16, 8), IDEAL_GAS), ((8, 8, 8), ISOTHERMAL)])
def test_cleaning_and_signal_speed_kernels_are_the_array_form(shape, equation_of_state):
    kernel_config, array_config, state, registered_variables = _setup(shape, equation_of_state, True)
    assert _periodic_kernels_supported(state, kernel_config)
    time_step = _cfl_time_step(state, array_config, PARAMS, registered_variables)
    cleaning_speed = _maximum_signal_speed_vl2(state, array_config, PARAMS, registered_variables)
    kernel = _parabolic_cleaning_pallas(state, cleaning_speed, time_step, kernel_config, registered_variables)
    array = _parabolic_cleaning(state, cleaning_speed, time_step, array_config, registered_variables)
    _assert_close(kernel, array, array_config)
    assert float(jnp.abs(array - state).max()) > 1e-4
    kernel_speed = _maximum_hll5r_speed_pallas(state, kernel_config, PARAMS, registered_variables)
    array_speed = _maximum_hll5r_speed(state, array_config, PARAMS, registered_variables)
    assert float(kernel_speed) == pytest.approx(float(array_speed), rel=ROUNDING_DOUBLE)
    kernel_time_step = _cfl_time_step(state, kernel_config, PARAMS, registered_variables)
    assert float(kernel_time_step) == pytest.approx(float(time_step), rel=ROUNDING_DOUBLE)


@pytest.mark.parametrize("periodic_layout", [True, False])
def test_a_run_with_the_kernels_is_the_run_of_the_array_form(periodic_layout):
    shape = (8, 8, 8)
    final_states = []
    for backend_config in (INTERPRETED, ARRAY_FORM):
        config = _configuration(
            shape,
            ISOTHERMAL,
            backend_config,
            periodic_layout,
            return_snapshots=True,
            num_snapshots=1,
            snapshot_settings=SnapshotSettings(return_final_state=True),
        )
        registered_variables = get_registered_variables(config)
        result = time_integration(_state(shape, ISOTHERMAL), config, PARAMS, registered_variables)
        final_states.append(np.asarray(result.final_state))
    kernel, array = final_states
    assert np.all(np.isfinite(kernel))
    assert np.abs(array - np.asarray(_state(shape, ISOTHERMAL))).max() > 0.1
    assert np.abs(kernel - array).max() <= 10.0 * ROUNDING_DOUBLE * np.abs(array).max()


def test_the_derivative_with_the_kernels_is_the_derivative_of_the_array_form():
    kernel_config, array_config, state, registered_variables = _setup((8, 8, 8), ISOTHERMAL, True)
    time_step = _cfl_time_step(state, array_config, PARAMS, registered_variables)
    generator = np.random.default_rng(3)
    tangent = jnp.asarray(generator.standard_normal(state.shape))
    cotangent = jnp.asarray(generator.standard_normal(state.shape))

    def step(config):
        arguments = (time_step, GAMMA, config, PARAMS, None, registered_variables, 0)
        return lambda values: muscl_hancock_step(values, *arguments)

    kernel_step, kernel_tangent = jax.jvp(step(kernel_config), (state,), (tangent,))
    array_step, array_tangent = jax.jvp(step(array_config), (state,), (tangent,))
    _assert_close(kernel_step, array_step, array_config)
    _assert_close(kernel_tangent, array_tangent, array_config)
    (kernel_cotangent,) = jax.vjp(step(kernel_config), state)[1](cotangent)
    assert np.all(np.isfinite(np.asarray(kernel_cotangent)))
    forward, backward = float(jnp.vdot(cotangent, kernel_tangent)), float(jnp.vdot(kernel_cotangent, tangent))
    assert forward == pytest.approx(backward, rel=1e-9)


def test_the_predicates_refuse_what_the_kernels_do_not_cover(monkeypatch):
    shape = (8, 8, 8)
    periodic, _, state, _ = _setup(shape, ISOTHERMAL, True)
    with_ghost_cells, _, padded_state, _ = _setup(shape, ISOTHERMAL, False)
    assert _reference_kernels_supported(state, periodic) and _periodic_kernels_supported(state, periodic)
    assert with_ghost_cells.boundary_handling == GHOST_CELLS and with_ghost_cells.num_ghost_cells == 2
    assert _reference_kernels_supported(padded_state, with_ghost_cells)
    assert not _periodic_kernels_supported(padded_state, with_ghost_cells)

    solver_config = periodic.solver_config
    refused = dict(
        array_backend=periodic._replace(backend_config=ARRAY_FORM),
        double_precision_compiled=periodic._replace(backend_config=COMPILED),
        block_shape=periodic._replace(backend_config=INTERPRETED._replace(pallas_block_shape=(3, 3, 3))),
        other_divergence_control=periodic._replace(
            solver_config=solver_config._replace(divergence_control=CONSTRAINED_TRANSPORT)
        ),
        first_order=periodic._replace(solver_config=solver_config._replace(reconstruction=DONOR_CELL)),
        polytropic_gas=periodic._replace(equation_of_state=POLYTROPIC),
    )
    for name, config in refused.items():
        assert not _reference_kernels_supported(state, config), name
        assert not _periodic_kernels_supported(state, config), name
    assert not _reference_kernels_supported(state[:, :, 0, 0], periodic._replace(dimensionality=1))
    assert not _reference_kernels_supported(padded_state, with_ghost_cells._replace(num_ghost_cells=1))

    monkeypatch.setattr(reference_pallas, "_current_pallas_mesh", lambda: SimpleNamespace(size=2))
    assert not _reference_kernels_supported(padded_state, with_ghost_cells)


def test_what_the_predicate_refuses_runs_in_array_form():
    shape = (8, 8, 8)
    refused = INTERPRETED._replace(pallas_block_shape=(3, 3, 3))
    kernel_config, array_config, state, registered_variables = _setup(shape, ISOTHERMAL, True, kernel_backend=refused)
    time_step = _cfl_time_step(state, kernel_config, PARAMS, registered_variables)
    arguments = (PARAMS, None, registered_variables, 0)
    fallback = muscl_hancock_step(state, time_step, GAMMA, kernel_config, *arguments)
    array = muscl_hancock_step(state, time_step, GAMMA, array_config, *arguments)
    assert np.array_equal(np.asarray(fallback), np.asarray(array))


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="the compiled kernels need a GPU")
@pytest.mark.parametrize("equation_of_state, periodic_layout", [(ISOTHERMAL, True), (IDEAL_GAS, False)])
def test_the_compiled_kernels_are_the_array_form_in_single_precision(equation_of_state, periodic_layout):
    shape = (32, 32, 32)
    with jax.enable_x64(False):
        kernel_config, array_config, state, registered_variables = _setup(
            shape, equation_of_state, periodic_layout, kernel_backend=COMPILED
        )
        assert state.dtype == jnp.float32
        assert _reference_kernels_supported(state, kernel_config)
        assert _periodic_kernels_supported(state, kernel_config) == periodic_layout
        states = {}
        for name, config in (("kernel", kernel_config), ("array", array_config)):
            time_step = _cfl_time_step(state, config, PARAMS, registered_variables)
            states[name] = muscl_hancock_step(state, time_step, GAMMA, config, PARAMS, None, registered_variables, 0)
        _assert_close(states["kernel"], states["array"], array_config, rounding=ROUNDING_SINGLE)
