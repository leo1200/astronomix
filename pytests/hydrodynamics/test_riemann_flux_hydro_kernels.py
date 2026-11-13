"""
The kernels of the hydrodynamic form of the finite-difference scheme with a Riemann flux
(``FDSolverConfig.scheme = RIEMANN_FLUX``, ``mhd=False``, ``BackendConfig.backend = PALLAS``): the two flux
kernels per axis of ``_riemann_flux_pallas.py`` and the stage combination, for the ideal, the isothermal and the
polytropic gas.

The array form is the definition of the scheme, so every check compares the kernels with it, in float32 (the
precision in which the kernels are compiled): two steps from a rough state with periodic boundaries and a few
steps of a smooth state with open boundaries, through the interpreter on any device
(``BackendConfig.pallas_interpret``) and compiled where a GPU is present. The two forms are different programs
of the same arithmetic (the kernels project on the characteristic variables in primitive form), so the
comparison is to rounding, not bitwise. What the predicate refuses runs in the array form.

    JAX_PLATFORMS=cpu python -m pytest pytests/hydrodynamics/test_riemann_flux_hydro_kernels.py
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
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    ISOTHERMAL,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    PALLAS,
    PERIODIC_BOUNDARY,
    POLYTROPIC,
)

# astronomix containers
from astronomix.option_classes.presets import FD_RIEMANN_IDEAL_GAS, FD_RIEMANN_ISOTHERMAL
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_difference._interface_fluxes._riemann_flux_pallas import _riemann_flux_pallas_supported
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

NUM_CELLS = 16
HAS_GPU = jax.default_backend() == "gpu"
THREE_GASES = pytest.mark.parametrize(
    "equation_of_state", [IDEAL_GAS, ISOTHERMAL, POLYTROPIC], ids=["ideal_gas", "isothermal", "polytropic"]
)
BOTH_PATHS = pytest.mark.parametrize(
    "interpret",
    [True, pytest.param(False, marks=pytest.mark.skipif(not HAS_GPU, reason="the compiled kernels need a GPU"))],
    ids=["interpreted", "compiled"],
)

#: Bound on the largest difference between the kernels and the array form after two steps from the rough state
#: (and after the steps with open boundaries), relative to the largest value of the state. Measured with the
#: interpreted kernels on a CPU (float32, 16 cells per axis): 2.5e-07 (ideal gas), 2.2e-07 (isothermal,
#: polytropic), at most 1.8e-07 in 1D and 2D; with open boundaries (smooth state, five steps) 2.0e-05, which in
#: float64 is 1.0e-15: single entries at which float32 rounding flips a selection. Compiled on a GH200: 2.9e-07 in
#: 3D, 2.7e-07 in 1D and 2D, 1.4e-05 with open boundaries. The bound is five times the largest.
STEP_TOLERANCE = 1e-4
#: The ghost layers of the hydrodynamic form (``finalize_config``): a block of 8 cells tiles 16 + 2 * 4.
GHOST_CELLS = 4


def _config(equation_of_state, backend, interpret=False, boundary=PERIODIC_BOUNDARY, dimensionality=3, mhd=False):
    """A finite-difference configuration of the hydrodynamic form with the preset of the gas."""
    side = BoundarySettings1D(boundary, boundary)
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=dimensionality,
        mhd=mhd,
        equation_of_state=equation_of_state,
        num_cells=NUM_CELLS,
        box_size=1.0,
        boundary_settings=side if dimensionality == 1 else BoundarySettings(*([side] * dimensionality)),
        backend_config=BackendConfig(backend=backend, pallas_interpret=interpret),
        solver_config=FD_RIEMANN_IDEAL_GAS if equation_of_state == IDEAL_GAS else FD_RIEMANN_ISOTHERMAL,
        progress_bar=False,
    )


def _params(t_end):
    """The parameters of a run to ``t_end`` at a CFL number of 0.5 (polytropic: Gamma 0.7, K 1)."""
    return SimulationParams(
        C_cfl=0.5, gamma=5.0 / 3.0, isothermal_sound_speed=1.0, polytropic_constant=1.0, polytropic_exponent=0.7,
        minimum_density=1e-10, minimum_pressure=1e-12, t_end=t_end,
    )


def _state(config, registered_variables, rough):
    """A rough state (density 0.1 to 1, velocities of rms 1, pressure 0.1 to 10 times rho) or a smooth one."""
    x = (np.arange(NUM_CELLS) + 0.5) / NUM_CELLS
    X, Y, Z = list(np.meshgrid(*([x] * config.dimensionality), indexing="ij")) + [0.0] * (3 - config.dimensionality)
    shape = X.shape
    if rough:
        rng = np.random.default_rng(3)
        density = 10.0 ** rng.uniform(-1.0, 0.0, shape)
        velocity = [rng.normal(size=shape) for _ in range(config.dimensionality)]
        pressure = density * 10.0 ** rng.uniform(-1.0, 1.0, shape)
    else:
        density = 1.0 + 0.3 * np.sin(2 * np.pi * (X + 2 * Y)) * np.cos(2 * np.pi * Z)
        velocity = [0.3 * np.sin(2 * np.pi * Y), 0.2 * np.cos(2 * np.pi * Z), 0.1 * np.sin(2 * np.pi * X)]
        velocity = [component + 0.0 * X for component in velocity[: config.dimensionality]]
        pressure = 1.0 + 0.2 * np.cos(2 * np.pi * (X - Z))
    fields = dict(density=jnp.asarray(density, jnp.float32))
    for axis, component in zip("xyz", velocity):
        fields[f"velocity_{axis}"] = jnp.asarray(component, jnp.float32)
    if config.equation_of_state == IDEAL_GAS:
        fields["gas_pressure"] = jnp.asarray(pressure, jnp.float32)
    return construct_primitive_state(config=config, registered_variables=registered_variables, **fields)


def _relative_difference(a, b):
    """max |a - b| relative to the largest value of b, and the number of differing entries."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b))), int(np.sum(a != b))


@functools.lru_cache(maxsize=None)
def _run(equation_of_state, backend, interpret, boundary, dimensionality=3):
    """(initial state, final state): two steps from the rough state (periodic) or the smooth state (open)."""
    config = _config(equation_of_state, backend, interpret, boundary, dimensionality)
    registered_variables = get_registered_variables(config)
    state = _state(config, registered_variables, rough=boundary == PERIODIC_BOUNDARY)
    config = finalize_config(config, state.shape)
    padded = jnp.zeros(
        (registered_variables.num_vars,) + (NUM_CELLS + 2 * config.num_ghost_cells,) * dimensionality, jnp.float32
    )
    assert bool(_riemann_flux_pallas_supported(padded, config)) == (backend == PALLAS)
    t_end = 0.004 if boundary == PERIODIC_BOUNDARY else 0.03
    final = time_integration(state, config, _params(t_end), registered_variables)
    return np.asarray(state), np.asarray(final)


@BOTH_PATHS
@THREE_GASES
def test_step_of_the_kernels_equals_the_array_form(equation_of_state, interpret):
    """Two steps from the rough state, periodic: the kernels against the array form, to rounding."""
    state, array_form = _run(equation_of_state, NATIVE_JAX, False, PERIODIC_BOUNDARY)
    _, kernels = _run(equation_of_state, PALLAS, interpret, PERIODIC_BOUNDARY)
    difference, differing = _relative_difference(kernels, array_form)
    change, _ = _relative_difference(array_form, state)
    print(f"kernel steps eos {equation_of_state} interpret {interpret}: {differing} of {kernels.size} entries "
          f"differ, largest difference {difference:.2e} of the largest value (the steps change {change:.2e})")
    assert kernels.dtype == np.float32 and np.all(np.isfinite(kernels))
    assert change > 1e-2
    assert difference < STEP_TOLERANCE


@BOTH_PATHS
@THREE_GASES
@pytest.mark.parametrize("dimensionality", [1, 2], ids=["1d", "2d"])
def test_lower_dimensional_steps_of_the_kernels_equal_the_array_form(dimensionality, equation_of_state, interpret):
    """1D and 2D, the same two steps: the kernels on a layer of one cell across against the array form."""
    state, array_form = _run(equation_of_state, NATIVE_JAX, False, PERIODIC_BOUNDARY, dimensionality)
    _, kernels = _run(equation_of_state, PALLAS, interpret, PERIODIC_BOUNDARY, dimensionality)
    difference, differing = _relative_difference(kernels, array_form)
    change, _ = _relative_difference(array_form, state)
    print(f"{dimensionality}D kernel steps eos {equation_of_state} interpret {interpret}: {differing} of "
          f"{kernels.size} entries differ, largest difference {difference:.2e} of the largest value (the steps "
          f"change {change:.2e})")
    assert kernels.dtype == np.float32 and np.all(np.isfinite(kernels))
    assert change > 1e-2
    assert difference < STEP_TOLERANCE


@BOTH_PATHS
def test_open_boundaries_with_the_kernels(interpret):
    """Open boundaries, ideal gas, smooth state: the kernels on the padded grid against the array form."""
    _, array_form = _run(IDEAL_GAS, NATIVE_JAX, False, OPEN_BOUNDARY)
    _, kernels = _run(IDEAL_GAS, PALLAS, interpret, OPEN_BOUNDARY)
    difference, differing = _relative_difference(kernels, array_form)
    print(f"kernels with open boundaries, interpret {interpret}: {differing} of {kernels.size} entries differ, "
          f"largest difference {difference:.2e} of the largest value")
    assert np.all(np.isfinite(kernels))
    assert difference < STEP_TOLERANCE


@THREE_GASES
def test_predicate(equation_of_state):
    """The kernels serve the hydrodynamic form in 1D to 3D on the Pallas backend; the native backend runs the
    array form, and so does the polytropic gas in the MHD form."""
    probe = jnp.zeros((5,) + (NUM_CELLS,) * 3, jnp.float32)
    assert _riemann_flux_pallas_supported(probe, _config(equation_of_state, PALLAS, interpret=True))
    assert not _riemann_flux_pallas_supported(probe, _config(equation_of_state, NATIVE_JAX))
    for dimensionality, served in ((1, True), (2, True)):
        config = _config(equation_of_state, PALLAS, interpret=True, dimensionality=dimensionality)
        probe_low = jnp.zeros((5,) + (NUM_CELLS,) * dimensionality, jnp.float32)
        assert bool(_riemann_flux_pallas_supported(probe_low, config)) == served
    mhd_config = _config(equation_of_state, PALLAS, interpret=True, mhd=True)
    assert bool(_riemann_flux_pallas_supported(probe, mhd_config)) == (equation_of_state != POLYTROPIC)
