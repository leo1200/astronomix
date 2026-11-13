"""
The fused stage kernel of the finite-volume framework (``_stage_pallas.py``) for a barotropic gas (isothermal or
polytropic, ``BackendConfig.backend = PALLAS``) against the native stage: ``VL2`` (hydrodynamics in 2D and 3D, GLM
MHD in 2D) and ``RK2_SSP`` (1D), with HLL, piecewise-linear reconstruction and the first-order flux correction.

The array form is the definition, so every check runs three steps of both forms from a rough state and compares
them to rounding (float64), through the interpreter on any device (``BackendConfig.pallas_interpret``) and compiled
where a GPU is present.

    JAX_PLATFORMS=cpu python -m pytest pytests/hydrodynamics/test_barotropic_stage_kernel.py
"""

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

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    GHOST_CELLS,
    GLM,
    HLL,
    ISOTHERMAL,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    PIECEWISE_LINEAR,
    POLYTROPIC,
    RK2_SSP,
    VAN_LEER,
    VL2,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_volume._stage_pallas import _fv_stage_pallas_supported
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

NUM_CELLS = 8
HAS_GPU = jax.default_backend() == "gpu"
#: (time integrator, dimensionality, mhd, equation of state, polytropic exponent): every gas in 1D and 2D, one in
#: 3D and in GLM MHD for both laws (the cases differ in the axes and rows the kernel loops over, not in the law).
CASES = pytest.mark.parametrize(
    "time_integrator, dimensionality, mhd, equation_of_state, exponent",
    [
        (VL2, 2, False, ISOTHERMAL, 1.0),
        (VL2, 2, False, POLYTROPIC, 0.7),
        (VL2, 2, False, POLYTROPIC, 5.0 / 3.0),
        (VL2, 3, False, POLYTROPIC, 0.7),
        (RK2_SSP, 1, False, ISOTHERMAL, 1.0),
        (RK2_SSP, 1, False, POLYTROPIC, 0.7),
        (RK2_SSP, 1, False, POLYTROPIC, 5.0 / 3.0),
        (RK2_SSP, 2, False, POLYTROPIC, 5.0 / 3.0),
        (VL2, 2, True, ISOTHERMAL, 1.0),
        (VL2, 2, True, POLYTROPIC, 0.7),
    ],
    ids=[
        "vl2_2d_isothermal", "vl2_2d_polytropic_0.7", "vl2_2d_polytropic_5/3", "vl2_3d_polytropic_0.7",
        "rk2_ssp_1d_isothermal", "rk2_ssp_1d_polytropic_0.7", "rk2_ssp_1d_polytropic_5/3",
        "rk2_ssp_2d_polytropic_5/3", "vl2_glm_2d_isothermal", "vl2_glm_2d_polytropic_0.7",
    ],
)
BOTH_PATHS = pytest.mark.parametrize(
    "interpret",
    [True, pytest.param(False, marks=pytest.mark.skipif(not HAS_GPU, reason="the compiled kernels need a GPU"))],
    ids=["interpreted", "compiled"],
)

#: Bound on the largest difference between the kernels and the array form after the steps, relative to the largest
#: value (measured on the CPU, interpreted: at most 7e-14 over the cases of this file).
STEP_TOLERANCE = 1e-11


def _fv_config(time_integrator, dimensionality, mhd, equation_of_state, backend, interpret=False):
    side = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    solver_config = FVSolverConfig(
        time_integrator=time_integrator, reconstruction=PIECEWISE_LINEAR, limiter=VAN_LEER, riemann_solver=HLL,
        divergence_control=GLM, first_order_flux_correction=True,
    )
    return SimulationConfig(
        solver_mode=FINITE_VOLUME, dimensionality=dimensionality, mhd=mhd, equation_of_state=equation_of_state,
        num_cells=NUM_CELLS, box_size=1.0,
        boundary_settings=side if dimensionality == 1 else BoundarySettings(*([side] * dimensionality)),
        solver_config=solver_config, backend_config=BackendConfig(backend=backend, pallas_interpret=interpret),
        fixed_timestep=True, num_timesteps=3, progress_bar=False,
    )


def _rough_state(config, registered_variables):
    """Density 0.1 to 1, velocities of rms 1; MHD a field (0.3 + 0.2 sin 2 pi y, 0.2 cos 2 pi z, 0.2 sin 2 pi x)."""
    dimensionality = config.dimensionality
    shape = (NUM_CELLS,) * dimensionality
    rng = np.random.default_rng(2)
    fields = dict(density=10.0 ** rng.uniform(-1.0, 0.0, shape))
    for axis in "xyz"[: 3 if config.mhd else dimensionality]:
        fields[f"velocity_{axis}"] = rng.normal(size=shape)
    if config.mhd:
        x = (np.arange(NUM_CELLS) + 0.5) / NUM_CELLS
        grid = list(np.meshgrid(*([x] * dimensionality), indexing="ij")) + [np.zeros(shape)] * (3 - dimensionality)
        X, Y, Z = grid
        field = (0.3 + 0.2 * np.sin(2 * np.pi * Y), 0.2 * np.cos(2 * np.pi * Z), 0.2 * np.sin(2 * np.pi * X))
        for axis, component in zip("xyz", field):
            # each component varies across one other axis only: face values equal cell values, div B = 0
            fields[f"magnetic_field_{axis}"] = component
    return construct_primitive_state(
        config=config, registered_variables=registered_variables, **{k: jnp.asarray(v) for k, v in fields.items()}
    )


def _steps(config, exponent):
    """Three fixed steps of the configuration from the rough state."""
    registered_variables = get_registered_variables(config)
    state = _rough_state(config, registered_variables)
    config = finalize_config(config, state.shape)
    params = SimulationParams(t_end=3 * 0.2 / (4.0 * NUM_CELLS), isothermal_sound_speed=1.0,
                              polytropic_constant=1.0, polytropic_exponent=exponent, minimum_density=1e-10)
    return config, state, np.asarray(time_integration(state, config, params, registered_variables))


def _compare(label, kernel_run, array_run):
    _, state, array_form = array_run
    _, _, kernels = kernel_run
    scale = np.max(np.abs(array_form))
    difference = np.max(np.abs(kernels - array_form)) / scale
    change = np.max(np.abs(array_form - np.asarray(state))) / scale
    print(f"{label}: largest difference {difference:.2e} of the largest value (the steps change {change:.2e})")
    assert np.all(np.isfinite(kernels))
    assert change > 1e-3
    assert difference < STEP_TOLERANCE


@BOTH_PATHS
@CASES
def test_fv_stage_kernel_equals_the_array_form(time_integrator, dimensionality, mhd, equation_of_state, exponent,
                                               interpret):
    kernel_run = _steps(_fv_config(time_integrator, dimensionality, mhd, equation_of_state, PALLAS, interpret),
                        exponent)
    config, state, _ = kernel_run
    # the stage reads the state with its ghost cells (one-dimensional periodic runs keep them)
    padded_shape = tuple(extent + 2 * config.num_ghost_cells for extent in state.shape[1:])
    if config.boundary_handling != GHOST_CELLS:
        padded_shape = state.shape[1:]
    assert _fv_stage_pallas_supported(jnp.zeros((state.shape[0],) + padded_shape), config)
    array_run = _steps(_fv_config(time_integrator, dimensionality, mhd, equation_of_state, NATIVE_JAX), exponent)
    _compare(f"FV stage {time_integrator} {dimensionality}D mhd {mhd} eos {equation_of_state} Gamma {exponent:.3f}",
             kernel_run, array_run)
