"""
The Pallas WENO kernels of the production finite-difference scheme (``WENO_FLUX_SPLITTING``, preset ``FD_PAPER``
with the positivity-preserving splitting) for a barotropic gas (isothermal or polytropic, ``BackendConfig.backend =
PALLAS``) against the array form: the hydrodynamic kernel in 1D to 3D and the MHD kernel in 3D (polytropic; its
isothermal form is the kernel of the isothermal gas), whose eigen-systems take the local sound speed of the law.

The array form is the definition, so every check runs three steps of both forms from a rough state and compares
them to rounding (float64), through the interpreter on any device (``BackendConfig.pallas_interpret``) and compiled
where a GPU is present.

    JAX_PLATFORMS=cpu python -m pytest pytests/hydrodynamics/test_barotropic_weno_kernels.py
"""

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
    ISOTHERMAL,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    POLYTROPIC,
)

# astronomix containers
from astronomix.option_classes.presets import FD_PAPER
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _hydro_pallas_flux_supported,
    _mhd_iso_pallas_flux_supported,
)
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

pytestmark = pytest.mark.x64

NUM_CELLS = 8
GASES = pytest.mark.parametrize(
    "equation_of_state, exponent",
    [(ISOTHERMAL, 1.0), (POLYTROPIC, 0.7), (POLYTROPIC, 5.0 / 3.0)],
    ids=["isothermal", "polytropic_0.7", "polytropic_5/3"],
)
BOTH_PATHS = pytest.mark.parametrize(
    "interpret",
    [True, pytest.param(False, marks=pytest.mark.gpu)],
    ids=["interpreted", "compiled"],
)

#: Bound on the largest difference between the kernels and the array form after the steps, relative to the largest
#: value (measured on the CPU, interpreted: at most 7e-14 over the cases of this file).
STEP_TOLERANCE = 1e-11


def _fd_config(dimensionality, mhd, equation_of_state, backend, interpret=False):
    side = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, dimensionality=dimensionality, mhd=mhd, equation_of_state=equation_of_state,
        num_cells=NUM_CELLS, box_size=1.0,
        boundary_settings=side if dimensionality == 1 else BoundarySettings(*([side] * dimensionality)),
        solver_config=FD_PAPER._replace(weno_positivity_preserving=True, ct_upwind_dissipation=mhd),
        backend_config=BackendConfig(backend=backend, pallas_interpret=interpret),
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
            fields[f"interface_magnetic_field_{axis}"] = component
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
@GASES
@pytest.mark.parametrize("dimensionality", [1, 2, 3], ids=["1d", "2d", "3d"])
def test_weno_hydro_kernel_equals_the_array_form(dimensionality, equation_of_state, exponent, interpret):
    kernel_run = _steps(_fd_config(dimensionality, False, equation_of_state, PALLAS, interpret), exponent)
    assert _hydro_pallas_flux_supported(jnp.asarray(kernel_run[1]), kernel_run[0])
    array_run = _steps(_fd_config(dimensionality, False, equation_of_state, NATIVE_JAX), exponent)
    _compare(f"WENO hydro {dimensionality}D eos {equation_of_state} Gamma {exponent:.3f}", kernel_run, array_run)


@pytest.mark.slow
@BOTH_PATHS
@pytest.mark.parametrize("exponent", [0.7, 5.0 / 3.0], ids=["polytropic_0.7", "polytropic_5/3"])
def test_weno_mhd_kernel_equals_the_array_form(exponent, interpret):
    kernel_run = _steps(_fd_config(3, True, POLYTROPIC, PALLAS, interpret), exponent)
    assert _mhd_iso_pallas_flux_supported(jnp.asarray(kernel_run[1]), kernel_run[0])
    array_run = _steps(_fd_config(3, True, POLYTROPIC, NATIVE_JAX), exponent)
    _compare(f"WENO MHD polytropic Gamma {exponent:.3f}", kernel_run, array_run)
