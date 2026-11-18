"""
The polytropic gas, P = K rho^Gamma, in the production finite-difference scheme (``FDSolverConfig.scheme =
WENO_FLUX_SPLITTING``, preset ``FD_PAPER``): the cell flux, the characteristic projection and the time step take
the local sound speed a^2 = Gamma K rho^(Gamma - 1) (in the cell, and at the averaged interface density).

Checks that Gamma = 1 with K = c_s^2 is the isothermal gas to rounding (hydrodynamics in 1D and 2D, MHD in 3D, with
and without the positivity-preserving splitting), polytropic shock tubes with Gamma = 0.7 and 5/3 against the exact
solution of the barotropic Riemann problem, the time step of the local sound speed, and the rules of
``finalize_config``.

    JAX_PLATFORMS=cpu python -m pytest pytests/hydrodynamics/test_weno_polytropic.py
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
    OPEN_BOUNDARY,
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
from astronomix._finite_difference._timestep import (
    _cfl_time_step_fd,
    _cfl_time_step_fd_hydro,
)
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

# the exact solution of the barotropic Riemann problem and the rough initial fields of the scheme with a Riemann flux
from test_riemann_flux_polytropic import _exact_barotropic, _rough_fields

pytestmark = pytest.mark.x64

SOUND_SPEED = 0.7


def _config(dimensionality, num_cells, equation_of_state=POLYTROPIC, boundary=PERIODIC_BOUNDARY, mhd=False,
            positivity_preserving=False):
    side = BoundarySettings1D(boundary, boundary)
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=dimensionality,
        mhd=mhd,
        equation_of_state=equation_of_state,
        num_cells=num_cells,
        box_size=1.0,
        boundary_settings=side if dimensionality == 1 else BoundarySettings(*([side] * dimensionality)),
        backend_config=BackendConfig(backend=NATIVE_JAX),
        solver_config=FD_PAPER._replace(weno_positivity_preserving=positivity_preserving, ct_upwind_dissipation=mhd),
        progress_bar=False,
    )


def _run(config, fields, params):
    registered_variables = get_registered_variables(config)
    state = construct_primitive_state(
        config=config, registered_variables=registered_variables, **{k: jnp.asarray(v) for k, v in fields.items()}
    )
    config = finalize_config(config, state.shape)
    return np.asarray(time_integration(state, config, params, registered_variables)), registered_variables


def _centres(num_cells):
    return (np.arange(num_cells) + 0.5) / num_cells


# ====================== Gamma = 1 is the isothermal gas ======================


@pytest.mark.parametrize(
    "dimensionality, num_cells, mhd, positivity_preserving, tolerance",
    [
        (1, 64, False, False, 1e-13),
        (1, 64, False, True, 1e-13),
        (2, 32, False, False, 1e-13),
        (2, 32, False, True, 1e-13),
        (3, 12, True, True, 1e-8),
    ],
    ids=["1d", "1d_pp", "2d", "2d_pp", "mhd_3d_pp"],
)
def test_exponent_one_is_the_isothermal_gas(dimensionality, num_cells, mhd, positivity_preserving, tolerance):
    """
    Gamma = 1, K = c_s^2 against the isothermal gas, relative to the largest value: measured 0 in 1D and 2D on the
    CPU and the GPU; MHD 3.4e-15 on the CPU and 3.6e-10 on the GPU, whose pow(rho, 1.0) differs from rho by an ulp:
    the square roots of differences in the MHD mode weightings (near-degenerate fast and slow speeds of the rough
    field) raise that to 4e-10 within five steps (1.8e-16 after one), hence the bound 1e-8 there.
    """
    params = SimulationParams(C_cfl=0.8, t_end=0.05, isothermal_sound_speed=SOUND_SPEED,
                              polytropic_constant=SOUND_SPEED**2, polytropic_exponent=1.0, minimum_density=1e-10)
    fields = _rough_fields(dimensionality, num_cells, mhd)
    isothermal, _ = _run(_config(dimensionality, num_cells, ISOTHERMAL, mhd=mhd,
                                 positivity_preserving=positivity_preserving), fields, params)
    polytropic, _ = _run(_config(dimensionality, num_cells, POLYTROPIC, mhd=mhd,
                                 positivity_preserving=positivity_preserving), fields, params)
    assert np.all(np.isfinite(polytropic))
    assert np.max(np.abs(isothermal - polytropic)) < tolerance * np.max(np.abs(isothermal))


# ============== polytropic shock tubes against the exact solution ==============

TUBES = {"rarefaction_and_shock": (1.0, 0.0, 0.125, 0.0), "collision": (1.0, 1.0, 1.0, -1.0)}
# bounds of the L1 errors of (rho, u) at 200 cells, t = 0.1, C_cfl 0.8, about 5 % above the measured values
# (without / with the positivity-preserving splitting; at 400 cells about half of these, first order):
# rarefaction and shock 2.00e-3, 5.08e-3 / 2.22e-3, 5.43e-3 (Gamma 0.7); 1.97e-3, 5.74e-3 / 2.00e-3, 5.79e-3 (5/3);
# collision 9.61e-3, 4.60e-3 / 1.30e-2, 5.79e-3 (Gamma 0.7); 4.10e-3, 4.48e-3 / 4.48e-3, 4.89e-3 (5/3)
TUBE_ERRORS = {
    ("rarefaction_and_shock", 0.7, False): (2.1e-3, 5.4e-3),
    ("rarefaction_and_shock", 0.7, True): (2.4e-3, 5.8e-3),
    ("rarefaction_and_shock", 5.0 / 3.0, False): (2.1e-3, 6.1e-3),
    ("rarefaction_and_shock", 5.0 / 3.0, True): (2.1e-3, 6.1e-3),
    ("collision", 0.7, False): (1.01e-2, 4.9e-3),
    ("collision", 0.7, True): (1.37e-2, 6.1e-3),
    ("collision", 5.0 / 3.0, False): (4.4e-3, 4.8e-3),
    ("collision", 5.0 / 3.0, True): (4.8e-3, 5.2e-3),
}


@pytest.mark.parametrize("positivity_preserving", [False, True], ids=["plain", "pp"])
@pytest.mark.parametrize("exponent", [0.7, 5.0 / 3.0], ids=["gamma_0.7", "gamma_5/3"])
@pytest.mark.parametrize("tube", list(TUBES))
def test_polytropic_shock_tube_against_the_exact_solution(tube, exponent, positivity_preserving):
    rho_L, u_L, rho_R, u_R = TUBES[tube]
    x = _centres(200)
    left = x < 0.5
    fields = dict(density=np.where(left, rho_L, rho_R), velocity_x=np.where(left, u_L, u_R))
    params = SimulationParams(C_cfl=0.8, t_end=0.1, polytropic_constant=1.0, polytropic_exponent=exponent)
    config = _config(1, 200, boundary=OPEN_BOUNDARY, positivity_preserving=positivity_preserving)
    final, registered_variables = _run(config, fields, params)
    rho, u = _exact_barotropic(rho_L, u_L, rho_R, u_R, 1.0, exponent, x, 0.1)
    bound = TUBE_ERRORS[(tube, exponent, positivity_preserving)]
    assert np.min(final[registered_variables.density_index]) > 0.124
    assert np.mean(np.abs(final[registered_variables.density_index] - rho)) < bound[0]
    assert np.mean(np.abs(final[registered_variables.velocity_index] - u)) < bound[1]


# ============================ time step ===============================


@pytest.mark.parametrize("mhd", [False, True], ids=["hydrodynamics", "mhd"])
def test_time_step_takes_the_local_sound_speed(mhd):
    """At rest (MHD: without a field): dt = C dx / (d max a), a^2 = Gamma K rho^(Gamma - 1), largest at low density."""
    dimensionality = 3 if mhd else 2
    num_cells = 16
    grid = np.meshgrid(*([_centres(num_cells)] * dimensionality), indexing="ij")
    density = 1.0 + 0.9 * np.sin(2 * np.pi * grid[0]) * np.sin(2 * np.pi * grid[1])
    config = _config(dimensionality, num_cells, mhd=mhd)
    registered_variables = get_registered_variables(config)
    zeros = jnp.zeros((num_cells,) * dimensionality)
    fields = {f"velocity_{axis}": zeros for axis in "xyz"[: 3 if mhd else dimensionality]}
    state = construct_primitive_state(config=config, registered_variables=registered_variables,
                                      density=jnp.asarray(density), **fields)
    config = finalize_config(config, state.shape)
    params = SimulationParams(C_cfl=0.8, polytropic_constant=2.0, polytropic_exponent=0.7)
    time_step = _cfl_time_step_fd if mhd else _cfl_time_step_fd_hydro
    dt = time_step(state, config.grid_spacing, params.dt_max, params.gamma, config, params, registered_variables,
                   params.C_cfl)
    speed = np.max(np.sqrt(0.7 * 2.0 * np.maximum(density, params.minimum_density) ** (0.7 - 1.0)))
    assert abs(float(dt) - 0.8 * (1.0 / num_cells) / (dimensionality * speed)) < 1e-14


# ========================= validity rules ============================


@pytest.mark.parametrize("mhd", [False, True], ids=["hydrodynamics", "mhd"])
def test_polytropic_layout_and_rules(mhd):
    """No pressure slot, the layout of the isothermal gas; the production scheme accepts the polytropic gas."""
    dimensionality = 3 if mhd else 2
    config = _config(dimensionality, 16, mhd=mhd)
    registered_variables = get_registered_variables(config)
    isothermal = get_registered_variables(config._replace(equation_of_state=ISOTHERMAL))
    assert registered_variables == isothermal
    assert registered_variables.pressure_index == -1
    shape = (registered_variables.num_vars,) + (16,) * dimensionality
    assert finalize_config(config, shape).equation_of_state == POLYTROPIC
