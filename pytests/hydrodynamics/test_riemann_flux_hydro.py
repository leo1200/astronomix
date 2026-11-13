"""
The hydrodynamic form of the finite-difference scheme with a Riemann flux (``FDSolverConfig.scheme =
RIEMANN_FLUX`` with ``mhd=False``), in the configurations of the presets ``FD_RIEMANN_IDEAL_GAS`` (HLLC) and
``FD_RIEMANN_ISOTHERMAL`` (the zero-field form of HLLEM_T).

Checks that the hydrodynamic form is the MHD form without a field (to rounding on a problem that varies along x
only), the Sod shock tube against the exact solution, the order of a sound wave, the isothermal shock tubes
against the exact solution, that a stationary shear layer is kept by HLLEM_T and spread by HLL, that strong
rarefactions, an expansion into near-vacuum and the double blast stay positive in float64 and float32, that more
ghost cells than the rule of ``finalize_config`` give the same result with open and reflective boundaries, and
the rules of ``finalize_config``.

    JAX_PLATFORMS=cpu python -m pytest pytests/hydrodynamics/test_riemann_flux_hydro.py
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
    FINITE_DIFFERENCE,
    HLL,
    HLLC,
    HLLD_BOUNDED,
    HLLEM_T,
    IDEAL_GAS,
    ISOTHERMAL,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    RIEMANN_FLUX,
)

# astronomix containers
from astronomix.option_classes.presets import FD_RIEMANN_IDEAL_GAS, FD_RIEMANN_ISOTHERMAL
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.test_setups.reference_solutions.riemann_solver import _exact_riemann_ideal_gas
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

BOTH_GASES = pytest.mark.parametrize("equation_of_state", [ISOTHERMAL, IDEAL_GAS], ids=["isothermal", "ideal_gas"])
BOTH_PRECISIONS = pytest.mark.parametrize("double", [True, False], ids=["float64", "float32"])


def _preset(equation_of_state):
    return FD_RIEMANN_IDEAL_GAS if equation_of_state == IDEAL_GAS else FD_RIEMANN_ISOTHERMAL


def _config(dimensionality, num_cells, boundary=PERIODIC_BOUNDARY, equation_of_state=IDEAL_GAS, **fields):
    """A finite-difference configuration of the hydrodynamic form with the preset of the gas (unit length along x)."""
    side = BoundarySettings1D(boundary, boundary) if isinstance(boundary, int) else boundary
    settings = dict(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=dimensionality,
        mhd=False,
        equation_of_state=equation_of_state,
        num_cells=num_cells,
        box_size=1.0,
        boundary_settings=side if dimensionality == 1 else BoundarySettings(*([side] * dimensionality)),
        backend_config=BackendConfig(backend=NATIVE_JAX),
        solver_config=_preset(equation_of_state),
        progress_bar=False,
    )
    settings.update(fields)
    return SimulationConfig(**settings)


def _run(config, fields, params):
    """The final primitive state and the registered variables of a run from the primitive ``fields``."""
    registered_variables = get_registered_variables(config)
    state = construct_primitive_state(
        config=config, registered_variables=registered_variables, **{k: jnp.asarray(v) for k, v in fields.items()}
    )
    config = finalize_config(config, state.shape)
    return np.asarray(time_integration(state, config, params, registered_variables)), registered_variables


def _centres(num_cells):
    return (np.arange(num_cells) + 0.5) / num_cells


# ================== the MHD form without a field =====================


def _slab_run(mhd, equation_of_state):
    """20 steps of a 32 x 4 x 4 problem that varies along x (transverse velocities proportional to rho)."""
    shape = (32, 4, 4)
    config = _config(
        3,
        StaticIntVector(*shape),
        equation_of_state=equation_of_state,
        mhd=mhd,
        box_size=StaticFloatVector(1.0, 0.125, 0.125),
        fixed_timestep=True,
        num_timesteps=20,
    )
    x = _centres(shape[0])[:, None, None] * np.ones(shape)
    density = np.where(np.abs(x - 0.5) < 0.25, 1.0, 0.125)
    fields = dict(density=density, velocity_x=0.3 * np.sin(2 * np.pi * x), velocity_y=0.2 * density,
                  velocity_z=-0.2 * density)
    if equation_of_state == IDEAL_GAS:
        fields["gas_pressure"] = np.where(np.abs(x - 0.5) < 0.25, 1.0, 0.1)
    if mhd:
        for axis in "xyz":
            fields[f"magnetic_field_{axis}"] = np.zeros(shape)
            fields[f"interface_magnetic_field_{axis}"] = np.zeros(shape)
    params = SimulationParams(C_cfl=0.5, t_end=0.05, gamma=5.0 / 3.0, isothermal_sound_speed=1.0)
    final, registered_variables = _run(config, fields, params)
    rows = [registered_variables.density_index, *registered_variables.momentum_index]
    if equation_of_state == IDEAL_GAS:
        rows.append(registered_variables.energy_index)
    return final[rows]


@BOTH_GASES
def test_hydrodynamic_form_is_the_mhd_form_without_a_field(equation_of_state):
    """The fluid rows of the two forms agree to rounding (relative 1e-13) when the field is zero."""
    hydrodynamic = _slab_run(False, equation_of_state)
    magnetohydrodynamic = _slab_run(True, equation_of_state)
    scale = np.max(np.abs(magnetohydrodynamic))
    assert np.max(np.abs(hydrodynamic - magnetohydrodynamic)) < 1e-13 * scale


# ========================= ideal gas =================================


def test_sod_shock_tube_against_the_exact_solution():
    """Sod's tube at 200 cells, t = 0.2: the L1 errors of the round-11 measurement and no density overshoot."""
    x = _centres(200)
    left = x < 0.5
    fields = dict(density=np.where(left, 1.0, 0.125), velocity_x=np.zeros(200), gas_pressure=np.where(left, 1.0, 0.1))
    final, registered_variables = _run(
        _config(1, 200, OPEN_BOUNDARY), fields, SimulationParams(C_cfl=1.0, t_end=0.2, gamma=5.0 / 3.0)
    )
    exact = _exact_riemann_ideal_gas(
        rho_L=1.0, u_L=0.0, p_L=1.0, rho_R=0.125, u_R=0.0, p_R=0.1, gamma=5.0 / 3.0, x=jnp.asarray(x), t=0.2, x0=0.5
    )
    variables = registered_variables
    rows = (variables.density_index, variables.velocity_index, variables.pressure_index)
    errors = [np.mean(np.abs(final[row] - np.asarray(value))) for row, value in zip(rows, exact)]
    # measured 2.724e-03, 3.498e-03, 1.704e-03
    assert errors[0] < 2.8e-3 and errors[1] < 3.6e-3 and errors[2] < 1.75e-3
    density = final[registered_variables.density_index]
    assert density.max() < 1.0 + 1e-10 and density.min() > 0.125 - 1e-10


def _sound_wave_error(num_cells):
    """L1 density error of a linear sound wave (amplitude 1e-6, c = 1) after one period."""
    x = _centres(num_cells)
    amplitude, gamma = 1e-6, 5.0 / 3.0
    wave = amplitude * np.sin(2 * np.pi * x)
    fields = dict(density=1.0 + wave, velocity_x=wave, gas_pressure=1.0 / gamma + wave)
    final, registered_variables = _run(
        _config(1, num_cells), fields, SimulationParams(C_cfl=1.0, t_end=1.0, gamma=gamma)
    )
    return np.mean(np.abs(final[registered_variables.density_index] - 1.0 - wave))


def test_sound_wave_error_and_order():
    errors = [_sound_wave_error(cells) for cells in (16, 32)]
    # measured 6.90e-10 and 2.94e-11 (order 4.55)
    assert np.log2(errors[0] / errors[1]) > 4.3
    assert errors[1] < 3.1e-11


# ======================== isothermal gas =============================


def _isothermal_exact(rho_L, u_L, rho_R, u_R, x, t):
    """The exact isothermal (c_s = 1) Riemann solution: one root for the middle density, then the two waves."""

    def velocity_change(rho, rho_side):  # u_side - u* for the wave of one side
        if rho > rho_side:
            return (rho - rho_side) / np.sqrt(rho * rho_side)
        return np.log(rho / rho_side)

    low, high = -40.0, 40.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        rho = np.exp(middle)
        mismatch = u_R + velocity_change(rho, rho_R) - (u_L - velocity_change(rho, rho_L))
        low, high = (low, middle) if mismatch > 0.0 else (middle, high)
    rho_star = np.exp(0.5 * (low + high))
    u_star = u_L - velocity_change(rho_star, rho_L)
    xi = (x - 0.5) / t
    rho, u = np.empty_like(xi), np.empty_like(xi)
    for side, rho_s, u_s in ((-1.0, rho_L, u_L), (1.0, rho_R, u_R)):
        here = xi < u_star if side < 0 else xi >= u_star
        if rho_star > rho_s:
            speed = (rho_star * u_star - rho_s * u_s) / (rho_star - rho_s)
            outside = (xi < speed) if side < 0 else (xi >= speed)
            fan = np.zeros_like(xi, bool)
        else:
            outside = (xi < u_s - 1.0) if side < 0 else (xi >= u_s + 1.0)
            fan = ~outside & ((xi < u_star - 1.0) if side < 0 else (xi >= u_star + 1.0))
        rho[here] = np.where(outside[here], rho_s, rho_star)
        u[here] = np.where(outside[here], u_s, u_star)
        # inside the fan u = xi -+ 1 and u -+ ln rho = u_s -+ ln rho_s
        fan &= here
        u[fan] = xi[fan] - side
        rho[fan] = rho_s * np.exp(side * (u[fan] - u_s))
    return rho, u


@pytest.mark.parametrize(
    "tube, bound",
    [((1.0, 0.0, 0.125, 0.0), (2.8e-3, 6.6e-3)), ((1.0, 5.0, 1.0, -5.0), (1.2e-1, 6.5e-3))],
    ids=["rarefaction_and_shock", "mach_5_collision"],
)
def test_isothermal_shock_tube_against_the_exact_solution(tube, bound):
    """200 cells, t = 0.1, c_s = 1: the L1 errors of density and velocity (measured 2.69e-3, 6.38e-3 and
    0.116, 6.28e-3)."""
    rho_L, u_L, rho_R, u_R = tube
    x = _centres(200)
    left = x < 0.5
    fields = dict(density=np.where(left, rho_L, rho_R), velocity_x=np.where(left, u_L, u_R))
    params = SimulationParams(C_cfl=1.0, t_end=0.1, isothermal_sound_speed=1.0)
    final, registered_variables = _run(_config(1, 200, OPEN_BOUNDARY, ISOTHERMAL), fields, params)
    rho, u = _isothermal_exact(rho_L, u_L, rho_R, u_R, x, 0.1)
    assert np.mean(np.abs(final[registered_variables.density_index] - rho)) < bound[0]
    assert np.mean(np.abs(final[registered_variables.velocity_index] - u)) < bound[1]


@pytest.mark.parametrize("riemann_solver", [HLLEM_T, HLL], ids=["hllem_t", "hll"])
def test_stationary_shear_layer(riemann_solver):
    """A step of v_y at rest: HLLEM_T keeps it exactly (the shear waves carry no dissipation), HLL spreads it."""
    shape = (64, 4)
    x = _centres(64)[:, None] * np.ones(shape)
    step = np.where(np.abs(x - 0.5) < 0.25, 0.5, -0.5)
    config = _config(
        2,
        StaticIntVector(*shape),
        equation_of_state=ISOTHERMAL,
        box_size=StaticFloatVector(1.0, 4.0 / 64.0),
        solver_config=FD_RIEMANN_ISOTHERMAL._replace(riemann_solver=riemann_solver),
    )
    fields = dict(density=np.ones(shape), velocity_x=np.zeros(shape), velocity_y=step)
    final, registered_variables = _run(config, fields, SimulationParams(C_cfl=1.0, t_end=1.0))
    error = np.mean(np.abs(final[registered_variables.velocity_index.y] - step))
    if riemann_solver == HLLEM_T:
        assert error < 1e-14
    else:
        assert error > 1e-2


# =========================== positivity ==============================


def _strong_cases():
    """(name, cells, gamma, t_end, density, velocity, pressure) of the field-free guard cases of round 11."""
    x = _centres(200)
    inside = (x > 0.25) & (x < 0.75)
    return [
        ("rare123", 1.4, 0.15, np.ones(200), np.where(x < 0.5, -2.0, 2.0), 0.4 * np.ones(200)),
        ("vacuum", 1.4, 0.1, np.ones(200), np.where(x < 0.5, -5.0, 5.0), 0.4 * np.ones(200)),
        ("expand", 5.0 / 3.0, 0.1, np.where(inside, 1.0, 1e-6), np.zeros(200), np.where(inside, 1.0, 1e-8)),
    ]


@BOTH_PRECISIONS
def test_strong_rarefactions_and_expansion_into_near_vacuum_stay_positive(double):
    """Einfeldt's 1-2-3 problem, its vacuum version and an expansion into rho 1e-6, p 1e-8 (periodic)."""
    with jax.enable_x64(double):
        for name, gamma, t_end, density, velocity, pressure in _strong_cases():
            fields = dict(density=density, velocity_x=velocity, gas_pressure=pressure)
            params = SimulationParams(C_cfl=1.0, t_end=t_end, gamma=gamma, minimum_density=1e-14,
                                      minimum_pressure=1e-14)
            final, registered_variables = _run(_config(1, 200), fields, params)
            assert final.dtype == (np.float64 if double else np.float32)
            assert np.all(np.isfinite(final)), name
            assert final[registered_variables.density_index].min() > 1e-9, name
            assert final[registered_variables.pressure_index].min() > 1e-11, name


@BOTH_PRECISIONS
def test_double_blast_stays_positive(double):
    """The double blast of Woodward & Colella (400 cells, reflective walls, t = 0.038)."""
    x = _centres(400)
    fields = dict(density=np.ones(400), velocity_x=np.zeros(400),
                  gas_pressure=np.where(x > 0.9, 100.0, np.where(x < 0.1, 1000.0, 0.01)))
    with jax.enable_x64(double):
        final, registered_variables = _run(
            _config(1, 400, REFLECTIVE_BOUNDARY), fields, SimulationParams(C_cfl=1.0, t_end=0.038, gamma=1.4)
        )
    assert final.dtype == (np.float64 if double else np.float32)
    assert np.all(np.isfinite(final))
    # measured: minimum density 0.147, minimum pressure 0.01 (the initial one)
    assert final[registered_variables.density_index].min() > 0.1
    assert final[registered_variables.pressure_index].min() > 0.005
    # the peak of the density between the shocks (6.03; the reference 6.45, the production scheme 6.13)
    assert 5.9 < final[registered_variables.density_index][x > 0.4].max() < 6.5


# ========================== ghost cells ==============================


@pytest.mark.parametrize("boundary", [OPEN_BOUNDARY, REFLECTIVE_BOUNDARY], ids=["open", "reflective"])
def test_more_ghost_cells_give_the_same_result(boundary):
    """A 2D blast with open or reflective boundaries on every side: the rule of finalize_config (4 layers) against
    4 more."""
    shape = (32, 32)
    x, y = np.meshgrid(_centres(32), _centres(32), indexing="ij")
    radius = np.sqrt((x - 0.4) ** 2 + (y - 0.45) ** 2)
    fields = dict(density=1.0 + 0.5 * (x > 0.6), velocity_x=0.3 * np.ones(shape), velocity_y=-0.2 * np.ones(shape),
                  gas_pressure=np.where(radius < 0.15, 10.0, 0.1))
    params = SimulationParams(C_cfl=1.0, t_end=0.03, gamma=1.4)
    results = []
    for extra in (0, 4):
        config = _config(2, StaticIntVector(*shape), boundary)
        registered_variables = get_registered_variables(config)
        state = construct_primitive_state(
            config=config, registered_variables=registered_variables, **{k: jnp.asarray(v) for k, v in fields.items()}
        )
        config = finalize_config(config, state.shape)
        assert config.num_ghost_cells == 4
        config = config._replace(num_ghost_cells=config.num_ghost_cells + extra)
        results.append(np.asarray(time_integration(state, config, params, registered_variables)))
    assert np.max(np.abs(results[0] - results[1])) < 1e-12 * np.max(np.abs(results[1]))


# ========================= validity rules ============================

_SERVED = {
    "ideal gas, HLLC": (IDEAL_GAS, FD_RIEMANN_IDEAL_GAS._replace(riemann_solver=HLLC)),
    "ideal gas, HLL": (IDEAL_GAS, FD_RIEMANN_IDEAL_GAS._replace(riemann_solver=HLL)),
    "isothermal gas, HLLEM_T": (ISOTHERMAL, FD_RIEMANN_ISOTHERMAL._replace(riemann_solver=HLLEM_T)),
    "isothermal gas, HLL": (ISOTHERMAL, FD_RIEMANN_ISOTHERMAL._replace(riemann_solver=HLL)),
}
_REJECTED = {
    "HLLD_BOUNDED without a field": (IDEAL_GAS, FD_RIEMANN_IDEAL_GAS._replace(riemann_solver=HLLD_BOUNDED)),
    "HLLEM_T with the ideal gas": (IDEAL_GAS, FD_RIEMANN_IDEAL_GAS._replace(riemann_solver=HLLEM_T)),
    "HLLC with the isothermal gas": (ISOTHERMAL, FD_RIEMANN_ISOTHERMAL._replace(riemann_solver=HLLC)),
}


@pytest.mark.parametrize("dimensionality", [1, 2, 3])
@pytest.mark.parametrize("name", list(_SERVED))
def test_finalize_config_serves_the_hydrodynamic_form(name, dimensionality):
    equation_of_state, solver = _SERVED[name]
    config = _config(dimensionality, 16, OPEN_BOUNDARY, equation_of_state, solver_config=solver)
    config = finalize_config(config, (get_registered_variables(config).num_vars,) + (16,) * dimensionality)
    assert config.solver_config.scheme == RIEMANN_FLUX
    assert config.num_ghost_cells == 4


@pytest.mark.parametrize("name", list(_REJECTED))
def test_finalize_config_rejects_the_fluxes_of_the_other_forms(name):
    equation_of_state, solver = _REJECTED[name]
    config = _config(2, 16, OPEN_BOUNDARY, equation_of_state, solver_config=solver)
    with pytest.raises(ValueError):
        finalize_config(config, (get_registered_variables(config).num_vars, 16, 16))


def test_finalize_config_rejects_hllc_with_a_field():
    config = _config(3, 16, OPEN_BOUNDARY, mhd=True, solver_config=FD_RIEMANN_IDEAL_GAS._replace(riemann_solver=HLLC))
    with pytest.raises(ValueError):
        finalize_config(config, (get_registered_variables(config).num_vars,) + (16,) * 3)
