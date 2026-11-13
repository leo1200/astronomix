"""
The polytropic gas, P = K rho^Gamma, in the finite-difference scheme with a Riemann flux
(``FDSolverConfig.scheme = RIEMANN_FLUX``, preset ``FD_RIEMANN_ISOTHERMAL``): the flux, the characteristic
projection and the time step take the local sound speed a^2 = Gamma K rho^(Gamma - 1).

Checks that Gamma = 1 with K = c_s^2 is the isothermal gas to rounding in the hydrodynamic and the MHD form,
polytropic shock tubes with Gamma = 0.7 and 5/3 against the exact solution of the barotropic Riemann problem,
that decaying 3D turbulence with Gamma = 0.7 stays positive, the time step of the local sound speed, and the
rules of ``finalize_config``.

    JAX_PLATFORMS=cpu python -m pytest pytests/hydrodynamics/test_riemann_flux_polytropic.py
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
from astronomix.option_classes.presets import FD_RIEMANN_ISOTHERMAL
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_difference._timestep import _cfl_time_step_fd_hydro
from astronomix._spatial_operators._differencing import finite_difference_int6
from astronomix._spatial_operators._interpolate import interp_face_to_center
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

pytestmark = pytest.mark.x64

SOUND_SPEED = 0.7


def _config(dimensionality, num_cells, equation_of_state=POLYTROPIC, boundary=PERIODIC_BOUNDARY, mhd=False, **fields):
    side = BoundarySettings1D(boundary, boundary)
    settings = dict(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=dimensionality,
        mhd=mhd,
        equation_of_state=equation_of_state,
        num_cells=num_cells,
        box_size=1.0,
        boundary_settings=side if dimensionality == 1 else BoundarySettings(*([side] * dimensionality)),
        backend_config=BackendConfig(backend=NATIVE_JAX),
        solver_config=FD_RIEMANN_ISOTHERMAL,
        progress_bar=False,
    )
    settings.update(fields)
    return SimulationConfig(**settings)


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


def _rough_fields(dimensionality, num_cells, mhd):
    """Density from 0.1 to 1, velocities of rms 0.5 (Mach 0.7), for MHD a rough solenoidal field of mean 0.3."""
    shape = (num_cells,) * dimensionality
    keys = jax.random.split(jax.random.key(5), 8)
    fields = dict(density=10.0 ** jax.random.uniform(keys[0], shape, minval=-1.0, maxval=0.0))
    for axis, key in zip("xyz"[: 3 if mhd else dimensionality], keys[1:4]):
        fields[f"velocity_{axis}"] = 0.5 * jax.random.normal(key, shape)
    if mhd:
        spacing = 1.0 / num_cells
        a_x, a_y, a_z = (0.3 * spacing * jax.random.normal(key, shape) for key in keys[5:8])
        faces = (
            (finite_difference_int6(a_z, 1) - finite_difference_int6(a_y, 2)) / spacing + 0.3,
            (finite_difference_int6(a_x, 2) - finite_difference_int6(a_z, 0)) / spacing,
            (finite_difference_int6(a_y, 0) - finite_difference_int6(a_x, 1)) / spacing,
        )
        for axis, face in enumerate(faces):
            fields[f"magnetic_field_{'xyz'[axis]}"] = interp_face_to_center(face, axis)
            fields[f"interface_magnetic_field_{'xyz'[axis]}"] = face
    return fields


@pytest.mark.parametrize(
    "dimensionality, num_cells, mhd", [(1, 64, False), (2, 32, False), (3, 12, True)], ids=["1d", "2d", "mhd_3d"]
)
@pytest.mark.slow
def test_exponent_one_is_the_isothermal_gas(dimensionality, num_cells, mhd):
    """Gamma = 1, K = c_s^2 against the isothermal gas: relative 1e-13 (measured 0, 9e-16, 5e-15)."""
    params = SimulationParams(C_cfl=1.0, t_end=0.05, isothermal_sound_speed=SOUND_SPEED,
                              polytropic_constant=SOUND_SPEED**2, polytropic_exponent=1.0, minimum_density=1e-10)
    fields = _rough_fields(dimensionality, num_cells, mhd)
    isothermal, _ = _run(_config(dimensionality, num_cells, ISOTHERMAL, mhd=mhd), fields, params)
    polytropic, _ = _run(_config(dimensionality, num_cells, POLYTROPIC, mhd=mhd), fields, params)
    assert np.max(np.abs(isothermal - polytropic)) < 1e-13 * np.max(np.abs(isothermal))


# ============== the exact solution of the barotropic Riemann problem ==============


def _sound_speed(rho, constant, exponent):
    return np.sqrt(exponent * constant * rho ** (exponent - 1.0))


def _invariant(rho, constant, exponent):
    """G(rho) with dG / drho = a / rho."""
    a = _sound_speed(rho, constant, exponent)
    return a * np.log(rho) if exponent == 1.0 else 2.0 * a / (exponent - 1.0)


def _velocity_change(rho_star, rho_side, constant, exponent):
    """|u_side - u*| across the wave of one side: shock (rho* > rho_side) or rarefaction."""
    if rho_star > rho_side:
        jump = constant * (rho_star**exponent - rho_side**exponent)
        return np.sqrt(jump * (rho_star - rho_side) / (rho_star * rho_side))
    return _invariant(rho_star, constant, exponent) - _invariant(rho_side, constant, exponent)


def _bisection(function, low, high):
    """The root of an increasing ``function`` in [low, high]."""
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (low, middle) if function(middle) > 0.0 else (middle, high)
    return 0.5 * (low + high)


def _exact_barotropic(rho_L, u_L, rho_R, u_R, constant, exponent, x, t):
    """
    Density and velocity of the exact solution: the middle state from u_R + f_R(rho*) = u_L - f_L(rho*), shocks
    with the speed of the mass flux, fans with u -+ a = x / t and the Riemann invariant u +- G(rho) of the side.
    """

    def mismatch(log_rho):
        rho = np.exp(log_rho)
        return u_R + _velocity_change(rho, rho_R, constant, exponent) - (
            u_L - _velocity_change(rho, rho_L, constant, exponent)
        )

    rho_star = np.exp(_bisection(mismatch, -40.0, 40.0))
    u_star = u_L - _velocity_change(rho_star, rho_L, constant, exponent)
    a_star = _sound_speed(rho_star, constant, exponent)
    xi = (x - 0.5) / t
    rho, u = np.empty_like(xi), np.empty_like(xi)
    for side, rho_s, u_s in ((-1.0, rho_L, u_L), (1.0, rho_R, u_R)):
        here = (xi < u_star) if side < 0 else (xi >= u_star)
        if rho_star > rho_s:
            speed = (rho_star * u_star - rho_s * u_s) / (rho_star - rho_s)
            outside = (xi < speed) if side < 0 else (xi >= speed)
            fan = np.zeros_like(here)
        else:
            a_s = _sound_speed(rho_s, constant, exponent)
            outside = (xi < u_s - a_s) if side < 0 else (xi >= u_s + a_s)
            fan = ~outside & ((xi < u_star - a_star) if side < 0 else (xi >= u_star + a_star))
        rho[here] = np.where(outside[here], rho_s, rho_star)
        u[here] = np.where(outside[here], u_s, u_star)
        invariant = u_s - side * _invariant(rho_s, constant, exponent)
        for i in np.nonzero(fan & here)[0]:
            # u + side a = xi with u = invariant + side G(rho); side (G + a) increases with rho
            def residual(log_rho, i=i):
                r = np.exp(log_rho)
                return side * (invariant + side * (_invariant(r, constant, exponent)
                                                   + _sound_speed(r, constant, exponent)) - xi[i])

            r = np.exp(_bisection(residual, np.log(min(rho_star, rho_s)), np.log(max(rho_star, rho_s))))
            rho[i], u[i] = r, invariant + side * _invariant(r, constant, exponent)
    return rho, u


@pytest.mark.slow
def test_exact_solution_conserves_mass_and_momentum():
    """The sampled solution changes the mass and the momentum by the fluxes of the two outer states."""
    x = np.linspace(0.0, 1.0, 200001)
    for exponent in (0.7, 5.0 / 3.0):
        for rho_L, u_L, rho_R, u_R in ((1.0, 0.0, 0.125, 0.0), (1.0, 1.0, 1.0, -1.0), (1.0, -1.0, 1.0, 1.0)):
            rho, u = _exact_barotropic(rho_L, u_L, rho_R, u_R, 1.0, exponent, x, 0.1)
            left = x < 0.5
            mass_rate = (np.trapezoid(rho, x) - np.trapezoid(np.where(left, rho_L, rho_R), x)) / 0.1
            momentum_rate = (np.trapezoid(rho * u, x) - np.trapezoid(np.where(left, rho_L * u_L, rho_R * u_R), x)) / 0.1
            flux = (rho_L * u_L - rho_R * u_R, rho_L * u_L**2 + rho_L**exponent - rho_R * u_R**2 - rho_R**exponent)
            assert abs(mass_rate - flux[0]) < 1e-4 and abs(momentum_rate - flux[1]) < 1e-4


TUBES = {"rarefaction_and_shock": (1.0, 0.0, 0.125, 0.0), "collision": (1.0, 1.0, 1.0, -1.0)}
# bounds of the L1 errors of (rho, u) at 200 cells, t = 0.1, about 5 % above the measured values
# (2.49e-3, 6.02e-3; 2.59e-3, 6.16e-3; 6.27e-3, 4.27e-3; 3.42e-3, 3.63e-3; at 400 cells about half of these)
TUBE_ERRORS = {
    ("rarefaction_and_shock", 0.7): (2.6e-3, 6.3e-3),
    ("rarefaction_and_shock", 5.0 / 3.0): (2.7e-3, 6.4e-3),
    ("collision", 0.7): (6.6e-3, 4.5e-3),
    ("collision", 5.0 / 3.0): (3.6e-3, 3.8e-3),
}


@pytest.mark.parametrize("exponent", [0.7, 5.0 / 3.0], ids=["gamma_0.7", "gamma_5/3"])
@pytest.mark.parametrize("tube", list(TUBES))
def test_polytropic_shock_tube_against_the_exact_solution(tube, exponent):
    rho_L, u_L, rho_R, u_R = TUBES[tube]
    x = _centres(200)
    left = x < 0.5
    fields = dict(density=np.where(left, rho_L, rho_R), velocity_x=np.where(left, u_L, u_R))
    params = SimulationParams(C_cfl=1.0, t_end=0.1, polytropic_constant=1.0, polytropic_exponent=exponent)
    final, registered_variables = _run(_config(1, 200, boundary=OPEN_BOUNDARY), fields, params)
    rho, u = _exact_barotropic(rho_L, u_L, rho_R, u_R, 1.0, exponent, x, 0.1)
    bound = TUBE_ERRORS[(tube, exponent)]
    assert np.mean(np.abs(final[registered_variables.density_index] - rho)) < bound[0]
    assert np.mean(np.abs(final[registered_variables.velocity_index] - u)) < bound[1]


# ============================ turbulence ==============================


def _solenoidal_velocity(num_cells, rms, seed=1):
    """A random solenoidal velocity in the shells 1 <= |k| <= 4, scaled to the rms speed."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(num_cells, 1.0 / num_cells)
    wave_vector = np.meshgrid(k, k, k, indexing="ij")
    k2 = sum(component**2 for component in wave_vector)
    shell = (k2 >= 1.0) & (k2 <= 16.0)
    modes = [np.fft.fftn(rng.normal(size=(num_cells,) * 3)) * shell for _ in range(3)]
    projection = sum(K * mode for K, mode in zip(wave_vector, modes)) / np.where(k2 > 0, k2, 1.0)
    velocity = [np.real(np.fft.ifftn(mode - projection * K)) for mode, K in zip(modes, wave_vector)]
    return [v * rms / np.sqrt(np.mean(sum(w**2 for w in velocity))) for v in velocity]


def test_decaying_turbulence_with_a_soft_polytrope_stays_positive():
    """Gamma = 0.7, rms Mach 3 at 16^3 to t = 0.1: finite, positive, the kinetic energy decays."""
    velocity = _solenoidal_velocity(16, 3.0)
    fields = dict(density=np.ones((16,) * 3), velocity_x=velocity[0], velocity_y=velocity[1], velocity_z=velocity[2])
    params = SimulationParams(C_cfl=1.0, t_end=0.1, polytropic_constant=1.0, polytropic_exponent=0.7,
                              minimum_density=1e-10)
    final, registered_variables = _run(_config(3, 16), fields, params)
    rho = final[registered_variables.density_index]
    kinetic = 0.5 * np.mean(rho * sum(final[index] ** 2 for index in registered_variables.velocity_index))
    assert np.all(np.isfinite(final)) and rho.min() > 1e-3
    assert kinetic < 0.5 * 9.0 and abs(np.mean(rho) - 1.0) < 1e-12


# ============================ time step ===============================


def test_time_step_takes_the_local_sound_speed():
    """At rest: dt = C dx / (d max a) with a^2 = Gamma K rho^(Gamma - 1), largest where the density is smallest."""
    x, y = np.meshgrid(_centres(32), _centres(32), indexing="ij")
    density = 1.0 + 0.9 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    config = _config(2, 32)
    registered_variables = get_registered_variables(config)
    state = construct_primitive_state(config=config, registered_variables=registered_variables,
                                      density=jnp.asarray(density), velocity_x=jnp.zeros((32, 32)),
                                      velocity_y=jnp.zeros((32, 32)))
    config = finalize_config(config, state.shape)
    params = SimulationParams(C_cfl=0.8, polytropic_constant=2.0, polytropic_exponent=0.7)
    dt = _cfl_time_step_fd_hydro(state, config.grid_spacing, params.dt_max, params.gamma, config, params,
                                 registered_variables, params.C_cfl)
    speed = np.max(np.sqrt(0.7 * 2.0 * np.maximum(density, params.minimum_density) ** (0.7 - 1.0)))
    assert abs(float(dt) - 0.8 * (1.0 / 32) / (2.0 * speed)) < 1e-14


# ========================= validity rules ============================


@pytest.mark.parametrize("mhd", [False, True], ids=["hydrodynamics", "mhd"])
def test_polytropic_layout_and_rules(mhd):
    """No pressure slot; served by RIEMANN_FLUX and by the production scheme."""
    dimensionality = 3 if mhd else 2
    config = _config(dimensionality, 16, mhd=mhd)
    registered_variables = get_registered_variables(config)
    assert registered_variables.pressure_index == -1
    assert registered_variables.num_vars == (10 if mhd else 3)
    shape = (registered_variables.num_vars,) + (16,) * dimensionality
    assert finalize_config(config, shape).equation_of_state == POLYTROPIC
    assert finalize_config(config._replace(solver_config=FDSolverConfig()), shape).equation_of_state == POLYTROPIC
