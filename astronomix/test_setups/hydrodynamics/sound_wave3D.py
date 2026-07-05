"""
3D linear sound wave.

Three-dimensional convergence test using a small-amplitude acoustic
eigenmode of the linearised Euler equations propagating obliquely with
respect to the grid axes. This is the pure-hydro analogue of the CP Alfvén
wave methods-paper test.

Eigenmode (right-going acoustic wave, propagation along
``n_hat = (1, 2, 2) / 3``):

    rho   = rho_0   + eps * rho_0           * cos(k . x - omega t)
    v_par = 0       + eps * c_s             * cos(k . x - omega t)
    p     = p_0     + eps * rho_0 * c_s^2   * cos(k . x - omega t)

with the velocity perturbation parallel to ``n_hat``, ``k = k_wave * n_hat``,
and ``omega = c_s * |k|``. The background state is chosen so that the sound
speed is unity: ``c_s = sqrt(gamma p_0 / rho_0) = 1``.

The simulation domain is the periodic box (3, 3/2, 3/2) with grid
(2N, N, N) so that the wave fits exactly one wavelength along each axis
and the grid spacing is uniform. After ``n_periods`` full periods the
analytic state has returned to the initial condition.
"""

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import CARTESIAN
from astronomix.option_classes.simulation_config import (
    PERIODIC_BOUNDARY,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.variable_registry.registered_variables import get_registered_variables


# Propagation direction in the simulation frame: (1, 2, 2)/3 (matches the
# CP Alfvén wave test geometry so the convergence pictures are directly
# comparable). Box size (3, 1.5, 1.5) is required for periodicity.
_N_HAT = jnp.array([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])


class SoundWave3DSettings(NamedTuple):
    """Problem constants for the 3D linear sound-wave test."""

    #: Simulation-frame box size ``(L_x, L_y, L_z)``.
    box_size: tuple = (3.0, 1.5, 1.5)

    #: Number of full wave periods to integrate over.
    n_periods: int = 5

    #: Adiabatic index of the gas.
    gamma: float = 5.0 / 3.0

    #: Background density ``rho_0``.
    rho_0: float = 1.0

    #: Background pressure ``p_0`` (chosen so that ``c_s = 1`` with the
    #: default ``rho_0`` and ``gamma``).
    p_0: float = 3.0 / 5.0

    #: Wavenumber along the unrotated axis; wavelength = 2 pi / k_wave.
    k_wave: float = 2.0 * jnp.pi

    #: Perturbation amplitude ``eps`` (small enough to remain linear).
    eps: float = 1.0e-6


def _c_s(settings: SoundWave3DSettings) -> float:
    return float(jnp.sqrt(settings.gamma * settings.p_0 / settings.rho_0))


def _omega(settings: SoundWave3DSettings) -> float:
    return _c_s(settings) * settings.k_wave


def _phase(X, Y, Z, t, settings: SoundWave3DSettings):
    """Wave phase ``k * (n_hat . x) - omega t`` at simulation-frame
    coordinates (X, Y, Z) and time ``t``."""
    x_par = _N_HAT[0] * X + _N_HAT[1] * Y + _N_HAT[2] * Z
    return settings.k_wave * x_par - _omega(settings) * t


def _wave_primitive_state(X, Y, Z, t, settings: SoundWave3DSettings):
    """Cell-centered primitive state of the right-going sound wave."""
    c_s = _c_s(settings)
    c = jnp.cos(_phase(X, Y, Z, t, settings))
    drho = settings.eps * settings.rho_0 * c
    dv_par = settings.eps * c_s * c
    dp = settings.eps * settings.rho_0 * c_s * c_s * c

    rho = settings.rho_0 + drho
    v_x = _N_HAT[0] * dv_par
    v_y = _N_HAT[1] * dv_par
    v_z = _N_HAT[2] * dv_par
    p = settings.p_0 + dp

    # Broadcast scalar background velocity components to the grid shape.
    v_x = jnp.broadcast_to(v_x, X.shape)
    v_y = jnp.broadcast_to(v_y, X.shape)
    v_z = jnp.broadcast_to(v_z, X.shape)
    return rho, v_x, v_y, v_z, p


def _generate_state(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    t: float,
    settings: SoundWave3DSettings,
) -> STATE_TYPE:
    cell_centers = helper_data.geometric_centers
    Xc, Yc, Zc = cell_centers[..., 0], cell_centers[..., 1], cell_centers[..., 2]

    rho, v_x, v_y, v_z, p = _wave_primitive_state(Xc, Yc, Zc, t=t, settings=settings)

    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=v_x,
        velocity_y=v_y,
        velocity_z=v_z,
        gas_pressure=p,
    )


def setup_sound_wave(
    config: SimulationConfig,
    params: SimulationParams,
    settings: SoundWave3DSettings = SoundWave3DSettings(),
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Set up the 3D linear sound-wave test.

    Enforces the geometry (3D Cartesian), box size, periodic boundaries on
    all faces, end time = ``n_periods`` * T, gamma, and pure-hydro mode.
    The number of cells, solver mode, Riemann solver, slope limiter and CFL
    number are left untouched. The user is responsible for choosing
    ``num_cells = (2N, N, N)`` so that the grid spacing is uniform.
    """
    period = 2.0 * jnp.pi / _omega(settings)
    t_end = float(settings.n_periods * period)

    config = config._replace(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=StaticFloatVector(*settings.box_size),
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        mhd=False,
    )
    params = params._replace(t_end=t_end, gamma=settings.gamma)

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    state = _generate_state(
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        t=0.0,
        settings=settings,
    )

    config = finalize_config(config, state.shape)
    return state, config, params


def sound_wave_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    settings: SoundWave3DSettings = SoundWave3DSettings(),
) -> STATE_TYPE:
    """
    Exact sound-wave state at ``t = params.t_end``. Because the simulation
    runs for an integer number of full periods, this equals the initial
    condition.
    """
    return _generate_state(
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        t=params.t_end,
        settings=settings,
    )
