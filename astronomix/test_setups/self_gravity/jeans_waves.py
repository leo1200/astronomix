r"""
Jeans linear waves (3D).

Jeans Linear Waves are a self-gravity setup with
analytical solution.

\rho = \rho_B + \rho_B \eps sin(kx - wt)
v = \eps w k / k^2 sin(kx - wt)
P = c_s^2 \rho_B / \gamma + c_s^2 \rho_B \eps sin(kx - wt)

with

w = sqrt(c_s^2 k^2 - 4 \pi G \rho_B)

Here with \rho_B = 1, c_s^2 = 1, \gamma = 5/3,
4 \pi G = 1, k = 2 \pi (2,4,4)^T, \eps = 1e-6.

For the box length one must ensure proper periodicity
of the wave. The wavelength is given by

\lambda = 2 \pi / k

and for each dimension a multiple of the
wavelength must fit in the box.

For the above, e.g. L = 1.0 in all dimensions.

The period is T = 2 \pi / w, when we run for full
periods, we should see the initial conditions exactly reproduced.

We will run for N_periods = 3.

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


class JeansWaveSettings(NamedTuple):
    """Problem constants for the 3D Jeans linear wave test."""

    #: Wavenumber vector ``k`` (one wavelength fits along each axis).
    k_vec: jnp.array = jnp.array([2.0, 4.0, 4.0])

    #: Number of full wave periods to integrate over.
    n_periods: int = 3

    #: Adiabatic index of the gas.
    gamma: float = 5.0 / 3.0

    #: Background density ``rho_B``.
    rho_b: float = 1.0

    #: Background sound speed squared ``c_s^2``.
    c_s_squared: float = 1.0

    #: Gravitational coupling ``4 pi G``; the simulation uses
    #: ``G = four_pi_g / (4 pi)``.
    four_pi_g: float = 1.0

    #: Perturbation amplitude ``eps``.
    eps: float = 1e-6


def _derived(settings: JeansWaveSettings):
    """Return values derived from the primary settings."""
    k_squared = float(jnp.sum(settings.k_vec ** 2))
    # Dispersion relation: w = sqrt(c_s^2 k^2 - 4 pi G rho_B)
    omega = float(jnp.sqrt(settings.c_s_squared * k_squared - settings.four_pi_g * settings.rho_b))
    period = 2.0 * jnp.pi / omega
    t_end = settings.n_periods * period
    box_size = (
        float(2.0 * jnp.pi / settings.k_vec[0]),
        float(2.0 * jnp.pi / settings.k_vec[1]),
        float(2.0 * jnp.pi / settings.k_vec[2]),
    )
    g = settings.four_pi_g / (4.0 * jnp.pi)
    return k_squared, omega, period, t_end, box_size, g


def _phase(X, Y, Z, t, settings: JeansWaveSettings, omega: float):
    """Wave phase ``k . x - w t`` at simulation-frame coordinates."""
    return settings.k_vec[0] * X + settings.k_vec[1] * Y + settings.k_vec[2] * Z - omega * t


def _wave_primitive_state(X, Y, Z, t, settings: JeansWaveSettings):
    """
    Cell-centered primitive state of the Jeans linear wave.

    Returns ``(rho, v_x, v_y, v_z, p)`` at simulation-frame coordinates
    (X, Y, Z) and time ``t``. The velocity perturbation is parallel to
    ``k`` (longitudinal mode) with amplitude ``eps * w / k``.
    """
    k_squared, omega, _, _, _, _ = _derived(settings)
    s = jnp.sin(_phase(X, Y, Z, t, settings, omega))

    rho = settings.rho_b + settings.rho_b * settings.eps * s
    v_x = settings.eps * omega * settings.k_vec[0] / k_squared * s
    v_y = settings.eps * omega * settings.k_vec[1] / k_squared * s
    v_z = settings.eps * omega * settings.k_vec[2] / k_squared * s
    p   = settings.c_s_squared * settings.rho_b / settings.gamma \
        + settings.c_s_squared * settings.rho_b * settings.eps * s

    return rho, v_x, v_y, v_z, p


def _generate_state(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    t: float,
    settings: JeansWaveSettings,
) -> STATE_TYPE:
    """
    Generate the discrete primitive state for the Jeans wave at time ``t``.

    The hydrodynamic primitives are evaluated directly at the cell centers
    provided by ``helper_data``. No staggered initialization is required
    since the test is purely hydrodynamic with self-gravity.
    """
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


def setup_jeans_wave(
    config: SimulationConfig,
    params: SimulationParams,
    settings: JeansWaveSettings = JeansWaveSettings(),
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Set up the 3D Jeans linear wave test.

    Enforces the geometry (3D Cartesian), box size (2π/k_x, 2π/k_y, 2π/k_z)
    so that exactly one wavelength fits along each axis, periodic
    boundaries on all faces, end time t = N_periods * T, gamma, the
    gravitational constant G = four_pi_g/(4π), self-gravity enabled, and MHD
    disabled. The number of cells, solver mode, self-gravity version,
    Riemann solver, slope limiter and CFL number are left untouched. The
    user is responsible for choosing num_cells = (2N, N, N) so that the
    grid spacing is uniform across axes.

    Args:
        config: Simulation configuration.
        params: Simulation parameters.
        settings: Problem constants (defaults to the standard Jeans wave
            values).

    Returns:
        state: Initial primitive state of the simulation.
        config: Updated simulation configuration (CARTESIAN geometry,
            box_size = 2π/k per axis, 3D, periodic boundaries, self-gravity
            enabled, MHD disabled).
        params: Updated simulation parameters (t_end = N_periods * T,
            gamma, gravitational_constant = four_pi_g/(4π)).
    """
    _, _, _, t_end, box_size, g = _derived(settings)

    config = config._replace(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=StaticFloatVector(*box_size),
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        mhd=False,
        gravity_config=config.gravity_config._replace(self_gravity=True),
    )
    params = params._replace(
        t_end=t_end,
        gamma=settings.gamma,
        gravitational_constant=g,
    )

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


def jeans_wave_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    settings: JeansWaveSettings = JeansWaveSettings(),
) -> STATE_TYPE:
    """
    Exact Jeans linear wave state at t = ``params.t_end``, evaluated on
    the cell centers in ``helper_data``. Because the simulation runs for
    an integer number of full periods T = 2π/w, the analytic state at
    t = t_end is identical to the initial condition.

    Note: the reference state is generated through the same code path as
    ``setup_jeans_wave``, so that the convergence test measures only the
    evolution errors of the numerical scheme, free from grid
    initialization mismatches.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.
        settings: Problem constants (must match those used in
            :func:`setup_jeans_wave`).

    Returns:
        state: Exact primitive state at t = params.t_end.
    """
    return _generate_state(
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        t=params.t_end,
        settings=settings,
    )
