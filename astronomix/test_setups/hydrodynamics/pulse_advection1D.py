"""
# Gaussian Pulse Advection

A linear advection test in which a Gaussian density pulse is transported by a
uniform velocity field across a periodic domain. Under exact linear advection
the pulse retains its shape, so any deviation between the numerical and exact
solutions is purely numerical error. This makes the test well suited for
measuring the order of convergence of a hydrodynamics scheme.

The domain has length 1.0 with periodic boundaries. The initial state is

    rho(x, 0) = 1 + exp( -(x - x0)^2 / (2 * sigma^2) )
    u(x, 0)   = v_adv
    p(x, 0)   = p0

with x0 = 0.5, sigma = 0.0625, v_adv = 1.0, p0 = 10.0, and gamma = 1.4. The
test is typically run to t = 2.0, corresponding to two full traversals of
the periodic box, so the exact final density profile coincides with the
initial one.
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
    BoundarySettings1D,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config


class GaussianPulseAdvection1DSettings(NamedTuple):
    """Problem constants for the 1D Gaussian pulse advection test."""

    #: Length of the (periodic) simulation domain.
    box_size: float = 1.0

    #: Final time at which the solution is evaluated.
    t_end: float = 2.0

    #: Adiabatic index of the gas.
    gamma: float = 1.4

    #: Initial center of the Gaussian pulse.
    pulse_center: float = 0.5

    #: Standard deviation (width) of the Gaussian pulse.
    pulse_width: float = 0.0625

    #: Uniform advection velocity.
    advection_velocity: float = 1.0

    #: Uniform background pressure.
    pressure: float = 10.0


def _gaussian_pulse_density(
    r: jnp.ndarray,
    t: float,
    settings: GaussianPulseAdvection1DSettings,
) -> jnp.ndarray:
    """Exact density profile of the advected Gaussian pulse at time ``t``."""
    center = (settings.pulse_center + settings.advection_velocity * t) % settings.box_size
    distance = jnp.abs(r - center)
    return 1.0 + jnp.exp(-distance**2 / (2.0 * settings.pulse_width**2))


def setup_gaussian_pulse_advection(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    settings: GaussianPulseAdvection1DSettings = GaussianPulseAdvection1DSettings(),
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Set up the Gaussian pulse advection test.

    Enforces the geometry, box size, periodic boundaries, end time and gamma
    required by the standard problem. The number of cells, Riemann solver,
    slope limiter and CFL number are left untouched so that the caller can
    study their influence.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.
        settings: Problem constants (defaults to the standard pulse
            advection values).

    Returns:
        state: Initial primitive state of the simulation.
        config: Updated simulation configuration (CARTESIAN geometry,
            box_size from ``settings``, periodic boundaries).
        params: Updated simulation parameters (t_end and gamma from
            ``settings``).
    """
    config = config._replace(
        geometry = CARTESIAN,
        box_size = settings.box_size,
        dimensionality = 1,
        boundary_settings = BoundarySettings1D(
            left_boundary = PERIODIC_BOUNDARY,
            right_boundary = PERIODIC_BOUNDARY,
        )
    )
    params = params._replace(t_end = settings.t_end, gamma = settings.gamma)

    r = helper_data.geometric_centers
    rho = _gaussian_pulse_density(r, t = 0.0, settings = settings)
    u   = jnp.full_like(r, settings.advection_velocity)
    p   = jnp.full_like(r, settings.pressure)

    state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )

    config = finalize_config(config, state.shape)

    return state, config, params

def gaussian_pulse_advection_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    settings: GaussianPulseAdvection1DSettings = GaussianPulseAdvection1DSettings(),
) -> STATE_TYPE:
    """
    Exact solution for the Gaussian pulse advection test, evaluated on the
    cell centers in ``helper_data`` at ``params.t_end``.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.
        settings: Problem constants (must match those used in
            :func:`setup_gaussian_pulse_advection`).

    Returns:
        state: Exact primitive state at t = params.t_end.
    """
    r = helper_data.geometric_centers
    rho = _gaussian_pulse_density(r, t = params.t_end, settings = settings)
    u   = jnp.full_like(r, settings.advection_velocity)
    p   = jnp.full_like(r, settings.pressure)

    return construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )
