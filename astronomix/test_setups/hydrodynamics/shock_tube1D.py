"""
# Sod Shock Tube

The Sod shock tube (Sod 1978) is a classic 1D Riemann problem that simultaneously
exercises a fluid simulator's treatment of shocks, contact discontinuities, and
rarefaction waves. Two constant fluid states are separated by a diaphragm at
x = 0.5 inside a box of length 1.0:

    Left state  (x < 0.5): rho = 1.0,   u = 0.0, p = 1.0
    Right state (x > 0.5): rho = 0.125, u = 0.0, p = 0.1

Once the diaphragm is removed, the flow develops a leftward-moving rarefaction,
a rightward-moving contact discontinuity, and a rightward-moving shock. The
test is typically evaluated at t = 0.2 with gamma = 5/3.

## References

- Sod, G. A. (1978). "A survey of several finite difference methods for systems
  of nonlinear hyperbolic conservation laws". Journal of Computational Physics,
  27(1), 1-31.
- Toro, E. F. (2009). "Riemann Solvers and Numerical Methods for Fluid Dynamics",
  3rd ed., Springer.
"""

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import CARTESIAN
from astronomix.option_classes.simulation_config import (
    OPEN_BOUNDARY,
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
from astronomix.test_setups.reference_solutions.riemann_solver import _exact_riemann_ideal_gas


class ShockTube1DSettings(NamedTuple):
    """Problem constants for the 1D Sod shock tube test."""

    #: Position of the initial diaphragm inside the box.
    shock_pos: float = 0.5

    #: Adiabatic index of the gas.
    gamma: float = 5 / 3

    #: Length of the simulation domain.
    box_size: float = 1.0

    #: Final time at which the solution is evaluated.
    t_end: float = 0.2

    #: Left state density.
    rho_L: float = 1.0

    #: Left state velocity.
    u_L: float = 0.0

    #: Left state pressure.
    p_L: float = 1.0

    #: Right state density.
    rho_R: float = 0.125

    #: Right state velocity.
    u_R: float = 0.0

    #: Right state pressure.
    p_R: float = 0.1


def setup_sod_shock_tube(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    settings: ShockTube1DSettings = ShockTube1DSettings(),
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Set up the Sod shock tube test.

    Enforces the geometry, box size and end time required by the standard
    problem. The number of cells, Riemann solver, and slope limiter are
    left untouched so that the caller can study their influence.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.
        settings: Problem constants (defaults to the standard Sod values).

    Returns:
        state: Initial primitive state of the simulation.
        config: Updated simulation configuration (CARTESIAN geometry,
            box_size from ``settings``).
        params: Updated simulation parameters (t_end and gamma from
            ``settings``).
    """
    config = config._replace(
        geometry = CARTESIAN,
        box_size = settings.box_size,
        dimensionality = 1,
        boundary_settings = BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY,
        )
    )
    params = params._replace(t_end = settings.t_end, gamma = settings.gamma)

    r = helper_data.geometric_centers
    rho = jnp.where(r < settings.shock_pos, settings.rho_L, settings.rho_R)
    u   = jnp.where(r < settings.shock_pos, settings.u_L,   settings.u_R)
    p   = jnp.where(r < settings.shock_pos, settings.p_L,   settings.p_R)

    state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )

    config = finalize_config(config, state.shape)

    return state, config, params


def sod_shock_tube_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    settings: ShockTube1DSettings = ShockTube1DSettings(),
) -> STATE_TYPE:
    """
    Exact Riemann solution for the Sod shock tube, evaluated on the cell
    centers in ``helper_data`` at ``params.t_end``.

    Requires the ``exactpack`` package.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.
        settings: Problem constants (must match those used in
            :func:`setup_sod_shock_tube`).

    Returns:
        state: Exact primitive state at t = params.t_end.
    """

    rho, u, p = _exact_riemann_ideal_gas(
        rho_L = settings.rho_L,
        u_L = settings.u_L,
        p_L = settings.p_L,
        rho_R = settings.rho_R,
        u_R = settings.u_R,
        p_R = settings.p_R,
        gamma = params.gamma,
        x = helper_data.geometric_centers,
        t = params.t_end,
        x0 = settings.shock_pos,
    )

    return construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )
