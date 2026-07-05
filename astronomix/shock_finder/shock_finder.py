"""
Shock detection on a 1D fluid state.

Implements the smoothness sensor and the multi-criterion shock test of
Pfrommer et al. (2017) used to flag shocks (e.g. for diffusive shock
acceleration of cosmic rays), plus a routine that broadens the strongest shock
into a numerical shock zone.

NOTE: the routines here currently only support 1D setups; TODO: generalise.
"""

# general
from functools import partial

# typing
from typing import Tuple, Union
from jaxtyping import Array, Int, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FIELD_TYPE,
    SPHERICAL,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams


@partial(jax.jit, static_argnames=["config"])
def _calculate_1d_divergence(
    field: FIELD_TYPE, config: SimulationConfig, r: FIELD_TYPE
) -> FIELD_TYPE:
    """Central-difference 1D divergence of ``field`` for the active geometry.

    Args:
        field: The 1D field whose divergence is taken.
        config: The simulation configuration (selects the geometry).
        r: The radial cell-centre coordinates (used in spherical geometry).

    Returns:
        The divergence of the field; ghost cells at the ends are left at zero.
    """
    # Approximate the 1D divergence with a simple central difference.
    div_field = jnp.zeros_like(field)
    if config.geometry == CARTESIAN:
        div_field = div_field.at[1:-1].set(
            (field[2:] - field[:-2]) / (2 * config.grid_spacing)
        )
    elif config.geometry == SPHERICAL:
        div_field = jnp.zeros_like(field)
        # This is not exactly correct, since the field values live at the
        # volumetric rather than the geometric cell centres, but it is accurate
        # enough for the purposes of the shock finder.
        div_field = div_field.at[1:-1].set(
            (r[2:] ** 2 * field[2:] - r[:-2] ** 2 * field[:-2])
            / (2 * config.grid_spacing * r[1:-1] ** 2)
        )
    else:
        raise NotImplementedError(
            "Only Cartesian and Spherical geometry supported for the shock finder."
        )
    return div_field


@jax.jit
def shock_sensor(pressure: FIELD_TYPE) -> FIELD_TYPE:
    """
    WENO-JS 1D smoothness indicator for shock detection.

    Args:
        pressure: the 1d pressure

    Returns:
        shock sensors, high where large pressure jumps

    """

    shock_sensors = jnp.zeros_like(pressure)
    shock_sensors = shock_sensors.at[1:-1].set(
        1 / 4 * (pressure[2:] - pressure[:-2]) ** 2
        + 13 / 12 * (pressure[2:] - 2 * pressure[1:-1] + pressure[:-2])
    )

    return shock_sensors


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def shock_criteria(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
) -> jnp.ndarray:
    """
    Implement the shock criteria from Pfrommer et al, 2017.
    https://arxiv.org/abs/1604.07399

    # NOTE: for now only 1D

    """

    gamma_gas = 5 / 3
    gamma_cr = 4 / 3

    # get the velocity
    velocity = primitive_state[registered_variables.velocity_index]

    # get the cosmic ray pressure
    P_CRs = primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr

    # i) \nabla \cdot \vec{v} < 0
    div_v = _calculate_1d_divergence(velocity, config, helper_data.geometric_centers)
    converging_flow_criterion = div_v < 0

    # ii) \nabla T \cdot \nabla \rho > 0
    pseudo_temperature = (
        primitive_state[registered_variables.pressure_index]
        / primitive_state[registered_variables.density_index]
    )
    div_T = jnp.zeros_like(pseudo_temperature)
    div_T = div_T.at[1:-1].set((pseudo_temperature[2:] - pseudo_temperature[:-2]) / 2)
    div_rho = jnp.zeros_like(primitive_state[registered_variables.density_index])
    div_rho = div_rho.at[1:-1].set(
        (
            primitive_state[registered_variables.density_index][2:]
            - primitive_state[registered_variables.density_index][:-2]
        )
        / 2
    )
    no_spurious_shocks = div_T * div_rho > 0

    # iii) M1 > Mmin
    Mmin = 1.3
    # NOTE: currently we only consider shocks moving left to right
    P2 = primitive_state[registered_variables.pressure_index, :-2]
    P2_CRs = P_CRs[:-2]
    e2_gas = P2 - P2_CRs  # gas energy / volume
    e2_crs = P2_CRs / (gamma_cr - 1)  # cosmic ray energy / volume
    e2 = e2_gas + e2_crs  # total energy / volume
    rho2 = primitive_state[registered_variables.density_index, :-2]

    P1 = primitive_state[registered_variables.pressure_index, 2:]
    P1_CRs = P_CRs[2:]
    P1_gas = P1 - P1_CRs  # gas pressure
    e1_gas = P1_gas / (gamma_gas - 1)  # gas energy / volume
    e1_crs = P1_CRs / (gamma_cr - 1)  # cosmic ray energy / volume
    e1 = e1_gas + e1_crs  # total energy / volume
    rho1 = primitive_state[registered_variables.density_index, 2:]

    gamma_eff1 = (gamma_cr * P1_CRs + gamma_gas * P1_gas) / P1
    gamma_eff2 = (gamma_cr * P2_CRs + gamma_gas * P2) / P2

    gamma1 = P1 / e1 + 1
    gamma2 = P2 / e2 + 1

    gammat = P2 / P1

    C = ((gamma2 + 1) * gammat + gamma2 - 1) * (gamma1 - 1)

    # advanced Mach number calculation, formula 16 from Dubois et al, 2019
    denominator = jnp.where(
        jnp.abs(C - ((gamma1 + 1) + (gamma1 - 1) * gammat) * (gamma2 - 1)) > 1e-6,
        (C - ((gamma1 + 1) + (gamma1 - 1) * gammat) * (gamma2 - 1)),
        1e-6,
    )
    M1sq = 1 / gamma_eff2 * (gammat - 1) * C / denominator

    # simple Mach number calculation, crashes
    # the simulation where x_s = 1, better just evaluate
    # this where the other criterions hold / add a numerical
    # safeguard
    # x_s = rho2 / rho1
    # M1sq = (P2 / P1 - 1) * x_s / (gamma_eff1 * (x_s - 1))

    mach_number_criterion = jnp.zeros_like(converging_flow_criterion, dtype=jnp.bool_)

    mach_number_criterion = mach_number_criterion.at[1:-1].set(M1sq > Mmin**2)

    return converging_flow_criterion & no_spurious_shocks & mach_number_criterion


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def find_shock_zone(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
) -> Tuple[
    Union[int, Int[Array, ""]], Union[int, Int[Array, ""]], Union[int, Int[Array, ""]]
]:
    """
    Find a numerically broadened shock region based of the strongest shock based
    on the result of the shock_sensor function and the pressure difference
    between adjacent cells. Assumes a shock front moving left to right.

    Args:
        pressure: 1d pressure
        velocity: 1d velocity

    Returns:
        index of max shock sensor,
        left boundary of broadened shock,
        right boundary of broadened shock

    """

    pressure = primitive_state[registered_variables.pressure_index]
    num_cells = pressure.shape[0]

    # one can either use the maximum of the shock sensor
    sensors = shock_sensor(pressure)
    # or the cell with maximum compression, as in Pfrommer et al 2017
    # div_v = _calculate_1d_divergence(primitive_state[registered_variables.velocity_index], config, helper_data.geometric_centers)

    shock_crit = shock_criteria(
        primitive_state, config, registered_variables, helper_data
    )

    max_shock_idx = jnp.argmax(jnp.where(shock_crit, sensors, -1))
    # max_shock_idx = jnp.argmin(jnp.where(shock_crit, div_v, 1))

    # calculate differences in pressure
    pressure_differences = jnp.zeros_like(pressure)
    # 0 <- 1 - 0
    pressure_differences = pressure_differences.at[1:].set(pressure[1:] - pressure[:-1])

    # bound on the change in pressure between adjacent cells compared
    # to the pressure jump at the max_shock_index
    bound_diff = 0.1 * jnp.abs(pressure_differences[max_shock_idx])

    # left index: closest left index where |pressure_difference| < bound_diff or switched sign
    # right index: closest right index where |pressure_difference| < bound_diff or switched sign
    indices = jnp.arange(num_cells)
    left_indices = jnp.where(
        (indices < max_shock_idx)
        & ((jnp.abs(pressure_differences) < bound_diff) | (pressure_differences > 0)),
        indices,
        -1,
    )
    right_indices = jnp.where(
        (indices > max_shock_idx)
        & ((jnp.abs(pressure_differences) < bound_diff) | (pressure_differences < 0)),
        indices,
        num_cells,
    )
    left_idx = jnp.max(left_indices)
    right_idx = jnp.min(right_indices)

    return max_shock_idx, left_idx, right_idx
