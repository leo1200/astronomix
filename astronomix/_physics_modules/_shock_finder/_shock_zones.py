# ============================================================================
# PHASE 2: SHOCK ZONE IDENTIFICATION
# ============================================================================
#  goal is to produce a boolean field marking every cell that belongs to a shock zone
# 
# Identify cells that are in the shock zone using three criteria:
# 1. Converging flow: ∇·v < 0
# 2. Aligned gradients: ∇T·∇ρ > 0
# 3. Minimum Mach number: M > M_min = 1.3

# general
from functools import partial
import jax.numpy as jnp
import jax

# typing
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE, BOOL_FIELD_TYPE,
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)
import astronomix._physics_modules._shock_finder._helpers as helpers


@partial(jax.jit, static_argnames=["config"])
def _shock_zone_criterion_converging_flow(
    velocity: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None
) -> BOOL_FIELD_TYPE:
    """
    Check criterion 1: Converging flow (∇·v < 0).
    
    In 1D, it means checking if the velocity gradient is negative
    
    Args:
        velocity: Velocity field (1D)
        config: Simulation configuration
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Boolean field, True where flow is converging
    """
    div_v = helpers._calculate_gradient(velocity, config, r)
    return div_v < 0


@partial(jax.jit, static_argnames=["config"])
def _shock_zone_criterion_aligned_gradients(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None
) -> BOOL_FIELD_TYPE:
    """
    Check criterion 2: Aligned gradients (∇T·∇ρ > 0).
    
    Temperature and density gradients point in the same direction
    
    Args:
        pressure: Gas pressure field
        density: Density field
        config: Simulation configuration
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Boolean field, True where ∇T·∇ρ > 0
    """
    grad_T = helpers._calculate_temperature_gradient(pressure, density, config, r)
    grad_rho = helpers._calculate_density_gradient(density, config, r)

    dot_product = grad_T * grad_rho
    return dot_product > 0


def get_post_pre_shock_values(shock_direction, pressure, temperature):
    is_rightward = shock_direction[1:-1] > 0  # shock moves left to right

    p_left  = pressure[:-2]   # pressure at i-1
    p_right = pressure[2:]    # pressure at i+1
    T_left  = temperature[:-2]
    T_right = temperature[2:]

    # select post and pre shock values based on shock direction
    p_post = jnp.where(is_rightward, p_left,  p_right)
    p_pre  = jnp.where(is_rightward, p_right, p_left)
    T_post = jnp.where(is_rightward, T_left,  T_right)
    T_pre  = jnp.where(is_rightward, T_right, T_left)
    return p_post,p_pre,T_post,T_pre


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _shock_zone_criterion_minimum_mach(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    shock_direction: FIELD_TYPE,
    mach_min: float = 1.3
) -> BOOL_FIELD_TYPE:
    
    # compute minimum jumps from Rankine-Hugoniot relations
    # If a shock has Mach number M, how much pressure and temperature must jump to reach M?
    # -> p_ratio_min and T_ratio_min from Rankine-Hugoniot for given M and gamma
    gamma_gas = 5 / 3
    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    temperature = pressure / density  # pseudo-temperature T = P/rho

    # Minimum jumps at M = mach_min from Rankine-Hugoniot relations
    M2 = mach_min ** 2
    p_ratio_min = (2 * gamma_gas * M2 - (gamma_gas - 1)) / (gamma_gas + 1)
    T_ratio_min = p_ratio_min * ((gamma_gas - 1) * M2 + 2) / ((gamma_gas + 1) * M2)

    # convert thresholds to log space for numerical stability
    log_p_min = jnp.log(p_ratio_min)
    log_T_min = jnp.log(T_ratio_min)

    # use shock_direction to determine which neighbor is post-shock (upstream)
    # and which is pre-shock (downstream) at each interior cell.
    # shock_direction = -∇T/|∇T| points from hot (post) toward cold (pre) gas.
    # if d_s[i] > 0 → post-shock is on the left  (i-1), pre-shock on the right (i+1)
    # if d_s[i] < 0 → post-shock is on the right (i+1), pre-shock on the left  (i-1)
    p_post, p_pre, T_post, T_pre = get_post_pre_shock_values(shock_direction, pressure, temperature)

    # compute log jumps: log(post/pre) = log(post) - log(pre)
    # clamp to 1e-30 to avoid log(0) = -inf → NaN
    log_p_jump = jnp.zeros_like(pressure)
    log_T_jump = jnp.zeros_like(temperature)

    log_p_jump = log_p_jump.at[1:-1].set(
        jnp.log(jnp.maximum(p_post, 1e-30))
        - jnp.log(jnp.maximum(p_pre,  1e-30))
    )
    log_T_jump = log_T_jump.at[1:-1].set(
        jnp.log(jnp.maximum(T_post, 1e-30))
        - jnp.log(jnp.maximum(T_pre,  1e-30))
    )

    # paper criterion: Δlog T >= log(T2/T1)|_Mmin  AND  Δlog P >= log(P2/P1)|_Mmin
    return (log_p_jump >= log_p_min) & (log_T_jump >= log_T_min)



@partial(jax.jit, static_argnames=["registered_variables", "config"])
def identify_shock_zones(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    shock_direction: FIELD_TYPE,
    mach_min: float = 1.3
) -> BOOL_FIELD_TYPE:
    """
    Identify all cells in shock zones.
    
    A cell is in a shock zone if ALL three criteria are met:
    1. Converging flow: ∇·v < 0
    2. Aligned gradients: ∇T·∇ρ > 0
    3. Minimum Mach: Δlog T >= log(T2/T1)|_Mmin AND Δlog P >= log(P2/P1)|_Mmin
       (Pfrommer et al. 2017, jumps measured along shock_direction)
    
    Results in a shock zone thickness of ~3-4 cells per shock (Pfrommer et al. 2017).
    
    Args:
        primitive_state: Primitive state variables
        config: Simulation configuration
        registered_variables: Registered variables
        helper_data: Helper data
        shock_direction: Unit vector field d_s = -∇T/|∇T|, from Phase 1
        mach_min: Minimum Mach number threshold
    
    Returns:
        Boolean field marking all shock zone cells
    """
    pressure = primitive_state[registered_variables.pressure_index]
    density  = primitive_state[registered_variables.density_index]
    velocity = primitive_state[registered_variables.velocity_index]

    # get radial coordinates if needed
    r = helper_data.geometric_centers if config.geometry == SPHERICAL else None

    # evaluate all three criteria
    criterion_1 = _shock_zone_criterion_converging_flow(velocity, config, r)
    criterion_2 = _shock_zone_criterion_aligned_gradients(pressure, density, config, r)
    criterion_3 = _shock_zone_criterion_minimum_mach(
        primitive_state, config,registered_variables, helper_data,
        shock_direction,   # ← direction-aware jump measurement
        mach_min
    )

    # combine with AND logic (all must be satisfied)
    shock_zones = criterion_1 & criterion_2 & criterion_3

    return shock_zones