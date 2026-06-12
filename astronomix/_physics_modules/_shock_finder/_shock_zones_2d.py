# ============================================================================
# PHASE 2: SHOCK ZONE IDENTIFICATION
# ============================================================================

from functools import partial
import jax.numpy as jnp
import jax

from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE, BOOL_FIELD_TYPE,
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)
from astronomix._physics_modules._shock_finder._gradients import (
    _calculate_velocity_divergence,
    _calculate_temperature_gradient,
    _calculate_density_gradient,
)


"""
Criterion 1: Converging flow (∇·v < 0).
"""
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _shock_zone_criterion_converging_flow(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    r: FIELD_TYPE = None,
) -> BOOL_FIELD_TYPE:
    div_v = _calculate_velocity_divergence(primitive_state, config, registered_variables, r)
    return div_v < 0


"""
Criterion 2: Aligned gradients (∇T · ∇ρ > 0).
"""
@partial(jax.jit, static_argnames=["config"])
def _shock_zone_criterion_aligned_gradients(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
) -> BOOL_FIELD_TYPE:
    grad_T   = _calculate_temperature_gradient(pressure, density, config, r)
    grad_rho = _calculate_density_gradient(density, config, r)

    # dot product over the ndim axis (axis=0 of the vector fields)
    dot_product = jnp.sum(grad_T * grad_rho, axis=0)
    return dot_product > 0


"""
Criterion 3: Minimum Mach number
* pick minimum Mach number
* For each cell, 
    look at the two neighbors along the shock direction (one on each side), 
    compute the pressure and temperature jumps across them, 
    -> get_post_pre_shock_values

    and check if those jumps are large enough to correspond to a shock of at least Mach mach_min
"""
def get_post_pre_shock_values(shock_direction, pressure, temperature):
    
    # dominant axis per cell
    dominant_axis = jnp.argmax(jnp.abs(shock_direction), axis=0)  # (*spatial_shape)

    # gather the d_s component along the dominant axis
    # shapes: shock_direction (ndim, *spatial), dominant_axis (*spatial)
    # use jnp.take_along_axis with an extra leading dim
    ds_dominant = jnp.take_along_axis(
        shock_direction,
        dominant_axis[jnp.newaxis],   # (1, *spatial)
        axis=0,
    )[0]   # (*spatial_shape)

    step = jnp.sign(ds_dominant).astype(jnp.int32)  # (*spatial_shape), +1 or -1

    ndim = pressure.ndim

    def shift(field, s, axis):
        """Roll field by -s along axis (post-shock) or +s (pre-shock)."""
        return jnp.roll(field, shift=-s, axis=axis)

    # build post/pre for each possible dominant axis and select
    # using jnp.where broadcast over spatial dims
    p_post = pressure
    p_pre  = pressure
    T_post = temperature
    T_pre  = temperature

    for ax in range(ndim):
        is_dominant = (dominant_axis == ax)  # (*spatial_shape)

        # shift by -step along ax → post-shock (upstream, hot side)
        # shift by +step along ax → pre-shock  (downstream, cold side)
        # step varies per cell; we handle +1 and -1 cases separately
        # since jnp.roll requires a scalar shift

        # post-shock: cell in the direction shock came FROM → shift by -step
        # step=+1 means post is at i-1 → roll by +1
        # step=-1 means post is at i+1 → roll by -1
        p_post_ax_fwd = jnp.roll(pressure,     shift= 1, axis=ax)  # step=+1 case
        p_post_ax_bwd = jnp.roll(pressure,     shift=-1, axis=ax)  # step=-1 case
        p_pre_ax_fwd  = jnp.roll(pressure,     shift=-1, axis=ax)
        p_pre_ax_bwd  = jnp.roll(pressure,     shift= 1, axis=ax)

        T_post_ax_fwd = jnp.roll(temperature,  shift= 1, axis=ax)
        T_post_ax_bwd = jnp.roll(temperature,  shift=-1, axis=ax)
        T_pre_ax_fwd  = jnp.roll(temperature,  shift=-1, axis=ax)
        T_pre_ax_bwd  = jnp.roll(temperature,  shift= 1, axis=ax)

        is_fwd = is_dominant & (step > 0)   # dominant axis AND step=+1
        is_bwd = is_dominant & (step < 0)   # dominant axis AND step=-1

        p_post = jnp.where(is_fwd, p_post_ax_fwd,
                 jnp.where(is_bwd, p_post_ax_bwd, p_post))
        p_pre  = jnp.where(is_fwd, p_pre_ax_fwd,
                 jnp.where(is_bwd, p_pre_ax_bwd,  p_pre))
        T_post = jnp.where(is_fwd, T_post_ax_fwd,
                 jnp.where(is_bwd, T_post_ax_bwd, T_post))
        T_pre  = jnp.where(is_fwd, T_pre_ax_fwd,
                 jnp.where(is_bwd, T_pre_ax_bwd,  T_pre))

    return p_post, p_pre, T_post, T_pre


def _make_interior_mask(spatial_shape):
    """
    Build a boolean mask that is True for interior cells (not on any boundary).
    Shape: spatial_shape.
    """
    mask = jnp.ones(spatial_shape, dtype=jnp.bool_)
    for ax in range(len(spatial_shape)):
        sl_first = [slice(None)] * len(spatial_shape)
        sl_last  = [slice(None)] * len(spatial_shape)
        sl_first[ax] = 0
        sl_last[ax]  = -1
        mask = mask.at[tuple(sl_first)].set(False)
        mask = mask.at[tuple(sl_last)].set(False)
    return mask


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _shock_zone_criterion_minimum_mach(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    shock_direction: FIELD_TYPE,
    mach_min: float = 1.3,
) -> BOOL_FIELD_TYPE:
    gamma_gas = 5 / 3
    pressure    = primitive_state[registered_variables.pressure_index]
    density     = primitive_state[registered_variables.density_index]
    temperature = pressure / density

    # Rankine-Hugoniot thresholds at mach_min
    M2          = mach_min ** 2
    p_ratio_min = (2 * gamma_gas * M2 - (gamma_gas - 1)) / (gamma_gas + 1)
    T_ratio_min = p_ratio_min * ((gamma_gas - 1) * M2 + 2) / ((gamma_gas + 1) * M2)
    log_p_min   = jnp.log(p_ratio_min)
    log_T_min   = jnp.log(T_ratio_min)

    p_post, p_pre, T_post, T_pre = get_post_pre_shock_values(
        shock_direction, pressure, temperature
    )

    log_p_jump = jnp.log(jnp.maximum(p_post, 1e-30)) - jnp.log(jnp.maximum(p_pre, 1e-30))
    log_T_jump = jnp.log(jnp.maximum(T_post, 1e-30)) - jnp.log(jnp.maximum(T_pre, 1e-30))

    # zero out boundary cells (jnp.roll wraps around, those values are meaningless)
    interior = _make_interior_mask(pressure.shape)
    log_p_jump = jnp.where(interior, log_p_jump, 0.0)
    log_T_jump = jnp.where(interior, log_T_jump, 0.0)

    return (log_p_jump >= log_p_min) & (log_T_jump >= log_T_min)


# ============================================================================
# PUBLIC INTERFACE
# ============================================================================

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def identify_shock_zones(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    shock_direction: FIELD_TYPE,
    mach_min: float = 1.3,
) -> BOOL_FIELD_TYPE:
    """
    Identify all cells in shock zones (criteria 1 AND 2 AND 3).
    Results in ~3-4 cell thick zones per shock (Pfrommer et al. 2017).

    Args:
        primitive_state:      (num_vars, *spatial_shape)
        config:               simulation configuration
        registered_variables: registry of variable indices
        helper_data:          geometric centers etc.
        shock_direction:      unit vector field (ndim, *spatial_shape)
        mach_min:             minimum Mach threshold

    Returns:
        Boolean field, shape (*spatial_shape)
    """
    pressure = primitive_state[registered_variables.pressure_index]
    density  = primitive_state[registered_variables.density_index]
    r = helper_data.geometric_centers if config.geometry == SPHERICAL else None

    criterion_1 = _shock_zone_criterion_converging_flow(
        primitive_state, config, registered_variables, r
    )
    criterion_2 = _shock_zone_criterion_aligned_gradients(pressure, density, config, r)
    criterion_3 = _shock_zone_criterion_minimum_mach(
        primitive_state, config, registered_variables, helper_data,
        shock_direction, mach_min,
    )

    return criterion_1 & criterion_2 & criterion_3