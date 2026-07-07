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
def get_post_pre_shock_values(
    shock_direction,
    field_a,
    field_b,
    max_steps=1,
):
    """
    Sample two scalar fields on both sides of a candidate shock.

    The shock direction points from the hot/post-shock side toward the
    cold/pre-shock side. For every cell, the dominant component of the
    shock-direction vector determines which grid axis is used for sampling.

    Args:
        shock_direction:
            Unit-vector field with shape (ndim, *spatial_shape).

        field_a:
            First scalar field to sample, for example pressure.

        field_b:
            Second scalar field to sample, for example temperature or density.

        max_steps:
            Number of grid cells to move away from the candidate shock cell.

    Returns:
        field_a_post:
            field_a sampled on the post-shock side.

        field_a_pre:
            field_a sampled on the pre-shock side.

        field_b_post:
            field_b sampled on the post-shock side.

        field_b_pre:
            field_b sampled on the pre-shock side.
    """

    dominant_axis = jnp.argmax(
        jnp.abs(shock_direction),
        axis=0,
    )

    dominant_direction = jnp.take_along_axis(
        shock_direction,
        dominant_axis[jnp.newaxis],
        axis=0,
    )[0]

    step_sign = jnp.sign(
        dominant_direction
    ).astype(jnp.int32)

    ndim = field_a.ndim

    def shift_field(field, shift, axis):
        """
        Move a scalar field by a fixed number of cells.

        Note:
            jnp.roll wraps around at domain boundaries. Boundary cells must
            therefore be masked elsewhere before the sampled values are used.
        """

        shifted_field = field

        for _ in range(max_steps):
            shifted_field = jnp.roll(
                shifted_field,
                shift=shift,
                axis=axis,
            )

        return shifted_field

    # Default values are the original cell values. They are replaced only
    # along the locally selected dominant shock axis.
    field_a_post = field_a
    field_a_pre = field_a
    field_b_post = field_b
    field_b_pre = field_b

    for axis in range(ndim):
        uses_this_axis = dominant_axis == axis

        points_in_positive_direction = (
            uses_this_axis
            & (step_sign > 0)
        )

        points_in_negative_direction = (
            uses_this_axis
            & (step_sign < 0)
        )

        # If the shock direction points in the positive axis direction,
        # the pre-shock gas is ahead (+axis), while the post-shock gas is
        # behind (-axis).
        field_a_post_positive = shift_field(field_a, +1, axis)
        field_a_pre_positive  = shift_field(field_a, -1, axis)

        field_b_post_positive = shift_field(field_b, +1, axis)
        field_b_pre_positive  = shift_field(field_b, -1, axis)

        # Reverse the sampling sides if the shock direction points
        # in the negative axis direction.
        field_a_post_negative = shift_field(field_a, -1, axis)
        field_a_pre_negative  = shift_field(field_a, +1, axis)

        field_b_post_negative = shift_field(field_b, -1, axis)
        field_b_pre_negative  = shift_field(field_b, +1, axis)

        field_a_post = jnp.where(
            points_in_positive_direction,
            field_a_post_positive,
            jnp.where(
                points_in_negative_direction,
                field_a_post_negative,
                field_a_post,
            ),
        )

        field_a_pre = jnp.where(
            points_in_positive_direction,
            field_a_pre_positive,
            jnp.where(
                points_in_negative_direction,
                field_a_pre_negative,
                field_a_pre,
            ),
        )

        field_b_post = jnp.where(
            points_in_positive_direction,
            field_b_post_positive,
            jnp.where(
                points_in_negative_direction,
                field_b_post_negative,
                field_b_post,
            ),
        )

        field_b_pre = jnp.where(
            points_in_positive_direction,
            field_b_pre_positive,
            jnp.where(
                points_in_negative_direction,
                field_b_pre_negative,
                field_b_pre,
            ),
        )

    return (
        field_a_post,
        field_a_pre,
        field_b_post,
        field_b_pre,
    )

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