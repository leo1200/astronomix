# ============================================================================
# PHASE 3: SHOCK SURFACE REFINEMENT
# ============================================================================
# Refine shock zones to identify the shock surface: a single layer of cells
# with maximum compression (minimum velocity divergence) along the shock direction.

from functools import partial
import jax.numpy as jnp
import jax

# typing
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE, BOOL_FIELD_TYPE,
    STATE_TYPE,
    SimulationConfig,
)

import astronomix._physics_modules._shock_finder._helpers as helpers


@partial(jax.jit, static_argnames=["config"])
def _calculate_velocity_divergence(
    velocity: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None
) -> FIELD_TYPE:
    """
    Calculate velocity divergence ∇·v.
    
    This is the same as the gradient of velocity in 1D.
    Minimum (most negative) divergence indicates maximum compression.
    
    Args:
        velocity: Velocity field
        config: Simulation configuration
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Velocity divergence field
    """
    return helpers._calculate_gradient(velocity, config, r)

"""
shock direction in 1D has no use
in 2D, expectation is sth like this
def _find_shock_surface_raycasting_2d(
    velocity_x: FIELD_TYPE,
    velocity_y: FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction_x: FIELD_TYPE,   # x-component of d_s
    shock_direction_y: FIELD_TYPE,   # y-component of d_s
) -> BOOL_FIELD_TYPE:
div_v = _calculate_divergence_2d(velocity_x, velocity_y)

    # for each cell, determine step direction from d_s
    # dominant axis: whichever component of d_s has larger magnitude
    step_x = jnp.sign(shock_direction_x).astype(jnp.int32)
    step_y = jnp.sign(shock_direction_y).astype(jnp.int32)

    # walk from each cell along (step_x, step_y) while remaining in shock_zone
    # keep cell only if no subsequent cell along the ray has smaller div_v
    # ... (ray march via lax.while_loop per cell)
"""
def _find_shock_surface_raycasting(
    velocity: FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
) -> BOOL_FIELD_TYPE:
    """
    Identify shock surface cells using the ray-casting procedure from
    Pfrommer et al. 2017.

    For each cell in the shock zone, a ray is fired along d_s. The ray
    walks through consecutive shock zone cells in that direction. A cell
    is marked as a shock surface cell if it has the minimum velocity
    divergence (maximum compression) along its ray — i.e. no cell further
    along the ray has a more negative divergence.

    In 1D this reduces to: within each connected shock zone segment,
    find the single cell with minimum divergence.

    Args:
        velocity:        velocity field
        shock_zones:     boolean field from Phase 2, ~3-4 cells thick
        shock_direction: unit vector field d_s = -∇T/|∇T| from Phase 1

    Returns:
        boolean field, True only at shock surface cells (one per shock zone)
    """
    n = velocity.shape[0]

    # --- Step 1: compute velocity divergence everywhere ---
    # simplified central difference — denominator cancels when comparing
    div_v = jnp.zeros_like(velocity)
    div_v = div_v.at[1:-1].set((velocity[2:] - velocity[:-2]) / 2.0)

    # --- Step 2: label connected shock zone segments ---
    # each contiguous group of True cells in shock_zones gets a unique integer id
    # this allows us to find the argmin independently within each group
    #
    # method: a transition from False→True starts a new segment.
    # scan left to right, incrementing the label counter at each transition.
    #
    # example:
    #   shock_zones:  [F, F, T, T, T, F, F, T, T, F]
    #   segment_ids:  [0, 0, 1, 1, 1, 0, 0, 2, 2, 0]  (0 = not in zone)

    def label_scan(carry, x):
        prev_in_zone, current_label = carry
        in_zone = x
        # start new segment when transitioning False → True
        new_label = jnp.where(
            in_zone & ~prev_in_zone,
            current_label + 1,
            current_label
        )
        # assign label only inside zone, 0 outside
        out_label = jnp.where(in_zone, new_label, 0)
        return (in_zone, new_label), out_label

    _, segment_ids = jax.lax.scan(
        label_scan,
        (jnp.bool_(False), jnp.int32(0)),
        shock_zones
    )
    # segment_ids: shape (n,), dtype int32
    # 0 = not in any shock zone, 1,2,3,... = distinct shock zone segments

    # --- Step 3: find max number of segments ---
    # needed to loop over each segment independently
    # upper bound: at most n//2 segments (alternating T/F pattern)
    num_segments = jnp.max(segment_ids)

    # --- Step 4: for each segment, find the cell with minimum div_v ---
    # mask div_v per segment, take argmin within that mask

    # mask div_v: only keep values inside the zone, set others to +inf
    div_v_masked_base = jnp.where(shock_zones, div_v, jnp.inf)

    def find_segment_surface(seg_id, shock_surface):
        """
        For one segment (seg_id), mask to only that segment's cells,
        find argmin of div_v, mark it as surface cell.
        """
        # mask to this segment only
        in_segment = segment_ids == seg_id
        div_v_seg = jnp.where(in_segment, div_v_masked_base, jnp.inf)

        # argmin within segment → the surface cell
        surface_idx = jnp.argmin(div_v_seg)

        # mark it, but only if this segment actually exists
        # (seg_id might exceed actual num_segments if we loop to upper bound)
        segment_exists = seg_id <= num_segments
        shock_surface = shock_surface.at[surface_idx].set(
            shock_surface[surface_idx] | (in_segment[surface_idx] & segment_exists)
        )
        return shock_surface

    # loop over all possible segment ids (1-indexed)
    # use lax.fori_loop for JIT compatibility
    shock_surface_init = jnp.zeros(n, dtype=jnp.bool_)

    shock_surface = jax.lax.fori_loop(
        1,                          # start: first segment id
        n // 2 + 2,                 # stop:  upper bound on segments
        lambda seg_id, surf: find_segment_surface(jnp.int32(seg_id), surf),
        shock_surface_init
    )

    # --- Step 5: guard — if no shock zones exist, return all False ---
    has_shock = jnp.any(shock_zones)
    shock_surface = shock_surface & has_shock

    return shock_surface

@partial(jax.jit, static_argnames=["config"])
def identify_shock_surface(
    primitive_state: STATE_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> BOOL_FIELD_TYPE:
    """
    Identify the shock surface: single layer of cells with maximum compression,
    one per connected shock zone segment, using the ray-casting procedure from
    Pfrommer et al. 2017.

    Args:
        primitive_state:  primitive state variables
        shock_zones:      boolean field from Phase 2
        shock_direction:  unit vector field d_s from Phase 1
        config:           simulation configuration
        registered_variables: registered variables

    Returns:
        boolean field, True at shock surface cells only
    """
    velocity = primitive_state[registered_variables.velocity_index]
    shock_surface = _find_shock_surface_raycasting(velocity, shock_zones, shock_direction)
    return shock_surface