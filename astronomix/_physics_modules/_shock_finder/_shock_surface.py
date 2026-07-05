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

from astronomix._physics_modules._shock_finder._gradients import _calculate_velocity_divergence


"""
Identify the shock surface for 1D
* consider a cell X that already identified as the Shock Surface
    * walking along the ray,
        via label scan to split into segments of contiguous shock zone cells
    * As you move through the cells of the shock zone, you check the velocity divergence of each cell
        * only need to get the minimum div_v cell in each segment
        * for example
            - Cell 1: Divergence = -10 
            - Cell 2: Divergence = -50 
            - Cell 3: Divergence = -100 (The Peak! Maximum compression)
            - Cell 4: Divergence = -30  
    The Result: tag Cell 3 as the "Shock Surface Cell” + ignore the others for final map of surface        
* do this for every cell of shock zone

Args:
    * div_v:       velocity divergence, shape (nx,)
    * shock_zones: boolean field, shape (nx,)

* ray direction: left to right
"""
def _find_shock_surface_1d(
    div_v: FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
) -> BOOL_FIELD_TYPE:

    n = div_v.shape[0]

    # Before can "walk along a ray within a shock zone" -> need to know which cells belong to the same contiguous zone
    # label contiguous shock zone segments
    # assigns each run of True cells a unique integer ID
    # F,F,T,T,T,F,T,T → 0,0,1,1,1,0,2,2

    # label_scan check label value for 1 cell
    def label_scan(carry, zone):
        # was previous cell in zone + what label are we on?
        prev_in_zone, prev_label = carry
        in_zone = zone
        
        # if entering a zone (current=True, previous=False) → increment label
        # otherwise → keep previous label
        current_label = jnp.where(
            in_zone & ~prev_in_zone,
            prev_label + 1,
            prev_label,
        )

        # cells outside zone output 0, cells inside output their segment label
        out_current_label = jnp.where(in_zone, current_label, 0)
        return (in_zone, current_label), out_current_label

    # scan through shock_zones to get segment IDs for all cells
    _, segment_ids = jax.lax.scan(
        label_scan,
        (jnp.bool_(False), jnp.int32(0)),
        shock_zones,
    )

    max_segment_id = jnp.max(segment_ids)

    # mask div_v to only consider shock zone cells, 
    # set others to +inf so they won't be chosen as surfaces
    div_v_masked_base = jnp.where(shock_zones, div_v, jnp.inf)

    # for each segment, find the cell with minimum div_v and mark it as surface
    def find_segment_surface(segment_id, shock_surface):
        in_segment = segment_ids == segment_id
        div_v_seg  = jnp.where(in_segment, div_v_masked_base, jnp.inf)

        # find the cell with minimum div_v in this segment
        surface_idx = jnp.argmin(div_v_seg)
        segment_exists = segment_id <= max_segment_id

        # update shock_surface at surface_idx if segment exists and is in shock zone
        shock_surface = shock_surface.at[surface_idx].set(
            shock_surface[surface_idx] | (in_segment[surface_idx] & segment_exists)
        )
        return shock_surface

    shock_surface_init = jnp.zeros(n, dtype=jnp.bool_)
    """
    jax.lax.fori_loop(start, stop, body_fn, init_val)
    =
    val = init_val
    for i in range(start, stop):
        val = body_fn(i, val)
    """
    shock_surface = jax.lax.fori_loop(
        1,
        max_segment_id,
        lambda segment_id, surf: find_segment_surface(jnp.int32(segment_id), surf),
        shock_surface_init,
    )

    any_shock_zones_exist = jnp.any(shock_zones)

    """
    with no shock zones:
    div_v_masked_base = all inf       # because shock_zones all False
    div_v_seg         = all inf       # same
    surface_idx       = argmin(inf)   = 0   # silent wrong result
    shock_surface[0]  = True          # spurious tag
    -> shock_surface ends up with a wrong value even though there were no zones
    -> need to check if any shock zones exist at the end and mask out shock_surface if not
    """
    return shock_surface & any_shock_zones_exist



"""
similar logic, but differnt in find the ray + how walk it

Identify the shock surface for 2D
* consider a cell X that already identified as the Shock Surface
    * ray direction must be computed per cell
        * Each cell has a shock_direction vector (ds_x, ds_y). First pick the dominant axis
        * this is design decision for computational purpose,
            as if we walk digonal, thing more complex
    * walking along the ray
        walk the ray per cell by ray direction
    * As you move through the cells of the shock zone, you check the velocity divergence of each cell   
* do this for every cell of shock zone

Args:
    * div_v:           velocity divergence, shape (nx, ny)
    * shock_zones:     boolean field, shape (nx, ny)
    * shock_direction: unit vector field, shape (2, nx, ny)
"""
def _find_shock_surface_2d(
    div_v: FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
) -> BOOL_FIELD_TYPE:
    nx, ny = shock_zones.shape

    """
    Compute the ray direction
    """
    # pick the dominant axis per cell 
    # -> return dominant_axis array for all cell 
    # dominant_axis[cell] -> 0 then x dominant, 1 then y dominant
    abs_ds = jnp.abs(shock_direction)  # shape (2, nx, ny)
    # jnp.argmax(..., axis=0) means we get the max between dimension 0 (the 2 components) for each cell
    dominant_axis = jnp.argmax(abs_ds, axis=0)  # shape (nx, ny)

    # get the step along dominant axis: +1 or -1
    # shock_direction[axis, i, j] gives the component along that axis
    # each cell gets its own (+1, 0) or (-1, 0) or (0, +1) or (0, -1)
    ds_x = shock_direction[0]
    ds_y = shock_direction[1]
    step_x = jnp.where(dominant_axis == 0, jnp.sign(ds_x).astype(jnp.int32), 0)
    step_y = jnp.where(dominant_axis == 1, jnp.sign(ds_y).astype(jnp.int32), 0)

    div_v_in_zone = jnp.where(shock_zones, div_v, jnp.inf)

    """
    Given ray direction (step_x, step_y) for each cell, 
    -> "walk along the ray" by per cell by ray direction
    -> so we loop per cell
        walk along the ray by step_x, step_y
        check div_v of each cell we walk through
        execute is_surface_cell logic to check if current cell is surface cell by comparing div_v along the ray

    * Cell (i,j) is a surface cell if (in the shock zone) + (has smallest div_v along the ray in its shock zone segment)
    """

    """
    For cell i j, 
    we get all cells satisfy (along the ray) + (ahead it or itself)
        do this by while loop to walk along the ray until still_in_zone false || hit boundary (max steps)
    then we check if any cell (along the ray + ahead it) has smaller div_v
        due to integration nature:
            if cell prior found_smaller = true -> previous cell not smallest 
                then all following cells will have found_smaller = true, no need to check further
                early stop possible, 
            else if prior have found_smaller = false -> previous cell is smallest
                then we check current cell's div_v with my_div to update found_smaller
    
    For example cell (i,j)
    → step 0: check (i, j) itself → found_smaller = False (initially)
    → step 1: check (i+sx, j+sy)      → found_smaller = False
    → step 2: check (i+2sx, j+2sy)    → found_smaller = False
    → step 3: check (i+3sx, j+3sy)    → found_smaller = True   ← stops caring, already True
    → step 4: ...                      → found_smaller = True   (can't go back to False)
    → ...up to max_steps
    """
    
    def is_surface_cell(i, j):
        my_div = div_v[i, j]
        sx = step_x[i, j]
        sy = step_y[i, j]

        # walk up to max(nx, ny) steps along the ray
        max_steps = int(max(nx, ny))

        # we get all cells satisfy (along the ray) + (ahead it or itself)
        # then we check if any cell (along the ray + ahead it) has smaller div_v
        def ray_step(carry, _):
            ci, cj, found_smaller = carry
            ni = jnp.clip(ci + sx, 0, nx - 1)
            nj = jnp.clip(cj + sy, 0, ny - 1)

            # stop if we left the shock zone or hit a boundary (same cell after clip)
            still_in_zone = shock_zones[ni, nj]
            hit_boundary = (ni != ci) | (nj != cj)
            active = still_in_zone & hit_boundary

            neighbor_div = jnp.where(active, div_v[ni, nj], jnp.inf)
            found_smaller = found_smaller | (neighbor_div < my_div)

            next_i = jnp.where(active, ni, ci)
            next_j = jnp.where(active, nj, cj)
            return (next_i, next_j, found_smaller), None

        # while loop to walk along the ray to calculate found_smaller for all cells along the ray ahead of (i, j)
        (_, _, found_smaller), _ = jax.lax.scan(
            ray_step,
            (i, j, jnp.bool_(False)),
            None,
            length=max_steps,   # ← static Python int, fine
        )

        # if found_smaller is False after walking through the ray -> (i, j) is the smallest
        return shock_zones[i, j] & ~found_smaller

    # calculate is_surface_cell for all cells loop by 2 nested vmap for dim 2
    i_idx = jnp.arange(nx)
    j_idx = jnp.arange(ny)
    ii, jj = jnp.meshgrid(i_idx, j_idx, indexing="ij")  # (nx, ny) each

    shock_surface = jax.vmap(
        jax.vmap(is_surface_cell, in_axes=(0, 0)),
        in_axes=(0, 0),
    )(ii, jj)

    return shock_surface & jnp.any(shock_zones)




@partial(jax.jit, static_argnames=["config", "registered_variables"])
def identify_shock_surface(
    primitive_state: STATE_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> BOOL_FIELD_TYPE:
    div_v = _calculate_velocity_divergence(primitive_state, config, registered_variables)

    if config.dimensionality == 1:
        # div_v shape: (nx,), shock_direction shape: (1, nx)
        return _find_shock_surface_1d(div_v, shock_zones)

    elif config.dimensionality == 2:
        return _find_shock_surface_2d(div_v, shock_zones, shock_direction)

    else:
        raise NotImplementedError(
            f"Shock surface raycasting not implemented for dimensionality={config.dimensionality}"
        )