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


# ============================================================================
# 1D RAYCASTING
# ============================================================================

def _find_shock_surface_raycasting_1d(
    div_v: FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
) -> BOOL_FIELD_TYPE:
    """
    1D raycasting: within each connected shock zone segment, find the single
    cell with minimum velocity divergence (maximum compression).

    Args:
        div_v:       velocity divergence, shape (nx,)
        shock_zones: boolean field, shape (nx,)

    Returns:
        boolean field, shape (nx,), True at one surface cell per segment
    """
    n = div_v.shape[0]

    # label contiguous shock zone segments: F,F,T,T,T,F,T,T → 0,0,1,1,1,0,2,2
    def label_scan(carry, x):
        prev_in_zone, current_label = carry
        in_zone = x
        new_label = jnp.where(
            in_zone & ~prev_in_zone,
            current_label + 1,
            current_label,
        )
        out_label = jnp.where(in_zone, new_label, 0)
        return (in_zone, new_label), out_label

    _, segment_ids = jax.lax.scan(
        label_scan,
        (jnp.bool_(False), jnp.int32(0)),
        shock_zones,
    )

    num_segments = jnp.max(segment_ids)
    div_v_masked_base = jnp.where(shock_zones, div_v, jnp.inf)

    def find_segment_surface(seg_id, shock_surface):
        in_segment = segment_ids == seg_id
        div_v_seg  = jnp.where(in_segment, div_v_masked_base, jnp.inf)
        surface_idx = jnp.argmin(div_v_seg)
        segment_exists = seg_id <= num_segments
        shock_surface = shock_surface.at[surface_idx].set(
            shock_surface[surface_idx] | (in_segment[surface_idx] & segment_exists)
        )
        return shock_surface

    shock_surface_init = jnp.zeros(n, dtype=jnp.bool_)
    shock_surface = jax.lax.fori_loop(
        1,
        n // 2 + 2,
        lambda seg_id, surf: find_segment_surface(jnp.int32(seg_id), surf),
        shock_surface_init,
    )

    return shock_surface & jnp.any(shock_zones)


# ============================================================================
# 2D RAYCASTING
# ============================================================================

def _find_shock_surface_raycasting_2d(
    div_v: FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
) -> BOOL_FIELD_TYPE:
    """
    2D raycasting: for each cell in the shock zone, fire a ray along the
    dominant axis of d_s. Walk through consecutive shock zone cells in that
    direction. A cell is a surface cell iff no further cell along its ray
    has a more negative divergence.

    Dominant axis at cell (i,j): argmax(|d_s_x|, |d_s_y|).
    Step direction: sign of d_s along the dominant axis.

    Args:
        div_v:           velocity divergence, shape (nx, ny)
        shock_zones:     boolean field, shape (nx, ny)
        shock_direction: unit vector field, shape (2, nx, ny)

    Returns:
        boolean field, shape (nx, ny)
    """
    nx, ny = shock_zones.shape

    # dominant axis per cell: 0=x, 1=y
    # shape (nx, ny)
    abs_ds = jnp.abs(shock_direction)          # (2, nx, ny)
    dominant_axis = jnp.argmax(abs_ds, axis=0) # (nx, ny)

    # step direction along dominant axis: +1 or -1
    # shock_direction[axis, i, j] gives the component along that axis
    ds_x = shock_direction[0]   # (nx, ny)
    ds_y = shock_direction[1]   # (nx, ny)
    step_x = jnp.where(dominant_axis == 0, jnp.sign(ds_x).astype(jnp.int32), 0)
    step_y = jnp.where(dominant_axis == 1, jnp.sign(ds_y).astype(jnp.int32), 0)

    div_v_in_zone = jnp.where(shock_zones, div_v, jnp.inf)

    def is_surface_cell(i, j):
        """
        Cell (i,j) is a surface cell if it is in the shock zone AND
        no cell further along the ray (in the step direction) within
        the shock zone has a smaller div_v.
        """
        my_div = div_v[i, j]
        sx = step_x[i, j]
        sy = step_y[i, j]

        # walk up to max(nx, ny) steps along the ray
        max_steps = jnp.maximum(nx, ny)

        def ray_step(carry, _):
            ci, cj, found_smaller = carry
            ni = jnp.clip(ci + sx, 0, nx - 1)
            nj = jnp.clip(cj + sy, 0, ny - 1)

            # stop if we left the shock zone or hit a boundary (same cell after clip)
            still_in_zone = shock_zones[ni, nj]
            moved = (ni != ci) | (nj != cj)
            active = still_in_zone & moved

            neighbor_div = jnp.where(active, div_v[ni, nj], jnp.inf)
            found_smaller = found_smaller | (neighbor_div < my_div)

            next_i = jnp.where(active, ni, ci)
            next_j = jnp.where(active, nj, cj)
            return (next_i, next_j, found_smaller), None

        (_, _, found_smaller), _ = jax.lax.scan(
            ray_step,
            (i, j, jnp.bool_(False)),
            None,
            length=max_steps,
        )

        return shock_zones[i, j] & ~found_smaller

    # vectorize over all cells using vmap
    i_idx = jnp.arange(nx)
    j_idx = jnp.arange(ny)
    ii, jj = jnp.meshgrid(i_idx, j_idx, indexing="ij")  # (nx, ny) each

    shock_surface = jax.vmap(
        jax.vmap(is_surface_cell, in_axes=(0, 0)),
        in_axes=(0, 0),
    )(ii, jj)

    return shock_surface & jnp.any(shock_zones)


# ============================================================================
# 3D RAYCASTING
# ============================================================================

def _find_shock_surface_raycasting_3d(
    div_v: FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
) -> BOOL_FIELD_TYPE:
    """
    3D raycasting: same logic as 2D but with three axes.

    Args:
        div_v:           velocity divergence, shape (nx, ny, nz)
        shock_zones:     boolean field, shape (nx, ny, nz)
        shock_direction: unit vector field, shape (3, nx, ny, nz)

    Returns:
        boolean field, shape (nx, ny, nz)
    """
    nx, ny, nz = shock_zones.shape

    abs_ds = jnp.abs(shock_direction)           # (3, nx, ny, nz)
    dominant_axis = jnp.argmax(abs_ds, axis=0)  # (nx, ny, nz)

    ds_x = shock_direction[0]
    ds_y = shock_direction[1]
    ds_z = shock_direction[2]

    step_x = jnp.where(dominant_axis == 0, jnp.sign(ds_x).astype(jnp.int32), 0)
    step_y = jnp.where(dominant_axis == 1, jnp.sign(ds_y).astype(jnp.int32), 0)
    step_z = jnp.where(dominant_axis == 2, jnp.sign(ds_z).astype(jnp.int32), 0)

    def is_surface_cell(i, j, k):
        my_div = div_v[i, j, k]
        sx = step_x[i, j, k]
        sy = step_y[i, j, k]
        sz = step_z[i, j, k]

        max_steps = jnp.maximum(jnp.maximum(nx, ny), nz)

        def ray_step(carry, _):
            ci, cj, ck, found_smaller = carry
            ni = jnp.clip(ci + sx, 0, nx - 1)
            nj = jnp.clip(cj + sy, 0, ny - 1)
            nk = jnp.clip(ck + sz, 0, nz - 1)

            still_in_zone = shock_zones[ni, nj, nk]
            moved = (ni != ci) | (nj != cj) | (nk != ck)
            active = still_in_zone & moved

            neighbor_div = jnp.where(active, div_v[ni, nj, nk], jnp.inf)
            found_smaller = found_smaller | (neighbor_div < my_div)

            next_i = jnp.where(active, ni, ci)
            next_j = jnp.where(active, nj, cj)
            next_k = jnp.where(active, nk, ck)
            return (next_i, next_j, next_k, found_smaller), None

        (_, _, _, found_smaller), _ = jax.lax.scan(
            ray_step,
            (i, j, k, jnp.bool_(False)),
            None,
            length=max_steps,
        )

        return shock_zones[i, j, k] & ~found_smaller

    i_idx = jnp.arange(nx)
    j_idx = jnp.arange(ny)
    k_idx = jnp.arange(nz)
    ii, jj, kk = jnp.meshgrid(i_idx, j_idx, k_idx, indexing="ij")

    shock_surface = jax.vmap(
        jax.vmap(
            jax.vmap(is_surface_cell, in_axes=(0, 0, 0)),
            in_axes=(0, 0, 0),
        ),
        in_axes=(0, 0, 0),
    )(ii, jj, kk)

    return shock_surface & jnp.any(shock_zones)


# ============================================================================
# PUBLIC INTERFACE
# ============================================================================

@partial(jax.jit, static_argnames=["config", "registered_variables"])
def identify_shock_surface(
    primitive_state: STATE_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> BOOL_FIELD_TYPE:
    """
    Identify the shock surface: single layer of maximum-compression cells,
    one per shock zone (Pfrommer et al. 2017).

    Dispatches to the correct raycasting implementation based on
    config.dimensionality.

    Args:
        primitive_state:      (num_vars, *spatial_shape)
        shock_zones:          boolean field, shape (*spatial_shape)
        shock_direction:      unit vector field, shape (ndim, *spatial_shape)
        config:               simulation configuration
        registered_variables: registry of variable indices

    Returns:
        boolean field, shape (*spatial_shape)
    """
    div_v = _calculate_velocity_divergence(primitive_state, config, registered_variables)

    if config.dimensionality == 1:
        # div_v shape: (nx,), shock_direction shape: (1, nx)
        return _find_shock_surface_raycasting_1d(div_v, shock_zones)

    elif config.dimensionality == 2:
        return _find_shock_surface_raycasting_2d(div_v, shock_zones, shock_direction)

    elif config.dimensionality == 3:
        return _find_shock_surface_raycasting_3d(div_v, shock_zones, shock_direction)

    else:
        raise NotImplementedError(
            f"Shock surface raycasting not implemented for dimensionality={config.dimensionality}"
        )