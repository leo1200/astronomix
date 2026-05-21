# general
from functools import partial
from dataclasses import dataclass
import jax.numpy as jnp
import jax
from jax import tree_util

# typing
from typing import Tuple, Union, NamedTuple
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FIELD_TYPE, INT_FIELD_TYPE, BOOL_FIELD_TYPE,
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)
from jaxtyping import Array, Int, jaxtyped
from beartype import beartype as typechecker

from astronomix.option_classes.simulation_params import SimulationParams


@dataclass
class ShockFinderResult:
    """
    each field is a snapshot of a different layer of the shock analysis:
    * shock_surface_cells: 
        boolean array, 
        one True per shock (the single cell of maximum compression)
    * shock_direction: 
        the unit vector field 
        d_s = -∇T/|∇T| at every cell, pointing from hot toward cold gas
    * mach_numbers: 
        float array, 
        nonzero only at surface cells; holds the Rankine-Hugoniot Mach number
    * shock_zones: 
        boolean array 
        marking the broader 3-4 cell thick region around each shock
    * num_shocks: 
        scalar int32, 
        currently just sum(shock_surface_cells), so one count per surface cell
    * shock_ids: 
        integer array, 
        1 where shock_surface is True, 0 elsewhere (stub for future multi-shock labeling)
    * shock_zone_ids: same idea but for the broader zone
    """
    shock_surface_cells: BOOL_FIELD_TYPE
    shock_direction:     FIELD_TYPE
    mach_numbers:        FIELD_TYPE 
    shock_zones:         BOOL_FIELD_TYPE
    num_shocks:          int
    shock_ids:           INT_FIELD_TYPE
    shock_zone_ids:      INT_FIELD_TYPE


def _shockresult_flatten(r):
    children = (
        r.shock_surface_cells,
        r.shock_direction,
        r.mach_numbers,
        r.shock_zones,
        r.shock_ids,
        r.shock_zone_ids,
        r.num_shocks,          # moved from aux to children
    )
    aux = None                 # nothing truly static here
    return children, aux

def _shockresult_unflatten(aux, children):
    return ShockFinderResult(
        shock_surface_cells=children[0],
        shock_direction=children[1],
        mach_numbers=children[2],
        shock_zones=children[3],
        shock_ids=children[4],
        shock_zone_ids=children[5],
        num_shocks=children[6],
    )

tree_util.register_pytree_node(
    ShockFinderResult,
    _shockresult_flatten,
    _shockresult_unflatten,
)

# ==========================
# PHASE 1: HELPER FUNCTIONS
# ==========================

@partial(jax.jit, static_argnames=["config"])
def _calculate_gradient(
    field: FIELD_TYPE, 
    config: SimulationConfig, 
    r: FIELD_TYPE = None
) -> FIELD_TYPE:
    """
    Calculate the spatial gradient of a scalar field using central differences.
    
    For Cartesian geometry:
        grad[i] = (field[i+1] - field[i-1]) / (2 * dx)
    
    For Spherical geometry (1D):
        grad[i] = (r[i+1]^2 * field[i+1] - r[i-1]^2 * field[i-1]) / (2 * dx * r[i]^2)
    
    Args:
        field: Scalar field to differentiate
        config: Simulation configuration, only needed for geometry type and grid spacing
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Gradient field (boundaries set to zero)
    """
    grad_field = jnp.zeros_like(field)
    
    if config.geometry == CARTESIAN:
        # grad_field[i] is immutable, so we use .at[].set() to update the interior points
        # grad_field[1:-1] means consider all points except the first and last (boundary points)
        grad_field = grad_field.at[1:-1].set(
            (field[2:] - field[:-2]) / (2 * config.grid_spacing)
        )
    elif config.geometry == SPHERICAL:
        if r is None:
            raise ValueError("Radial coordinates r required for spherical geometry")
        grad_field = grad_field.at[1:-1].set(
            (r[2:] ** 2 * field[2:] - r[:-2] ** 2 * field[:-2])
            / (2 * config.grid_spacing * r[1:-1] ** 2)
        )
    else:
        raise NotImplementedError(
            "Only Cartesian and Spherical geometry supported for shock finder."
        )
    
    return grad_field


@partial(jax.jit, static_argnames=["config"])
def _calculate_temperature_gradient(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None
) -> FIELD_TYPE:
    """
    Computes ∇T across the grid
    
    T is not the true thermodynamic temperature but a pseudo-temperature
    defined as T_eff[i] = P[i] / ρ[i]
    
    then gradient is computed via _calculate_gradient for T_eff field
    
    Args:
        pressure: Gas pressure field
        density: Density field
        config: Simulation configuration
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Temperature gradient field
    """
    pseudo_temperature = pressure / density
    grad_T = _calculate_gradient(pseudo_temperature, config, r)
    return grad_T


@partial(jax.jit, static_argnames=["config"])
def _calculate_density_gradient(
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None
) -> FIELD_TYPE:
    """
    Calculate the density gradient ∇ρ across the grid
    by using _calculate_gradient on the density field.
    
    Args:
        density: Density field
        config: Simulation configuration
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Density gradient field
    """
    grad_rho = _calculate_gradient(density, config, r)
    return grad_rho


@partial(jax.jit)
def _normalize_vector(
    vector: FIELD_TYPE,
    epsilon: float = 1e-12
) -> FIELD_TYPE:
    """
    Converts a vector field to unit magnitude everywhere:
    
    normalized[i] = vector[i] / (|vector[i]| + epsilon)
    
    The epsilon prevents division by zero when the vector is near zero.
    
    Args:
        vector: Vector field to normalize
        epsilon: Small constant for numerical stability
    
    Returns:
        Normalized vector field (magnitude 1 except where input is near-zero)
    """
    ## TODO: do this for 2D and 3D also, now only works for 1D scalar fields (shock direction in 1D is just a sign)
    magnitude = jnp.abs(vector) + epsilon
    normalized = vector / magnitude
    return normalized


@partial(jax.jit, static_argnames=["config"])
def _calculate_shock_direction(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None
) -> FIELD_TYPE:
    """
    Calculate the shock direction vector at each cell.
    Shock direction points from the hot (shocked) gas toward the cold (pre-shocked) gas.
    
    The shock direction is defined as:
        d_s = -∇T / |∇T|
    ∇T from _calculate_temperature_gradient with given pressure and density fields.

    step: compute ∇T, negate it, normalize it
    
    Physical interpretation:
        - Negative gradient means temperature decreases in that direction
        - The negative sign points in the direction of decreasing temperature
    
    Args:
        pressure: Gas pressure field
        density: Density field
        config: Simulation configuration
        r: Radial coordinates (required for spherical geometry)
    
    Returns:
        Normalized shock direction field (unit vector at each cell)
    """
    grad_T = _calculate_temperature_gradient(pressure, density, config, r)
    
    # Shock direction is -∇T (points toward cold gas)
    shock_dir = -grad_T
    
    shock_dir_normalized = _normalize_vector(shock_dir)
    
    return shock_dir_normalized


# ============================================================================
# PHASE 2: SHOCK ZONE IDENTIFICATION
# ============================================================================
#  goal is to produce a boolean field marking every cell that belongs to a shock zone
# 
# Identify cells that are in the shock zone using three criteria:
# 1. Converging flow: ∇·v < 0
# 2. Aligned gradients: ∇T·∇ρ > 0
# 3. Minimum Mach number: M > M_min = 1.3


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
    div_v = _calculate_gradient(velocity, config, r)
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
    grad_T = _calculate_temperature_gradient(pressure, density, config, r)
    grad_rho = _calculate_density_gradient(density, config, r)
    
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


# ============================================================================
# PHASE 3: SHOCK SURFACE REFINEMENT
# ============================================================================
# Refine shock zones to identify the shock surface: a single layer of cells
# with maximum compression (minimum velocity divergence) along the shock direction.


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
    return _calculate_gradient(velocity, config, r)

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


# ============================================================================
# PHASE 4: MACH CALCULATION & STRUCTURED OUTPUT
# ============================================================================
# Calculate Mach numbers at shock surfaces and return structured result.


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _calculate_mach_at_surface(
    primitive_state: STATE_TYPE,
    shock_surface: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,        # ← needed for direction-aware p_post/p_pre
    config: SimulationConfig,
    registered_variables: RegisteredVariables
) -> FIELD_TYPE:
    gamma_gas = 5 / 3

    pressure = primitive_state[registered_variables.pressure_index]
    density  = primitive_state[registered_variables.density_index]
    temperature = pressure / density

    # direction-aware post/pre selection — same helper as criterion 3
    p_post, p_pre, _, _ = get_post_pre_shock_values(
        shock_direction, pressure, temperature
    )

    # calculate Mach number for all cells
    # p₂/p₁ = p_post/p_pre, but clamp to 1 to avoid numerical issues with very weak shocks
    p_ratio = jnp.maximum(p_post / jnp.maximum(p_pre, 1e-30), 1.0)
    # as p₂/p₁ = (2γM² − (γ−1)) / (γ+1) so M = √[ (p₂/p₁ · (γ+1) + (γ−1)) / (2γ) ]
    M = jnp.sqrt((p_ratio * (gamma_gas + 1) + (gamma_gas - 1)) / (2 * gamma_gas))

    # write Mach only at surface cells, zero elsewhere
    mach_array = jnp.zeros_like(pressure)
    mach_array = mach_array.at[1:-1].set(
        jnp.where(shock_surface[1:-1], M, 0.0)
    )

    return mach_array

##### public entry point ######
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def find_shocks_pfrommer(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    mach_min: float = 1.3
) -> ShockFinderResult:
    """
    Main entry point: Identify shocks using Pfrommer et al. 2017 methodology.
    
    This orchestrates all phases:
    1. Calculate shock direction
    2. Identify shock zones (3-4 cells thick)
    3. Refine to shock surface (single layer)
    4. Calculate Mach numbers
    5. Return structured result
    
    Args:
        primitive_state: Primitive state variables [num_variables, num_cells]
        config: Simulation configuration
        registered_variables: Registry of variable indices
        helper_data: Helper data (geometric centers, etc.)
        mach_min: Minimum Mach number threshold (default 1.3)
    
    Returns:
        ShockFinderResult with shock metrics
    """
    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    r = helper_data.geometric_centers if config.geometry == SPHERICAL else None
    
    # Phase 1: Calculate shock direction
    shock_direction = _calculate_shock_direction(pressure, density, config, r)
    
    # Phase 2: Identify shock zones
    shock_zones = identify_shock_zones(
        primitive_state, config, registered_variables,
        helper_data, shock_direction, mach_min
    )
    
    # Phase 3: Identify shock surface
    shock_surface = identify_shock_surface(
        primitive_state, shock_zones, shock_direction,
        config, registered_variables
    )
    
    # Phase 4: Calculate Mach numbers at surface
    mach_numbers = _calculate_mach_at_surface(
        primitive_state, shock_surface, shock_direction,
        config, registered_variables
    )
    
    # num_shocks counts distinct surface cells
    num_shocks = jnp.sum(shock_surface, dtype=jnp.int32)
    
    # Shock IDs: for now, single label per cell on surface
    shock_ids = jnp.where(shock_surface, 1, 0)
    shock_zone_ids = jnp.where(shock_zones, 1, 0)
    
    return ShockFinderResult(
        shock_surface_cells=shock_surface,
        shock_direction=shock_direction,
        mach_numbers=mach_numbers,
        shock_zones=shock_zones,
        num_shocks=num_shocks,
        shock_ids=shock_ids,
        shock_zone_ids=shock_zone_ids
    )