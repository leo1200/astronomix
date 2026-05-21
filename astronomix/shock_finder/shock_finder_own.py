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
    registered_variables: RegisteredVariables,
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
        primitive_state, registered_variables,
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


# Replace _find_shock_surface_simple entirely — remove its @jax.jit decorator
# and rewrite to avoid lax.cond

def _find_shock_surface_simple(
    velocity: FIELD_TYPE,
    shock_zones: FIELD_TYPE
) -> FIELD_TYPE:
    """
    Identify shock surface as the cell with minimum velocity divergence
    within each shock zone.
    
    Always-compute pattern: calculate argmin unconditionally, 
    then zero out result when no shock zones exist. This avoids
    TracerBoolConversionError from nested JIT + lax.cond interaction.
    """
    div_v = jnp.zeros_like(velocity)
    div_v = div_v.at[1:-1].set((velocity[2:] - velocity[:-2]) / 2.0)

    # Always compute — safe because argmin on all-inf is defined (returns 0)
    masked_div_v = jnp.where(shock_zones, div_v, jnp.inf)
    max_compression_idx = jnp.argmin(masked_div_v)
    
    shock_surface = jnp.zeros_like(shock_zones, dtype=jnp.bool_)
    shock_surface = shock_surface.at[max_compression_idx].set(True)
    
    # Only keep result if at least one shock zone cell exists
    has_shock = jnp.any(shock_zones)
    shock_surface = shock_surface & has_shock
    
    return shock_surface

@partial(jax.jit, static_argnames=["config"])
def identify_shock_surface(
    primitive_state: STATE_TYPE,
    shock_zones: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables
) -> FIELD_TYPE:
    """
    Identify the shock surface: single layer of cells with maximum compression.
    
    From Pfrommer et al. 2017:
    "Each ray stores the velocity divergence of the cell from which it started.
     If a ray traverses a cell with a smaller divergence, the ray is discarded.
     We call these cells with minimum velocity divergence (i.e., maximum compression)
     across the shock zone the shock surface cells."
    
    For 1D, we find cells where velocity divergence is minimum (most negative)
    within the shock zone.
    
    Args:
        primitive_state: Primitive state variables
        shock_zones: Boolean array marking shock zone cells
        config: Simulation configuration
        registered_variables: Registered variables
    
    Returns:
        Boolean array marking shock surface cells (single layer per shock zone)
    """
    velocity = primitive_state[registered_variables.velocity_index]
    
    # For now, use simple approach
    # TODO: Extend with connected component analysis for multiple shocks
    shock_surface = _find_shock_surface_simple(velocity, shock_zones)
    
    return shock_surface


# ============================================================================
# PHASE 4: MACH CALCULATION & STRUCTURED OUTPUT
# ============================================================================
# Calculate Mach numbers at shock surfaces and return structured result.


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _calculate_mach_at_surface(
    primitive_state: STATE_TYPE,
    shock_surface: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables
) -> FIELD_TYPE:
    gamma_gas = 5 / 3

    pressure = primitive_state[registered_variables.pressure_index]
    mach_array = jnp.zeros_like(pressure)

    # p2 = post-shock (left), p1 = pre-shock (right)
    p2 = pressure[:-2]
    p1 = pressure[2:]

    # Simple Rankine-Hugoniot — correct for pure gas, handles p2/p1=1 edge case
    p_ratio = jnp.maximum(p2 / jnp.maximum(p1, 1e-30), 1.0)
    M = jnp.sqrt((p_ratio * (gamma_gas + 1) + (gamma_gas - 1)) / (2 * gamma_gas))

    mach_array = mach_array.at[1:-1].set(
        jnp.where(shock_surface[1:-1], M, 0.0)
    )

    return mach_array


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
        primitive_state, config, registered_variables, helper_data, mach_min
    )
    
    # Phase 3: Identify shock surface
    shock_surface = identify_shock_surface(
        primitive_state, shock_zones, config, registered_variables
    )
    
    # Phase 4: Calculate Mach numbers at surface
    mach_numbers = _calculate_mach_at_surface(
        primitive_state, shock_surface, config, registered_variables
    )
    
    # Count number of distinct shocks (simplified: count connected components of shock_surface)
    # For 1D, this is straightforward
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


# ============================================================================
# LEGACY FUNCTIONS (KEPT FOR REFERENCE/TRANSITION)
# ============================================================================
# These functions were from the original implementation.
# They are being replaced by the new Pfrommer et al. methodology,
# but are kept here for reference and gradual migration.


@partial(jax.jit, static_argnames=["config"])
def _calculate_1d_divergence(
    field: FIELD_TYPE, config: SimulationConfig, r: FIELD_TYPE
) -> FIELD_TYPE:
    # calculate the 1d divergence by a simple
    # central difference approximation
    div_field = jnp.zeros_like(field)
    if config.geometry == CARTESIAN:
        div_field = div_field.at[1:-1].set(
            (field[2:] - field[:-2]) / (2 * config.grid_spacing)
        )
    elif config.geometry == SPHERICAL:
        div_field = jnp.zeros_like(field)
        # this is not exactly correct, as our field values are
        # defined at the volumetric not geometric cell centers etc
        # but should be fine for the shock finder
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
        + 13 / 12 * (pressure[2:] - 2 * pressure[1:-1] + pressure[:-2]) ** 2
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
    
    # --- Downstream State (Region 2) ---
    P2 = primitive_state[registered_variables.pressure_index, :-2]
    P2_CRs = P_CRs[:-2]
    P2_gas = P2 - P2_CRs  # Define gas pressure component
    
    # FIX: Compute energy density using (P / (gamma - 1))
    e2_gas = P2_gas / (gamma_gas - 1)  
    e2_crs = P2_CRs / (gamma_cr - 1)
    e2 = e2_gas + e2_crs
    rho2 = primitive_state[registered_variables.density_index, :-2]

    # --- Upstream State (Region 1) ---
    P1 = primitive_state[registered_variables.pressure_index, 2:]
    P1_CRs = P_CRs[2:]
    P1_gas = P1 - P1_CRs
    e1_gas = P1_gas / (gamma_gas - 1)
    e1_crs = P1_CRs / (gamma_cr - 1)
    e1 = e1_gas + e1_crs
    rho1 = primitive_state[registered_variables.density_index, 2:]

    # --- Consistent Effective Gamma Calculation ---
    # FIX: Use gas-only pressure (P_gas) for both gamma_eff1 and gamma_eff2
    gamma_eff1 = (gamma_cr * P1_CRs + gamma_gas * P1_gas) / P1
    gamma_eff2 = (gamma_cr * P2_CRs + gamma_gas * P2_gas) / P2

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

