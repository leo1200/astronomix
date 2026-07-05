from dataclasses import dataclass
from jax import tree_util
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE, INT_FIELD_TYPE, BOOL_FIELD_TYPE,
)

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
    * thermal_energy_flux: 
        float array, 
        nonzero only at surface cells; holds the energy flux through the shock
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
    thermal_energy_flux: FIELD_TYPE 
    shock_zones:         BOOL_FIELD_TYPE
    num_shocks:          int
    shock_ids:           INT_FIELD_TYPE
    shock_zone_ids:      INT_FIELD_TYPE


def _shockresult_flatten(result):
    children = (
        result.shock_surface_cells,
        result.shock_direction,
        result.mach_numbers,
        result.thermal_energy_flux,
        result.shock_zones,
        result.shock_ids,
        result.shock_zone_ids,
        result.num_shocks,
    )

    return children, None

def _shockresult_unflatten(aux, children):
    return ShockFinderResult(
        shock_surface_cells=children[0],
        shock_direction=children[1],
        mach_numbers=children[2],
        thermal_energy_flux=children[3],
        shock_zones=children[4],
        shock_ids=children[5],
        shock_zone_ids=children[6],
        num_shocks=children[7],
    )

tree_util.register_pytree_node(
    ShockFinderResult,
    _shockresult_flatten,
    _shockresult_unflatten,
)