"""Confirm the Pallas FD WENO flux predicate actually fires (i.e. the AD
agreement is Pallas-vs-native, not native-vs-native via silent fallback)."""
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from astronomix import get_registered_variables
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _hydro_pallas_flux_supported,
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN, FINITE_DIFFERENCE, PALLAS, PERIODIC_BOUNDARY, PERIODIC_ROLL,
    BoundarySettings, BoundarySettings1D, SimulationConfig, finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)


def check(dim, N):
    if dim == 1:
        bs = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    else:
        bs = BoundarySettings(*([BoundarySettings1D(PERIODIC_BOUNDARY,
                                                    PERIODIC_BOUNDARY)] * 3))
    cfg = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, geometry=CARTESIAN, progress_bar=False,
        boundary_handling=PERIODIC_ROLL, num_ghost_cells=0, mhd=False,
        dimensionality=dim, box_size=1.0, num_cells=N, boundary_settings=bs,
        backend=PALLAS, pallas_block_shape=(4, 4, 8),
    )
    rv = get_registered_variables(cfg)
    shape = (rv.num_vars,) + (N,) * dim
    ones = jnp.ones((N,) * dim)
    ps = construct_primitive_state(
        cfg, rv, density=ones, gas_pressure=ones,
        **({"velocity_x": jnp.zeros_like(ones)} if dim == 1 else {
            "velocity_x": jnp.zeros_like(ones),
            "velocity_y": jnp.zeros_like(ones),
            "velocity_z": jnp.zeros_like(ones)}),
    )
    cfg = finalize_config(cfg, ps.shape)
    # The predicate only inspects ndim / spatial-shape divisibility, so the
    # primitive state (same ndim and spatial shape as the conserved state)
    # is sufficient to decide whether the Pallas kernel fires.
    sup = _hydro_pallas_flux_supported(ps, cfg)
    print(f"  dim={dim} N={N}: _hydro_pallas_flux_supported = {sup}")
    return sup


print("=== Pallas FD WENO support predicate ===")
ok1 = check(1, 64)
ok3 = check(3, 16)
print(f"RESULT: 1D={'PALLAS' if ok1 else 'FALLBACK'}  "
      f"3D={'PALLAS' if ok3 else 'FALLBACK'}")
