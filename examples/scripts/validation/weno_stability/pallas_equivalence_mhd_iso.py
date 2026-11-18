"""Pallas isothermal-MHD WENO kernel vs native with PP-WENO (interpret mode)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import jax
import jax.numpy as jnp

from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    SimulationConfig,
    SimulationParams,
    finalize_config,
    get_registered_variables,
)
from astronomix.option_classes.simulation_config import ISOTHERMAL
from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_native_for_axis
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _mhd_iso_pallas_flux_supported,
    _weno_flux_mhd_iso_pallas,
)

periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
n = 16
shape = (n, n, n)
vscale = float(os.environ.get("EQUIV_VSCALE", "1.0"))
key = jax.random.PRNGKey(3)
k1, k2, k3 = jax.random.split(key, 3)
rho = jnp.exp(3.0 * jax.random.normal(k1, shape))
velocity = vscale * jax.random.normal(k2, (3,) + shape)
magnetic = 0.5 * jax.random.normal(k3, (3,) + shape)
worst = 0.0
for variant in ["base", "pp"]:
    options = {} if variant == "base" else dict(weno_positivity_preserving=True)
    configs = {}
    for backend in (NATIVE_JAX, PALLAS):
        config = SimulationConfig(
            dimensionality=3, num_cells=n, mhd=True, equation_of_state=ISOTHERMAL,
            boundary_settings=BoundarySettings(periodic, periodic, periodic),
            backend_config=BackendConfig(backend=backend, pallas_interpret=True, pallas_block_shape=(8, 8, 8)),
            solver_config=FDSolverConfig(**options),
        )
        registered_variables = get_registered_variables(config)
        configs[backend] = finalize_config(config, (registered_variables.num_vars,) + shape)
    params = SimulationParams(isothermal_sound_speed=0.3, minimum_density=1e-10)
    state = jnp.concatenate([rho[None], rho[None] * velocity, magnetic], axis=0)
    assert _mhd_iso_pallas_flux_supported(state, configs[PALLAS])
    for axis in range(3):
        native = _weno_flux_native_for_axis(axis)(state, params, configs[NATIVE_JAX], registered_variables)
        pallas = _weno_flux_mhd_iso_pallas(state, params, configs[PALLAS], registered_variables, axis=axis)
        scale = jnp.max(jnp.abs(native), axis=(1, 2, 3), keepdims=True)
        error = float(jnp.max(jnp.abs(pallas - native) / jnp.maximum(scale, 1e-30)))
        worst = max(worst, error)
        print(f"iso-MHD {variant:4s} axis {axis}: max |pallas - native| / max|native| = {error:.2e}", flush=True)
print(f"WORST {worst:.2e}")
