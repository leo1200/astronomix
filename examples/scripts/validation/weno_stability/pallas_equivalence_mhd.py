"""Pallas ideal-MHD WENO kernel vs native with the new options (interpret mode, x64)."""
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
from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_native_for_axis
from astronomix._finite_difference._interface_fluxes._weno_positivity import mhd_inflow_reference
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _mhd_pallas_flux_supported,
    _weno_flux_mhd_pallas,
)

GAMMA = 5.0 / 3.0
periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
n = 16
shape = (n, n, n)
key = jax.random.PRNGKey(5)
k1, k2, k3, k4 = jax.random.split(key, 4)
rho = jnp.exp(3.0 * jax.random.normal(k1, shape))
pressure = jnp.exp(2.0 * jax.random.normal(k2, shape))
velocity = jax.random.normal(k3, (3,) + shape)
magnetic = 0.5 * jax.random.normal(k4, (3,) + shape)
energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (velocity**2).sum(0) + 0.5 * (magnetic**2).sum(0)
state = jnp.concatenate([rho[None], rho[None] * velocity, energy[None], magnetic], axis=0)
internal = pressure / (GAMMA - 1.0)
worst = 0.0
for variant in ["base", "face", "pp"]:
    options = dict(base={}, face=dict(weno_admissible_face_state=True), pp=dict(weno_positivity_preserving=True))[variant]
    configs = {}
    for backend in (NATIVE_JAX, PALLAS):
        config = SimulationConfig(
            dimensionality=3, num_cells=n, mhd=True,
            boundary_settings=BoundarySettings(periodic, periodic, periodic),
            backend_config=BackendConfig(backend=backend, pallas_interpret=True, pallas_block_shape=(8, 8, 8)),
            solver_config=FDSolverConfig(**options),
        )
        registered_variables = get_registered_variables(config)
        configs[backend] = finalize_config(config, (registered_variables.num_vars,) + shape)
    params = SimulationParams(gamma=GAMMA, minimum_density=1e-10, minimum_pressure=1e-10)
    assert _mhd_pallas_flux_supported(state, configs[PALLAS])
    for axis in range(3):
        for dual in (False, True):
            g = internal if dual else None
            native = _weno_flux_native_for_axis(axis)(
                state, params, configs[NATIVE_JAX], registered_variables, internal_energy_density=g)
            pallas = _weno_flux_mhd_pallas(
                state, params, configs[PALLAS], registered_variables, axis=axis, internal_energy_density=g)
            scale = jnp.max(jnp.abs(native), axis=(1, 2, 3), keepdims=True)
            error = float(jnp.max(jnp.abs(pallas - native) / jnp.maximum(scale, 1e-30)))
            worst = max(worst, error)
            print(f"MHD {variant:4s} axis {axis} dual={int(dual)}: max |pallas - native| / max|native| = {error:.2e}", flush=True)
        if variant == "pp":
            # joint per-cell inflow limiting (the production path for ideal MHD + PP)
            reference = mhd_inflow_reference(state, params, configs[NATIVE_JAX], registered_variables)
            native = _weno_flux_native_for_axis(axis)(
                state, params, configs[NATIVE_JAX], registered_variables, inflow_reference=reference)
            pallas = _weno_flux_mhd_pallas(
                state, params, configs[PALLAS], registered_variables, axis=axis, inflow_reference=reference)
            scale = jnp.max(jnp.abs(native), axis=(1, 2, 3), keepdims=True)
            error = float(jnp.max(jnp.abs(pallas - native) / jnp.maximum(scale, 1e-30)))
            worst = max(worst, error)
            print(f"MHD pp-joint axis {axis}: max |pallas - native| / max|native| = {error:.2e}", flush=True)
print(f"WORST {worst:.2e}")
