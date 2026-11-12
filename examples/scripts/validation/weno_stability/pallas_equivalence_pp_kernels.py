"""The fused PP-recombination and inflow-reference kernels against the array
forms (interpret mode, x64, a rough 3D ideal-MHD state).

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/pallas_equivalence_pp_kernels.py
"""
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
from astronomix._finite_difference._interface_fluxes._weno_pallas import _weno_flux_mhd_pallas
from astronomix._finite_difference._interface_fluxes._weno_positivity import mhd_inflow_reference
from astronomix._finite_difference._interface_fluxes._weno_positivity_pallas import mhd_inflow_reference_pallas

GAMMA = 5.0 / 3.0
n = 16
shape = (n, n, n)
k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(7), 4)
rho = jnp.exp(3.0 * jax.random.normal(k1, shape))
pressure = jnp.exp(2.0 * jax.random.normal(k2, shape))
velocity = jax.random.normal(k3, (3,) + shape)
magnetic = 0.5 * jax.random.normal(k4, (3,) + shape)
energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (velocity**2).sum(0) + 0.5 * (magnetic**2).sum(0)
state = jnp.concatenate([rho[None], rho[None] * velocity, energy[None], magnetic], axis=0)
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
configs = {}
for backend in (NATIVE_JAX, PALLAS):
    config = SimulationConfig(
        dimensionality=3,
        num_cells=n,
        mhd=True,
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        backend_config=BackendConfig(backend=backend, pallas_interpret=True, pallas_block_shape=(8, 8, 8)),
        solver_config=FDSolverConfig(weno_positivity_preserving=True),
    )
    rv = get_registered_variables(config)
    configs[backend] = finalize_config(config, (rv.num_vars,) + shape)
params = SimulationParams(gamma=GAMMA, minimum_density=1e-10, minimum_pressure=1e-10)

reference = mhd_inflow_reference(state, params, configs[NATIVE_JAX], rv)
fused = mhd_inflow_reference_pallas(state, params, configs[PALLAS], rv)
worst = 0.0
for name, a, b in (("reference B", reference[0], fused[0]), ("reference sum S", reference[1], fused[1])):
    error = float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a)))
    worst = max(worst, error)
    print(f"{name:18s}: max |pallas - array| / max|array| = {error:.2e}")
for axis in range(3):
    native = _weno_flux_native_for_axis(axis)(state, params, configs[NATIVE_JAX], rv, inflow_reference=reference)
    pallas = _weno_flux_mhd_pallas(state, params, configs[PALLAS], rv, axis=axis, inflow_reference=reference)
    scale = jnp.max(jnp.abs(native), axis=(1, 2, 3), keepdims=True)
    error = float(jnp.max(jnp.abs(pallas - native) / jnp.maximum(scale, 1e-30)))
    worst = max(worst, error)
    print(f"joint flux axis {axis}: max |pallas - native| / max|native| = {error:.2e}")
print(f"WORST {worst:.2e}")
