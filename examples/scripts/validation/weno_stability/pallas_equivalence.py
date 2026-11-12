"""Pallas hydro WENO kernel vs native kernel with the new options (interpret mode).

States are rough, high-contrast and supersonic so the admissibility scaling is
active at many interfaces; the script also reports how many faces had theta < 1.

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/pallas_equivalence.py
"""

# general
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if os.environ.get("EQUIV_X64"):
    os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import numpy as np
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
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_native_for_axis,
)
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _hydro_pallas_flux_supported,
    _weno_flux_hydro_pallas,
)

periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
GAMMA = 5.0 / 3.0


def rough_state(shape, seed, registered_variables, dimensionality):
    """Log-normal density (contrast ~1e4), Mach ~10 velocities, mixed pressure."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    rho = jnp.exp(3.0 * jax.random.normal(k1, shape))
    pressure = float(os.environ.get("EQUIV_PSCALE", "1e-2")) * jnp.exp(2.0 * jax.random.normal(k2, shape))
    velocity = float(os.environ.get("EQUIV_VSCALE", "3.0")) * jax.random.normal(k3, (dimensionality,) + shape)
    kinetic = 0.5 * rho * (velocity**2).sum(0)
    parts = [rho[None], rho[None] * velocity, (pressure / (GAMMA - 1.0) + kinetic)[None]]
    dtype = jnp.float64 if os.environ.get("EQUIV_X64") else jnp.float32
    return jnp.concatenate(parts, axis=0).astype(dtype), (pressure / (GAMMA - 1.0)).astype(dtype)


worst = 0.0
for dimensionality, n in [(1, 64), (2, 32), (3, 16)]:
    shape = (n,) * dimensionality
    boundaries = periodic if dimensionality == 1 else BoundarySettings(*([periodic] * dimensionality))
    for variant in ["base", "face", "pp"]:
        options = dict(
            base={}, face=dict(weno_admissible_face_state=True), pp=dict(weno_positivity_preserving=True)
        )[variant]
        configs = {}
        for backend in (NATIVE_JAX, PALLAS):
            config = SimulationConfig(
                dimensionality=dimensionality,
                num_cells=n,
                boundary_settings=boundaries,
                backend_config=BackendConfig(
                    backend=backend,
                    pallas_interpret=True,
                    pallas_block_shape=(8, 8, 8)[:dimensionality] + (1,) * (3 - dimensionality),
                ),
                solver_config=FDSolverConfig(**options),
            )
            registered_variables = get_registered_variables(config)
            configs[backend] = finalize_config(config, (registered_variables.num_vars,) + shape)
        params = SimulationParams(gamma=GAMMA, minimum_density=1e-10, minimum_pressure=1e-10)
        state, internal = rough_state(shape, 7, registered_variables, dimensionality)
        assert _hydro_pallas_flux_supported(state, configs[PALLAS])
        for axis in range(dimensionality):
            for dual in (False, True):
                g = internal if dual else None
                native = _weno_flux_native_for_axis(axis)(
                    state, params, configs[NATIVE_JAX], registered_variables, internal_energy_density=g,
                )
                pallas = _weno_flux_hydro_pallas(
                    state, params, configs[PALLAS], registered_variables, axis=axis,
                    internal_energy_density=g,
                )
                scale = jnp.max(jnp.abs(native), axis=tuple(range(1, native.ndim)), keepdims=True)
                error = float(jnp.max(jnp.abs(pallas - native) / jnp.maximum(scale, 1e-30)))
                worst = max(worst, error)
                print(f"{dimensionality}D {variant:4s} axis {axis} dual={int(dual)}: "
                      f"max |pallas - native| / max|native| = {error:.2e}", flush=True)
print(f"WORST {worst:.2e}")
