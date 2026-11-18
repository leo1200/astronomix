"""Wall-clock cost of the WENO variants: one native WENO flux evaluation per axis.

    PYTHONPATH=. python examples/scripts/validation/weno_stability/bench.py out/turb/lastgood_bare64_current.npz
"""

# general
import os
import sys
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)

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
    PERIODIC_BOUNDARY,
    SimulationConfig,
    SimulationParams,
    finalize_config,
    get_registered_variables,
)
from astronomix.option_classes.simulation_config import ISOTHERMAL, IDEAL_GAS
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x_native,
)

state_file = sys.argv[1]
eos = sys.argv[2] if len(sys.argv) > 2 else "iso"
state = jnp.asarray(np.load(state_file)["state"], dtype=jnp.float32)
n = state.shape[-1]
periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
results = {}
for variant, kwargs in [
    ("baseline", {}),
    ("face", dict(weno_admissible_face_state=True)),
    ("pp", dict(weno_positivity_preserving=True)),
]:
    if eos == "iso" and variant == "face":
        continue
    config = SimulationConfig(
        equation_of_state=ISOTHERMAL if eos == "iso" else IDEAL_GAS,
        dimensionality=3,
        num_cells=n,
        mhd=True,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        solver_config=FDSolverConfig(**kwargs),
    )
    config = finalize_config(config, state.shape)
    registered_variables = get_registered_variables(config)
    params = SimulationParams(isothermal_sound_speed=0.1, minimum_density=1e-10, minimum_pressure=1e-10)
    # conserved variables from the isothermal primitive snapshot (rho, v, B);
    # the ideal-gas variant gets an energy at the isothermal temperature
    rho, velocity, magnetic = state[0], state[1:4], state[4:7]
    parts = [rho[None], rho[None] * velocity]
    if eos == "iso":
        parts.append(magnetic)
    else:
        pressure = rho * 0.01
        energy = pressure / (params.gamma - 1.0) + 0.5 * rho * (velocity**2).sum(0) + 0.5 * (magnetic**2).sum(0)
        parts += [energy[None], magnetic]
    conserved = jnp.concatenate(parts, axis=0)
    flux = jax.jit(lambda q: _weno_flux_x_native(q, params, config, registered_variables))
    jax.block_until_ready(flux(conserved))
    start = time.perf_counter()
    repeats = 20
    for _ in range(repeats):
        out = flux(conserved)
    jax.block_until_ready(out)
    results[variant] = (time.perf_counter() - start) / repeats
    print(f"{variant:9s} {eos}: {1e3 * results[variant]:.2f} ms per x-flux at {n}^3", flush=True)
print("relative to baseline:", {k: f"{v / results['baseline']:.2f}x" for k, v in results.items()})
