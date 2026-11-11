"""Convergence of the 3D circularly polarised Alfven wave with and without PP-WENO.

At low beta the Lax-Friedrichs split state q +- F/alpha of ideal MHD is not
admissible even for alpha = |v_n| + c_f (with B along the normal and v = 0,
p_split / (gamma - 1) = p / (gamma - 1) - (p - B^2/2)^2 / (2 rho alpha^2) < 0).
The theta-scaling then has no admissible base and returns theta = 0: the face
falls back to first-order Rusanov although the flow is smooth. This measures
whether that happens on the standard wave (beta = 0.2) and at beta = 0.02.

    WENO_VARIANT=pp PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/alfven_pp.py --p0 0.01
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--p0", type=float, default=0.1)
parser.add_argument("--resolutions", default="8,16,32")
parser.add_argument("--t-end", type=float, default=1.0)
parser.add_argument("--cfl", type=float, default=0.75)
parser.add_argument("--backend", choices=["native", "pallas"], default="native")
args = parser.parse_args()

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs, weno_variant_name

from astronomix import (
    BackendConfig, NATIVE_JAX, PALLAS, PositivityConfig, SimulationConfig, SimulationParams,
    SnapshotSettings, get_helper_data, get_registered_variables, time_integration,
)
from astronomix.option_classes.simulation_config import StaticIntVector
from astronomix.test_setups.mhd.alfven_wave3D import (
    CPAlfvenWave3DSettings, cp_alfven_wave_solution, setup_cp_alfven_wave,
)

settings = CPAlfvenWave3DSettings(p_0=args.p0, t_end=args.t_end)
beta = 2.0 * args.p0 / settings.b_parallel**2
errors = []
resolutions = [int(r) for r in args.resolutions.split(",")]
for n in resolutions:
    config = SimulationConfig(
        num_cells=StaticIntVector(2 * n, n, n),
        backend_config=BackendConfig(backend=NATIVE_JAX if args.backend == "native" else PALLAS),
        positivity_config=PositivityConfig(clamp_in_estimates=False),
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_states=False, return_final_state=True),
        num_snapshots=2,
        **weno_variant_kwargs(),
    )
    params = SimulationParams(C_cfl=args.cfl, minimum_density=1e-30, minimum_pressure=1e-30)
    state, config, params = setup_cp_alfven_wave(config, params, settings)
    registered_variables = get_registered_variables(config)
    final = jax.block_until_ready(time_integration(state, config, params, registered_variables)).final_state
    exact = cp_alfven_wave_solution(config, registered_variables, params, get_helper_data(config), settings)
    indices = (registered_variables.density_index, *registered_variables.velocity_index,
               registered_variables.pressure_index, *registered_variables.magnetic_index)
    error = float(np.mean([jnp.mean(jnp.abs(final[i] - exact[i])) for i in indices]))
    errors.append(error)
    print(f"[{weno_variant_name()} beta={beta:.3g}] N={n:4d} L1={error:.3e}", flush=True)

orders = [np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]
print(f"ALFVEN-ORDER [{weno_variant_name()} beta={beta:.3g} cfl={args.cfl}] "
      + " ".join(f"{e:.2e}" for e in errors) + " | orders " + " ".join(f"{o:.2f}" for o in orders), flush=True)
