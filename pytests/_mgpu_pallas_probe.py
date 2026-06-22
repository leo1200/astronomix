"""
Probe: does the FD **Pallas** backend work under multi-GPU NamedSharding on
this JAX build?  (JAX 0.10 reportedly has multi-GPU Pallas issues.)

Single process, ``--gpus`` devices, X-sharded mesh.  Builds the sound-wave IC
sharded, runs a few FD/Pallas timesteps, and checks the sharded result against
a single-GPU reference for the same grid.  Prints a clear PASS/FAIL so we can
decide whether to proceed on JAX 0.10 or fall back to a JAX 0.9 environment.

    python pytests/_mgpu_pallas_probe.py --gpus 2 --N 128
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, default=2)
parser.add_argument("--N", type=int, default=128, help="grid is 2N x N x N")
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--block-shape", type=str, default="8,8,8")
args = parser.parse_args()
_BLOCK = tuple(int(x) for x in args.block_shape.split(","))

from autocvd import autocvd  # noqa: E402
autocvd(num_gpus=args.gpus)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)

from astronomix import get_registered_variables, time_integration  # noqa: E402
from astronomix.option_classes.simulation_config import (  # noqa: E402
    FINITE_DIFFERENCE,
    PALLAS,
    RK4_LSRK,
    SimulationConfig,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams  # noqa: E402
from astronomix.test_setups.hydrodynamics.sound_wave3D import (  # noqa: E402
    build_sound_wave_state_sharded,
)
from _benchmark_utils import _build_global_sharding  # noqa: E402

print("devices:", jax.devices(), flush=True)

N = args.N
GRID = StaticIntVector(2 * N, N, N)


def _make_config():
    return SimulationConfig(
        backend=PALLAS, solver_mode=FINITE_DIFFERENCE, time_integrator=RK4_LSRK,
        pallas_block_shape=_BLOCK, pallas_use_triton=True, pallas_interpret=False,
        mhd=False, dimensionality=3, num_cells=GRID,
        fixed_timestep=True, num_timesteps=args.steps,
        progress_bar=False, donate_state=False,
    )


def _run(sharding):
    cfg = _make_config()
    state, cfg, params = build_sound_wave_state_sharded(
        cfg, SimulationParams(C_cfl=1.5), sharding
    )
    # STABLE fixed dt: box=(3,1.5,1.5), grid (2N,N,N) -> dx = 1.5/N, c_s=1.
    # dt = 0.4*dx keeps CFL ~ 0.4 (the default probe previously used dt=0.4,
    # i.e. CFL ~ 25, which blew up to NaN for BOTH ref and sharded).
    dx = 1.5 / N
    dt = 0.4 * dx
    params = params._replace(t_end=dt * args.steps)
    rv = get_registered_variables(cfg)
    out = time_integration(state, cfg, params, rv, sharding=sharding)
    final = out.final_state if hasattr(out, "final_state") else out
    final.block_until_ready()
    return final


def main():
    print(f"--- 1-GPU reference (Pallas), grid {2*N}x{N}x{N} ---", flush=True)
    ref = _run(None)
    print("reference ok", flush=True)

    print(f"--- {args.gpus}-GPU sharded (Pallas) ---", flush=True)
    sharding = _build_global_sharding((1, args.gpus, 1, 1))
    shr = _run(sharding)
    print("sharded run ok", flush=True)

    ref_h = jax.device_get(ref)
    shr_h = jax.device_get(shr)
    ref_finite = bool(jnp.all(jnp.isfinite(ref_h)))
    shr_finite = bool(jnp.all(jnp.isfinite(shr_h)))
    print(f"reference finite={ref_finite}  sharded finite={shr_finite}", flush=True)
    if not ref_finite:
        print("FAIL (reference itself is non-finite -- unstable dt, not a "
              "sharding issue)", flush=True)
        return 2
    max_abs = float(jnp.max(jnp.abs(ref_h - shr_h)))
    ref_scale = float(jnp.max(jnp.abs(ref_h)))
    rel = max_abs / (ref_scale + 1e-30)
    print(f"max|ref - sharded| = {max_abs:.3e}  (rel {rel:.3e})", flush=True)
    # fp32 + different reduction order across shards -> allow a loose tolerance.
    ok = shr_finite and rel < 1e-3
    print("PASS" if ok else "FAIL", "multi-GPU Pallas sharding", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
