"""Multi-GPU disk-checkpointing (Orbax) smoke + restart-continuity test.

Runs a small 3D shock on a 2-GPU mesh with ``snapshot_storage_mode == TO_DISK``,
then restarts from the latest on-disk checkpoint and continues — checking that

  (A) one uninterrupted TO_DISK run over [0, t_end] in 4 segments

equals

  (B) the same run split in two: TO_DISK over [0, t_end/2], then a fresh-process
      style restart from the latest checkpoint continuing to t_end,

bit-for-bit. Each device writes its own shard, so this also exercises the
multi-device Orbax save / restore path.

Run with:  PYTHONPATH=<repo root> python tests/multi_gpu/orbax_checkpoint.py
"""

# On boxes where GPU peer-to-peer (NVLink/PCIe P2P) is unavailable or broken,
# NCCL collectives — used both by the sharded solver (halo exchange) and by a
# sharded Orbax restore — deadlock. Disabling P2P/SHM forces NCCL onto a
# working fallback path. Harmless where P2P works; must be set before JAX init.
import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=2)
# =======================

import shutil

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

from astronomix import (
    SimulationConfig,
    SimulationParams,
    time_integration,
    get_registered_variables,
    restart_from_latest_checkpoint,
)
from astronomix.option_classes.simulation_config import (
    finalize_config,
    FORWARDS,
    HLL,
    TO_DISK,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)
from astronomix.setup_helpers import latest_checkpoint_step

N = 64
T_END = 0.1

# 2-GPU mesh, sharded along the x axis (as in tests/multi_gpu/scaling.py).
mesh = jax.make_mesh((1, 2, 1, 1), (VARAXIS, XAXIS, YAXIS, ZAXIS))
sharding = NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))


def make_ic():
    rho = jnp.ones((N, N, N)) * 0.125
    u = jnp.zeros((N, N, N))
    p = jnp.ones((N, N, N)) * 0.1
    c = N // 2
    sl = slice(c - 4, c + 4)
    rho = rho.at[sl, sl, sl].set(1.0)
    p = p.at[sl, sl, sl].set(1.0)
    ps = jnp.stack([rho, u, u, u, p], axis=0)
    return jax.device_put(ps, sharding)


def cfg(path, num_snapshots):
    c = SimulationConfig(
        dimensionality=3, box_size=1.0, num_cells=N,
        differentiation_mode=FORWARDS, riemann_solver=HLL,
        snapshot_storage_mode=TO_DISK, snapshot_storage_path=path,
        num_snapshots=num_snapshots,
    )
    return finalize_config(c, (5, N, N, N))


ps = make_ic()
rv = get_registered_variables(cfg("/tmp/junk", 4))
print("input sharding:")
jax.debug.visualize_array_sharding(ps[0, :, :, 0])

# --- Run A: full TO_DISK over [0, T_END] in 4 segments, sharded ---
dirA = "/tmp/mgpu_ckptA"
shutil.rmtree(dirA, ignore_errors=True)
params = SimulationParams(t_end=T_END, C_cfl=0.4)
finalA = time_integration(ps, cfg(dirA, 4), params, rv, sharding=sharding)
print("Run A done. latest step:", latest_checkpoint_step(dirA))
print("final-state sharding (should still be split over 2 GPUs):")
jax.debug.visualize_array_sharding(finalA[0, :, :, 0])

# --- Run B1: TO_DISK over [0, T_END/2] in 2 segments ---
dirB1 = "/tmp/mgpu_ckptB1"
shutil.rmtree(dirB1, ignore_errors=True)
pB1 = SimulationParams(t_end=T_END / 2, C_cfl=0.4)
_ = time_integration(ps, cfg(dirB1, 2), pB1, rv, sharding=sharding)

# --- Run B2: restart (sharded) and continue to T_END in 2 segments ---
dirB2 = "/tmp/mgpu_ckptB2"
shutil.rmtree(dirB2, ignore_errors=True)
pB2 = SimulationParams(t_end=T_END, C_cfl=0.4)
ps_r, pB2, restart = restart_from_latest_checkpoint(dirB1, pB2, sharding=sharding)
print("restored state sharding:")
jax.debug.visualize_array_sharding(ps_r[0, :, :, 0])
finalB = time_integration(ps_r, cfg(dirB2, 2), pB2, rv, sharding=sharding,
                          restart_state=restart)

finalA = jax.device_get(finalA)
finalB = jax.device_get(finalB)
maxdiff = float(jnp.max(jnp.abs(finalA - finalB)))
print("max|A-B| =", maxdiff)
print("BITEXACT restart across 2 GPUs:", bool(jnp.all(finalA == finalB)))
