"""
Minimal multi-process / multi-node jax.distributed sanity check.

Validates the rendezvous + a cross-process collective before committing a big
multi-node weak-scaling run.  Launch with srun, one task per GPU::

    srun --ntasks=<G> --ntasks-per-node=4 --gpus-per-task=1 \
        python pytests/_dist_sanity.py
"""

import os

import jax

# Diagnostic: each rank's view of the GPUs, BEFORE touching the backend.
print(
    f"[pre-init procid={os.environ.get('SLURM_PROCID','?')} "
    f"localid={os.environ.get('SLURM_LOCALID','?')}] "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}",
    flush=True,
)

if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    # HoreKa binds ONE GPU per task (--gpus-per-task=1), remapped so each
    # process sees its unique physical GPU as local ordinal 0.  So every rank
    # must use local_device_ids=[0]; passing SLURM_LOCALID instead points
    # ranks 1..n at ordinals that don't exist -> "invalid device ordinal".
    jax.distributed.initialize(local_device_ids=[0])

import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils as mh  # noqa: E402

pc = jax.process_count()
pi = jax.process_index()
print(
    f"[rank {pi}/{pc}] host={os.environ.get('SLURMD_NODENAME','?')} "
    f"local_devices={jax.local_device_count()} global_devices={jax.device_count()} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}",
    flush=True,
)

# Cross-process collective: each rank contributes its index; all should agree.
gathered = mh.process_allgather(jnp.array([float(pi)]))
if pi == 0:
    print(f"[rank 0] allgather = {gathered.ravel()}  (expect 0..{pc-1})", flush=True)
    print(f"[rank 0] PASS distributed rendezvous: {pc} processes, "
          f"{jax.device_count()} devices", flush=True)

mh.sync_global_devices("dist_sanity_done")
print(f"[rank {pi}] done", flush=True)
