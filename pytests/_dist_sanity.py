"""
Minimal multi-process / multi-node jax.distributed sanity check.

Validates the rendezvous + a cross-process collective before committing a big
multi-node weak-scaling run.  Launch with srun, one task per GPU::

    srun --ntasks=<G> --ntasks-per-node=4 --gpus-per-task=1 \
        python pytests/_dist_sanity.py
"""

import os

import jax

if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    # HoreKa does not constrain CUDA_VISIBLE_DEVICES per task, so every rank
    # sees all 4 node GPUs.  Pin each process to the GPU matching its node-local
    # rank, else bare initialize() claims all 4 -> "invalid device ordinal".
    _local_id = int(os.environ.get("SLURM_LOCALID", "0"))
    jax.distributed.initialize(local_device_ids=[_local_id])

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
