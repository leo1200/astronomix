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

def _local_device_ids():
    """Pick this rank's local device id robustly for either Slurm GPU binding.

    - Cgroup-bound (--gpus-per-task=1): each task sees ONE GPU as ordinal 0 ->
      use [0].  (But intra-node NCCL P2P then fails -- prefer --gpu-bind=none.)
    - All-visible (--gpu-bind=none): each task sees ALL node GPUs -> select the
      one matching this task's node-local rank, [SLURM_LOCALID].
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [x for x in cvd.split(",") if x != ""]
    localid = int(os.environ.get("SLURM_LOCALID", "0"))
    return [localid] if len(visible) > 1 else [0]


if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize(local_device_ids=_local_device_ids())

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
