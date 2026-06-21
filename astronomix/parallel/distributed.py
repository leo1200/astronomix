"""Multi-process (multi-node / multi-GPU) bootstrap for astronomix.

JAX needs ``jax.distributed.initialize()`` to run *before* any JAX
computation when running across multiple processes (one process per GPU is
the model we use under Slurm ``srun``).  On HoreKa this is launched with::

    srun --ntasks=<#GPUs> --ntasks-per-node=4 --gpus-per-task=1 \
        python driver.py --weak

Under Slurm-launched GPU jobs JAX (>= 0.4) auto-detects the coordinator,
process count, process id and assigns **one device per process**, so
``jax.distributed.initialize()`` can be called with no arguments.

``init_distributed`` is a safe no-op when the program is run as a single
process (no ``SLURM_NTASKS`` > 1 and no Open-MPI rank), so the same driver
scripts work both on a single GPU and across nodes.

IMPORTANT: call :func:`init_distributed` before importing ``jax.numpy`` or
creating any array / calling ``jax.devices()``.
"""

from __future__ import annotations

import os
from typing import NamedTuple


class DistInfo(NamedTuple):
    """Summary of the multi-process topology after initialisation."""

    #: Total number of JAX processes (1 when single-process).
    process_count: int
    #: This process's index in ``[0, process_count)``.
    process_index: int
    #: Number of devices visible to *this* process (1 under one-GPU-per-rank).
    local_device_count: int
    #: Total number of devices across all processes.
    global_device_count: int
    #: True when ``process_count > 1``.
    is_distributed: bool


# Module-level guard so a second call is a cheap no-op instead of an error
# (``jax.distributed.initialize`` raises if called twice).
_INITIALIZED = False


def _looks_distributed() -> bool:
    """Heuristic: are we launched as a multi-process Slurm / MPI job?"""
    slurm = (
        "SLURM_PROCID" in os.environ
        and int(os.environ.get("SLURM_NTASKS", "1")) > 1
    )
    ompi = (
        "OMPI_COMM_WORLD_RANK" in os.environ
        and int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1
    )
    return slurm or ompi


def init_distributed(verbose: bool = True) -> DistInfo:
    """Initialise JAX multi-process mode if launched across ranks.

    Safe no-op when single-process.  Returns a :class:`DistInfo` describing
    the topology.  Must be called before any other JAX use.
    """
    global _INITIALIZED

    # Import jax lazily so importing THIS module does not create the backend.
    #
    # NOTE: importing the wider ``astronomix`` package *does* create the JAX
    # backend at import time (some option NamedTuples have ``jnp.array``
    # defaults).  ``jax.distributed.initialize()`` must run before the backend
    # is created, so a multi-node driver should bootstrap distributed mode
    # (call ``jax.distributed.initialize()`` or this function) *before*
    # importing astronomix.  This function tolerates being called after the
    # client was already bootstrapped by the driver.
    import jax

    if _looks_distributed() and not _INITIALIZED:
        try:
            # No arguments: Slurm / Open-MPI auto-detection (JAX >= 0.4).
            jax.distributed.initialize()
        except (RuntimeError, ValueError) as exc:
            # Already initialised by the driver -> not an error.
            if "already" not in str(exc).lower():
                raise
        _INITIALIZED = True

    info = DistInfo(
        process_count=jax.process_count(),
        process_index=jax.process_index(),
        local_device_count=jax.local_device_count(),
        global_device_count=jax.device_count(),
        is_distributed=jax.process_count() > 1,
    )

    if verbose and info.process_index == 0:
        print(
            f"[distributed] processes={info.process_count} "
            f"global_devices={info.global_device_count} "
            f"local_devices/proc={info.local_device_count} "
            f"distributed={info.is_distributed}",
            flush=True,
        )
    return info
