"""
Extract mid-plane magnetic-energy slices from EVERY ``turbulence_dist.py``
checkpoint, for animating the field over time.

Same restore machinery as ``turb_slices.py`` (each checkpoint step is
restored sharded across the launched processes, so a hero-size state never
has to fit one device), but sweeping the whole checkpoint sequence in one
job and writing one small NPZ per step into ``--outdir``.  Steps whose NPZ
already exists are skipped, so a job that hits its walltime can simply be
resubmitted and continues where it left off.

Launch (2 nodes are enough for a 2048^3 run)::

    srun --ntasks=8 --ntasks-per-node=4 --gpu-bind=none \
        python examples/scripts/forward/mhd/turbulence/turb_slices_series.py \
            --ckpt /e/scratch/astronomix/turb_hero2048 --N 2048 \
            --outdir /e/project1/astronomix/storcks1/runs/slices_series_icm2048

Animate the result with ``animate_slices.py``.
"""

# --- bootstrap multi-process mode BEFORE importing astronomix ---
import argparse  # noqa: E402
import os  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--N", type=int, required=True, help="cells per dimension")
parser.add_argument("--outdir", type=str, required=True,
                    help="directory for the per-step slice NPZs")
parser.add_argument("--every", type=int, default=1,
                    help="process every k-th checkpoint step")
args = parser.parse_args()

_multi = "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1

import jax  # noqa: E402

if _multi:
    _cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    _localid = int(os.environ.get("SLURM_LOCALID", "0"))
    jax.distributed.initialize(
        local_device_ids=[_localid] if len(_cvd) > 1 else [0]
    )

jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)

# general
from pathlib import Path  # noqa: E402

# third-party
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import AxisType, PartitionSpec  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402

# astronomix
from astronomix.option_classes.simulation_config import (  # noqa: E402
    ISOTHERMAL,
    SimulationConfig,
    StaticIntVector,
)
from astronomix.variable_registry.registered_variables import (  # noqa: E402
    get_registered_variables,
)
from astronomix._snapshotting._orbax_storage import (  # noqa: E402
    load_loop_checkpoint,
)

G = jax.device_count()
rank = jax.process_index()

mesh = jax.make_mesh((1, G, 1, 1), (0, 1, 2, 3), axis_types=(AxisType.Auto,) * 4)
sharding = jax.NamedSharding(mesh, PartitionSpec(0, 1, 2, 3))

# Variable indices for the isothermal-MHD layout of the hero run.
_cfg = SimulationConfig(
    equation_of_state=ISOTHERMAL,
    dimensionality=3,
    num_cells=StaticIntVector(args.N, args.N, args.N),
    mhd=True,
)
rv = get_registered_variables(_cfg)


@jax.jit
def _magnetic_energy(state):
    """Magnetic energy density B^2/2 as a full 3D field (sharded)."""
    return 0.5 * (
        state[rv.magnetic_index.x] ** 2
        + state[rv.magnetic_index.y] ** 2
        + state[rv.magnetic_index.z] ** 2
    )


def _checkpoint_steps(directory):
    """All checkpoint step numbers in ``directory``, ascending.

    Args:
        directory: The Orbax checkpoint root.

    Returns:
        Sorted list of integer step numbers (the numeric subdirectories).
    """
    root = Path(directory)
    return sorted(
        int(child.name)
        for child in root.iterdir()
        if child.is_dir() and child.name.isdigit()
    )


def main():
    outdir = Path(args.outdir)
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    steps = _checkpoint_steps(args.ckpt)[:: args.every]
    if rank == 0:
        print(f"[slices-series] {len(steps)} checkpoint steps in {args.ckpt}",
              flush=True)

    mid = args.N // 2
    for step in steps:
        out_path = outdir / f"slices_step_{step:03d}.npz"
        # All ranks see the same shared filesystem, so this skip decision is
        # collective by construction — no rank diverges from the others.
        if out_path.exists():
            continue

        ckpt = load_loop_checkpoint(
            args.ckpt, step, sharding=sharding
        )
        magnetic_energy = _magnetic_energy(ckpt.primitive_state)

        # z mid-plane keeps the X sharding (slice along unsharded z), so the
        # gather moves only an N x N plane; the x mid-plane lives on one
        # shard and gathering it is equally cheap.
        eb_xy = np.asarray(
            multihost_utils.process_allgather(magnetic_energy[:, :, mid], tiled=True)
        )
        eb_yz = np.asarray(
            multihost_utils.process_allgather(magnetic_energy[mid, :, :], tiled=True)
        )

        if rank == 0:
            np.savez(
                out_path,
                step=step,
                time=float(ckpt.time),
                N=args.N,
                EB_xy=eb_xy,
                EB_yz=eb_yz,
            )
            print(f"[slices-series] step {step} t={float(ckpt.time):.4f} "
                  f"-> {out_path}", flush=True)

        # Drop the (large) restored state before the next restore so its
        # device buffers are freed back to the preallocated pool.
        del ckpt, magnetic_energy

    if _multi:
        multihost_utils.sync_global_devices("slices_series_done")


if __name__ == "__main__":
    main()
