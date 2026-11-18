"""
Extract 2D mid-plane slices from a ``turbulence_dist.py`` checkpoint.

Restores one checkpoint step sharded across the launched processes (so a
hero-size state never has to fit one device), slices density, magnetic
energy and velocity magnitude at the mid-plane of each axis, gathers the
(tiny) slices to rank 0 and writes them to an NPZ for plotting.

Launch::

    srun --ntasks=8 --ntasks-per-node=4 --gpu-bind=none \
        python examples/scripts/forward/mhd/turbulence/turb_slices.py \
            --ckpt /e/scratch/astronomix/turb_hero2048 --step 63 --N 2048 \
            --out slices_icm2048_final.npz
"""

# --- bootstrap multi-process mode BEFORE importing astronomix ---
import argparse  # noqa: E402
import os  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: latest)")
parser.add_argument("--N", type=int, required=True, help="cells per dimension")
parser.add_argument("--out", type=str, required=True, help="output NPZ path")
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

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import AxisType, PartitionSpec  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402

from astronomix.option_classes.simulation_config import (  # noqa: E402
    ISOTHERMAL,
    SimulationConfig,
    StaticIntVector,
)
from astronomix.variable_registry.registered_variables import (  # noqa: E402
    get_registered_variables,
)
from astronomix._snapshotting._orbax_storage import (  # noqa: E402
    latest_step,
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
def _derived_fields(state):
    """Density, magnetic energy and |v| as full 3D fields (sharded)."""
    rho = state[rv.density_index]
    magnetic_energy = 0.5 * (
        state[rv.magnetic_index.x] ** 2
        + state[rv.magnetic_index.y] ** 2
        + state[rv.magnetic_index.z] ** 2
    )
    speed = jnp.sqrt(
        state[rv.velocity_index.x] ** 2
        + state[rv.velocity_index.y] ** 2
        + state[rv.velocity_index.z] ** 2
    )
    return rho, magnetic_energy, speed


def main():
    step = args.step if args.step is not None else latest_step(args.ckpt)
    ckpt = load_loop_checkpoint(
        args.ckpt, step, sharding=sharding
    )
    rho, magnetic_energy, speed = _derived_fields(ckpt.primitive_state)

    mid = args.N // 2
    slices = {}
    for name, field in [("rho", rho), ("EB", magnetic_energy), ("v", speed)]:
        # z mid-plane keeps the X sharding (slice along unsharded z), so the
        # gather moves only an N x N plane; the x mid-plane lives on one
        # shard and gathering it is equally cheap.
        slices[f"{name}_xy"] = np.asarray(
            multihost_utils.process_allgather(field[:, :, mid], tiled=True)
        )
        slices[f"{name}_yz"] = np.asarray(
            multihost_utils.process_allgather(field[mid, :, :], tiled=True)
        )

    if rank == 0:
        np.savez(
            args.out,
            step=step,
            time=float(ckpt.time),
            N=args.N,
            **slices,
        )
        print(f"[slices] step {step} t={float(ckpt.time):.4f} -> {args.out}",
              flush=True)

    if _multi:
        multihost_utils.sync_global_devices("slices_done")


if __name__ == "__main__":
    main()
