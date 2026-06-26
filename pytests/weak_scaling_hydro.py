"""
Multi-node WEAK-scaling driver: pure hydrodynamics, FD solver, Pallas backend.

Weak scaling keeps the per-GPU sub-domain fixed while growing the global grid
with the device count, so ideal scaling is a flat wall-clock curve.  The state
is built already globally-sharded (each process materialises only its local
shard via ``build_sound_wave_state_sharded``), which is what makes the largest
runs (up to 2048^3 on 16 GPUs / 4 nodes) feasible -- the full grid never lives
on one host.

Launch (one process per GPU) under Slurm::

    srun --ntasks=<G> --ntasks-per-node=4 --gpus-per-task=1 \
        python pytests/weak_scaling_hydro.py --bx 128 --by 2048 --bz 2048 \
            --steps 10 --block-shape 8,8,8

The mesh shards the X axis only: global grid = (bx * G, by, bz).  The box is
set numerically equal to the grid so the spacing stays uniform (h = 1) at every
rung; this is a timing/throughput benchmark, so wave periodicity is irrelevant.
With (bx, by, bz) = (128, 2048, 2048) the ladder G = 1,2,4,8,16 culminates in a
2048^3 cube on 16 GPUs.
"""

# --- IMPORTANT: bootstrap multi-process mode BEFORE importing astronomix. ---
# Importing astronomix creates the JAX backend (NamedTuple jnp.array defaults),
# and jax.distributed.initialize() must run before the backend exists.
import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--bx", type=int, default=128, help="per-GPU cells along sharded X")
parser.add_argument("--by", type=int, default=2048, help="cells along Y (global)")
parser.add_argument("--bz", type=int, default=2048, help="cells along Z (global)")
parser.add_argument("--steps", type=int, default=10, help="fixed number of timesteps")
parser.add_argument("--dt", type=float, default=0.4, help="fixed timestep (stable for h=1, c_s=1)")
parser.add_argument("--block-shape", type=str, default="8,8,8")
parser.add_argument("--cfl", type=float, default=1.5)
parser.add_argument("--tag", type=str, default="h200")
parser.add_argument("--gpus", type=int, default=0,
                    help="SINGLE-process mode: grab this many local GPUs via "
                         "autocvd (no jax.distributed). Ignored under srun "
                         "multi-process (SLURM_NTASKS>1).")
args = parser.parse_args()

_BLOCK = tuple(int(x) for x in args.block_shape.split(","))

_multi = "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1

# Single-process multi-GPU: select GPUs via autocvd BEFORE importing jax.  This
# path runs weak scaling on one node across --gpus devices in a single process
# (no jax.distributed rendezvous needed -- works around the multi-process init
# timeout, ideal for up to 8 GPUs on a Teal H200 node).
if not _multi and args.gpus > 0:
    from autocvd import autocvd
    autocvd(num_gpus=args.gpus)

# Bootstrap distributed mode first (raw jax, no astronomix import yet).
import jax  # noqa: E402

# CRITICAL: jax.distributed.initialize() must run before ANYTHING touches the
# backend, otherwise each process eagerly grabs all local GPUs (invalid device
# ordinal -> 5-min topology-gather deadlock).  Even jax.config.update() can
# trigger backend init, so do the rendezvous FIRST and configure afterwards.
# (The GSPMD/Shardy choice is already set via JAX_USE_SHARDY_PARTITIONER in
# _env.sh, so no pre-init config.update is needed here.)
#
# Pick this rank's local device id robustly for either Slurm GPU binding:
#   --gpus-per-task=1 (cgroup-bound): one GPU per task as ordinal 0 -> [0]
#     (but intra-node NCCL P2P fails -- launch with --gpu-bind=none instead);
#   --gpu-bind=none (all-visible): each task sees all node GPUs -> [SLURM_LOCALID].
if _multi:
    _cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    _localid = int(os.environ.get("SLURM_LOCALID", "0"))
    _local_ids = [_localid] if len(_cvd) > 1 else [0]
    jax.distributed.initialize(local_device_ids=_local_ids)

# JAX 0.10: GSPMD partitioner (redundant with the env var, but explicit). Safe
# to set now that the backend has been initialized distributed-aware.
jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)  # fp32

# Now safe to import astronomix.
from astronomix.parallel.distributed import init_distributed  # noqa: E402
from astronomix.option_classes.simulation_config import (  # noqa: E402
    FINITE_DIFFERENCE,
    PALLAS,
    RK4_LSRK,
    SimulationConfig,
    SnapshotSettings,
)
from astronomix.test_setups.hydrodynamics.sound_wave3D import (  # noqa: E402
    SoundWave3DSettings,
    build_sound_wave_state_sharded,
)
from _benchmark_utils import run_weak_scaling_point  # noqa: E402

info = init_distributed()
G = info.global_device_count

DATA_DIR = os.path.join(_HERE, "scaling_results", "weak_scaling")

# Global grid: shard X across all G devices, keep Y/Z fixed per process.
GLOBAL = (args.bx * G, args.by, args.bz)
MESH_SHAPE = (1, G, 1, 1)
# Box numerically equal to the grid -> uniform spacing h = 1 at every rung.
BOX = (float(GLOBAL[0]), float(GLOBAL[1]), float(GLOBAL[2]))

# A memory-lean FD/Pallas config; LSRK4 + donate keep the per-GPU footprint low.
base_config = SimulationConfig(
    backend=PALLAS,
    solver_mode=FINITE_DIFFERENCE,
    time_integrator=RK4_LSRK,
    pallas_block_shape=_BLOCK,
    pallas_use_triton=True,
    pallas_interpret=False,
    mhd=False,
    dimensionality=3,
    donate_state=True,
    progress_bar=False,
    print_elapsed_time=True,
)

# settings.box_size drives grid_spacing inside build_sound_wave_state_sharded.
settings = SoundWave3DSettings(box_size=BOX)

if info.process_index == 0:
    total = GLOBAL[0] * GLOBAL[1] * GLOBAL[2]
    print(
        f"[weak] G={G} mesh={MESH_SHAPE} global_grid={GLOBAL} "
        f"({total:.3e} cells, ~{total / G:.3e} cells/GPU) block={_BLOCK} "
        f"steps={args.steps} dt={args.dt}",
        flush=True,
    )

run_weak_scaling_point(
    build_sound_wave_state_sharded,
    base_config,
    settings,
    mesh_shape=MESH_SHAPE,
    global_cells=GLOBAL,
    box_size=BOX,
    cfl=args.cfl,
    dt=args.dt,
    num_timesteps=args.steps,
    name=f"weak_hydro_pallas_{args.tag}",
    data_dir=DATA_DIR,
)
