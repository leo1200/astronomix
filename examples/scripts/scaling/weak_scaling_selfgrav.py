"""
Multi-node WEAK-scaling driver: hydro + self-gravity, FD solver, Pallas
backend.

Weak scaling keeps the per-GPU sub-domain fixed while growing the global grid
with the device count.  The state is built already globally-sharded via
``build_jeans_state_sharded`` (each process materialises only its local
shard).

CAVEAT (measured, not assumed): the Poisson step solves in Fourier space via
``jnp.fft.fftn`` on the full grid, and the XLA SPMD partitioner replicates
the operand of an FFT along its transform dimensions.  Per-device FFT work
and memory therefore grow with the *global* grid, so the default per-GPU
sub-domain here is much smaller than the pure-hydro one (64x1024x1024) and
the recorded per-device memory quantifies the replication overhead at every
rung.

Launch (one process per GPU) under Slurm::

    srun --ntasks=<G> --ntasks-per-node=4 --gpu-bind=none \
        python examples/scripts/scaling/weak_scaling_selfgrav.py --bx 64 --by 1024 --bz 1024 \
            --steps 10 --block-shape 4,4,8
"""

# --- IMPORTANT: bootstrap multi-process mode BEFORE importing astronomix. ---
import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--bx", type=int, default=64, help="per-GPU cells along sharded X")
parser.add_argument("--by", type=int, default=1024, help="cells along Y (global)")
parser.add_argument("--bz", type=int, default=1024, help="cells along Z (global)")
parser.add_argument("--steps", type=int, default=10, help="fixed number of timesteps")
parser.add_argument("--dt", type=float, default=0.4,
                    help="fixed timestep (stable for h=1, c_s=1)")
parser.add_argument("--block-shape", type=str, default="4,4,8")
parser.add_argument("--cfl", type=float, default=1.5)
parser.add_argument("--tag", type=str, default="gh200")
parser.add_argument("--gpus", type=int, default=0,
                    help="SINGLE-process mode: grab this many local GPUs via "
                         "autocvd (no jax.distributed). Ignored under srun "
                         "multi-process (SLURM_NTASKS>1).")
args = parser.parse_args()

_BLOCK = tuple(int(x) for x in args.block_shape.split(","))

_multi = "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1

if not _multi and args.gpus > 0:
    from autocvd import autocvd
    autocvd(num_gpus=args.gpus)

# Bootstrap distributed mode first (raw jax, no astronomix import yet); see
# weak_scaling_hydro.py for the full rationale.
import jax  # noqa: E402

if _multi:
    _cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    _localid = int(os.environ.get("SLURM_LOCALID", "0"))
    _local_ids = [_localid] if len(_cvd) > 1 else [0]
    jax.distributed.initialize(local_device_ids=_local_ids)

jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)  # fp32

# Now safe to import astronomix.
from astronomix.parallel.distributed import init_distributed  # noqa: E402
from astronomix.option_classes.simulation_config import (  # noqa: E402
    FINITE_DIFFERENCE,
    PALLAS,
    LSRK4,
    SimulationConfig,
    BackendConfig,
    FDSolverConfig,
)
from astronomix.test_setups.self_gravity.jeans_waves import (  # noqa: E402
    JeansWaveSettings,
    build_jeans_state_sharded,
)

_PYTESTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "pytests"
)
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)

from _benchmark_utils import run_weak_scaling_point  # noqa: E402

info = init_distributed()
G = info.global_device_count

DATA_DIR = os.path.join(_HERE, "data", "weak_scaling")

# Global grid: shard X across all G devices, keep Y/Z fixed per process.
GLOBAL = (args.bx * G, args.by, args.bz)
MESH_SHAPE = (1, G, 1, 1)
# Box numerically equal to the grid -> uniform spacing h = 1 at every rung.
BOX = (float(GLOBAL[0]), float(GLOBAL[1]), float(GLOBAL[2]))

base_config = SimulationConfig(
    backend_config=BackendConfig(
        backend=PALLAS,
        pallas_block_shape=_BLOCK,
        pallas_use_triton=True,
        pallas_interpret=False,
    ),
    solver_mode=FINITE_DIFFERENCE,
    solver_config=FDSolverConfig(time_integrator=LSRK4),
    mhd=False,
    dimensionality=3,
    donate_state=True,
    progress_bar=False,
    print_elapsed_time=True,
)

# Box = grid; the Jeans wave then provides a smooth, tiny (eps = 1e-6)
# perturbation on the uniform background -- an exact eigenmode is irrelevant
# for a throughput benchmark.  Stable: c_s^2 k^2 = 36 > 4 pi G rho = 1.
settings = JeansWaveSettings(box_size=BOX)

if info.process_index == 0:
    total = GLOBAL[0] * GLOBAL[1] * GLOBAL[2]
    print(
        f"[weak-sg] G={G} mesh={MESH_SHAPE} global_grid={GLOBAL} "
        f"({total:.3e} cells, ~{total / G:.3e} cells/GPU) block={_BLOCK} "
        f"steps={args.steps} dt={args.dt}",
        flush=True,
    )

run_weak_scaling_point(
    build_jeans_state_sharded,
    base_config,
    settings,
    mesh_shape=MESH_SHAPE,
    global_cells=GLOBAL,
    box_size=BOX,
    cfl=args.cfl,
    dt=args.dt,
    num_timesteps=args.steps,
    name=f"weak_selfgrav_pallas_{args.tag}",
    data_dir=DATA_DIR,
    setup_key="selfgrav_weak",
)
