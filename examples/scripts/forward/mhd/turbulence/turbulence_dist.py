"""
Multi-node driven MHD turbulence (hero-run driver), FD solver, Pallas backend.

Scales the HOW-MHD-style OU-driven turbulence of ``paper_turbulence.py`` to
sharded multi-node grids (2048^3 and beyond):

* The initial condition (uniform density and B_z, gas at rest) is built
  globally sharded -- each process materialises only its local shard.
* The TurbGen driving (as in ``paper_turbulence.py``) keeps a few hundred
  replicated mode amplitudes; each device sums the modes on its own shard,
  so no full-grid FFT and no communication is needed for the acceleration.
* Snapshots stream to disk (Orbax, one shard per device) via the TO_DISK
  segment loop, which doubles as checkpoint/restart across Slurm walltime
  limits: resubmitting the same command auto-resumes from the latest
  checkpoint in ``--ckpt``.

Launch (one process per GPU) under Slurm::

    srun --ntasks=<G> --ntasks-per-node=4 --gpu-bind=none \
        python examples/scripts/forward/mhd/turbulence/turbulence_dist.py \
            --mturb 0.5 --beta 1e6 --N 2048 --tcross 5 --nseg 100 \
            --ckpt /e/scratch/astronomix/turb2048 --tag icm2048

Diagnostics (spectra, time series) are computed offline from the checkpoint
sequence; this driver keeps the hot loop free of full-grid analysis.
"""

# --- IMPORTANT: bootstrap multi-process mode BEFORE importing astronomix. ---
import argparse  # noqa: E402
import os  # noqa: E402
import time as walltime  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mturb", type=float, default=0.5, help="target turbulent Mach number")
parser.add_argument("--beta", type=float, default=1e6, help="initial plasma beta")
parser.add_argument("--N", type=int, required=True, help="cells per dimension")
parser.add_argument("--tcross", type=float, default=5.0, help="t_end in crossing times")
parser.add_argument("--kdrive", type=float, default=1.5,
                    help="peak of the driving in units of 2 pi / L (= 0.75 k_exp); band kdrive -+ 0.5")
parser.add_argument("--amplitude-factor", type=float, default=1.0,
                    help="TurbGen amplitude factor at the start (adjusted automatically)")
parser.add_argument("--cfl", type=float, default=1.5)
parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
parser.add_argument("--nseg", type=int, default=50,
                    help="number of checkpoint segments spanning [0, t_end]")
parser.add_argument("--block-shape", type=str, default="4,4,8")
parser.add_argument("--ckpt", type=str, required=True,
                    help="checkpoint/snapshot directory (scratch); auto-resume "
                         "from its latest step")
parser.add_argument("--tag", type=str, required=True)
args = parser.parse_args()

_BLOCK = tuple(int(x) for x in args.block_shape.split(","))

_multi = "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1

# Bootstrap distributed mode first (raw jax, no astronomix import yet); see
# examples/scripts/scaling/weak_scaling_hydro.py for the full rationale.
import jax  # noqa: E402

if _multi:
    _cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    _localid = int(os.environ.get("SLURM_LOCALID", "0"))
    jax.distributed.initialize(
        local_device_ids=[_localid] if len(_cvd) > 1 else [0]
    )

jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)  # fp32

import jax.numpy as jnp  # noqa: E402
from jax.sharding import AxisType, PartitionSpec  # noqa: E402

from astronomix import (  # noqa: E402
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    PERIODIC_BOUNDARY,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (  # noqa: E402
    FINITE_DIFFERENCE,
    ISOTHERMAL,
    PALLAS,
    LSRK4,
    TO_DISK,
    BackendConfig,
    StaticIntVector,
)
from astronomix._modules._turbulence_driving import (  # noqa: E402
    TurbulenceDrivingConfig,
    TurbulenceDrivingParams,
)
from astronomix._snapshotting._orbax_storage import (  # noqa: E402
    latest_step,
    load_loop_checkpoint,
)
from astronomix.time_stepping.time_integration import LoopState  # noqa: E402
from astronomix.parallel.distributed import init_distributed  # noqa: E402

info = init_distributed()
G = info.global_device_count
rank = info.process_index

# ---------------------------------------------------------------------------
# ============ ↓ Problem normalisation (vrms1, as in the paper) ↓ ===========
# ---------------------------------------------------------------------------
rho0 = 1.0
a = 1.0 / args.mturb          # isothermal sound speed; drives v_rms ~ 1
vrms_target = 1.0
P_thermal = a ** 2 * rho0
B0 = float((2.0 * P_thermal / args.beta) ** 0.5)

L_inj = 0.5                    # injection scale = box/2 (box = 1)
t_cross = L_inj / vrms_target
t_end = args.tcross * t_cross

# Low-Mach runs need none of the supersonic robustness machinery (mirrors the
# auto switches in paper_turbulence.py for M_turb < 2).
config = SimulationConfig(
    backend_config=BackendConfig(
        backend=PALLAS,
        pallas_block_shape=_BLOCK,
        pallas_use_triton=True,
        pallas_interpret=False,
    ),
    solver_mode=FINITE_DIFFERENCE,
    solver_config=FDSolverConfig(time_integrator=LSRK4),
    equation_of_state=ISOTHERMAL,
    dimensionality=3,
    num_cells=StaticIntVector(args.N, args.N, args.N),
    box_size=1.0,
    mhd=True,
    donate_state=True,
    progress_bar=False,
    boundary_settings=BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    turbulence_driving_config=TurbulenceDrivingConfig(
        turbulence_driving=True,
        k_min=args.kdrive - 0.5,
        k_max=args.kdrive + 0.5,
        amplitude_auto_adjust=True,
    ),
    # Subsonic turbulence needs no positivity machinery: no floors, plain WENO.
    positivity_config=PositivityConfig(),
    snapshot_storage_mode=TO_DISK,
    snapshot_storage_path=args.ckpt,
    num_snapshots=args.nseg,
)

params = SimulationParams(
    C_cfl=args.cfl,
    gamma=args.gamma,
    isothermal_sound_speed=a,
    t_end=t_end,
    turbulence_driving_params=TurbulenceDrivingParams(
        velocity=vrms_target,
        k_driving=args.kdrive,
        solenoidal_weight=1.0,
        amplitude_factor=args.amplitude_factor,
    ),
)

# ---------------------------------------------------------------------------
# ================= ↓ Globally-sharded uniform initial state ↓ ==============
# ---------------------------------------------------------------------------
registered_variables = get_registered_variables(config)

mesh = jax.make_mesh(
    (1, G, 1, 1), (0, 1, 2, 3), axis_types=(AxisType.Auto,) * 4
)
sharding = jax.NamedSharding(mesh, PartitionSpec(0, 1, 2, 3))
spatial_sharding = jax.NamedSharding(mesh, PartitionSpec(1, 2, 3))

N = args.N


def _uniform_fields():
    # Uniform fields: the constant-B interface fields equal the cell-centered
    # values exactly (any consistent interpolation of a constant is the
    # constant), so no staggered construction is needed.
    density = jnp.ones((N, N, N), dtype=jnp.float32) * rho0
    zero = jnp.zeros((N, N, N), dtype=jnp.float32)
    b_z = jnp.ones((N, N, N), dtype=jnp.float32) * B0
    out = (density, zero, b_z)
    return tuple(
        jax.lax.with_sharding_constraint(f, spatial_sharding) for f in out
    )


density, zero, b_z = jax.jit(
    _uniform_fields, out_shardings=(spatial_sharding,) * 3
)()

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=density,
    velocity_x=zero,
    velocity_y=zero,
    velocity_z=zero,
    magnetic_field_x=zero,
    magnetic_field_y=zero,
    magnetic_field_z=b_z,
    interface_magnetic_field_x=zero,
    interface_magnetic_field_y=zero,
    interface_magnetic_field_z=b_z,
    sharding=sharding,
)
config = finalize_config(config, initial_state.shape)
# Free the IC helper fields: at hero scale every stray per-shard buffer
# (three fields here ~ 6 GB/GPU at 2048^3 on 32 GPUs) eats into the
# integration's temp headroom.
del density, zero, b_z

# ---------------------------------------------------------------------------
# ==================== ↓ Auto-resume from the checkpoints ↓ =================
# ---------------------------------------------------------------------------
restart_state = None
resume_step = latest_step(args.ckpt)
if resume_step is not None:
    ckpt = load_loop_checkpoint(args.ckpt, sharding=sharding)
    initial_state = ckpt.primitive_state
    # The restart carry only needs the driving state; holding
    # the restored primitive state here as well would pin its buffer for
    # the whole run and defeat donate_state (the extra live reference forces
    # XLA to copy the ~12 GB/GPU state, which OOMed resumed hero runs).
    restart_state = LoopState(primitive_state=None, driving_state=ckpt.driving_state)
    params = params._replace(t_start=float(ckpt.time))
    if rank == 0:
        print(
            f"[{args.tag}] RESUME from step {resume_step} at t = "
            f"{float(ckpt.time):.5f} (t/tc = {float(ckpt.time) / t_cross:.3f})",
            flush=True,
        )
    del ckpt

if rank == 0:
    total = N ** 3
    print(
        f"[{args.tag}] G={G} grid={N}^3 ({total:.3e} cells, "
        f"{total / G:.3e}/GPU) M_turb={args.mturb} beta={args.beta:g} "
        f"a={a:.3f} B0={B0:.4g} kdrive={args.kdrive} amplitude_factor={args.amplitude_factor} "
        f"cfl={args.cfl} t_end={t_end:.4f} "
        f"segments={args.nseg} ckpt={args.ckpt}",
        flush=True,
    )

start = walltime.time()
# Hand the state over through a single-element container so no module-level
# reference survives into the call -- that is what lets donate_state
# actually donate the ~12 GB/GPU buffer instead of copying it.
_state_container = [initial_state]
del initial_state
final_state = time_integration(
    _state_container.pop(),
    config,
    params,
    registered_variables,
    sharding=sharding,
    restart_state=restart_state,
)
final_state.block_until_ready()

if rank == 0:
    print(
        f"[{args.tag}] reached t_end, wall {walltime.time() - start:.1f}s, "
        f"checkpoints in {args.ckpt}",
        flush=True,
    )
