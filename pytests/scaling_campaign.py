"""
Single-GPU + multi-GPU scaling campaign driver for the astronomix methods
paper (HoreKa).  Covers, with a standardised re-plottable (NPZ + JSON) schema
written under ``pytests/scaling_results/``:

  * ``--phase single``  : single-GPU runtime + per-device memory sweep over N,
                          for every (setup x solver-mode).
  * ``--phase block``   : pallas_block_shape sweep at a fixed grid (hydro,
                          FD Pallas) to find the best block.
  * ``--phase strong``  : strong-scaling sweep (1 GPU vs --gpus GPUs).

All scaling runs use fp32 + a memory-saving configuration (LSRK4 for FD,
donate_state, fixed number of timesteps so wall-clock is bounded and
time-per-step is comparable across grid sizes).  Existing convergence /
strong-scaling data files are never touched (distinct ``name=`` prefixes and a
separate results directory).

Examples
--------
    # single-GPU sweep, all setups (1 GPU)
    python pytests/scaling_campaign.py --phase single --setup all --gpus 1
    # pallas block-shape sweep (1 GPU)
    python pytests/scaling_campaign.py --phase block --gpus 1
    # strong scaling on 4 GPUs, hydro only, using the chosen block shape
    python pytests/scaling_campaign.py --phase strong --setup hydro --gpus 4 \
        --block-shape 8,8,8
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---- CLI (parsed before importing jax so we can size the GPU request) ----
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["single", "block", "strong"], required=True)
parser.add_argument("--setup", choices=["hydro", "mhd", "selfgrav", "all"], default="all")
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--block-shape", type=str, default="4,4,8",
                    help="comma-separated pallas block shape used for Pallas runs")
parser.add_argument("--steps", type=int, default=10,
                    help="fixed number of timesteps per run (bounds walltime)")
parser.add_argument("--nmax", type=int, default=None,
                    help="cap the N sweep at this value (for cheap smoke runs)")
parser.add_argument("--tag", type=str, default="h200",
                    help="filename tag, e.g. the GPU family (h100/h200)")
args = parser.parse_args()

BEST_BLOCK = tuple(int(x) for x in args.block_shape.split(","))

# ==== GPU selection ====
from autocvd import autocvd  # noqa: E402
autocvd(num_gpus=args.gpus)
# =======================

import jax  # noqa: E402

# JAX 0.10: use the GSPMD partitioner (the code uses integer mesh-axis names +
# with_sharding_constraint, which the default Shardy partitioner rejects).
jax.config.update("jax_use_shardy_partitioner", False)
# fp32 for all scaling runs (halves memory + compute; values are irrelevant to
# the runtime/memory measurement).
jax.config.update("jax_enable_x64", False)

from astronomix.option_classes.simulation_config import (  # noqa: E402
    FD_FLUX_GRAVITY,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    NATIVE_JAX,
    PALLAS,
    RK4_LSRK,
    SIMPLE_SOURCE_TERM,
    WENO_FLUX_GRAVITY,
    SimulationConfig,
    SnapshotSettings,
    StaticFloatVector,
)
from astronomix.test_setups.hydrodynamics.sound_wave3D import setup_sound_wave  # noqa: E402
from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave  # noqa: E402
from astronomix.test_setups.self_gravity.jeans_waves import setup_jeans_wave  # noqa: E402

from _benchmark_utils import (  # noqa: E402
    BenchmarkSpec,
    _gpu_metadata,
    run_block_shape_sweep,
    run_single_gpu_bench,
    run_strong_scaling,
)

RESULTS = os.path.join(_HERE, "scaling_results")
SINGLE_DIR = os.path.join(RESULTS, "single_gpu")
BLOCK_DIR = os.path.join(RESULTS, "block_shape")
STRONG_DIR = os.path.join(RESULTS, "strong_scaling")
FIG_DIR = os.path.join(RESULTS, "figures")


# --------------------------------------------------------------------------
# Memory-saving config building blocks.
# --------------------------------------------------------------------------
_snap = dict(
    progress_bar=False,
    memory_analysis=True,
    print_elapsed_time=True,
    return_snapshots=True,
    snapshot_settings=SnapshotSettings(return_final_state=True),
    donate_state=True,
)


def _fv(**extra):
    return SimulationConfig(backend=NATIVE_JAX, solver_mode=FINITE_VOLUME, **_snap, **extra)


def _fd_jax(**extra):
    return SimulationConfig(
        backend=NATIVE_JAX, solver_mode=FINITE_DIFFERENCE,
        time_integrator=RK4_LSRK, **_snap, **extra,
    )


def _fd_pallas(block=BEST_BLOCK, **extra):
    return SimulationConfig(
        backend=PALLAS, solver_mode=FINITE_DIFFERENCE, time_integrator=RK4_LSRK,
        pallas_block_shape=block, pallas_use_triton=True, pallas_interpret=False,
        **_snap, **extra,
    )


_HYDRO = dict(box_size=StaticFloatVector(3.0, 1.5, 1.5), mhd=False, dimensionality=3)
_MHD = dict(box_size=StaticFloatVector(3.0, 1.5, 1.5), mhd=True, dimensionality=3)
_SG = dict(mhd=False, self_gravity=True, dimensionality=3)


def _hydro_specs():
    return [
        BenchmarkSpec("FV (JAX)", _fv(**_HYDRO), 0.4),
        BenchmarkSpec("FD (JAX)", _fd_jax(**_HYDRO), 1.5),
        BenchmarkSpec("FD (Pallas)", _fd_pallas(**_HYDRO), 1.5),
    ]


def _mhd_specs():
    return [
        BenchmarkSpec("FV (JAX)", _fv(**_MHD), 0.4),
        BenchmarkSpec("FD (JAX)", _fd_jax(**_MHD), 1.5),
        BenchmarkSpec("FD (Pallas)", _fd_pallas(**_MHD), 1.5),
    ]


def _selfgrav_specs():
    # Self-gravity is FD-centric; FV support is probed empirically (the
    # benchmark records a NaN row if a config is unsupported).
    return [
        BenchmarkSpec("FD (JAX)", _fd_jax(self_gravity_version=WENO_FLUX_GRAVITY, **_SG), 1.5),
        BenchmarkSpec("FD (Pallas)", _fd_pallas(self_gravity_version=WENO_FLUX_GRAVITY, **_SG), 1.5),
        BenchmarkSpec("FV (JAX)", _fv(self_gravity_version=SIMPLE_SOURCE_TERM, **_SG), 0.4),
    ]


# (setup_key, setup_fn, specs_fn, single_N, strong_N)
REGISTRY = {
    "hydro": dict(
        setup_fn=setup_sound_wave, specs=_hydro_specs,
        single_N=[64, 128, 256, 384, 512, 640, 768, 896, 1024],
        strong_N=[128, 256, 384, 512, 640],
    ),
    "mhd": dict(
        setup_fn=setup_cp_alfven_wave, specs=_mhd_specs,
        single_N=[48, 64, 128, 192, 256, 384, 512],
        strong_N=[96, 128, 192, 256, 384],
    ),
    "selfgrav": dict(
        setup_fn=setup_jeans_wave, specs=_selfgrav_specs,
        single_N=[48, 64, 96, 128, 192, 256, 384],
        strong_N=[64, 96, 128, 192, 256],
    ),
}


def _setups():
    return list(REGISTRY) if args.setup == "all" else [args.setup]


def _cap(N_values):
    if args.nmax is None:
        return N_values
    capped = [n for n in N_values if n <= args.nmax]
    return capped or N_values[:1]


def run_single():
    for key in _setups():
        info = REGISTRY[key]
        print(f"\n========== SINGLE-GPU [{key}] ({_gpu_metadata(1)['gpu_model']}) ==========")
        run_single_gpu_bench(
            info["specs"](), _cap(info["single_N"]), info["setup_fn"],
            name=f"{key}_{args.tag}", setup_key=key, data_dir=SINGLE_DIR,
            num_timesteps=args.steps,
        )


def run_block():
    cfg = _fd_pallas(**_HYDRO)
    block_shapes = [
        (2, 2, 4), (4, 4, 8), (8, 8, 8), (2, 4, 8),
        (4, 8, 16), (8, 8, 16), (8, 16, 16), (16, 16, 16),
    ]
    print(f"\n========== BLOCK-SHAPE SWEEP hydro FD(Pallas) ({_gpu_metadata(1)['gpu_model']}) ==========")
    run_block_shape_sweep(
        cfg, block_shapes, N=256, setup_fn=setup_sound_wave,
        name=f"hydro_block_shape_{args.tag}", setup_key="hydro",
        data_dir=BLOCK_DIR, cfl=1.5, num_timesteps=max(args.steps, 20),
    )


def run_strong():
    for key in _setups():
        info = REGISTRY[key]
        name = f"{key}_{args.tag}"
        print(f"\n========== STRONG SCALING [{key}] 1 vs {args.gpus} GPUs ==========")
        run_strong_scaling(
            info["specs"](), _cap(info["strong_N"]), info["setup_fn"],
            num_gpus=args.gpus, name=name, title=f"{key} strong scaling",
            data_dir=STRONG_DIR, figure_dir=FIG_DIR, num_timesteps=args.steps,
        )
        # JSON metadata sidecar (run_strong_scaling itself only writes the NPZ).
        meta = dict(
            _gpu_metadata(args.gpus), setup=key, phase="strong_scaling",
            block_shape=list(BEST_BLOCK), num_timesteps=args.steps,
            strong_N=info["strong_N"],
        )
        with open(os.path.join(STRONG_DIR, f"{name}_strong_scaling.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    if args.phase == "single":
        run_single()
    elif args.phase == "block":
        run_block()
    elif args.phase == "strong":
        run_strong()
