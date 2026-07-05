"""
Extended strong-scaling sweep for the 3D sound-wave benchmark:
FV (JAX) and FD (Pallas) only, up to N=512.

Runs on 2 GPUs. Outputs to `data/astronomix/sound_wave3D_strong_scaling_extended.npz`
and `figures/sound_wave3D_strong_scaling_extended.svg`.
"""

import os
import sys

# Allocate 2 GPUs for the scaling sweep.
from autocvd import autocvd
autocvd(num_gpus=2)
# ruff: noqa: E402

import jax

# Strong-scaling benchmark runs in x32 — the comparison is wall-clock /
# memory across backends, not numerical convergence (which is exercised
# by ``sound_wave3D.py``), and x32 cuts both per-step compute and temp
# memory roughly in half.
jax.config.update("jax_enable_x64", False)

from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    NATIVE_JAX,
    PALLAS,
    SimulationConfig,
    SnapshotSettings,
    StaticFloatVector,
)
from astronomix.test_setups.hydrodynamics.sound_wave3D import setup_sound_wave

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTESTS_DIR = os.path.dirname(_HERE)
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)
from _benchmark_utils import BenchmarkSpec, run_strong_scaling  # noqa: E402

DATA_DIR = os.path.join(_HERE, "data", "astronomix")
FIG_DIR = os.path.join(_HERE, "figures")

_common_kwargs = dict(
    box_size=StaticFloatVector(3.0, 1.5, 1.5),
    mhd=False,
    dimensionality=3,
    progress_bar=False,
    memory_analysis=True,
    print_elapsed_time=True,
    return_snapshots=True,
    snapshot_settings=SnapshotSettings(return_final_state=True),
)

BENCHMARKS = [
    BenchmarkSpec(
        label="FV (JAX)",
        base_config=SimulationConfig(
            backend=NATIVE_JAX,
            solver_mode=FINITE_VOLUME,
            **_common_kwargs,
        ),
        cfl=0.4,
    ),
    BenchmarkSpec(
        label="FD (JAX)",
        base_config=SimulationConfig(
            backend=NATIVE_JAX,
            solver_mode=FINITE_DIFFERENCE,
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
    BenchmarkSpec(
        label="FD (Pallas)",
        base_config=SimulationConfig(
            backend=PALLAS,
            pallas_block_shape=(4, 4, 8),
            pallas_use_triton=True,
            pallas_interpret=False,
            solver_mode=FINITE_DIFFERENCE,
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
]

run_strong_scaling(
    BENCHMARKS,
    N_values=[32, 64, 128, 256],
    setup_fn=setup_sound_wave,
    num_gpus=2,
    name="sound_wave3D_extended",
    title="3D linear sound wave (extended)",
    data_dir=DATA_DIR,
    figure_dir=FIG_DIR,
)
