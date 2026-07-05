"""
3D linear sound-wave benchmark (pure-hydro methods-paper test).

Configurations:
    - FV  (NATIVE_JAX)
    - FD  (NATIVE_JAX)
    - FD  (Pallas)

Modes:
    Default (convergence): L1 error and runtime plots across a resolution
        sweep.
    --scaling: strong-scaling sweep on every config (1 GPU vs
        ``NUM_GPUS_SCALING`` GPUs) producing runtime, speedup and per-device
        memory plots.
"""

import os
import sys

NUM_GPUS_SCALING = 2

RUN_SCALING = "--scaling" in sys.argv
RUN_CONVERGENCE = "--convergence" in sys.argv or not RUN_SCALING

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=NUM_GPUS_SCALING if RUN_SCALING else 1)
# ruff: noqa: E402
# =======================

import jax

# Double precision for tiny perturbation amplitudes.
jax.config.update("jax_enable_x64", True)

from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    NATIVE_JAX,
    PALLAS,
    SimulationConfig,
    SnapshotSettings,
    StaticFloatVector,
)
from astronomix.test_setups.hydrodynamics.sound_wave3D import (
    setup_sound_wave,
    sound_wave_solution,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTESTS_DIR = os.path.dirname(_HERE)
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)
from _benchmark_utils import (  # noqa: E402
    BenchmarkSpec,
    run_convergence_and_runtime,
    run_strong_scaling,
)

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


def _error_indices(rv):
    return (
        rv.density_index,
        rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z,
        rv.pressure_index,
    )


def test_sound_wave_convergence():
    run_convergence_and_runtime(
        BENCHMARKS,
        N_values=[8, 16, 32, 64, 128],
        setup_fn=setup_sound_wave,
        analytic_fn=sound_wave_solution,
        error_var_indices_fn=_error_indices,
        name="sound_wave3D",
        title="3D linear sound wave",
        data_dir=DATA_DIR,
        figure_dir=FIG_DIR,
    )


def test_sound_wave_strong_scaling():
    run_strong_scaling(
        BENCHMARKS,
        N_values=[16, 32, 64, 128],
        setup_fn=setup_sound_wave,
        num_gpus=NUM_GPUS_SCALING,
        name="sound_wave3D",
        title="3D linear sound wave",
        data_dir=DATA_DIR,
        figure_dir=FIG_DIR,
    )


if __name__ == "__main__":
    if RUN_CONVERGENCE:
        test_sound_wave_convergence()
    if RUN_SCALING:
        test_sound_wave_strong_scaling()
