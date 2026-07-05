"""
3D Jeans linear-wave benchmark (self-gravity methods-paper test).

Configurations: three FD self-gravity treatments
    - simple source term
    - flux-based source (FD)
    - corrected flux-based source (WENO)

Modes:
    Default (convergence): L1 error and runtime plots across a resolution
        sweep.
    --scaling: strong-scaling sweep on every config (1 GPU vs
        ``NUM_GPUS_SCALING`` GPUs) producing runtime, speedup and per-device
        memory plots.
"""

import os
import sys

NUM_GPUS_SCALING = 4

RUN_SCALING = "--scaling" in sys.argv
RUN_CONVERGENCE = "--convergence" in sys.argv or not RUN_SCALING

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=NUM_GPUS_SCALING if RUN_SCALING else 1)
# ruff: noqa: E402
# =======================

import jax

# Double precision: the perturbation amplitude is eps = 1e-6.
jax.config.update("jax_enable_x64", True)

from astronomix.option_classes.simulation_config import (
    GravityConfig,
    SECOND_ORDER_CONSERVATIVE,
    FINITE_DIFFERENCE,
    SIMPLE_SOURCE,
    FOURTH_ORDER_CONSERVATIVE,
    SimulationConfig,
    SnapshotSettings,
)
from astronomix.test_setups.self_gravity.jeans_waves import (
    jeans_wave_solution,
    setup_jeans_wave,
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
    solver_mode=FINITE_DIFFERENCE,
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
        label="FD, simple source",
        base_config=SimulationConfig(
            gravity_config=GravityConfig(self_gravity=True, self_gravity_version=SIMPLE_SOURCE),
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
    BenchmarkSpec(
        label="FD, flux-based source",
        base_config=SimulationConfig(
            gravity_config=GravityConfig(self_gravity=True, self_gravity_version=SECOND_ORDER_CONSERVATIVE),
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
    BenchmarkSpec(
        label="FD, corrected flux-based source",
        base_config=SimulationConfig(
            gravity_config=GravityConfig(self_gravity=True, self_gravity_version=FOURTH_ORDER_CONSERVATIVE),
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


def test_jeans_wave_convergence():
    run_convergence_and_runtime(
        BENCHMARKS,
        N_values=[8, 16, 32, 48],
        setup_fn=setup_jeans_wave,
        analytic_fn=jeans_wave_solution,
        error_var_indices_fn=_error_indices,
        name="jeans_waves3D",
        title="3D Jeans linear wave",
        data_dir=DATA_DIR,
        figure_dir=FIG_DIR,
    )


def test_jeans_wave_strong_scaling():
    run_strong_scaling(
        BENCHMARKS,
        N_values=[16, 32, 48],
        setup_fn=setup_jeans_wave,
        num_gpus=NUM_GPUS_SCALING,
        name="jeans_waves3D",
        title="3D Jeans linear wave",
        data_dir=DATA_DIR,
        figure_dir=FIG_DIR,
    )


if __name__ == "__main__":
    if RUN_CONVERGENCE:
        test_jeans_wave_convergence()
    if RUN_SCALING:
        test_jeans_wave_strong_scaling()
