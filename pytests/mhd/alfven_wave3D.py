"""
3D CP Alfvén wave benchmark (MHD methods-paper test).

Configurations:
    - FV  (NATIVE_JAX)
    - FD  (Pallas)
    - AthenaPK overlay (loaded from npz)

Modes:
    Default (convergence): runs both single (x32) and double (x64) precision
        convergence sweeps, producing one figure per precision with the
        matching AthenaPK overlay (athenapk_alfven_convergence_{sp,dp}.npz).
    --sp / --dp: restrict to one precision.
    --scaling: strong-scaling sweep on every config (1 GPU vs
        ``NUM_GPUS_SCALING`` GPUs) producing runtime, speedup and per-device
        memory plots.

Examples:
    python pytests/mhd/alfven_wave3D.py
    python pytests/mhd/alfven_wave3D.py --sp
    python pytests/mhd/alfven_wave3D.py --dp
    python pytests/mhd/alfven_wave3D.py --scaling
    python pytests/mhd/alfven_wave3D.py --convergence --scaling
"""

import os
import sys

NUM_GPUS_SCALING = 4

RUN_SCALING = "--scaling" in sys.argv
RUN_CONVERGENCE = "--convergence" in sys.argv or not RUN_SCALING

PRECISIONS = []
if "--sp" in sys.argv:
    PRECISIONS.append("sp")
if "--dp" in sys.argv:
    PRECISIONS.append("dp")
if not PRECISIONS:
    PRECISIONS = ["sp", "dp"]

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=NUM_GPUS_SCALING if RUN_SCALING else 1)
# ruff: noqa: E402
# =======================

import jax

from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    NATIVE_JAX,
    PALLAS,
    SimulationConfig,
    SnapshotSettings,
    StaticFloatVector,
)
from astronomix.test_setups.mhd.alfven_wave3D import (
    setup_cp_alfven_wave,
    cp_alfven_wave_solution,
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
ATHENAPK_DIR = os.path.join(_HERE, "data", "athenapk")


_common_kwargs = dict(
    box_size=StaticFloatVector(3.0, 1.5, 1.5),
    mhd=True,
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
        rv.magnetic_index.x, rv.magnetic_index.y, rv.magnetic_index.z,
    )


def _precision_label(precision: str) -> str:
    return "double" if precision == "dp" else "single"


def test_alfven_wave_convergence():
    for precision in PRECISIONS:
        jax.config.update("jax_enable_x64", precision == "dp")
        jax.clear_caches()
        athenapk_npz = os.path.join(
            ATHENAPK_DIR, f"athenapk_alfven_convergence_{precision}.npz"
        )
        run_convergence_and_runtime(
            BENCHMARKS,
            N_values=[8, 16, 32, 64, 128],
            setup_fn=setup_cp_alfven_wave,
            analytic_fn=cp_alfven_wave_solution,
            error_var_indices_fn=_error_indices,
            name=f"alfven_wave3D_{precision}",
            title=f"3D CP Alfvén wave ({_precision_label(precision)} precision)",
            data_dir=DATA_DIR,
            figure_dir=FIG_DIR,
            athenapk_npz=athenapk_npz if os.path.exists(athenapk_npz) else None,
        )


def test_alfven_wave_strong_scaling():
    run_strong_scaling(
        BENCHMARKS,
        N_values=[16, 32, 64, 128],
        setup_fn=setup_cp_alfven_wave,
        num_gpus=NUM_GPUS_SCALING,
        name="alfven_wave3D",
        title="3D CP Alfvén wave",
        data_dir=DATA_DIR,
        figure_dir=FIG_DIR,
    )


if __name__ == "__main__":
    if RUN_CONVERGENCE:
        test_alfven_wave_convergence()
    if RUN_SCALING:
        test_alfven_wave_strong_scaling()
