"""
3D CP Alfvén wave benchmark (MHD methods-paper test).

Configurations:
    - FV  (VL2 with GLM cleaning, NATIVE_JAX)
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
    python pytests/mhd/test_alfven_wave_3d.py
    python pytests/mhd/test_alfven_wave_3d.py --sp
    python pytests/mhd/test_alfven_wave_3d.py --dp
    python pytests/mhd/test_alfven_wave_3d.py --scaling
    python pytests/mhd/test_alfven_wave_3d.py --convergence --scaling
"""

# general
# NOTE: os/sys and the argv parsing below must run before autocvd, because the
# number of GPUs requested from autocvd is derived from the --scaling flag. The
# testing
import pytest

# jax / astronomix imports are therefore deliberately deferred until after the
# autocvd call (see the E402 waiver below).
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

# ==== GPU selection (a run as a script; under pytest the conftest selects) ====
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=NUM_GPUS_SCALING if RUN_SCALING else 1)
# ruff: noqa: E402
# =======================

# jax
import jax

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    NATIVE_JAX,
    VL2,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SnapshotSettings,
    BackendConfig,
    FVSolverConfig,
)
from astronomix.option_classes.simulation_config import StaticFloatVector

# astronomix functions
from astronomix.test_setups.mhd.alfven_wave3D import (
    setup_cp_alfven_wave,
    cp_alfven_wave_solution,
)


# benchmark helpers
# The shared benchmark module lives in the pytests/ root; make sure that
# directory is importable before pulling it in (it may not be on sys.path when
# this file is run directly rather than through pytest).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTESTS_DIR = os.path.dirname(_HERE)
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)
from _benchmark_utils import (  # noqa: E402
    BenchmarkSpec,
    assert_correctness_at_resolution,
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
        label="FV VL2 (JAX)",
        base_config=SimulationConfig(
            backend_config=BackendConfig(backend=NATIVE_JAX),
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(time_integrator=VL2),
            **_common_kwargs,
        ),
        cfl=0.3,
    ),
    BenchmarkSpec(
        label="FD (Pallas)",
        base_config=SimulationConfig(
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
]


def _error_indices(registered_variables):
    """State-array indices of the variables entered into the L1 error norm.

    The convergence metric averages the error over density, the three velocity
    components, pressure and the three magnetic-field components. Indexing goes
    through ``registered_variables`` so the layout stays in sync with the solver.
    """
    return (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
        registered_variables.magnetic_index.x,
        registered_variables.magnetic_index.y,
        registered_variables.magnetic_index.z,
    )


def _precision_label(precision: str) -> str:
    """Human-readable precision name for plot titles ("dp" -> "double")."""
    return "double" if precision == "dp" else "single"


@pytest.mark.slow
def test_alfven_wave_convergence():
    """Run the convergence + runtime sweep for each requested precision.

    Each precision is run in its own x64 setting (``jax.enable_x64``) with the
    JIT caches cleared in between (single and double precision compile to
    different kernels), and the matching AthenaPK reference overlay is attached
    when its NPZ is present.
    """
    for precision in PRECISIONS:
        with jax.enable_x64(precision == "dp"):
            jax.clear_caches()
            # Per-benchmark tolerances: at N=16 the 2nd-order VL2 scheme sits at
            # L1 ~ 2.2e-2 while the 5th-order FD scheme reaches ~1.2e-3 (both
            # precisions).
            for spec, tol in zip(BENCHMARKS, (0.05, 0.005), strict=True):
                assert_correctness_at_resolution(
                    [spec],
                    N=16,
                    setup_fn=setup_cp_alfven_wave,
                    analytic_fn=cp_alfven_wave_solution,
                    error_var_indices_fn=_error_indices,
                    name=f"alfven_wave3D_{precision}",
                    tol=tol,
                )


if __name__ == "__main__":
    test_alfven_wave_convergence()
