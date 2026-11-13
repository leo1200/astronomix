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

# general
import os
import sys

# The number of GPUs to allocate for the scaling sweep is decided from the
# command line before autocvd runs, since autocvd needs the final GPU count.
NUM_GPUS_SCALING = 2

RUN_SCALING = "--scaling" in sys.argv
RUN_CONVERGENCE = "--convergence" in sys.argv or not RUN_SCALING

# ==== GPU selection (a run as a script; under pytest the conftest selects) ====
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=NUM_GPUS_SCALING if RUN_SCALING else 1)
# ruff: noqa: E402
# =======================

# testing
import pytest

# jax
import jax

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    NATIVE_JAX,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SnapshotSettings,
    BackendConfig,
)
from astronomix.option_classes.simulation_config import StaticFloatVector

# astronomix functions
from astronomix.test_setups.hydrodynamics.sound_wave3D import (
    setup_sound_wave,
    sound_wave_solution,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTESTS_DIR = os.path.dirname(_HERE)
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)

# shared benchmark harness (containers + drivers) living one directory up
from _benchmark_utils import (  # noqa: E402
    BenchmarkSpec,
    assert_correctness_at_resolution,
)

# The tiny perturbation amplitudes of a linear sound wave need double precision
# to stay above float rounding noise.
pytestmark = pytest.mark.x64

DATA_DIR = os.path.join(_HERE, "data", "astronomix")
FIG_DIR = os.path.join(_HERE, "figures")


# Options shared by every benchmark configuration; only the backend, solver
# mode and CFL number differ between them.
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
            backend_config=BackendConfig(backend=NATIVE_JAX),
            solver_mode=FINITE_VOLUME,
            **_common_kwargs,
        ),
        cfl=0.4,
    ),
    BenchmarkSpec(
        label="FD (JAX)",
        base_config=SimulationConfig(
            backend_config=BackendConfig(backend=NATIVE_JAX),
            **_common_kwargs,
        ),
        cfl=1.5,
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
    """Return the state indices whose L1 error the convergence test tracks.

    Args:
        registered_variables: The registered variables for the run, mapping
            each primitive field to its index in the state array.

    Returns:
        A tuple of density, the three velocity components and pressure indices.
    """
    return (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
    )


@pytest.mark.slow
def test_sound_wave_convergence():
    assert_correctness_at_resolution(
        BENCHMARKS,
        N=16,
        setup_fn=setup_sound_wave,
        analytic_fn=sound_wave_solution,
        error_var_indices_fn=_error_indices,
        name="sound_wave3D",
        tol=0.005,
    )


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_sound_wave_convergence()
