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

# general
import os
import sys

# The number of GPUs requested from autocvd depends on the run mode, which is
# parsed from argv here — before the GPU block and the first jax import.
NUM_GPUS_SCALING = 4

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
from astronomix import FINITE_DIFFERENCE
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix import (
    GravityConfig,
    SimulationConfig,
    SnapshotSettings,
)

# astronomix functions
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
    assert_correctness_at_resolution,
)

# The perturbation amplitude is eps = 1e-6, so we run in double precision to keep
# the linear wave from being swamped by round-off.
pytestmark = pytest.mark.x64

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
            gravity_config=GravityConfig(
                self_gravity=True,
                self_gravity_version=SIMPLE_SOURCE,
            ),
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
    BenchmarkSpec(
        label="FD, flux-based source",
        base_config=SimulationConfig(
            gravity_config=GravityConfig(
                self_gravity=True,
                self_gravity_version=SECOND_ORDER_CONSERVATIVE,
            ),
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
    BenchmarkSpec(
        label="FD, corrected flux-based source",
        base_config=SimulationConfig(
            gravity_config=GravityConfig(
                self_gravity=True,
                self_gravity_version=FOURTH_ORDER_CONSERVATIVE,
            ),
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
]


def _error_indices(registered_variables):
    """Return the state-array indices whose L1 error the benchmark tracks.

    Args:
        registered_variables: The registered variables of the run.

    Returns:
        A tuple of the density, three velocity component and pressure indices.
    """
    return (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
    )


@pytest.mark.slow
def test_jeans_wave_convergence():
    """Run the resolution sweep and emit the L1-error and runtime plots."""
    assert_correctness_at_resolution(
        BENCHMARKS,
        N=16,
        setup_fn=setup_jeans_wave,
        analytic_fn=jeans_wave_solution,
        error_var_indices_fn=_error_indices,
        name="jeans_waves3D",
        tol=0.05,
    )


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_jeans_wave_convergence()
