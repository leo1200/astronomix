"""Compare the two FD WENO paths (weno_low_memory False vs True).

Runs both at N=64 and reports:
  * L1 error vs analytic alfven solution (correctness)
  * second-run per-iter time (warm)
  * compiled temp memory
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    SimulationConfig,
    SnapshotSettings,
    StaticFloatVector,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.test_setups.mhd.alfven_wave3D import (
    cp_alfven_wave_solution,
    setup_cp_alfven_wave,
)
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables


def run(N: int, low_memory: bool):
    base_config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        memory_analysis=True,
        print_elapsed_time=False,
        box_size=StaticFloatVector(3.0, 1.5, 1.5),
        mhd=True,
        dimensionality=3,
        progress_bar=False,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
        weno_low_memory=low_memory,
    )
    config = base_config._replace(num_cells=StaticIntVector(2 * N, N, N))
    initial_state, config, params = setup_cp_alfven_wave(
        config, SimulationParams(C_cfl=1.5)
    )
    rv = get_registered_variables(config)
    helper_data = get_helper_data(config)

    # warm-up run (JIT compile)
    result = time_integration(initial_state, config, params, rv)
    jax.block_until_ready(result.final_state)

    # measured run
    t0 = time.time()
    result = time_integration(initial_state, config, params, rv)
    jax.block_until_ready(result.final_state)
    wall = time.time() - t0

    truth = cp_alfven_wave_solution(config, rv, params, helper_data)
    final = result.final_state
    err = sum(
        float(jnp.mean(jnp.abs(final[i] - truth[i])))
        for i in range(initial_state.shape[0] - 3)  # skip the 3 staggered B fields
    ) / (initial_state.shape[0] - 3)

    return {
        "low_memory": low_memory,
        "L1": err,
        "runtime": float(result.runtime),
        "iters": int(result.num_iterations),
        "per_iter_ms": float(result.runtime) / int(result.num_iterations) * 1000.0,
        "wall": wall,
    }


if __name__ == "__main__":
    N = 64
    print(f"=== N={N} comparison ===")
    for lm in (False, True):
        r = run(N, lm)
        label = "low_memory" if lm else "performance"
        print(
            f"  [{label:>11}] L1={r['L1']:.3e}  "
            f"per-iter={r['per_iter_ms']:.2f} ms  "
            f"iters={r['iters']}  wall={r['wall']:.1f}s"
        )
