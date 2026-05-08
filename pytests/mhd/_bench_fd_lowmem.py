"""Single-resolution FD bench with weno_low_memory=True. Mirrors _bench_fd.py
otherwise."""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import sys
import time

import jax
jax.config.update("jax_enable_x64", True)

from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    SimulationConfig,
    SnapshotSettings,
    StaticFloatVector,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping.time_integration import time_integration
from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave
from astronomix.variable_registry.registered_variables import get_registered_variables


def bench(N: int):
    base_config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        memory_analysis=True,
        print_elapsed_time=True,
        box_size=StaticFloatVector(3.0, 1.5, 1.5),
        mhd=True,
        dimensionality=3,
        progress_bar=False,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
        weno_low_memory=True,
    )
    config = base_config._replace(num_cells=StaticIntVector(2 * N, N, N))

    initial_state, config, params = setup_cp_alfven_wave(
        config,
        SimulationParams(C_cfl=1.5),
    )
    rv = get_registered_variables(config)

    t0 = time.time()
    result = time_integration(initial_state, config, params, rv)
    jax.block_until_ready(result.final_state)
    wall = time.time() - t0
    print(f"  [first run]  N={N:>3} runtime={result.runtime:.3f}s "
          f"iters={int(result.num_iterations)} "
          f"per-iter={result.runtime/int(result.num_iterations)*1000:.2f} ms "
          f"wall={wall:.2f}s")

    t0 = time.time()
    result = time_integration(initial_state, config, params, rv)
    jax.block_until_ready(result.final_state)
    wall = time.time() - t0
    print(f"  [second run] N={N:>3} runtime={result.runtime:.3f}s "
          f"iters={int(result.num_iterations)} "
          f"per-iter={result.runtime/int(result.num_iterations)*1000:.2f} ms "
          f"wall={wall:.2f}s")
    return result


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    bench(N)
