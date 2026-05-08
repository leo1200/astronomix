"""FD-only correctness check on the 3D Alfven wave.

Verifies that the optimized WENO refactor still matches a 5th-order
convergence rate (typical FD slope on this test).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

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
from astronomix.time_stepping.time_integration import time_integration
from astronomix.test_setups.mhd.alfven_wave3D import (
    setup_cp_alfven_wave,
    cp_alfven_wave_solution,
)
from astronomix.variable_registry.registered_variables import get_registered_variables


def run(N: int):
    base_config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        memory_analysis=False,
        print_elapsed_time=False,
        box_size=StaticFloatVector(3.0, 1.5, 1.5),
        mhd=True,
        dimensionality=3,
        progress_bar=False,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    config = base_config._replace(num_cells=StaticIntVector(2 * N, N, N))
    initial_state, config, params = setup_cp_alfven_wave(
        config, SimulationParams(C_cfl=1.5)
    )
    rv = get_registered_variables(config)
    helper_data = get_helper_data(config)

    result = time_integration(initial_state, config, params, rv)
    final = result.final_state
    truth = cp_alfven_wave_solution(config, rv, params, helper_data)

    err_rho = jnp.mean(jnp.abs(final[rv.density_index] - truth[rv.density_index]))
    err_vx = jnp.mean(jnp.abs(final[rv.velocity_index.x] - truth[rv.velocity_index.x]))
    err_vy = jnp.mean(jnp.abs(final[rv.velocity_index.y] - truth[rv.velocity_index.y]))
    err_vz = jnp.mean(jnp.abs(final[rv.velocity_index.z] - truth[rv.velocity_index.z]))
    err_p = jnp.mean(jnp.abs(final[rv.pressure_index] - truth[rv.pressure_index]))
    err_bx = jnp.mean(jnp.abs(final[rv.magnetic_index.x] - truth[rv.magnetic_index.x]))
    err_by = jnp.mean(jnp.abs(final[rv.magnetic_index.y] - truth[rv.magnetic_index.y]))
    err_bz = jnp.mean(jnp.abs(final[rv.magnetic_index.z] - truth[rv.magnetic_index.z]))
    total = (err_rho + err_vx + err_vy + err_vz + err_p + err_bx + err_by + err_bz) / 8
    return float(total), float(result.runtime), int(result.num_iterations)


if __name__ == "__main__":
    Ns = [8, 16, 32, 64]
    rows = []
    for N in Ns:
        err, rt, it = run(N)
        rows.append((N, err, rt, it))
        if len(rows) > 1:
            ratio = rows[-2][1] / rows[-1][1]
            slope = np.log2(ratio)
        else:
            slope = float("nan")
        print(
            f"N={N:>3}  L1={err:.3e}  rate={slope:.2f}  "
            f"runtime={rt:.2f}s  iters={it}  per-iter={rt/it*1000:.2f}ms"
        )
