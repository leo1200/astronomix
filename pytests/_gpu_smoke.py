"""Single-GPU smoke test: tiny FD (Pallas) sound-wave run on one GPU.

Verifies the freshly built environment end-to-end: CUDA-enabled jaxlib,
Triton/Pallas kernel compilation on the GH200, and the astronomix FD solver
pipeline.  Launch through autocvd so a free GPU is picked on shared nodes::

    autocvd -n 1 -- python pytests/_gpu_smoke.py
"""

# general
import time

# third-party
import jax

# The scaling campaign convention: fp32 + GSPMD partitioner.
jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)

# astronomix
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    PALLAS,
    LSRK4,
    BackendConfig,
    SimulationConfig,
    SnapshotSettings,
    FDSolverConfig,
)
from astronomix.test_setups.hydrodynamics.sound_wave3D import (
    SoundWave3DSettings,
    build_sound_wave_state_sharded,
)
from astronomix.option_classes.simulation_config import StaticIntVector
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import (
    get_registered_variables,
)
from astronomix.time_stepping.time_integration import time_integration


def main():
    print("devices:", jax.devices(), flush=True)

    config = SimulationConfig(
        backend_config=BackendConfig(
            backend=PALLAS,
            pallas_block_shape=(4, 4, 8),
            pallas_use_triton=True,
            pallas_interpret=False,
        ),
        solver_mode=FINITE_DIFFERENCE,
        solver_config=FDSolverConfig(time_integrator=LSRK4),
        mhd=False,
        dimensionality=3,
        donate_state=True,
        progress_bar=False,
        print_elapsed_time=True,
        num_cells=StaticIntVector(64, 64, 64),
        fixed_timestep=True,
        num_timesteps=10,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    params = SimulationParams(C_cfl=1.5)
    settings = SoundWave3DSettings(box_size=(64.0, 64.0, 64.0))

    state, config, params = build_sound_wave_state_sharded(
        config, params, sharding=None, settings=settings
    )
    params = params._replace(t_end=0.4 * 10)

    registered_variables = get_registered_variables(config)
    start = time.time()
    result = time_integration(state, config, params, registered_variables)
    result.final_state.block_until_ready()
    elapsed = time.time() - start

    finite = bool(jax.numpy.all(jax.numpy.isfinite(result.final_state)))
    print(f"ran {int(result.num_iterations)} steps in {elapsed:.2f}s, "
          f"finite={finite}", flush=True)
    assert finite, "non-finite state after smoke run"
    print("SMOKE PASS", flush=True)


if __name__ == "__main__":
    main()
