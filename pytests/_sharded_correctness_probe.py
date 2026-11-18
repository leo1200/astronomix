"""Numerical-correctness probe: sharded multi-GPU vs single-GPU FD (Pallas).

Runs the same small problem twice -- once globally sharded across all ranks,
once single-device on rank 0 -- and compares the final states.  Covers all
three weak-scaling setups (hydro sound wave, MHD Alfvén wave, self-gravity
Jeans wave), so a silent halo-exchange / FFT-partitioning bug cannot slip
into the scaling campaign.

Launch under Slurm (one process per GPU)::

    srun --ntasks=4 --ntasks-per-node=4 --gpu-bind=none \
        python pytests/_sharded_correctness_probe.py

Expected output on rank 0: three ``max|diff|`` lines and ``PROBE PASS``.
"""

# general
import os

# third-party (raw jax only before distributed init)
import jax

_cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
_localid = int(os.environ.get("SLURM_LOCALID", "0"))
jax.distributed.initialize(
    local_device_ids=[_localid] if len(_cvd) > 1 else [0]
)

jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import AxisType, PartitionSpec  # noqa: E402

from astronomix.option_classes.simulation_config import (  # noqa: E402
    FINITE_DIFFERENCE,
    PALLAS,
    LSRK4,
    BackendConfig,
    SimulationConfig,
    SnapshotSettings,
    StaticIntVector,
    FDSolverConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams  # noqa: E402
from astronomix.time_stepping.time_integration import time_integration  # noqa: E402
from astronomix.variable_registry.registered_variables import (  # noqa: E402
    get_registered_variables,
)
from astronomix.test_setups.hydrodynamics.sound_wave3D import (  # noqa: E402
    SoundWave3DSettings,
    build_sound_wave_state_sharded,
)
from astronomix.test_setups.mhd.alfven_wave3D import (  # noqa: E402
    CPAlfvenWave3DSettings,
    build_cp_alfven_state_sharded,
)
from astronomix.test_setups.self_gravity.jeans_waves import (  # noqa: E402
    JeansWaveSettings,
    build_jeans_state_sharded,
)

# Small enough that the reference run trivially fits one GPU, large enough
# that every rank owns several stencil widths along the sharded axis.
GRID = (64, 32, 32)
BOX = tuple(float(g) for g in GRID)
STEPS = 20
DT = 0.3


def _base_config(mhd, pallas_ct=False):
    return SimulationConfig(
        backend_config=BackendConfig(
            backend=PALLAS,
            pallas_block_shape=(4, 4, 8),
            pallas_use_triton=True,
            pallas_interpret=False,
            pallas_ct=pallas_ct,
        ),
        solver_mode=FINITE_DIFFERENCE,
        solver_config=FDSolverConfig(time_integrator=LSRK4),
        mhd=mhd,
        dimensionality=3,
        donate_state=False,
        progress_bar=False,
        num_cells=StaticIntVector(*GRID),
        fixed_timestep=True,
        num_timesteps=STEPS,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )


def _run(builder, settings, mhd, sharding, pallas_ct=False):
    config = _base_config(mhd, pallas_ct=pallas_ct)
    params = SimulationParams(C_cfl=1.5)
    state, config, params = builder(config, params, sharding, settings)
    params = params._replace(t_end=DT * STEPS)
    registered_variables = get_registered_variables(config)
    result = time_integration(
        state, config, params, registered_variables, sharding=sharding
    )
    return result.final_state


def main():
    rank = jax.process_index()
    G = jax.device_count()
    mesh = jax.make_mesh(
        (1, G, 1, 1), (0, 1, 2, 3), axis_types=(AxisType.Auto,) * 4
    )
    sharding = jax.NamedSharding(mesh, PartitionSpec(0, 1, 2, 3))

    cases = [
        ("hydro", build_sound_wave_state_sharded,
         SoundWave3DSettings(box_size=BOX), False, False),
        ("mhd", build_cp_alfven_state_sharded,
         CPAlfvenWave3DSettings(box_size=BOX, amplitude=1.0e-6), True, False),
        # Full-amplitude wave with the staged Pallas-CT kernels sharded;
        # the single-device reference below runs the same pallas_ct path.
        ("mhd_pallas_ct", build_cp_alfven_state_sharded,
         CPAlfvenWave3DSettings(box_size=BOX, amplitude=0.1), True, True),
        ("selfgrav", build_jeans_state_sharded,
         JeansWaveSettings(box_size=BOX), False, False),
    ]

    failures = []
    for name, builder, settings, mhd, pallas_ct in cases:
        sharded_final = _run(builder, settings, mhd, sharding, pallas_ct=pallas_ct)
        # Gather the sharded result to every host for comparison.
        gathered = multihost_utils.process_allgather(sharded_final, tiled=True)

        # Single-device reference, computed identically on every rank (cheap
        # at this size and avoids cross-host control flow).
        reference = np.asarray(
            _run(builder, settings, mhd, None, pallas_ct=pallas_ct)
        )

        max_diff = float(np.max(np.abs(np.asarray(gathered) - reference)))
        max_ref = float(np.max(np.abs(reference)))
        finite = bool(np.all(np.isfinite(np.asarray(gathered))))
        if rank == 0:
            print(
                f"[{name}] max|sharded - single| = {max_diff:.3e} "
                f"(max|ref| = {max_ref:.3e}, finite={finite})",
                flush=True,
            )
        # fp32 tolerance: reduction orders differ across the mesh, so demand
        # agreement to ~1e-5 relative, not bit equality.
        if not finite or max_diff > 1.0e-5 * max_ref:
            failures.append((name, max_diff))

    multihost_utils.sync_global_devices("probe_done")
    if rank == 0:
        if failures:
            print(f"PROBE FAIL: {failures}", flush=True)
            raise SystemExit(1)
        print("PROBE PASS", flush=True)


if __name__ == "__main__":
    main()
