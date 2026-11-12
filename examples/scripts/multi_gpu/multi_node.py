"""
Multi-node: run a 3D driven-turbulence simulation sharded across GPUs on
several nodes.

This is the multi-node counterpart of ``multi_gpu.py``. The differences are all
about the multi-process launch (one process per GPU, under a scheduler such as
Slurm); the physics and the sharded initial-condition construction are identical.
A ready-to-adapt Slurm runner lives next to this file: ``multi_node.sh``.

Launch model: one process per GPU. Under Slurm::

    srun --gpu-bind=none python multi_node.py

``jax.distributed.initialize()`` auto-detects the coordinator, process count and
process id from the Slurm environment. Two things are essential and are handled
below:

  * Init ordering — ``jax.distributed.initialize()`` must run before ANY JAX
    call that creates the backend, and *importing astronomix already creates the
    backend* (some option NamedTuples have ``jnp.array`` defaults). So this file
    bootstraps distributed mode with raw ``jax`` FIRST, then imports astronomix.

  * GPU binding — launch with ``srun --gpu-bind=none`` so every task sees all of
    its node's GPUs (needed for intra-node NCCL peer-to-peer), and let each rank
    pick its own device via ``local_device_ids=[SLURM_LOCALID]``. The tempting
    ``--gpus-per-task=1`` instead cgroup-binds each task to a single GPU that
    shows up as ordinal 0, which breaks NCCL P2P and *deadlocks* the topology
    exchange with ``invalid device ordinal``.

As in ``multi_gpu.py`` (see its docstring for the three initial-state
strategies) the state is built directly on the individual devices with
``jax.jit(build_initial_state, out_shardings=sharding)`` — in a multi-controller
run this makes every process compute only its own shards, so the full state is
never replicated on any node. ``donate_state`` and the low-storage ``LSRK4``
integrator further cut the peak memory.
"""

# ==== distributed initialization (must precede any other JAX use — and any
# astronomix import, which itself creates the JAX backend) ====
import os
import jax


def _init_distributed():
    """Bootstrap JAX multi-process mode when launched across ranks (no-op on a
    single process, so the same file also runs on one GPU).

    Robust to both Slurm GPU-binding modes: with ``--gpu-bind=none`` all node
    GPUs are visible and this rank selects its device by node-local rank
    (``SLURM_LOCALID``); with a single cgroup-bound GPU it falls back to the only
    visible device (ordinal 0).
    """
    multiprocess = (
        int(os.environ.get("SLURM_NTASKS", "1")) > 1
        or int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1
    )
    if not multiprocess:
        return
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    local_rank = int(
        os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or "0"
    )
    local_device_ids = [local_rank] if len(visible) > 1 else [0]
    jax.distributed.initialize(local_device_ids=local_device_ids)


_init_distributed()

# JAX >= 0.10 defaults to the Shardy partitioner, which rejects this codebase's
# integer mesh-axis names and its ``with_sharding_constraint`` usage; fall back
# to the GSPMD partitioner the sharding code was written for. (Set AFTER init —
# touching jax.config before initialize() can create the backend early.)
try:
    jax.config.update("jax_use_shardy_partitioner", False)
except Exception:
    pass
# =====================================================================

# jax
import jax.numpy as jnp
from jax.sharding import (
    AxisType,
    PartitionSpec as P,
    NamedSharding,
)

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import (
    ISOTHERMAL,
    LSRK4,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)

# astronomix functions
from astronomix import (
    time_integration,
    get_helper_data,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
    initialize_interface_fields,
)


# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

if jax.process_index() == 0:
    print(f"Running on {jax.process_count()} processes, "
          f"{jax.device_count()} devices in total.")

# configure a 3D driven-turbulence box
num_cells = 256
sound_speed = 0.5
B_0 = 0.1

config = SimulationConfig(
    equation_of_state = ISOTHERMAL,
    mhd = True,
    progress_bar = jax.process_index() == 0,
    dimensionality = 3,
    num_cells = num_cells,
    box_size = 1.0,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    turbulent_forcing_config = TurbulentForcingConfig(turbulent_forcing = True),
    # let the integrator reuse the state buffer and use the 2N-storage LSRK4
    # integrator — both cut the peak memory of the run
    donate_state = True,
    solver_config=FDSolverConfig(time_integrator=LSRK4),
)

registered_variables = get_registered_variables(config)

params = SimulationParams(
    C_cfl = 1.5,
    isothermal_sound_speed = sound_speed,
    t_end = 2.0 * 0.5,
    minimum_density = 0.02,
    turbulent_forcing_params = TurbulentForcingParams(energy_injection_rate = 1.65),
)

# domain-decompose across ALL global devices along the x axis. The mesh axes are
# Auto (not the JAX>=0.10 default Explicit) so with_sharding_constraint works.
mesh = jax.make_mesh(
    (1, jax.device_count(), 1, 1),
    (VARAXIS, XAXIS, YAXIS, ZAXIS),
    axis_types=(AxisType.Auto,) * 4,
)
sharding = NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))


def build_initial_state():
    """Assemble the initial primitive state (uniform medium threaded by a
    uniform B_z field, at rest).

    The intermediate fields are local to this function and freed once the state
    is built. Under the ``out_shardings`` jit below each process computes only
    the shards it owns, so no node holds a full replicated copy of the state.
    """
    rho = jnp.ones((num_cells, num_cells, num_cells))
    u_x = jnp.zeros_like(rho)
    u_y = jnp.zeros_like(rho)
    u_z = jnp.zeros_like(rho)
    B_x = jnp.zeros_like(rho)
    B_y = jnp.zeros_like(rho)
    B_z = B_0 * jnp.ones_like(rho)
    bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

    return construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u_x,
        velocity_y = u_y,
        velocity_z = u_z,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        interface_magnetic_field_x = bxb,
        interface_magnetic_field_y = byb,
        interface_magnetic_field_z = bzb,
        sharding = sharding,
    )


# build the state directly on the individual devices across all nodes
initial_state = jax.jit(build_initial_state, out_shardings=sharding)()

config = finalize_config(config, initial_state.shape)
helper_data = get_helper_data(config, sharding)

# run the simulation across all nodes
final_state = time_integration(
    initial_state, config, params, registered_variables, sharding=sharding
)

# gather a central density slice to process 0 and plot it there
z = num_cells // 2
density_slice = jax.device_get(final_state[registered_variables.density_index][:, :, z])

if jax.process_index() == 0:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(density_slice.T, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(figures_dir / "multi_node_density.png", dpi=200, bbox_inches="tight")
