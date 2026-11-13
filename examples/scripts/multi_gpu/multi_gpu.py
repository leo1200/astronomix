"""
Multi-GPU: run a 3D driven-turbulence simulation sharded across several GPUs.

The primitive state and helper data are domain-decomposed with a NamedSharding
(split along the x axis); astronomix performs the inter-device halo exchange
automatically. Change ``NUM_GPUS`` and the mesh split to scale to more devices.

Building the sharded initial state — three strategies, from simplest to most
memory-efficient:

1. Generate on one GPU, then shard::

       state = construct_primitive_state(...)        # full state on GPU 0
       state = jax.device_put(state, sharding)        # redistribute

   Simplest, but the entire state (and every intermediate field) must fit on a
   single device first and is then copied device-to-device — a memory and
   bandwidth bottleneck that caps the resolution at what one GPU can hold.

2. Generate on the host (CPU), then distribute to all GPUs::

       with jax.default_device(jax.devices("cpu")[0]):
           state = construct_primitive_state(...)     # full state in host RAM
       state = jax.device_put(state, sharding)         # scatter to the GPUs

   Lifts the single-GPU memory limit (host RAM is usually much larger), but
   builds the state on the slow CPU and pays a full host->device transfer.

3. Generate directly on the individual GPUs (used here)::

       initial_state = jax.jit(build_initial_state, out_shardings=sharding)()

   ``jit`` + ``out_shardings`` lets XLA partition the whole construction, so
   every device only ever computes and stores *its own shard* of each field and
   of the assembled state. No single device holds the full state and there is no
   post-hoc reshuffle — the most memory-efficient and fastest option at scale.

Two further memory savers used below:
  * the initial state is assembled inside ``build_initial_state`` so the
    intermediate density / velocity / magnetic-field arrays are local and freed
    once the state is built (only the final state survives the ``jit``);
  * ``donate_state=True`` lets the integrator reuse the input state buffer, and
    the low-storage ``LSRK4`` integrator keeps one fewer full-state buffer
    than the default SSPRK4 (trading a little CFL margin for memory).
"""

# ==== GPU selection ====
from autocvd import autocvd
NUM_GPUS = 2
autocvd(num_gpus=NUM_GPUS)
# ruff: noqa: E402
# =======================

# jax
import jax
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
from astronomix import TurbulenceDrivingConfig, TurbulenceDrivingParams

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

# configure a 3D driven-turbulence box
num_cells = 128
sound_speed = 0.5
B_0 = 0.1

config = SimulationConfig(
    equation_of_state = ISOTHERMAL,
    mhd = True,
    progress_bar = True,
    dimensionality = 3,
    num_cells = num_cells,
    box_size = 1.0,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    turbulence_driving_config = TurbulenceDrivingConfig(turbulence_driving = True),
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
    # target velocity dispersion 1, rms Mach number about 2
    turbulence_driving_params = TurbulenceDrivingParams(velocity = 1.0),
)

# domain-decompose along the x axis across the GPUs; the mesh axes must be
# Auto (newer jax defaults to Explicit), since the helper data is placed with
# with_sharding_constraint
mesh = jax.make_mesh(
    (1, NUM_GPUS, 1, 1),
    (VARAXIS, XAXIS, YAXIS, ZAXIS),
    axis_types=(AxisType.Auto,) * 4,
)
sharding = NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))


def build_initial_state():
    """Assemble the initial primitive state (uniform medium threaded by a
    uniform B_z field, at rest).

    The density / velocity / magnetic-field arrays are local to this function,
    so they are freed once the state has been assembled; only the returned state
    survives. Passing ``sharding`` constrains the assembled state to the device
    layout, and (under the ``out_shardings`` jit below) lets XLA build each field
    directly on the GPU that owns its shard.
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


# strategy 3: build the state directly on the individual GPUs — XLA partitions
# the whole construction, so no single device ever holds the full state
initial_state = jax.jit(build_initial_state, out_shardings=sharding)()

config = finalize_config(config, initial_state.shape)
helper_data = get_helper_data(config, sharding)

jax.debug.visualize_array_sharding(initial_state[registered_variables.density_index, :, :, 0])

# run the simulation across the GPUs
final_state = time_integration(
    initial_state, config, params, registered_variables, sharding=sharding
)

# plot a central density slice
z = num_cells // 2
density = final_state[registered_variables.density_index][:, :, z]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(density.T, origin="lower", cmap="viridis")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(figures_dir / "multi_gpu_density.png", dpi=200, bbox_inches="tight")
