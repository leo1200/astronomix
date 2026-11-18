"""Magnetically driven jet (3D MHD) — paper figure, finite-difference solver.

A magnetic tower launched from a magnetized central region in an initially
uniform medium (vector-potential initial condition), evolved with the
finite-difference scheme (WENO-Z weights, positivity-preserving WENO,
constrained transport) at C_cfl = 0.75 with SSPRK4, the combination for which
positivity preservation is provable.

The figure is a density slice through the jet axis at t = 5.0. Only that
slice is cached under ``data/``, so the figure can be re-styled without
re-running the (expensive) simulation.

The box is periodic. Nothing reaches its faces before t = 5: the bow shock,
the fastest signal, stays well inside, and the initial field decays as
exp(-r^2), so open boundaries would give the same solution.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/mhd_jet.py [--res N] [--gpus M] [--block bx,by,bz] [--lsrk4] [--rerun]

The run is domain-decomposed along x across ``--gpus M`` devices of one
process, or, when launched by ``srun`` with one task per GPU, across all GPUs
of all tasks (``pytests/runners/mhd_jet.sh``). The positivity-preserving MHD
fluxes need about 100 GB of temporary memory per 2.7e8 cells (fp32, SSPRK4),
so 768^3 is the largest grid on the four GPUs of one node and 1024^3 needs at
least two nodes:

    sbatch --nodes=1 --ntasks-per-node=1 pytests/runners/mhd_jet.sh --res 768 --gpus 4 --rerun
    sbatch --nodes=4 --time=12:00:00 pytests/runners/mhd_jet.sh --res 1024 --rerun

``--lsrk4`` selects the 2N-storage LSRK4 integrator, one full-state buffer
fewer than SSPRK4; it is not strong stability preserving, so positivity
preservation is then no longer provable.

``--ct-upwind`` adds the upwind dissipation of the CT edge EMFs at its default
strength (the setup of the supersonic turbulence); the run is cached and drawn
under ``mhd_jet_fd_<N>_uct`` and compared with the plain run of the same
resolution when that is cached.
"""

# general
import os
import sys


def _num_gpus_from_argv():
    """Return the GPU count requested via ``--gpus N`` (default 1)."""
    if "--gpus" in sys.argv:
        return int(sys.argv[sys.argv.index("--gpus") + 1])
    return 1


def _block_shape_from_argv():
    """Return the Pallas block shape requested via ``--block bx,by,bz``.

    ``None`` (the default) lets astronomix pick its own default block shape.
    A smaller block trades a little kernel throughput for a lower peak-memory
    footprint (smaller halo padding + per-program tile temporaries) and is
    bit-identical, so it is a safe knob for fitting a large run on-device.
    """
    if "--block" in sys.argv:
        raw = sys.argv[sys.argv.index("--block") + 1]
        return tuple(int(p) for p in raw.split(","))
    return None


# Under srun with several tasks every task drives one GPU, and JAX's
# distributed runtime must be initialised before anything (astronomix
# included) creates the backend. Every task sees all GPUs of its node
# (``--gpu-bind=none``) and picks its own by its local rank.
MULTI_PROCESS = int(os.environ.get("SLURM_NTASKS", "1")) > 1
BLOCK_SHAPE = _block_shape_from_argv()

# ==== GPU selection ====
if MULTI_PROCESS:
    import jax
    visible_devices = [
        device for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if device
    ]
    local_device_ids = [int(os.environ["SLURM_LOCALID"])] if len(visible_devices) > 1 else [0]
    jax.distributed.initialize(local_device_ids=local_device_ids)
    NUM_GPUS = jax.device_count()
else:
    from autocvd import autocvd
    NUM_GPUS = _num_gpus_from_argv()
    autocvd(num_gpus=NUM_GPUS)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# jax
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import AxisType, PartitionSpec as P, NamedSharding

# astronomix constants
from astronomix import (
    PALLAS,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import (
    LSRK4,
    SSPRK54,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)
from astronomix.plotting_helpers.paper_style import (
    SEQUENTIAL_COLORMAP,
    COLUMN_WIDTH,
    TEXT_WIDTH,
    TWO_COLUMN,
    layout_path,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
    BoundarySettings,
    BoundarySettings1D,
    BackendConfig,
    FDSolverConfig,
)

# astronomix functions
from astronomix import (
    get_registered_variables,
    construct_primitive_state,
    time_integration,
    finalize_config,
    setup_magnetic_fields_from_vector_potential,
)
from astronomix.plotting_helpers.paper_style import (
    attach_colorbar,
    use_paper_style,
)


# shared figure helpers
from _common import DATA_DIR
from _common import FIG_DIR
from _common import rerun_requested


GAMMA = 5.0 / 3.0
BOX_SIZE = 24.0
T_END = 5.0
RHO_0 = 1.0
P_0 = 1.0
A0 = 20.0

# The CFL number up to which positivity-preserving WENO is provable with
# SSPRK4.
C_CFL = 0.75

# Opt into the low-storage, non-SSP LSRK4 integrator.
USE_LSRK4 = "--lsrk4" in sys.argv
CT_UPWIND = "--ct-upwind" in sys.argv
RUN_SUFFIX = "_uct" if CT_UPWIND else ""


def _resolution_from_argv():
    """Return the per-dimension resolution requested via ``--res N`` (default 256)."""
    if "--res" in sys.argv:
        return int(sys.argv[sys.argv.index("--res") + 1])
    return 256


def cache_path(num_cells, suffix=RUN_SUFFIX):
    """Return the cache path for the FD jet run at ``num_cells``^3."""
    return DATA_DIR / f"mhd_jet_fd_{num_cells}{suffix}.npz"


def make_config(num_cells, boundary_type=PERIODIC_BOUNDARY):
    """Build the jet configuration at ``num_cells``^3.

    Args:
        num_cells: The per-dimension resolution of the cubic grid.
        boundary_type: The boundary condition on every face.

    Returns:
        The (not yet finalized) simulation configuration.
    """
    axis_boundaries = BoundarySettings1D(boundary_type, boundary_type)
    return SimulationConfig(
        mhd=True,
        solver_config=FDSolverConfig(
            time_integrator=LSRK4 if USE_LSRK4 else SSPRK54,
            weno_z=True,
            weno_positivity_preserving=True,
            ct_upwind_dissipation=CT_UPWIND,
        ),
        progress_bar=True,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        grid_spacing=BOX_SIZE / num_cells,
        boundary_settings=BoundarySettings(
            axis_boundaries,
            axis_boundaries,
            axis_boundaries,
        ),
        # Donating the state buffer to the integration is numerically neutral
        # and saves one state-sized buffer; it is used for the multi-GPU
        # (high-resolution) runs, where memory is tight.
        donate_state=(NUM_GPUS > 1),
        # Optional smaller Pallas block shape (``--block bx,by,bz``): a
        # numerics-preserving lever to shave peak device memory (smaller halo
        # padding + tile temporaries). ``None`` keeps the astronomix default.
        backend_config=BackendConfig(
            backend=PALLAS,
            pallas_block_shape=BLOCK_SHAPE,
        ),
        # Return the final state together with the step count, the runtime
        # and the compiled memory footprint.
        return_snapshots=True,
        num_snapshots=2,
        print_elapsed_time=True,
        memory_analysis=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )


def gather_to_host(array):
    """Return the whole of a (possibly multi-process) sharded array as numpy.

    Args:
        array: A jax array, sharded over the devices of one or several
            processes.

    Returns:
        The full array as a numpy array, on every process.
    """
    return np.asarray(multihost_utils.process_allgather(array, tiled=True))


def make_sharding(num_gpus):
    """Return the sharding that splits the x axis across ``num_gpus`` devices.

    With one GPU this is a trivial single-device mesh, so the single- and
    multi-GPU paths share exactly the same code. The mesh axes are Auto
    because newer JAX defaults to Explicit, which the helper data's
    ``with_sharding_constraint`` does not accept.

    Args:
        num_gpus: The number of devices.

    Returns:
        The NamedSharding of the ``(variable, x, y, z)`` state.
    """
    mesh = jax.make_mesh(
        (1, num_gpus, 1, 1),
        (VARAXIS, XAXIS, YAXIS, ZAXIS),
        axis_types=(AxisType.Auto,) * 4,
    )
    return NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))


def build_initial_state(config, registered_variables, sharding, num_cells):
    """Build the sharded initial state of the jet on the devices.

    The magnetic field is seeded from a vector potential so that the
    constrained-transport solver starts from a divergence-free field. The
    construction runs under a jit with ``out_shardings``, so XLA partitions it
    and every device only computes and stores its own shard of each field; no
    single device holds the full state, which is what makes the 1024^3 run fit.

    Args:
        config: The simulation configuration.
        registered_variables: The registered variables.
        sharding: The sharding of the state.
        num_cells: The per-dimension resolution of the cubic grid.

    Returns:
        The initial primitive state, sharded over the devices.
    """
    center = BOX_SIZE / 2.0

    def jet_vector_potential(X, Y, Z):
        r = jnp.sqrt((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)
        A_x = -jnp.exp(-r**2) * (Y - center)
        A_y = jnp.exp(-r**2) * (X - center)
        A_z = 0.5 * A0 * jnp.exp(-r**2)
        return A_x, A_y, A_z

    def build():
        B_x, B_y, B_z, bxb, byb, bzb = setup_magnetic_fields_from_vector_potential(
            config=config,
            vector_potential_func=jet_vector_potential,
        )

        shape = (num_cells, num_cells, num_cells)
        rho = jnp.ones(shape) * RHO_0
        zeros = jnp.zeros(shape)
        p = jnp.ones(shape) * P_0

        return construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=rho,
            velocity_x=zeros,
            velocity_y=zeros,
            velocity_z=zeros,
            gas_pressure=p,
            magnetic_field_x=B_x,
            magnetic_field_y=B_y,
            magnetic_field_z=B_z,
            interface_magnetic_field_x=bxb,
            interface_magnetic_field_y=byb,
            interface_magnetic_field_z=bzb,
            sharding=sharding,
        )

    return jax.jit(build, out_shardings=sharding)()


def simulate(num_cells):
    """Run the finite-difference MHD jet simulation at ``num_cells``^3.

    Args:
        num_cells: The per-dimension resolution of the cubic grid.

    Returns:
        A dict with the density slice through the jet axis (the x-z plane at
        mid-y), the extrema of the final state, the number of steps, the
        runtime and the compiled temporary memory per device.
    """
    config = make_config(num_cells)
    registered_variables = get_registered_variables(config)
    sharding = make_sharding(NUM_GPUS)

    initial_state = build_initial_state(config, registered_variables, sharding, num_cells)

    params = SimulationParams(
        C_cfl=C_CFL,
        dt_max=0.1,
        t_end=T_END,
        gamma=GAMMA,
    )

    config = finalize_config(config, initial_state.shape)
    result = time_integration(
        initial_state,
        config,
        params,
        registered_variables,
        sharding=sharding,
    )
    final_state = result.final_state

    density = final_state[registered_variables.density_index]
    pressure = final_state[registered_variables.pressure_index]
    summary = dict(
        density_slice=gather_to_host(density[:, num_cells // 2, :]),
        minimum_density=float(jnp.min(density)),
        minimum_pressure=float(jnp.min(pressure)),
        all_finite=bool(jnp.all(jnp.isfinite(final_state))),
        num_iterations=int(result.num_iterations),
        runtime=float(result.runtime),
        temporary_memory_bytes=int(result.temporary_memory_bytes),
    )
    print(
        f"  {summary['num_iterations']} steps in {summary['runtime']:.1f} s, "
        f"temporary memory {summary['temporary_memory_bytes'] / 1024**3:.1f} GiB per device, "
        f"min density {summary['minimum_density']:.3e}, "
        f"min pressure {summary['minimum_pressure']:.3e}, "
        f"finite: {summary['all_finite']}"
    )
    return summary


def get_run(num_cells, rerun):
    """Return the jet-axis density slice, from cache or by (re-)running.

    Args:
        num_cells: The per-dimension resolution of the run.
        rerun: When True, ignore any cache and re-run the simulation.

    Returns:
        The density slice through the jet axis as a numpy array.
    """
    path = cache_path(num_cells)
    if path.exists() and not rerun:
        return np.load(path)["density_slice"]
    print(f"running FD MHD jet at {num_cells}^3 on {NUM_GPUS} GPU(s) ...")
    summary = simulate(num_cells)
    # Every process holds the gathered slice; one writes the cache.
    if jax.process_index() == 0:
        np.savez(path, **summary)
        print(f"  cached -> {path}")
    return summary["density_slice"]


def plot(num_cells, rerun):
    """Render the jet-axis density slice figure.

    Loads (or regenerates) the cached density slice through the jet axis (the
    x-z plane at mid-y) and saves the figure as ``.png``, ``.pdf`` and
    ``.svg`` (``mhd_jet_fd_{num_cells}``).

    Args:
        num_cells: The per-dimension resolution of the run to plot.
        rerun: When True, force the underlying simulation to be re-run.
    """
    density_slice = get_run(num_cells, rerun).T
    plain_path = cache_path(num_cells, suffix="")
    if CT_UPWIND and plain_path.exists():
        plain_slice = np.load(plain_path)["density_slice"].T
        print(
            f"with vs without the upwind EMF dissipation at {num_cells}^3: mean |density difference| "
            f"{np.mean(np.abs(density_slice - plain_slice)):.3e} (mean density {np.mean(plain_slice):.3e}, "
            f"max |difference| {np.max(np.abs(density_slice - plain_slice)):.3e})"
        )

    use_paper_style()
    width = COLUMN_WIDTH if TWO_COLUMN else 0.7 * TEXT_WIDTH
    fig, ax = plt.subplots(1, 1, figsize=(width, 0.886 * width))
    image = ax.imshow(
        density_slice,
        origin="lower",
        extent=(0, BOX_SIZE, 0, BOX_SIZE),
        cmap=SEQUENTIAL_COLORMAP,
    )
    ax.set_aspect("equal", adjustable="box")
    # Bare density panel: no axis labels or tick numbers.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    attach_colorbar(fig, image, ax, label="density")
    fig.tight_layout()

    # High raster resolution so the density panel stays crisp, including the
    # image embedded (rasterized) inside the PDF/SVG vector outputs.
    dpi = 600
    out = layout_path(FIG_DIR / f"mhd_jet_fd_{num_cells}{RUN_SUFFIX}.png")
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out.with_suffix(suffix), dpi=dpi)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    num_cells = _resolution_from_argv()
    if MULTI_PROCESS and jax.process_index() != 0:
        # The other tasks only take part in the simulation.
        get_run(num_cells, rerun_requested())
    else:
        plot(num_cells, rerun_requested())
