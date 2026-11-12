"""
Shared set-up of the sharded scaling benchmarks (weak scaling, strong scaling
and the sharded-correctness check).

All of them run the same four configurations, each on the Pallas backend in
single precision with a 1D decomposition along x and one process per node
driving all of its GPUs:

- ``fd``: the finite-difference scheme (WENO-Z weights, admissible face
  state, LSRK4); for MHD with constrained transport on the staged Pallas CT
  kernels (``pallas_ct``).
- ``fv``: AthenaPK's VL2 finite-volume scheme (PLM with harmonic van Leer
  slopes, HLLC for hydrodynamics, HLLD + GLM divergence cleaning for MHD).

The hydro benchmark is the 3D sound wave, the MHD benchmark the CP Alfvén
wave, both built already globally sharded (each process materialises only
its own shard), with the box numerically equal to the grid so the cell size
is h = 1 at every grid size.

The distributed bootstrap must run before astronomix is imported (importing
astronomix creates the JAX backend, and ``jax.distributed.initialize`` must
come first), so this module imports astronomix only inside the functions that
build configurations and states.
"""

# general
import os

# jax
import jax


#: The two schemes of the benchmarks and their labels in tables and figures.
SCHEME_LABELS = {
    "fd": "FD (WENO-Z, Pallas)",
    "fv": "FV (VL2, Pallas)",
}

#: Pallas block shape per scheme: ``None`` selects the library default,
#: 128 cells filled from the contiguous z axis, i.e. (1, 1, 128) on these
#: grids. On GH200 it is the fastest shape measured for both schemes (FD MHD
#: 7 % and FD hydro 3 % faster than (1, 4, 32), ~30 % faster than (4, 4, 8)),
#: and its block extent of 1 along the decomposed x axis keeps the
#: block-rounded halo at the stencil reach.
DEFAULT_BLOCK_SHAPES = {
    "fd": None,
    "fv": None,
}

#: Fixed time step per scheme for the h = 1 benchmark grids (the largest
#: signal speed is 1 for the sound wave and ~1.08 for the Alfvén-wave
#: background). The FD scheme with LSRK4 is stable well beyond this; for the
#: unsplit VL2 scheme 0.3 is AthenaPK's 3D Courant number. The cost of a step
#: does not depend on dt.
DEFAULT_TIME_STEPS = {
    "fd": 0.4,
    "fv": 0.3,
}


def _tasks_on_this_node() -> int:
    """
    The number of processes of this Slurm step on this node (1 if unknown).

    Slurm writes the step layout as e.g. ``"1(x8)"`` or ``"4,2"``; all of our
    launches place the same number of tasks on every node, so the leading
    count applies.
    """
    layout = os.environ.get(
        "SLURM_STEP_TASKS_PER_NODE",
        os.environ.get("SLURM_TASKS_PER_NODE", "1"),
    )
    leading_digits = ""
    for character in layout:
        if not character.isdigit():
            break
        leading_digits += character
    return int(leading_digits) if leading_digits else 1


def bootstrap_distributed(single_process_gpus: int = 0) -> None:
    """
    Start JAX in multi-process mode under ``srun``, or claim local GPUs for a
    single process, and fix the JAX options the sharded runs need. Must be
    called before astronomix is imported.

    Two process layouts are supported. With one process per node (the
    default of the runners) every process drives all GPUs of its node:
    intra-node halo exchanges then stay inside one process, which on
    JUPITER makes the multi-node rungs about 30% faster than with one
    process per GPU. With one process per GPU (``srun --gpu-bind=none``, all
    node GPUs visible to every task) each process selects its own GPU
    through ``local_device_ids``; with a cgroup binding
    (``--gpus-per-task=1``) the single visible GPU is ordinal 0.
    ``jax.config.update`` may itself initialise the backend, so it runs only
    after the rendezvous.

    Args:
        single_process_gpus: Outside a multi-task Slurm step, claim this many
            free local GPUs via autocvd (0 leaves the visible devices alone).
    """
    multi_process = (
        "SLURM_PROCID" in os.environ
        and int(os.environ.get("SLURM_NTASKS", "1")) > 1
    )

    if not multi_process and single_process_gpus > 0:
        # local import: autocvd is only needed in the single-process mode
        from autocvd import autocvd

        autocvd(num_gpus=single_process_gpus)

    if multi_process:
        visible_devices = [
            device
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device
        ]
        if len(visible_devices) <= 1:
            local_device_ids = [0]
        elif _tasks_on_this_node() == 1:
            local_device_ids = list(range(len(visible_devices)))
        else:
            local_device_ids = [int(os.environ.get("SLURM_LOCALID", "0"))]
        jax.distributed.initialize(local_device_ids=local_device_ids)

    # The code uses integer mesh-axis names with with_sharding_constraint
    # (GSPMD style), which the Shardy partitioner rejects.
    jax.config.update("jax_use_shardy_partitioner", False)
    # All scaling runs are single precision.
    jax.config.update("jax_enable_x64", False)


def benchmark_configuration(
    scheme: str,
    physics: str,
    block_shape: tuple,
    pallas_ct: bool = True,
    halo_exchange: str = "pair_all_to_all",
):
    """
    The memory-lean simulation configuration of one benchmark.

    Args:
        scheme: ``"fd"`` or ``"fv"``.
        physics: ``"hydro"`` or ``"mhd"``.
        block_shape: The Pallas block shape ``(bx, by, bz)``, or None for the
            library default.
        pallas_ct: Route the FD constrained-transport helpers through their
            Pallas kernels (one halo exchange per kernel under sharding
            instead of one collective permute per stencil shift). Only used
            by FD MHD.
        halo_exchange: ``BackendConfig.halo_exchange`` by name: ``"ppermute"``
            (ring, ``PPERMUTE``) or ``"pair_all_to_all"`` (disjoint neighbour
            pairs, ``PAIR_ALL_TO_ALL``).

    Returns:
        The SimulationConfig without the grid (set per run).
    """
    # astronomix is imported here, after the distributed bootstrap.
    from astronomix.option_classes.simulation_config import (
        FINITE_DIFFERENCE,
        FINITE_VOLUME,
        HLLC,
        HLLD,
        PAIR_ALL_TO_ALL,
        PALLAS,
        PPERMUTE,
        LSRK4,
        VL2,
    )
    from astronomix.option_classes.simulation_config import (
        BackendConfig,
        FDSolverConfig,
        FVSolverConfig,
        SimulationConfig,
    )

    if physics not in ("hydro", "mhd"):
        raise ValueError(f"unknown physics {physics!r} (expected 'hydro' or 'mhd')")
    mhd = physics == "mhd"

    backend_config = BackendConfig(
        backend=PALLAS,
        pallas_block_shape=None if block_shape is None else tuple(block_shape),
        pallas_use_triton=True,
        pallas_interpret=False,
        pallas_ct=bool(pallas_ct and mhd and scheme == "fd"),
        halo_exchange={"ppermute": PPERMUTE, "pair_all_to_all": PAIR_ALL_TO_ALL}[halo_exchange],
    )
    # Donating the state and the fixed step count keep the per-GPU footprint
    # and the wall clock bounded; the timer excludes compilation.
    common_options = dict(
        backend_config=backend_config,
        mhd=mhd,
        dimensionality=3,
        donate_state=True,
        progress_bar=False,
        print_elapsed_time=True,
    )

    if scheme == "fd":
        return SimulationConfig(
            solver_mode=FINITE_DIFFERENCE,
            solver_config=FDSolverConfig(time_integrator=LSRK4, weno_z=True),
            **common_options,
        )
    if scheme == "fv":
        return SimulationConfig(
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(
                time_integrator=VL2,
                riemann_solver=HLLD if mhd else HLLC,
            ),
            **common_options,
        )
    raise ValueError(f"unknown scheme {scheme!r} (expected 'fd' or 'fv')")


def benchmark_state_builder(physics: str):
    """
    The sharding-aware initial-condition builder of a benchmark.

    Args:
        physics: ``"hydro"`` or ``"mhd"``.

    Returns:
        ``builder(config, params, sharding, settings) -> (state, config,
        params)``.
    """
    if physics == "hydro":
        from astronomix.test_setups.hydrodynamics.sound_wave3D import (
            build_sound_wave_state_sharded,
        )

        return build_sound_wave_state_sharded
    if physics == "mhd":
        from astronomix.test_setups.mhd.alfven_wave3D import (
            build_cp_alfven_state_sharded,
        )

        return build_cp_alfven_state_sharded
    raise ValueError(f"unknown physics {physics!r} (expected 'hydro' or 'mhd')")


def benchmark_settings(physics: str, box_size: tuple, amplitude: float = 1.0e-6):
    """
    The problem settings of a benchmark on the given box.

    The benchmark boxes equal the grid (h = 1) and are not wave-periodic, so
    the default amplitude keeps the wave a harmless linear perturbation of the
    uniform background; the correctness check passes a resolved box and a
    finite amplitude instead.

    Args:
        physics: ``"hydro"`` or ``"mhd"``.
        box_size: The box ``(L_x, L_y, L_z)``.
        amplitude: The wave amplitude (``eps`` of the sound wave, the
            transverse amplitude of the Alfvén wave).

    Returns:
        The settings NamedTuple of the test setup.
    """
    if physics == "hydro":
        from astronomix.test_setups.hydrodynamics.sound_wave3D import (
            SoundWave3DSettings,
        )

        return SoundWave3DSettings(box_size=tuple(box_size), eps=amplitude)
    if physics == "mhd":
        from astronomix.test_setups.mhd.alfven_wave3D import (
            CPAlfvenWave3DSettings,
        )

        return CPAlfvenWave3DSettings(box_size=tuple(box_size), amplitude=amplitude)
    raise ValueError(f"unknown physics {physics!r} (expected 'hydro' or 'mhd')")


def parse_block_shape(text: str, scheme: str) -> tuple:
    """
    Parse a ``"bx,by,bz"`` block shape; an empty string selects the scheme's
    default.

    Args:
        text: The command-line value.
        scheme: ``"fd"`` or ``"fv"``.

    Returns:
        The block shape as a 3-tuple of ints, or None for the library default.
    """
    if not text:
        return DEFAULT_BLOCK_SHAPES[scheme]
    return tuple(int(extent) for extent in text.replace("x", ",").split(","))
