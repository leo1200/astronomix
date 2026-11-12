"""
Pallas block-shape and warp-count tuning on one GPU: time per step of whole
``time_integration`` steps for every Pallas kernel family.

The Pallas kernels tile the grid in blocks of ``(bx, by, bz)`` cells and run
each block with ``num_warps`` warps. Both choices change the memory access
pattern (the state is stored as (variable, x, y, z) with z contiguous, and one
memory transaction reads 128 contiguous bytes) and the register budget per
thread, and with them the speed of the step by tens of percent. This script
measures the steady-state time per step of the program ``time_integration``
compiles, for

* the finite-difference scheme (WENO-Z) for ideal-gas hydrodynamics, ideal MHD
  and isothermal MHD, with LSRK4 or SSPRK4, with or without the
  positivity-preserving WENO and (MHD) with the native or the Pallas
  constrained transport;
* the finite-volume scheme VL2 (HLLC for hydrodynamics, HLLD with GLM cleaning
  for MHD);

in single or double precision, on any grid, for any block shape and warp
count (or the library defaults).

How a point is measured. The initial state is a uniform background with a
small smooth perturbation (the cost of a step does not depend on the data:
the kernels are branch free), built directly on the device. The time step is
the CFL step of the initial state and the end time is set for a fixed number
of steps, so every step includes its CFL reduction, as in a production run.
``time_integration`` is compiled once ahead of time (compilation is never
timed), run once untimed, and the fastest of the following repetitions is
reported. Every point runs in a fresh subprocess, so that an out-of-memory
failure, a compiler crash or the allocator state of one point cannot affect
the next.

Usage:
    # one point (inside a GPU job), prints one JSON result line
    python pallas_block_tuning.py --point '{"family": "fd_mhd", ...}'
    # a named sweep, appending to a JSON-lines cache and skipping cached points
    python pallas_block_tuning.py --sweep blocks --family fd_mhd --precision fp32 \
        --cells 256,128,128 --output results.jsonl
    # tables from a cache
    python pallas_block_tuning.py --report results.jsonl
"""

# general
import argparse
import json
import os
import subprocess
import sys
import threading
import time

# The driver must not open a GPU context of its own: only the per-point
# subprocess (``--point``) uses the GPU. The driver hands it the original
# platform setting back.
ORIGINAL_JAX_PLATFORMS = os.environ.get("JAX_PLATFORMS")
if "--point" not in sys.argv and "--points" not in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

# jax
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.extend.core import ClosedJaxpr, Jaxpr  # noqa: E402

# numerics
import numpy as np  # noqa: E402


# -------------------------------------------------------------
# ===================== ↓ Settings ↓ ==========================
# -------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

FAMILIES = ("fd_hydro", "fd_mhd", "fd_mhd_iso", "fv_hydro", "fv_mhd")

#: CFL numbers: the finite-difference value of the smooth tests, the bound of
#: the positivity-preserving WENO, AthenaPK's 3D value for VL2. The time step
#: only sets the physical time covered; the cost of a step does not depend on
#: it.
FD_CFL = 1.5
POSITIVITY_PRESERVING_CFL = 0.75
FV_CFL = 0.3

#: Steps per run: about this many cell updates, between the two bounds.
CELL_UPDATES_PER_RUN = 2.5e8
MIN_STEPS = 3
MAX_STEPS = 100

#: Timed repetitions after the discarded first execution: at least the
#: minimum, then on until the timed runs add up to the target.
MIN_TIMED_REPETITIONS = 2
TARGET_TIMED_SECONDS = 3.0
MAX_TIMED_REPETITIONS = 10

GPU_MEMORY_FRACTION = "0.95"
POINT_TIMEOUT_SECONDS = 1500.0
RESULT_PREFIX = "PALLAS_TUNING_RESULT "

#: The fields that identify a measurement point (the cache key).
POINT_FIELDS = (
    "family",
    "integrator",
    "positivity_preserving",
    "pallas_ct",
    "cells",
    "precision",
    "block",
    "num_warps",
    "label",
)

#: Candidate (block shape, warp counts) of the block sweep, grouped by the
#: number of cells per block. A block of 32 * num_warps cells holds one cell
#: per thread; fewer cells per block leave threads without work, more give
#: each thread several cells (more registers).
BLOCK_CANDIDATES = (
    # 128 cells
    ((1, 4, 32), (2, 4, 8)),
    ((1, 2, 64), (2, 4, 8)),
    ((1, 1, 128), (2, 4, 8)),
    ((1, 8, 16), (2, 4, 8)),
    ((2, 2, 32), (2, 4, 8)),
    ((4, 4, 8), (2, 4, 8)),
    ((2, 4, 16), (2, 4, 8)),
    ((1, 16, 8), (2, 4, 8)),
    # 64 cells
    ((1, 2, 32), (2, 4)),
    ((1, 1, 64), (2, 4)),
    ((1, 4, 16), (2, 4)),
    ((2, 2, 16), (2, 4)),
    # 32 cells
    ((1, 1, 32), (1, 2)),
    # 256 cells
    ((1, 4, 64), (4, 8)),
    ((1, 8, 32), (4, 8)),
    ((2, 2, 64), (4, 8)),
    ((2, 4, 32), (4, 8)),
)

#: The 2D candidates (y is the contiguous axis).
BLOCK_CANDIDATES_2D = (
    ((1, 128), (4,)),
    ((2, 64), (4,)),
    ((4, 32), (4,)),
    ((8, 16), (4,)),
    ((16, 16), (4, 8)),
    ((1, 64), (2,)),
    ((2, 32), (2,)),
    ((1, 256), (8,)),
    ((2, 128), (8,)),
)

# -------------------------------------------------------------
# ===================== ↑ Settings ↑ ==========================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============= ↓ Measurement of a single point ↓ =============
# -------------------------------------------------------------


def build_simulation_config(point: dict):
    """
    The simulation configuration of a measurement point (before
    ``finalize_config``).

    Args:
        point: The measurement point (see ``POINT_FIELDS``).

    Returns:
        The simulation configuration and the simulation parameters.
    """
    from astronomix.option_classes.simulation_config import (
        CARTESIAN,
        FINITE_DIFFERENCE,
        FINITE_VOLUME,
        HLLC,
        HLLD,
        IDEAL_GAS,
        ISOTHERMAL,
        PALLAS,
        PERIODIC_BOUNDARY,
        LSRK4,
        SSPRK54,
        VL2,
    )
    from astronomix.option_classes.simulation_config import (
        BackendConfig,
        BoundarySettings,
        BoundarySettings1D,
        FDSolverConfig,
        FVSolverConfig,
        SimulationConfig,
        SnapshotSettings,
        StaticFloatVector,
        StaticIntVector,
    )
    from astronomix.option_classes.simulation_params import SimulationParams

    family = point["family"]
    mhd = family in ("fd_mhd", "fd_mhd_iso", "fv_mhd")
    cells = tuple(int(extent) for extent in point["cells"])

    backend_options = dict(backend=PALLAS, pallas_ct=bool(point["pallas_ct"]))
    if point["block"] is not None:
        backend_options["pallas_block_shape"] = tuple(int(size) for size in point["block"])
    if point["num_warps"] is not None:
        backend_options["pallas_num_warps"] = int(point["num_warps"])

    if family.startswith("fd"):
        cfl = POSITIVITY_PRESERVING_CFL if point["positivity_preserving"] else FD_CFL
        scheme_options = dict(
            solver_mode=FINITE_DIFFERENCE,
            solver_config=FDSolverConfig(
                time_integrator=LSRK4 if point["integrator"] == "LSRK4" else SSPRK54,
                weno_z=True,
                weno_positivity_preserving=bool(point["positivity_preserving"]),
            ),
            equation_of_state=ISOTHERMAL if family == "fd_mhd_iso" else IDEAL_GAS,
        )
    else:
        cfl = FV_CFL
        scheme_options = dict(
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(
                time_integrator=VL2,
                riemann_solver=HLLD if mhd else HLLC,
            ),
        )

    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    dimensionality = len(cells)
    boundary_settings = BoundarySettings(x=periodic, y=periodic, z=periodic)
    if dimensionality == 2:
        boundary_settings = BoundarySettings(x=periodic, y=periodic)
    # The box equals the grid (cell size 1), so the spacing is uniform for any
    # grid shape. Two snapshots, the minimum: the snapshot grid of the default
    # ten would shorten the time step of a run this short.
    config = SimulationConfig(
        backend_config=BackendConfig(**backend_options),
        geometry=CARTESIAN,
        mhd=mhd,
        dimensionality=dimensionality,
        num_cells=StaticIntVector(*cells),
        box_size=StaticFloatVector(*(float(extent) for extent in cells)),
        boundary_settings=boundary_settings,
        donate_state=True,
        return_snapshots=True,
        num_snapshots=2,
        snapshot_settings=SnapshotSettings(return_final_state=True),
        **scheme_options,
    )
    params = SimulationParams(C_cfl=cfl, gamma=5.0 / 3.0, isothermal_sound_speed=1.0)
    return config, params


def build_initial_state(config, params, registered_variables):
    """
    A uniform background with a small smooth perturbation, built on the
    device: density and pressure 1 (sound speed ~1.3), a velocity of 1e-3 and,
    for MHD, a uniform oblique magnetic field (divergence free on any grid).

    Args:
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The primitive state.
    """
    from astronomix.initial_condition_generation.construct_primitive_state import (
        construct_primitive_state,
    )

    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    two_pi = 2.0 * np.pi
    if config.dimensionality == 2:
        nx, ny = config.num_cells.x, config.num_cells.y
        x = ((jnp.arange(nx, dtype=dtype) + 0.5) / nx)[:, None]
        y = ((jnp.arange(ny, dtype=dtype) + 0.5) / ny)[None, :]
        shape = (nx, ny)
        wave = jnp.sin(two_pi * x) * jnp.cos(two_pi * y)
    else:
        nx, ny, nz = config.num_cells.x, config.num_cells.y, config.num_cells.z
        x = ((jnp.arange(nx, dtype=dtype) + 0.5) / nx)[:, None, None]
        y = ((jnp.arange(ny, dtype=dtype) + 0.5) / ny)[None, :, None]
        z = ((jnp.arange(nz, dtype=dtype) + 0.5) / nz)[None, None, :]
        shape = (nx, ny, nz)
        wave = jnp.sin(two_pi * x) * jnp.cos(two_pi * y) * jnp.sin(two_pi * z)

    fields = dict(
        density=jnp.broadcast_to(1.0 + 1.0e-3 * wave, shape),
        velocity_x=jnp.broadcast_to(1.0e-3 * wave, shape),
        velocity_y=jnp.broadcast_to(-0.5e-3 * wave, shape),
        velocity_z=jnp.broadcast_to(0.25e-3 * wave, shape),
        gas_pressure=jnp.broadcast_to(1.0 + 1.0e-3 * wave, shape),
    )
    if config.dimensionality == 2 and not config.mhd:
        # 2D hydrodynamics has no third velocity component.
        del fields["velocity_z"]
    if config.mhd:
        fields.update(
            magnetic_field_x=jnp.full(shape, 1.0, dtype),
            magnetic_field_y=jnp.full(shape, 0.5, dtype),
            magnetic_field_z=jnp.full(shape, 0.25, dtype),
        )
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        **fields,
    )


def initial_time_step(primitive_state, config, params, registered_variables) -> float:
    """
    The CFL time step of a state, from the estimator the time loop uses.

    Args:
        primitive_state: The primitive state.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The time step.
    """
    from astronomix.option_classes.simulation_config import FINITE_VOLUME
    from astronomix._finite_volume._timestep_estimation._timestep_estimator import (
        _cfl_time_step,
    )
    from astronomix._finite_difference._timestep_estimation._timestep_estimator import (
        _cfl_time_step_fd,
        _cfl_time_step_fd_hydro,
    )

    if config.solver_mode == FINITE_VOLUME:
        return float(_cfl_time_step(primitive_state, config, params, registered_variables))
    estimator = _cfl_time_step_fd if config.mhd else _cfl_time_step_fd_hydro
    return float(
        estimator(
            primitive_state,
            config.grid_spacing,
            params.dt_max,
            params.gamma,
            config,
            params,
            registered_variables,
            params.C_cfl,
        )
    )


def count_pallas_calls(jaxpr: Jaxpr) -> int:
    """
    The number of ``pallas_call`` equations in a jaxpr and all its sub-jaxprs.
    Zero for a Pallas configuration means that the backend silently fell back
    to native JAX (e.g. a block that does not tile the grid).

    Args:
        jaxpr: The jaxpr.

    Returns:
        The number of Pallas kernel calls in the program text.
    """
    number_of_calls = 0
    for equation in jaxpr.eqns:
        if equation.primitive.name == "pallas_call":
            number_of_calls += 1
        for parameter in equation.params.values():
            candidates = parameter if isinstance(parameter, (tuple, list)) else (parameter,)
            for candidate in candidates:
                if isinstance(candidate, ClosedJaxpr):
                    number_of_calls += count_pallas_calls(candidate.jaxpr)
                elif isinstance(candidate, Jaxpr):
                    number_of_calls += count_pallas_calls(candidate)
    return number_of_calls


def gpu_status() -> dict:
    """
    The name, clocks, power draw and temperature of the visible GPU, so that a
    throttled measurement can be recognised.

    Returns:
        The ``nvidia-smi`` fields by name (an ``error`` entry if the query
        failed).
    """
    fields = ("name", "clocks.sm", "clocks.max.sm", "power.draw", "temperature.gpu")
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader",
                "-i",
                visible_device,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(error=repr(error))
    values = [value.strip() for value in query.stdout.strip().split(",")]
    if query.returncode != 0 or len(values) != len(fields):
        return dict(error=(query.stdout + query.stderr).strip())
    return dict(zip(fields, values))


def steps_per_run(cells) -> int:
    """The number of time steps of a run on the grid (see ``CELL_UPDATES_PER_RUN``)."""
    number_of_cells = int(np.prod(cells))
    return int(min(MAX_STEPS, max(MIN_STEPS, CELL_UPDATES_PER_RUN // number_of_cells)))


def measure_point(point: dict) -> dict:
    """
    Measure one point: compile the time integration once, run it once untimed
    and then time the repetitions.

    Args:
        point: The measurement point (see ``POINT_FIELDS``).

    Returns:
        The point with the measurement: ``status`` (``ok``, ``oom`` or
        ``error``), the timings (``time_per_step`` = fastest timed run /
        steps), the Pallas-call count, the compiled temporary memory and the
        GPU status.
    """
    from astronomix import get_registered_variables, time_integration
    from astronomix.option_classes.simulation_config import finalize_config
    from astronomix._pallas_helpers import _as_3tuple_block_shape

    record = dict(point)
    stage = "setup"
    try:
        if bool(jax.config.jax_enable_x64) != (point["precision"] == "fp64"):
            raise RuntimeError("JAX_ENABLE_X64 does not match the requested precision.")
        gpu_device = jax.devices()[0]
        record["device_kind"] = gpu_device.device_kind
        record["jax_version"] = jax.__version__

        config, params = build_simulation_config(point)
        registered_variables = get_registered_variables(config)
        state = build_initial_state(config, params, registered_variables)
        config = finalize_config(config, state.shape)
        time_step = initial_time_step(state, config, params, registered_variables)
        number_of_steps = steps_per_run(point["cells"])
        # End half a step after the last full step, so that round-off in the
        # clock cannot add a sliver of a step.
        params = params._replace(t_end=(number_of_steps - 0.5) * time_step)
        record["resolved_block"] = list(
            _as_3tuple_block_shape(
                config.backend_config.pallas_block_shape,
                len(point["cells"]),
                spatial_shape=tuple(point["cells"]),
            )
        )
        record["resolved_num_warps"] = int(config.backend_config.pallas_num_warps)
        state_shape, state_dtype = state.shape, state.dtype
        host_state = np.asarray(state)
        del state

        stage = "compile"
        plain_config = config._replace(memory_analysis=False, print_elapsed_time=False, progress_bar=False)

        def integrate(primitive_state, simulation_params):
            return time_integration(
                primitive_state,
                plain_config,
                simulation_params,
                registered_variables,
            )

        compile_start = time.perf_counter()
        traced = jax.jit(integrate, donate_argnums=0).trace(
            jax.ShapeDtypeStruct(state_shape, state_dtype),
            params,
        )
        record["pallas_calls"] = count_pallas_calls(traced.jaxpr.jaxpr)
        compiled = traced.lower().compile()
        memory = compiled.memory_analysis()
        record["compile_seconds"] = time.perf_counter() - compile_start
        record["temp_bytes"] = int(memory.temp_size_in_bytes)

        stage = "run"

        def timed_run():
            """One run from a fresh device copy of the initial state."""
            device_state = jax.device_put(host_state, gpu_device)
            device_state.block_until_ready()
            start = time.perf_counter()
            result = compiled(device_state, params)
            jax.block_until_ready(result)
            return time.perf_counter() - start, result

        first_run_seconds, result = timed_run()
        number_of_iterations = int(result.num_iterations)
        final_state_finite = bool(jax.jit(lambda final_state: jnp.all(jnp.isfinite(final_state)))(result.final_state))
        del result

        timed_seconds = []
        while len(timed_seconds) < MIN_TIMED_REPETITIONS or (
            sum(timed_seconds) < TARGET_TIMED_SECONDS and len(timed_seconds) < MAX_TIMED_REPETITIONS
        ):
            run_seconds, result = timed_run()
            del result
            timed_seconds.append(run_seconds)

        number_of_cells = int(np.prod(point["cells"]))
        time_per_step = min(timed_seconds) / number_of_iterations
        record.update(
            status="ok",
            iterations=number_of_iterations,
            final_state_finite=final_state_finite,
            first_run_seconds=first_run_seconds,
            timed_seconds=timed_seconds,
            time_per_step=time_per_step,
            median_time_per_step=float(np.median(timed_seconds)) / number_of_iterations,
            mcell_updates_per_second=number_of_cells / time_per_step / 1.0e6,
            gpu=gpu_status(),
        )
    except Exception as error:  # noqa: BLE001 (a failed point is a result too)
        message = repr(error)
        out_of_memory = any(marker in message for marker in ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory"))
        record.update(
            status="oom" if out_of_memory else "error",
            failed_stage=stage,
            error=message[:2000],
        )
    return record


# -------------------------------------------------------------
# ============= ↑ Measurement of a single point ↑ =============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================ ↓ Sweep driver and cache ↓ =================
# -------------------------------------------------------------


def point_key(point: dict) -> str:
    """The cache key of a point."""
    return json.dumps([point.get(field) for field in POINT_FIELDS])


def make_point(
    family,
    precision,
    cells,
    block=None,
    num_warps=None,
    integrator=None,
    positivity_preserving=False,
    pallas_ct=False,
    label="",
) -> dict:
    """
    A measurement point.

    Args:
        family: One of ``FAMILIES``.
        precision: ``fp32`` or ``fp64``.
        cells: The grid ``(nx, ny, nz)``.
        block: The block shape, or None for the library default.
        num_warps: The warp count, or None for the library default.
        integrator: ``LSRK4``/``SSPRK4`` (FD) or ``VL2`` (FV); None picks
            LSRK4 / VL2.
        positivity_preserving: Positivity-preserving WENO (FD).
        pallas_ct: Pallas constrained transport (FD MHD).
        label: A free tag (e.g. the code version measured).

    Returns:
        The point.
    """
    if integrator is None:
        integrator = "VL2" if family.startswith("fv") else "LSRK4"
    return dict(
        family=family,
        integrator=integrator,
        positivity_preserving=bool(positivity_preserving),
        pallas_ct=bool(pallas_ct),
        cells=[int(extent) for extent in cells],
        precision=precision,
        block=None if block is None else [int(size) for size in block],
        num_warps=None if num_warps is None else int(num_warps),
        label=label,
    )


def block_sweep_points(family, precision, cells, **options) -> list:
    """The block-shape x warp-count sweep (``BLOCK_CANDIDATES``) of one configuration."""
    points = [make_point(family, precision, cells, **options)]
    candidates = BLOCK_CANDIDATES if len(cells) == 3 else BLOCK_CANDIDATES_2D
    for block, warp_counts in candidates:
        if any(int(extent) % size for extent, size in zip(cells, block)):
            continue
        for num_warps in warp_counts:
            points.append(make_point(family, precision, cells, block=block, num_warps=num_warps, **options))
    return points


def read_cache(path: str) -> dict:
    """The cached results by point key (later lines override earlier ones)."""
    results = {}
    if os.path.exists(path):
        with open(path) as cache_file:
            for line in cache_file:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    results[point_key(record)] = record
    return results


def measurement_environment(precision: str) -> dict:
    """The environment of a measurement subprocess (precision fixed at import)."""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "1" if precision == "fp64" else "0"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = GPU_MEMORY_FRACTION
    if ORIGINAL_JAX_PLATFORMS is None:
        environment.pop("JAX_PLATFORMS", None)
    else:
        environment["JAX_PLATFORMS"] = ORIGINAL_JAX_PLATFORMS
    return environment


def run_batch_subprocess(points: list, on_result) -> list:
    """
    Measure points of one precision in one subprocess (one JAX start for
    many points), handing every result to ``on_result`` as it arrives.

    A point that does not report within ``POINT_TIMEOUT_SECONDS`` kills the
    subprocess; the points after a crash or a timeout are left unmeasured.

    Args:
        points: The measurement points (all of one precision).
        on_result: Called with each result record.

    Returns:
        The points that were not measured, the failed one first.
    """
    command = [sys.executable, "-u", os.path.abspath(__file__), "--points", json.dumps(points)]
    process = subprocess.Popen(
        command,
        env=measurement_environment(points[0]["precision"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    finished = []
    last_lines = []
    deadline = [time.monotonic() + POINT_TIMEOUT_SECONDS]
    timed_out = [False]

    def watchdog():
        """Kill the subprocess when the current point exceeds its time."""
        while process.poll() is None:
            if time.monotonic() > deadline[0]:
                timed_out[0] = True
                process.kill()
                return
            time.sleep(5.0)

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()
    for line in process.stdout:
        if line.startswith(RESULT_PREFIX):
            record = json.loads(line[len(RESULT_PREFIX):])
            finished.append(record)
            on_result(record)
            deadline[0] = time.monotonic() + POINT_TIMEOUT_SECONDS
        else:
            last_lines = (last_lines + [line])[-30:]
    process.wait()
    remaining = points[len(finished):]
    if remaining:
        status = "timeout" if timed_out[0] else "crash"
        on_result(dict(remaining[0], status=status, error="".join(last_lines)[-2000:]))
        remaining = remaining[1:]
    return remaining


def run_points(points: list, output_path: str) -> None:
    """Measure every uncached point and append it to the cache."""
    cached = read_cache(output_path)
    pending = [
        point
        for point in points
        if not (point_key(point) in cached and cached[point_key(point)].get("status") == "ok")
    ]

    def on_result(record):
        with open(output_path, "a") as cache_file:
            cache_file.write(json.dumps(record) + "\n")
        summary = (
            f"{record.get('time_per_step', float('nan')) * 1e3:9.3f} ms/step "
            f"{record.get('mcell_updates_per_second', float('nan')):8.1f} Mcell/s"
            if record.get("status") == "ok"
            else record.get("status") + " " + str(record.get("error", ""))[-300:]
        )
        print(
            f"{record['family']:10s} {record['integrator']:6s} pp={int(record['positivity_preserving'])} "
            f"ct={int(record['pallas_ct'])} {record['precision']} {record['cells']} "
            f"block={record['block'] or record.get('resolved_block')} "
            f"warps={record['num_warps'] or record.get('resolved_num_warps')} "
            f"calls={record.get('pallas_calls')} {summary}",
            flush=True,
        )

    for precision in ("fp32", "fp64"):
        remaining = [point for point in pending if point["precision"] == precision]
        while remaining:
            remaining = run_batch_subprocess(remaining, on_result)


def print_report(path: str) -> None:
    """Print the cached points grouped by configuration, fastest first."""
    groups = {}
    for record in read_cache(path).values():
        key = (
            record["family"],
            record["integrator"],
            record["positivity_preserving"],
            record["pallas_ct"],
            tuple(record["cells"]),
            record["precision"],
            record.get("label", ""),
        )
        groups.setdefault(key, []).append(record)
    for key in sorted(groups, key=str):
        records = groups[key]
        print("== " + " ".join(str(part) for part in key))
        default = [record for record in records if record["block"] is None and record["num_warps"] is None]
        default_time = default[0].get("time_per_step") if default and default[0].get("status") == "ok" else None
        ok_records = sorted(
            (record for record in records if record.get("status") == "ok"),
            key=lambda record: record["time_per_step"],
        )
        for record in ok_records:
            relative = f"{record['time_per_step'] / default_time:6.3f}" if default_time else "     -"
            print(
                f"   block={str(record['block'] or 'default'):14s} warps={str(record['num_warps'] or '-'):3s} "
                f"{record['time_per_step'] * 1e3:9.3f} ms/step {record['mcell_updates_per_second']:8.1f} Mcell/s "
                f"rel={relative} calls={record.get('pallas_calls')}"
            )
        for record in records:
            if record.get("status") != "ok":
                print(f"   block={record['block']} warps={record['num_warps']} {record.get('status')}")


# -------------------------------------------------------------
# ================ ↑ Sweep driver and cache ↑ =================
# -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", type=str, default=None, help="measure one point (JSON) in this process")
    parser.add_argument("--points", type=str, default=None, help="measure a JSON list of points in this process")
    parser.add_argument("--sweep", choices=("blocks", "default", "list"), default=None)
    parser.add_argument("--family", choices=FAMILIES, default="fd_hydro")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--cells", type=str, default="256,128,128")
    parser.add_argument("--integrator", type=str, default=None)
    parser.add_argument("--pp", action="store_true", help="positivity-preserving WENO")
    parser.add_argument("--pallas-ct", action="store_true")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument(
        "--blocks",
        type=str,
        default="",
        help="for --sweep list: semicolon-separated bx,by,bz/warps entries ('default' = library default)",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    if args.point is not None:
        record = measure_point(json.loads(args.point))
        print(RESULT_PREFIX + json.dumps(record), flush=True)
        return
    if args.points is not None:
        for point in json.loads(args.points):
            record = measure_point(point)
            print(RESULT_PREFIX + json.dumps(record), flush=True)
            # Release the compiled programs and buffers of the point.
            jax.clear_caches()
        return
    if args.report is not None:
        print_report(args.report)
        return

    cells = tuple(int(extent) for extent in args.cells.split(","))
    options = dict(
        integrator=args.integrator,
        positivity_preserving=args.pp,
        pallas_ct=args.pallas_ct,
        label=args.label,
    )
    if args.sweep == "blocks":
        points = block_sweep_points(args.family, args.precision, cells, **options)
    elif args.sweep == "default":
        points = [make_point(args.family, args.precision, cells, **options)]
    else:
        points = []
        for entry in args.blocks.split(";"):
            entry = entry.strip()
            if not entry or entry == "default":
                points.append(make_point(args.family, args.precision, cells, **options))
                continue
            block_text, warps_text = entry.split("/")
            block = tuple(int(size) for size in block_text.split(","))
            points.append(make_point(args.family, args.precision, cells, block=block, num_warps=int(warps_text), **options))
    run_points(points, args.output)


if __name__ == "__main__":
    main()
