"""
Pallas versus native-JAX backend on one GPU: time per step and compiled memory.

For both astronomix schemes -- the finite-difference scheme (WENO-Z, low-storage
LSRK4, donated state, as in the scaling sections) and AthenaPK's finite-volume
scheme (VL2: HLLC for hydrodynamics, HLLD with GLM cleaning for MHD) -- on the
native-JAX and on the Pallas backend, this script measures

* the steady-state wall-clock time per step of the 3D sound wave
  (hydrodynamics) and of the 3D circularly polarized Alfvén wave (MHD) on the
  grid 2N x N x N, from N = 16 up to the largest N that fits on the device;
* XLA's compiled memory of the same program: argument, output, aliased and
  temporary buffers, and the transient-to-argument ratio;

and plots the Pallas speedup figure and the single-GPU memory figure of the
methods paper. A few extra finite-difference Pallas points give the cost of the
positivity-preserving WENO and of the SSPRK4 integrator.

How a point is measured. The program is exactly the one ``time_integration``
compiles: ``time_integration`` jitted with the state and the parameters as
arguments and the state donated reproduces its compiled program and memory
analysis bit for bit. It is compiled once ahead of time, so compilation is
never timed. The time step is the CFL step of the initial condition and the end
time is set for a fixed number of steps; every step includes its CFL reduction,
as in a production run. The first execution (module loading and other one-time
costs) is discarded and the fastest of the following repetitions is reported,
since a sporadic disturbance only ever adds time. Every point runs in a fresh
subprocess, so that an out-of-memory failure, the allocator state or the
compiled programs of one point cannot affect the next.

Usage:
    # on a GPU node, one physics and precision per job; fills the cache in
    # data/pallas_speedup/ and resumes from it
    python pallas_speedup.py --measure --physics hydro --precision single
    # anywhere: the figures and the tables from the cache
    python pallas_speedup.py
"""

# general
import argparse
import json
import os
import subprocess
import sys
import time

# The driver and the plotting must not open a GPU context: importing astronomix
# initialises a JAX backend, and the driver runs on the node whose GPU the
# measurements use. Only the per-point subprocess (``--point``) uses the GPU;
# the driver hands it the original platform setting back.
ORIGINAL_JAX_PLATFORMS = os.environ.get("JAX_PLATFORMS")
if "--point" not in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

# jax
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.extend.core import ClosedJaxpr, Jaxpr  # noqa: E402

# numerics
import numpy as np  # noqa: E402

# plotting
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

# astronomix constants
from astronomix.plotting_helpers.paper_style import (  # noqa: E402
    FD_COLOR,
    FV_COLOR,
    GUIDE_COLOR,
    COLUMN_WIDTH,
    TEXT_WIDTH,
    TWO_COLUMN,
    layout_path,
)
from astronomix.option_classes.simulation_config import (  # noqa: E402
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLC,
    HLLD,
    NATIVE_JAX,
    PALLAS,
    LSRK4,
    SSPRK54,
    VL2,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (  # noqa: E402
    BackendConfig,
    FDSolverConfig,
    FVSolverConfig,
    SimulationConfig,
    SnapshotSettings,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams  # noqa: E402

# astronomix functions
from astronomix import get_registered_variables, time_integration  # noqa: E402
from astronomix.test_setups.hydrodynamics.sound_wave3D import setup_sound_wave  # noqa: E402
from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave  # noqa: E402
from astronomix._finite_volume._timestep import (  # noqa: E402
    _cfl_time_step,
)
from astronomix._finite_difference._timestep import (  # noqa: E402
    _cfl_time_step_fd,
    _cfl_time_step_fd_hydro,
)
from astronomix._pallas_helpers import _as_3tuple_block_shape  # noqa: E402
from astronomix.plotting_helpers.paper_style import (  # noqa: E402
    figure_legend,
    use_paper_style,
    variant_style,
)


# -------------------------------------------------------------
# ===================== ↓ Settings ↓ ==========================
# -------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "data", "pallas_speedup")
DEFAULT_FIGURE_DIR = os.path.join(HERE, "figures")

PHYSICS = ("hydro", "mhd")
PRECISIONS = ("single", "double")
PRECISION_SUFFIX = {"single": "fp32", "double": "fp64"}
PHYSICS_TITLE = {
    "hydro": "hydrodynamics (3D sound wave)",
    "mhd": "MHD (3D CP Alfvén wave)",
}

#: The resolution ladder N (grid 2N x N x N); each configuration climbs it
#: until the first failure (out of memory). The Pallas backend runs its
#: default block, which the library reduces to a tiling of every grid; a
#: silent fall-back to native JAX would show as a Pallas-call count of zero.
RESOLUTIONS = (16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 576, 640, 768, 896, 1024)
#: Resolutions of the extra finite-difference Pallas points (PP-WENO, SSPRK4).
EXTRA_RESOLUTIONS = (128, 256)

#: CFL numbers: the finite-difference value of the paper's smooth tests, the
#: 3D value AthenaPK uses for VL2, and the bound under which the
#: positivity-preserving WENO is provable (SSPRK4). The time step only sets
#: the physical time covered; the cost of a step does not depend on it.
FD_CFL = 1.5
FV_CFL = 0.3
POSITIVITY_PRESERVING_CFL = 0.75

#: Steps per run: about this many cell updates, between the two bounds. Small
#: grids take many steps so that the per-run overhead (dispatch, the loop
#: setup) is negligible; large grids few, since a step there already takes
#: seconds.
CELL_UPDATES_PER_RUN = 1.0e9
MIN_STEPS = 3
MAX_STEPS = 200

#: Timed repetitions after the discarded first execution: at least the
#: minimum, then on until the timed runs add up to the target.
MIN_TIMED_REPETITIONS = 2
TARGET_TIMED_SECONDS = 3.0
MAX_TIMED_REPETITIONS = 20

#: The measurement subprocess preallocates this fraction of the device memory
#: as one pool: on-demand allocation fragments the pool and fails on large
#: temporary buffers well before the device is full.
GPU_MEMORY_FRACTION = "0.95"
POINT_TIMEOUT_SECONDS = 3600.0
RESULT_PREFIX = "PALLAS_SPEEDUP_RESULT "

#: The fields that identify a measurement point (the cache key).
POINT_FIELDS = (
    "physics",
    "scheme",
    "backend",
    "integrator",
    "positivity_preserving",
    "cfl",
    "resolution",
    "precision",
)

GIB = 1024**3
GIGABYTE = 1.0e9

#: Plot style of the four configurations, in the colours of all figures of
#: the paper: the Pallas backend solid with filled markers, native JAX as the
#: dashed, hollow variant.
CONFIGURATION_STYLE = {
    ("FD", "jax"): dict(label="FD (JAX)", marker="o", **variant_style(FD_COLOR)),
    ("FD", "pallas"): dict(label="FD (Pallas)", color=FD_COLOR, marker="o", linestyle="-"),
    ("FV", "jax"): dict(label="FV (JAX)", marker="s", **variant_style(FV_COLOR)),
    ("FV", "pallas"): dict(label="FV (Pallas)", color=FV_COLOR, marker="s", linestyle="-"),
}
SPEEDUP_COLOR = {"FD": FD_COLOR, "FV": FV_COLOR}
SPEEDUP_MARKER = {"FD": "o", "FV": "s"}
MAIN_INTEGRATOR = {"FD": "LSRK4", "FV": "VL2"}
MAIN_CFL = {"FD": FD_CFL, "FV": FV_CFL}
RESOLUTION_TICKS = (16, 32, 64, 128, 256, 512, 1024)

#: The resolution range of the speedup figure of the paper: from where the
#: kernels no longer run at launch-overhead-dominated sizes to the grid of
#: the memory table.
SPEEDUP_FIGURE_RESOLUTIONS = (64, 256)

#: The resolution of the memory table of the paper (grid 2N x N x N).
MEMORY_TABLE_RESOLUTION = 256

# -------------------------------------------------------------
# ===================== ↑ Settings ↑ ==========================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============= ↓ Measurement of a single point ↓ =============
# -------------------------------------------------------------


def build_simulation_config(point: dict) -> SimulationConfig:
    """
    The simulation configuration of a measurement point, before the test
    setup completes it (box, boundaries, end time).

    Args:
        point: The measurement point (see ``POINT_FIELDS``).

    Returns:
        The simulation configuration.
    """
    resolution = point["resolution"]
    mhd = point["physics"] == "mhd"

    if point["scheme"] == "FD":
        time_integrator = LSRK4 if point["integrator"] == "LSRK4" else SSPRK54
        scheme_options = dict(
            solver_mode=FINITE_DIFFERENCE,
            solver_config=FDSolverConfig(
                time_integrator=time_integrator,
                weno_z=True,
                weno_positivity_preserving=point["positivity_preserving"],
            ),
        )
    else:
        scheme_options = dict(
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(
                time_integrator=VL2,
                riemann_solver=HLLD if mhd else HLLC,
            ),
        )

    backend = PALLAS if point["backend"] == "pallas" else NATIVE_JAX

    # Two snapshots, the minimum: the snapshot grid of the default ten would
    # shorten the time step of a run this short.
    return SimulationConfig(
        backend_config=BackendConfig(backend=backend),
        mhd=mhd,
        dimensionality=3,
        num_cells=StaticIntVector(2 * resolution, resolution, resolution),
        donate_state=True,
        return_snapshots=True,
        num_snapshots=2,
        snapshot_settings=SnapshotSettings(return_final_state=True),
        **scheme_options,
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
    The number of ``pallas_call`` equations in a jaxpr and all its sub-jaxprs
    (loop bodies, custom-derivative wrappers). Zero for a Pallas configuration
    means that the backend silently fell back to native JAX.

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
    The name, clocks, power draw, temperature and memory of the visible GPU,
    so that a throttled measurement can be recognised.

    Returns:
        The ``nvidia-smi`` fields by name (an ``error`` entry if the query
        failed).
    """
    fields = (
        "name",
        "clocks.sm",
        "clocks.max.sm",
        "power.draw",
        "temperature.gpu",
        "memory.total",
    )
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


def is_out_of_memory(error: Exception) -> bool:
    """Whether an exception reports that the device ran out of memory."""
    message = repr(error)
    return any(marker in message for marker in ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory"))


def steps_per_run(resolution: int) -> int:
    """The number of time steps of a run at resolution N (see ``CELL_UPDATES_PER_RUN``)."""
    number_of_cells = 2 * resolution**3
    return int(min(MAX_STEPS, max(MIN_STEPS, CELL_UPDATES_PER_RUN // number_of_cells)))


def measure_point(point: dict) -> dict:
    """
    Measure one point: compile the time integration once, read its compiled
    memory, run it once untimed and then time the repetitions.

    The initial state is built on the host, so that the device holds only
    what the run needs and the largest feasible N is that of the run, not of
    the test setup; every run gets a fresh copy, since the state is donated.

    Args:
        point: The measurement point (see ``POINT_FIELDS``).

    Returns:
        The point with the measurement: ``status`` (``ok``, ``oom`` or
        ``error``, with the failed stage and the error), the timings
        (``time_per_step`` = fastest timed run / steps), the compiled memory
        in bytes, the Pallas-call count and the GPU status.
    """
    record = dict(point)
    stage = "setup"

    try:
        # The precision is fixed at import time through JAX_ENABLE_X64, so
        # that the module-level arrays of the test setups have it as well.
        double_precision = point["precision"] == "double"
        if bool(jax.config.jax_enable_x64) != double_precision:
            raise RuntimeError("JAX_ENABLE_X64 does not match the requested precision.")

        gpu_device = jax.devices()[0]
        record["device_kind"] = gpu_device.device_kind
        record["jax_version"] = jax.__version__

        # -------------------------------------------------------------
        # ============ ↓ Initial state and time step ↓ ================
        # -------------------------------------------------------------

        setup_function = setup_cp_alfven_wave if point["physics"] == "mhd" else setup_sound_wave
        setup_start = time.perf_counter()
        with jax.default_device(jax.devices("cpu")[0]):
            state, config, params = setup_function(
                build_simulation_config(point),
                SimulationParams(C_cfl=point["cfl"]),
            )
            registered_variables = get_registered_variables(config)
            time_step = initial_time_step(state, config, params, registered_variables)
        host_state = np.asarray(state)
        del state

        # End half a step after the last full step, so that round-off in the
        # clock cannot add a sliver of a step.
        number_of_steps = steps_per_run(point["resolution"])
        params = params._replace(t_end=(number_of_steps - 0.5) * time_step)

        record.update(
            setup_seconds=time.perf_counter() - setup_start,
            state_shape=list(host_state.shape),
            state_dtype=str(host_state.dtype),
            state_bytes=int(host_state.nbytes),
            time_step=time_step,
            # The block shape the Pallas kernels run on this grid; recorded for
            # the native backend too, where it is unused.
            pallas_block_shape=list(
                _as_3tuple_block_shape(
                    config.backend_config.pallas_block_shape,
                    3,
                    spatial_shape=host_state.shape[1:],
                )
            ),
        )

        # -------------------------------------------------------------
        # ============ ↑ Initial state and time step ↑ ================
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # ========= ↓ Compilation and compiled memory ↓ ===============
        # -------------------------------------------------------------

        stage = "compile"
        # time_integration's own timing and memory analysis are replaced by
        # the ones below; everything else of the configuration is kept.
        plain_config = config._replace(memory_analysis=False, print_elapsed_time=False)

        def integrate(primitive_state, simulation_params):
            return time_integration(
                primitive_state,
                plain_config,
                simulation_params,
                registered_variables,
            )

        compile_start = time.perf_counter()
        traced = jax.jit(integrate, donate_argnums=0).trace(
            jax.ShapeDtypeStruct(host_state.shape, host_state.dtype),
            params,
        )
        number_of_pallas_calls = count_pallas_calls(traced.jaxpr.jaxpr)
        compiled = traced.lower().compile()
        memory = compiled.memory_analysis()
        record.update(
            compile_seconds=time.perf_counter() - compile_start,
            pallas_calls=number_of_pallas_calls,
            temp_bytes=int(memory.temp_size_in_bytes),
            argument_bytes=int(memory.argument_size_in_bytes),
            output_bytes=int(memory.output_size_in_bytes),
            alias_bytes=int(memory.alias_size_in_bytes),
            code_bytes=int(memory.generated_code_size_in_bytes),
        )
        # The total of the earlier memory figures (and of time_integration's
        # memory analysis): what the compiled program holds at its peak.
        record["total_bytes"] = (
            record["temp_bytes"] + record["argument_bytes"] + record["output_bytes"] - record["alias_bytes"]
        )

        # -------------------------------------------------------------
        # ========= ↑ Compilation and compiled memory ↑ ===============
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # ===================== ↓ Timed runs ↓ ========================
        # -------------------------------------------------------------

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
        # Free the outputs before the next run: a run near the memory limit has
        # no room for a second final state.
        del result

        timed_seconds = []
        while len(timed_seconds) < MIN_TIMED_REPETITIONS or (
            sum(timed_seconds) < TARGET_TIMED_SECONDS and len(timed_seconds) < MAX_TIMED_REPETITIONS
        ):
            run_seconds, result = timed_run()
            del result
            timed_seconds.append(run_seconds)

        record.update(
            status="ok",
            iterations=number_of_iterations,
            final_state_finite=final_state_finite,
            first_run_seconds=first_run_seconds,
            timed_seconds=timed_seconds,
            time_per_step=min(timed_seconds) / number_of_iterations,
            median_time_per_step=float(np.median(timed_seconds)) / number_of_iterations,
            gpu=gpu_status(),
        )

        # -------------------------------------------------------------
        # ===================== ↑ Timed runs ↑ ========================
        # -------------------------------------------------------------

    except Exception as error:  # noqa: BLE001 (a failed point is a result too)
        record.update(
            status="oom" if is_out_of_memory(error) else "error",
            failed_stage=stage,
            error=repr(error)[:3000],
        )

    return record


# -------------------------------------------------------------
# ============= ↑ Measurement of a single point ↑ =============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================ ↓ Sweep driver and cache ↓ =================
# -------------------------------------------------------------


def main_configurations(physics: str, precision: str) -> list:
    """
    The four scheme x backend configurations of the main sweep.

    Args:
        physics: ``hydro`` or ``mhd``.
        precision: ``single`` or ``double``.

    Returns:
        The configurations (points without the resolution).
    """
    return [
        dict(
            physics=physics,
            scheme=scheme,
            backend=backend,
            integrator=MAIN_INTEGRATOR[scheme],
            positivity_preserving=False,
            cfl=MAIN_CFL[scheme],
            precision=precision,
        )
        for scheme in ("FD", "FV")
        for backend in ("jax", "pallas")
    ]


def extra_configurations(physics: str, precision: str) -> list:
    """
    The extra finite-difference Pallas configurations: LSRK4 and SSPRK4, each
    without and with the positivity-preserving WENO, all at the CFL number of
    the positivity-preserving runs so that the pairs differ in one option only.

    Args:
        physics: ``hydro`` or ``mhd``.
        precision: ``single`` or ``double``.

    Returns:
        The configurations (points without the resolution).
    """
    return [
        dict(
            physics=physics,
            scheme="FD",
            backend="pallas",
            integrator=integrator,
            positivity_preserving=positivity_preserving,
            cfl=POSITIVITY_PRESERVING_CFL,
            precision=precision,
        )
        for integrator in ("LSRK4", "SSPRK4")
        for positivity_preserving in (False, True)
    ]


def point_key(point: dict) -> tuple:
    """The cache key of a point."""
    return tuple(point[field] for field in POINT_FIELDS)


def cache_path(data_dir: str, tag: str, precision: str, physics: str) -> str:
    """The cache file of one GPU, precision and physics (one per measurement job)."""
    return os.path.join(data_dir, f"{tag}_{PRECISION_SUFFIX[precision]}_{physics}.json")


def load_cache(path: str) -> list:
    """The cached point records of a file (empty if it does not exist)."""
    if not os.path.exists(path):
        return []
    with open(path) as cache_file:
        return json.load(cache_file)["points"]


def save_cache(path: str, records: list) -> None:
    """Write the point records, through a temporary file so a killed job never leaves a torn cache."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = sorted(records, key=lambda record: tuple(str(value) for value in point_key(record)))
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as cache_file:
        json.dump({"points": records}, cache_file, indent=1)
    os.replace(temporary_path, path)


def run_point_in_subprocess(point: dict, timeout_seconds: float) -> dict:
    """
    Measure a point in a fresh Python process on the GPU.

    Args:
        point: The measurement point.
        timeout_seconds: The wall-clock limit of the subprocess.

    Returns:
        The point record; ``status`` is ``crash`` or ``timeout`` if the
        subprocess died or hung without reporting.
    """
    environment = dict(os.environ)
    environment.pop("JAX_PLATFORMS", None)
    if ORIGINAL_JAX_PLATFORMS is not None:
        environment["JAX_PLATFORMS"] = ORIGINAL_JAX_PLATFORMS
    environment["JAX_ENABLE_X64"] = "1" if point["precision"] == "double" else "0"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = GPU_MEMORY_FRACTION

    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--point",
        json.dumps(point),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return dict(point, status="timeout", error=f"no result within {timeout_seconds:.0f} s")

    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])

    output_tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-40:])
    return dict(point, status="crash", error=f"exit code {completed.returncode}:\n{output_tail}")


def describe_record(record: dict) -> str:
    """A one-line summary of a point record for the job log."""
    name = (
        f"{record['physics']} {record['scheme']} ({record['backend']}, {record['integrator']}"
        f"{', PP' if record['positivity_preserving'] else ''}) N={record['resolution']} {record['precision']}"
    )
    if record["status"] != "ok":
        return f"{name}: {record['status'].upper()} at {record.get('failed_stage', '?')}: {record.get('error', '')[:300]}"
    return (
        f"{name}: {1e3 * record['time_per_step']:.4g} ms/step ({record['iterations']} steps, "
        f"{len(record['timed_seconds'])} timed runs, spread "
        f"{100 * (max(record['timed_seconds']) / min(record['timed_seconds']) - 1):.1f} %), "
        f"temp/arg {record['temp_bytes'] / record['argument_bytes']:.2f}, "
        f"total {record['total_bytes'] / GIB:.2f} GiB, pallas calls {record['pallas_calls']}, "
        f"compile {record['compile_seconds']:.0f} s, finite {record['final_state_finite']}"
    )


def measure(arguments: argparse.Namespace) -> None:
    """
    Run the sweep of one precision for the requested physics, resuming from
    the cache: every configuration climbs the resolution ladder until its
    first failure; then the extra points follow.

    Args:
        arguments: The command-line arguments.
    """
    for physics in arguments.physics:
        path = cache_path(arguments.data_dir, arguments.tag, arguments.precision, physics)
        records = load_cache(path)
        cached = {point_key(record): record for record in records}

        def measure_ladder(configurations, resolutions, stop_after_failure):
            """Measure the configurations over the resolutions, N outermost."""
            stopped = set()
            for resolution in resolutions:
                for configuration_index, configuration in enumerate(configurations):
                    if configuration_index in stopped:
                        continue
                    point = dict(configuration, resolution=resolution)
                    key = point_key(point)
                    if key in cached and not arguments.rerun:
                        if cached[key]["status"] != "ok" and stop_after_failure:
                            stopped.add(configuration_index)
                        continue

                    record = run_point_in_subprocess(point, arguments.timeout)
                    print(describe_record(record), flush=True)
                    if record["status"] == "ok" and arguments.tag not in record["device_kind"].lower().replace(" ", ""):
                        print(f"WARNING: tag '{arguments.tag}' does not match the device '{record['device_kind']}'.")
                    if record["status"] == "ok" and record["backend"] == "pallas" and record["pallas_calls"] == 0:
                        print("WARNING: a Pallas configuration ran without a single Pallas kernel.")

                    cached[key] = record
                    save_cache(path, list(cached.values()))
                    if record["status"] != "ok" and stop_after_failure:
                        stopped.add(configuration_index)

        measure_ladder(
            main_configurations(physics, arguments.precision),
            arguments.resolutions,
            stop_after_failure=True,
        )
        if not arguments.no_extras:
            measure_ladder(
                extra_configurations(physics, arguments.precision),
                arguments.extra_resolutions,
                stop_after_failure=False,
            )


# -------------------------------------------------------------
# ================ ↑ Sweep driver and cache ↑ =================
# -------------------------------------------------------------


# -------------------------------------------------------------
# =================== ↓ Figures and tables ↓ ==================
# -------------------------------------------------------------


def load_records(data_dir: str, tag: str, precision: str) -> list:
    """All cached point records of one GPU and precision (both physics)."""
    records = []
    for physics in PHYSICS:
        records += load_cache(cache_path(data_dir, tag, precision, physics))
    return records


def find_records(records: list, **selection) -> list:
    """The successful records matching every given field, sorted by resolution."""
    matching = [
        record
        for record in records
        if record["status"] == "ok" and all(record[field] == value for field, value in selection.items())
    ]
    return sorted(matching, key=lambda record: record["resolution"])


def main_series(records: list, physics: str, scheme: str, backend: str) -> dict:
    """
    The main-sweep records of one configuration, keyed by resolution.

    Args:
        records: The point records.
        physics: ``hydro`` or ``mhd``.
        scheme: ``FD`` or ``FV``.
        backend: ``jax`` or ``pallas``.

    Returns:
        The records by resolution.
    """
    selected = find_records(
        records,
        physics=physics,
        scheme=scheme,
        backend=backend,
        integrator=MAIN_INTEGRATOR[scheme],
        positivity_preserving=False,
        cfl=MAIN_CFL[scheme],
    )
    return {record["resolution"]: record for record in selected}


def speedup_series(records: list, physics: str, scheme: str, resolution_range: tuple = None) -> tuple:
    """
    The Pallas speedup T(JAX) / T(Pallas) of a scheme at every resolution
    both backends ran.

    Args:
        records: The point records.
        physics: ``hydro`` or ``mhd``.
        scheme: ``FD`` or ``FV``.
        resolution_range: Optional ``(smallest, largest)`` N to keep.

    Returns:
        The resolutions and the speedups (arrays).
    """
    native = main_series(records, physics, scheme, "jax")
    pallas = main_series(records, physics, scheme, "pallas")
    resolutions = sorted(set(native) & set(pallas))
    if resolution_range is not None:
        resolutions = [
            resolution for resolution in resolutions if resolution_range[0] <= resolution <= resolution_range[1]
        ]
    speedups = [native[resolution]["time_per_step"] / pallas[resolution]["time_per_step"] for resolution in resolutions]
    return np.array(resolutions), np.array(speedups)


def format_resolution_axis(axis, resolutions, ticks=RESOLUTION_TICKS) -> None:
    """Log resolution axis labelled at the given ticks within the measured range."""
    axis.set_xscale("log")
    ticks = [tick for tick in ticks if min(resolutions) <= tick <= max(resolutions)]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in ticks]))
    axis.xaxis.set_minor_locator(NullLocator())


def save_figure(figure, figure_dir: str, name: str) -> None:
    """Write a figure as SVG, PDF and PNG into the figure directory."""
    os.makedirs(figure_dir, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        figure.savefig(layout_path(os.path.join(figure_dir, f"{name}.{extension}")))
    plt.close(figure)
    print(f"wrote {os.path.join(figure_dir, name)}.{{svg,pdf,png}}")


def plot_pallas_speedup(
    records: list,
    figure_dir: str,
    name: str,
    overlay_records: list = None,
    overlay_label: str = None,
    main_label_suffix: str = None,
) -> None:
    """
    The Pallas speedup figure: columns hydrodynamics | MHD, the speedup of the
    Pallas over the native-JAX backend for FD and FV against N within
    ``SPEEDUP_FIGURE_RESOLUTIONS``. The time per step of the configurations is
    printed by ``print_main_table``.

    Args:
        records: The point records of the main precision.
        figure_dir: The directory of the figure files.
        name: The figure file name (without extension).
        overlay_records: Optional records of a second precision whose speedups
            are added as the hollow, dashed variant.
        overlay_label: The legend suffix of the overlay (e.g. "double").
        main_label_suffix: The legend suffix of the main precision (e.g. "single").
    """
    use_paper_style()
    # Side by side in the single-column layout, stacked in one column (sharing
    # N) in the two-column layout.
    if TWO_COLUMN:
        figure, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 4.8), sharex=True, layout="constrained")
    else:
        figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7), sharey=True, layout="constrained")

    maximum_speedup = 1.0
    for axis, physics in zip(axes, PHYSICS):
        measured_resolutions = []
        for scheme in ("FD", "FV"):
            main_label = f"{scheme}, {main_label_suffix}" if main_label_suffix else scheme
            series_of_precisions = [(records, dict(color=SPEEDUP_COLOR[scheme]), main_label)]
            if overlay_records:
                series_of_precisions.append(
                    (overlay_records, variant_style(SPEEDUP_COLOR[scheme]), f"{scheme}, {overlay_label}")
                )
            for precision_records, style, label in series_of_precisions:
                resolutions, speedups = speedup_series(
                    precision_records,
                    physics,
                    scheme,
                    SPEEDUP_FIGURE_RESOLUTIONS,
                )
                if not resolutions.size:
                    continue
                maximum_speedup = max(maximum_speedup, speedups.max())
                measured_resolutions += list(resolutions)
                axis.plot(
                    resolutions,
                    speedups,
                    marker=SPEEDUP_MARKER[scheme],
                    label=label,
                    **style,
                )
        axis.set_title(PHYSICS_TITLE[physics])
        if not TWO_COLUMN or axis is axes[-1]:
            axis.set_xlabel(r"$N$ (grid $2N \times N \times N$)")
        if measured_resolutions:
            format_resolution_axis(axis, measured_resolutions, ticks=sorted(set(measured_resolutions)))
    for axis in axes if TWO_COLUMN else axes[:1]:
        axis.set_ylabel(r"Pallas speedup $T_\mathrm{JAX} / T_\mathrm{Pallas}$")
        axis.set_ylim(0, 1.12 * maximum_speedup)
    figure_legend(figure, ncols=2 if TWO_COLUMN else None)
    save_figure(figure, figure_dir, name)


def plot_memory_breakdown(records: list, figure_dir: str, name: str) -> None:
    """
    The single-GPU memory figure: columns hydrodynamics | MHD; top row the
    total compiled memory versus N with the device memory, bottom row the
    transient-to-argument memory ratio. (The paper condenses it into the
    table of ``print_memory_table``.)

    Args:
        records: The point records of one precision.
        figure_dir: The directory of the figure files.
        name: The figure file name (without extension).
    """
    use_paper_style()
    figure, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.8), sharex="col", layout="constrained")

    device_memory_mib = [
        float(record["gpu"]["memory.total"].split()[0])
        for record in records
        if record["status"] == "ok" and "memory.total" in record.get("gpu", {})
    ]

    for column, physics in enumerate(PHYSICS):
        total_axis = axes[0, column]
        ratio_axis = axes[1, column]
        measured_resolutions = []

        for (scheme, backend), style in CONFIGURATION_STYLE.items():
            series = main_series(records, physics, scheme, backend)
            if not series:
                continue
            resolutions = np.array(sorted(series))
            total_gib = np.array([series[resolution]["total_bytes"] / GIB for resolution in resolutions])
            ratios = np.array(
                [series[resolution]["temp_bytes"] / series[resolution]["argument_bytes"] for resolution in resolutions]
            )
            measured_resolutions += list(resolutions)
            total_axis.loglog(resolutions, total_gib, **style)
            ratio_axis.plot(resolutions, ratios, **style)

        if device_memory_mib:
            total_axis.axhline(
                max(device_memory_mib) / 1024,
                color=GUIDE_COLOR,
                linestyle="-.",
                label="device memory (HBM)",
            )
        ratio_axis.axhline(2.0, color=GUIDE_COLOR, linestyle=":", label="ratio = 2")

        total_axis.set_title(PHYSICS_TITLE[physics])
        total_axis.set_ylabel("total compiled memory (GiB)")
        ratio_axis.set_ylabel("transient / argument memory")
        ratio_axis.set_xlabel(r"$N$ (grid $2N \times N \times N$)")
        ratio_axis.set_ylim(bottom=0)
        if measured_resolutions:
            format_resolution_axis(ratio_axis, measured_resolutions)

    figure_legend(figure, ncols=3)
    save_figure(figure, figure_dir, name)


def print_memory_table(records: list, resolution: int = MEMORY_TABLE_RESOLUTION) -> None:
    """
    Print the total compiled memory of one time step at one resolution as the
    rows of the LaTeX memory table of the paper: one row per scheme and
    backend, one column per physics, in GB (10^9 bytes).

    Args:
        records: The point records of one precision.
        resolution: The resolution N (grid 2N x N x N).
    """
    print(f"\n% total compiled memory at N = {resolution} (grid {2 * resolution} x {resolution} x {resolution}), GB")
    for scheme, backend in (("FD", "pallas"), ("FD", "jax"), ("FV", "pallas"), ("FV", "jax")):
        cells = []
        for physics in PHYSICS:
            record = main_series(records, physics, scheme, backend).get(resolution)
            cells.append(f"{record['total_bytes'] / GIGABYTE:.1f}" if record else "--")
        backend_name = "\\texttt{Pallas}" if backend == "pallas" else "native \\texttt{JAX}"
        print(f"{scheme} & {backend_name} & " + " & ".join(cells) + " \\\\")


def print_main_table(records: list, precision: str) -> None:
    """
    Print the main sweep as Markdown tables (one per physics): time per step,
    speedup, total compiled memory and transient-to-argument ratio of the four
    configurations at every N.

    Args:
        records: The point records of one precision.
        precision: The precision (for the heading).
    """
    configurations = list(CONFIGURATION_STYLE)
    for physics in PHYSICS:
        series = {configuration: main_series(records, physics, *configuration) for configuration in configurations}
        resolutions = sorted(set().union(*[set(entries) for entries in series.values()]))
        if not resolutions:
            continue
        labels = [CONFIGURATION_STYLE[configuration]["label"] for configuration in configurations]
        print(f"\n### {physics}, {precision} precision: time per step (ms), speedup, total memory (GiB), temp/arg")
        print(
            "| N | "
            + " | ".join(f"{label} ms" for label in labels)
            + " | FD speedup | FV speedup | "
            + " | ".join(f"{label} GiB" for label in labels)
            + " | "
            + " | ".join(f"{label} temp/arg" for label in labels)
            + " |"
        )
        print("|" + "---|" * (1 + 3 * len(labels) + 2))

        def cell(configuration, resolution, quantity):
            record = series[configuration].get(resolution)
            if record is None:
                return "-"
            if quantity == "milliseconds":
                return f"{1e3 * record['time_per_step']:.4g}"
            if quantity == "gibibytes":
                return f"{record['total_bytes'] / GIB:.3g}"
            return f"{record['temp_bytes'] / record['argument_bytes']:.2f}"

        for resolution in resolutions:
            speedups = []
            for scheme in ("FD", "FV"):
                native = series[(scheme, "jax")].get(resolution)
                pallas = series[(scheme, "pallas")].get(resolution)
                speedups.append(f"{native['time_per_step'] / pallas['time_per_step']:.2f}" if native and pallas else "-")
            print(
                f"| {resolution} | "
                + " | ".join(cell(configuration, resolution, "milliseconds") for configuration in configurations)
                + f" | {speedups[0]} | {speedups[1]} | "
                + " | ".join(cell(configuration, resolution, "gibibytes") for configuration in configurations)
                + " | "
                + " | ".join(cell(configuration, resolution, "ratio") for configuration in configurations)
                + " |"
            )


def print_extra_table(records: list, precision: str) -> None:
    """
    Print the extra finite-difference Pallas points: time per step and
    memory ratio of LSRK4 and SSPRK4 without and with the positivity-preserving
    WENO, and the relative cost of the positivity-preserving WENO.

    Args:
        records: The point records of one precision.
        precision: The precision (for the heading).
    """
    extras = find_records(records, scheme="FD", backend="pallas", cfl=POSITIVITY_PRESERVING_CFL)
    if not extras:
        return
    print(f"\n### FD (Pallas) extras, {precision} precision, C_cfl = {POSITIVITY_PRESERVING_CFL}")
    print("| physics | N | integrator | ms/step | ms/step with PP | PP cost | temp/arg | temp/arg with PP |")
    print("|---|---|---|---|---|---|---|---|")
    for physics in PHYSICS:
        for resolution in sorted({record["resolution"] for record in extras}):
            for integrator in ("LSRK4", "SSPRK4"):
                pair = {
                    positivity_preserving: find_records(
                        extras,
                        physics=physics,
                        resolution=resolution,
                        integrator=integrator,
                        positivity_preserving=positivity_preserving,
                    )
                    for positivity_preserving in (False, True)
                }
                if not pair[False] or not pair[True]:
                    continue
                without, with_pp = pair[False][0], pair[True][0]
                print(
                    f"| {physics} | {resolution} | {integrator} | {1e3 * without['time_per_step']:.4g} | "
                    f"{1e3 * with_pp['time_per_step']:.4g} | "
                    f"{100 * (with_pp['time_per_step'] / without['time_per_step'] - 1):+.0f} % | "
                    f"{without['temp_bytes'] / without['argument_bytes']:.2f} | "
                    f"{with_pp['temp_bytes'] / with_pp['argument_bytes']:.2f} |"
                )


def make_figures(arguments: argparse.Namespace) -> None:
    """
    The figures and tables from the cache: the Pallas speedup figure and the
    memory figure of each measured precision; the single-precision speedup
    figure carries the double-precision speedups as an overlay when they exist.

    Args:
        arguments: The command-line arguments.
    """
    records_by_precision = {
        precision: load_records(arguments.data_dir, arguments.tag, precision) for precision in PRECISIONS
    }
    single_records = records_by_precision["single"]
    double_records = records_by_precision["double"]

    if single_records:
        plot_pallas_speedup(
            single_records,
            arguments.figure_dir,
            "pallas_speedup",
            overlay_records=double_records or None,
            overlay_label="double",
            main_label_suffix="single",
        )
        plot_memory_breakdown(single_records, arguments.figure_dir, "single_gpu_memory_breakdown")
        print_memory_table(single_records)
    if double_records:
        plot_pallas_speedup(double_records, arguments.figure_dir, "pallas_speedup_fp64")
        plot_memory_breakdown(double_records, arguments.figure_dir, "single_gpu_memory_breakdown_fp64")

    for precision, records in records_by_precision.items():
        print_main_table(records, precision)
        print_extra_table(records, precision)


# -------------------------------------------------------------
# =================== ↑ Figures and tables ↑ ==================
# -------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """The command-line arguments (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure", action="store_true", help="run the sweep (on a GPU node) instead of plotting")
    parser.add_argument("--point", type=str, default=None, help="internal: measure the point given as JSON")
    parser.add_argument("--physics", nargs="+", choices=PHYSICS, default=list(PHYSICS))
    parser.add_argument("--precision", choices=PRECISIONS, default="single")
    parser.add_argument("--resolutions", nargs="+", type=int, default=list(RESOLUTIONS))
    parser.add_argument("--extra-resolutions", nargs="+", type=int, default=list(EXTRA_RESOLUTIONS))
    parser.add_argument("--no-extras", action="store_true", help="skip the PP-WENO / SSPRK4 points")
    parser.add_argument("--rerun", action="store_true", help="measure cached points again")
    parser.add_argument("--tag", default="gh200", help="GPU tag of the cache files")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="directory of the cache files")
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR, help="directory of the figure files")
    parser.add_argument("--timeout", type=float, default=POINT_TIMEOUT_SECONDS, help="seconds per point")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.point is not None:
        point_record = measure_point(json.loads(arguments.point))
        print(RESULT_PREFIX + json.dumps(point_record), flush=True)
    elif arguments.measure:
        measure(arguments)
    else:
        make_figures(arguments)
