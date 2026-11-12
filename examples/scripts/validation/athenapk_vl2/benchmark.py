"""
Performance of the VL2 scheme: astronomix (Pallas, optionally native JAX)
against AthenaPK, in double or single precision.

Both codes run the 3D circularly polarized Alfvén wave (grid 2N x N x N,
VL2 + PLM + HLLD + GLM, CFL 0.3) on the same GPU. For astronomix the time per
step is measured over a fixed number of complete steps (time-step reduction plus
both stages) inside one compiled loop; the compilation (and first run) is timed
separately. Every configuration is timed several times, interleaved with the
others, and the fastest repetition is reported (a sporadic slowdown on a shared
node only ever adds time). After the timing the state is checked: finite, and
the density of the wave (exactly 1) unchanged to truncation error, which would
catch a kernel that mis-addresses a large grid. AthenaPK's figure is its own
``zone-cycles/wallsecond`` over 100 cycles after two warm-up cycles, i.e.
without initialisation (P. Grete's tuned settings: single meshblock,
``minimum_number_of_teams_for_boundary_kernel = 256``, ``scratch_level = 1``).
If a single meshblock does not fit, AthenaPK is retried with smaller meshblocks
(recorded in the results). The GPU clock, power draw and throttle reasons are
logged with every repetition, so that a throttled measurement can be recognised.

One precision per process: single precision runs astronomix with
``jax_enable_x64`` off and AthenaPK's single-precision build
(``ATHENAPK_GPU_SINGLE_BIN``), double precision with x64 on and
``ATHENAPK_GPU_BIN``.

Usage:
    [ATHENAPK_GPU_BIN=... ATHENAPK_GPU_SINGLE_BIN=...] python benchmark.py
        --precision {double,single} --resolutions 16 32 64 128 256
        [--backends pallas native] [--output results.json]

On JUPITER, ``jupiter_athenapk.sh`` sets the AthenaPK variables.
"""

# general
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
PARSER.add_argument("--precision", choices=["double", "single"], default="double")
PARSER.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128])
PARSER.add_argument("--backends", nargs="+", default=["pallas"])
PARSER.add_argument("--repetitions", type=int, default=5)
PARSER.add_argument("--athenapk-repetitions", type=int, default=3)
PARSER.add_argument("--output", default=None)

# ==== GPU selection (skipped when a queue has already assigned one) ====
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =====================================================================

# jax
import jax

# The precision is global in JAX and has to be fixed before the first array.
jax.config.update("jax_enable_x64", PARSER.parse_known_args()[0].precision == "double")
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    HLLD,
    NATIVE_JAX,
    PALLAS,
    VAN_LEER,
    VL2,
    StaticIntVector,
)

# astronomix containers
from astronomix import BackendConfig, FVSolverConfig, SimulationConfig, SimulationParams

# astronomix functions
from astronomix import get_registered_variables
from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave
from astronomix._finite_volume._timestep_estimation._timestep_estimator import _cfl_time_step
from astronomix._finite_volume._state_evolution.evolve_state import _evolve_state_fv

# validation helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from athenapk_runner import (  # noqa: E402
    athenapk_binary,
    athenapk_environment,
    cp_alfven_input,
    performance_summary,
)

#: AthenaPK cycles per measurement, of which the first two are excluded.
ATHENAPK_CYCLES = 102


def gpu_status() -> str:
    """The current SM clock, power draw and throttle reasons of the visible GPU."""
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active",
            "--format=csv,noheader",
            "-i",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        ],
        capture_output=True,
        text=True,
    )
    return query.stdout.strip()


def astronomix_step_function(resolution: int, backend: int, steps: int):
    """
    A compiled function running ``steps`` complete VL2 steps, and its initial state.

    Args:
        resolution: N (grid 2N x N x N).
        backend: ``PALLAS`` or ``NATIVE_JAX``.
        steps: Steps per call.

    Returns:
        ``(run, state, temporary_bytes, density_index)``.
    """
    base_config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(
            time_integrator=VL2,
            limiter=VAN_LEER,
            riemann_solver=HLLD,
        ),
        mhd=True,
        dimensionality=3,
        num_cells=StaticIntVector(2 * resolution, resolution, resolution),
        backend_config=BackendConfig(backend=backend),
    )
    state, config, params = setup_cp_alfven_wave(base_config, SimulationParams(C_cfl=0.3, gamma=5.0 / 3.0))
    registered_variables = get_registered_variables(config)

    def step(_, primitive_state):
        dt = _cfl_time_step(primitive_state, config, params, registered_variables)
        return _evolve_state_fv(primitive_state, dt, params.gamma, config, params, None, registered_variables)

    run = jax.jit(lambda primitive_state: jax.lax.fori_loop(0, steps, step, primitive_state))
    memory = run.lower(state).compile().memory_analysis()
    return run, state, getattr(memory, "temp_size_in_bytes", -1), registered_variables.density_index


def athenapk_meshblocks(resolution: int):
    """
    The meshblock sizes to try for AthenaPK, best first: a single meshblock
    (fastest, as tuned by P. Grete), then halving the block until it fits.
    """
    candidates = [
        (2 * resolution, resolution, resolution),
        (resolution, resolution, resolution),
        (resolution, resolution, resolution // 2),
        (resolution // 2, resolution // 2, resolution // 2),
    ]
    return [candidate for candidate in candidates if min(candidate) >= 16]


def athenapk_run(resolution: int, meshblock, single_precision: bool):
    """
    One AthenaPK measurement of ``ATHENAPK_CYCLES`` cycles.

    Args:
        resolution: N.
        meshblock: The meshblock size.
        single_precision: Use the single-precision build.

    Returns:
        ``(zone_cycles_per_second, process_seconds)``, or ``None`` if AthenaPK failed.
    """
    with tempfile.TemporaryDirectory() as run_directory:
        input_file = os.path.join(run_directory, "cpaw.in")
        with open(input_file, "w") as file:
            file.write(cp_alfven_input(resolution, meshblock_cells=meshblock, cycle_limit=ATHENAPK_CYCLES))
        start = time.perf_counter()
        result = subprocess.run(
            [athenapk_binary(gpu=True, single_precision=single_precision), "-i", input_file],
            cwd=run_directory,
            capture_output=True,
            text=True,
            env=athenapk_environment(),
        )
        process_seconds = time.perf_counter() - start
    if result.returncode != 0:
        print(f"   AthenaPK failed with meshblock {meshblock}:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}", flush=True)
        return None
    zone_cycles_per_second, _, _ = performance_summary(result.stdout)
    return zone_cycles_per_second, process_seconds


def measure_astronomix(resolution: int, arguments, device: str):
    """
    Time every astronomix backend at one resolution (interleaved repetitions).

    Args:
        resolution: N.
        arguments: The command-line arguments.
        device: The device name.

    Returns:
        The result entries.
    """
    cells = 2 * resolution**3
    dtype = jnp.float64 if arguments.precision == "double" else jnp.float32
    # Enough steps for ~1 s of work per repetition, at least 10.
    steps = int(max(10, min(500, 2e9 / cells / 20)))
    variants = {}
    for backend_name in arguments.backends:
        backend = PALLAS if backend_name == "pallas" else NATIVE_JAX
        try:
            start = time.perf_counter()
            run, state, temporary_bytes, density_index = astronomix_step_function(resolution, backend, steps)
            compile_seconds = time.perf_counter() - start
            first_start = time.perf_counter()
            jax.block_until_ready(run(state))
            first_run_seconds = time.perf_counter() - first_start
        except Exception as error:  # an out-of-memory grid is recorded, not fatal
            print(f"N={resolution:4d} astronomix {backend_name}: failed ({type(error).__name__}: {str(error)[:300]})", flush=True)
            continue
        if state.dtype != dtype:
            raise RuntimeError(f"The state is {state.dtype}, expected {dtype}.")
        variants[backend_name] = dict(
            run=run,
            state=state,
            temporary_bytes=temporary_bytes,
            density_index=density_index,
            compile_seconds=compile_seconds,
            first_run_seconds=first_run_seconds,
            times=[],
        )

    statuses = []
    for _ in range(arguments.repetitions):
        for variant in variants.values():
            start = time.perf_counter()
            jax.block_until_ready(variant["run"](variant["state"]))
            variant["times"].append(1e3 * (time.perf_counter() - start) / steps)
        statuses.append(gpu_status())

    entries = []
    for backend_name, variant in variants.items():
        # The density of the CP Alfvén wave is exactly 1; after a few steps it
        # deviates only by truncation error.
        final_state = variant["run"](variant["state"])
        density_deviation = float(jnp.max(jnp.abs(final_state[variant["density_index"]] - 1.0)))
        finite = bool(jnp.all(jnp.isfinite(final_state)))
        milliseconds_per_step = min(variant["times"])
        entry = dict(
            code=f"astronomix {backend_name}",
            precision=arguments.precision,
            resolution=resolution,
            cells=cells,
            steps_per_repetition=steps,
            milliseconds_per_step=milliseconds_per_step,
            all_milliseconds_per_step=variant["times"],
            zone_cycles_per_second=cells / (milliseconds_per_step * 1e-3),
            temporary_memory_mib=variant["temporary_bytes"] / 2**20,
            compile_seconds=variant["compile_seconds"],
            first_run_seconds=variant["first_run_seconds"],
            density_deviation=density_deviation,
            finite=finite,
            gpu_status=statuses,
            device=device,
        )
        entries.append(entry)
        print(
            f"N={resolution:4d} {entry['code']:18s} {arguments.precision:6s}: {milliseconds_per_step:9.3f} ms/step "
            f"({entry['zone_cycles_per_second'] / 1e6:7.1f} Mzone-cycles/s, temp {entry['temporary_memory_mib']:8.0f} MiB, "
            f"compile {entry['compile_seconds']:5.1f} s, max|rho-1| {density_deviation:.1e}, finite {finite})",
            flush=True,
        )
        if not finite or density_deviation > 1e-2:
            print(f"   WARNING: the astronomix state at N={resolution} is wrong", flush=True)
    print(f"   GPU status during the repetitions: {statuses}", flush=True)
    variants.clear()
    jax.clear_caches()
    return entries


def measure_athenapk(resolution: int, arguments, device: str):
    """
    Time AthenaPK at one resolution, trying smaller meshblocks if one does not fit.

    Args:
        resolution: N.
        arguments: The command-line arguments.
        device: The device name.

    Returns:
        The result entries (empty if no meshblock size worked).
    """
    cells = 2 * resolution**3
    single_precision = arguments.precision == "single"
    for meshblock in athenapk_meshblocks(resolution):
        first = athenapk_run(resolution, meshblock, single_precision)
        if first is None:
            continue
        measurements = [first] + [
            athenapk_run(resolution, meshblock, single_precision)
            for _ in range(arguments.athenapk_repetitions - 1)
        ]
        throughputs = [measurement[0] for measurement in measurements]
        milliseconds = [1e3 * cells / throughput for throughput in throughputs]
        entry = dict(
            code="AthenaPK",
            precision=arguments.precision,
            resolution=resolution,
            cells=cells,
            meshblock=list(meshblock),
            meshblocks=cells // (meshblock[0] * meshblock[1] * meshblock[2]),
            cycles_timed=ATHENAPK_CYCLES - 2,
            milliseconds_per_step=min(milliseconds),
            all_milliseconds_per_step=milliseconds,
            zone_cycles_per_second=max(throughputs),
            process_seconds=[measurement[1] for measurement in measurements],
            gpu_status=[gpu_status()],
            device=device,
        )
        print(
            f"N={resolution:4d} {'AthenaPK':18s} {arguments.precision:6s}: {entry['milliseconds_per_step']:9.3f} ms/step "
            f"({entry['zone_cycles_per_second'] / 1e6:7.1f} Mzone-cycles/s, {entry['meshblocks']} meshblock(s) {meshblock})",
            flush=True,
        )
        return [entry]
    print(f"N={resolution:4d} AthenaPK: no meshblock size fits", flush=True)
    return []


def main():
    arguments = PARSER.parse_args()
    device = jax.devices()[0].device_kind
    print(f"device: {device}, precision: {arguments.precision}", flush=True)
    athenapk_available = bool(
        os.environ.get("ATHENAPK_GPU_SINGLE_BIN" if arguments.precision == "single" else "ATHENAPK_GPU_BIN")
    )
    results = []
    for resolution in arguments.resolutions:
        results += measure_astronomix(resolution, arguments, device)
        if athenapk_available:
            results += measure_athenapk(resolution, arguments, device)
        # Rewritten after every resolution, so an interrupted run keeps its data.
        if arguments.output:
            with open(arguments.output, "w") as file:
                json.dump(results, file, indent=1)


if __name__ == "__main__":
    main()
