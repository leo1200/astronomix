"""
Shared benchmark machinery for the methods-paper 3D linear-wave tests.

A "benchmark" here is the combination of:
- a problem-specific initial-state factory ``setup_fn`` and matching
  analytic-state factory ``analytic_fn``,
- one or more :class:`BenchmarkSpec` rows (a ``SimulationConfig`` template
  plus a label and CFL number),
- a sweep over grid resolutions.

The convergence/runtime pipeline and the strong-scaling+memory pipeline are
shared so that every physics-module benchmark in this paper produces the
same plot suite (see ``project_methods_paper_benchmarks`` memory).
"""

import json
import os
import re
from typing import Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.option_classes.simulation_config import (
    SimulationConfig,
    SnapshotSettings,
    StaticIntVector,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables


class BenchmarkSpec(NamedTuple):
    """A single curve in a benchmark sweep."""

    #: Human-readable legend label, e.g. "FD (Pallas)".
    label: str
    #: SimulationConfig with everything set except ``num_cells`` (injected
    #: per-N inside the sweep).
    base_config: SimulationConfig
    #: CFL number for this configuration.
    cfl: float


def grid_shape_3d(N: int) -> StaticIntVector:
    """Standard methods-paper grid shape: (2N, N, N)."""
    return StaticIntVector(2 * N, N, N)


def _slug(text: str) -> str:
    """Lowercase, alphanumeric-and-underscore version of ``text``."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _build_sharding(num_gpus: int):
    if num_gpus <= 1:
        return None
    # Auto axis types: JAX >= 0.10 defaults make_mesh to Explicit axes, but the
    # code relies on GSPMD `with_sharding_constraint` (which needs Auto axes).
    mesh = jax.make_mesh(
        (1, num_gpus, 1, 1), (VARAXIS, XAXIS, YAXIS, ZAXIS),
        axis_types=(AxisType.Auto,) * 4,
    )
    return jax.NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))


def _ensure_snapshot_config(config: SimulationConfig) -> SimulationConfig:
    """Force the flags the helpers need so callers can supply a minimal config."""
    updates = {}
    if not config.return_snapshots:
        updates["return_snapshots"] = True
    if not config.print_elapsed_time:
        updates["print_elapsed_time"] = True
    if not config.memory_analysis:
        updates["memory_analysis"] = True
    if not config.snapshot_settings.return_final_state:
        updates["snapshot_settings"] = config.snapshot_settings._replace(
            return_final_state=True
        )
    return config._replace(**updates) if updates else config


def _run_simulation(
    spec: BenchmarkSpec,
    N: int,
    setup_fn: Callable,
    num_gpus: int = 1,
    *,
    num_timesteps: Optional[int] = None,
    t_end: Optional[float] = None,
):
    """Build the per-N config, set up the IC, optionally shard, integrate.

    Returns ``(result, config, params, registered_variables, helper_data)``.

    ``num_timesteps`` (optional) switches the run to a fixed number of
    timesteps -- bounding walltime and making time-per-step directly
    comparable across grid sizes, which is what the scaling sweeps want.
    ``t_end`` (optional) overrides the end time set by ``setup_fn`` (with
    ``num_timesteps`` fixed, ``dt = t_end / num_timesteps``).
    """
    # Clear JIT/compilation caches between successive runs. Reusing the same
    # cached compile across (sharded, unsharded) inputs surfaces as
    # ``AttributeError: 'UnspecifiedValue' object has no attribute
    # addressable_devices_indices_map`` when the array sharding diverges
    # from what the cached trace expects.
    jax.clear_caches()

    base = _ensure_snapshot_config(spec.base_config)
    base = base._replace(num_cells=grid_shape_3d(N))
    initial_state, config, params = setup_fn(base, SimulationParams(C_cfl=spec.cfl))
    if num_timesteps is not None:
        config = config._replace(fixed_timestep=True, num_timesteps=int(num_timesteps))
    if t_end is not None:
        params = params._replace(t_end=float(t_end))
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    sharding = _build_sharding(num_gpus)
    if sharding is not None:
        initial_state = jax.device_put(initial_state, sharding)

    result = time_integration(
        initial_state,
        config,
        params,
        registered_variables,
        sharding=sharding,
    )
    # Make sure compute has finished before we read runtime/memory back.
    if hasattr(result, "final_state") and result.final_state is not None:
        result.final_state.block_until_ready()
    return result, config, params, registered_variables, helper_data


def _mean_l1_error(final_state, true_state, indices: Sequence[int]) -> float:
    errs = [jnp.mean(jnp.abs(final_state[i] - true_state[i])) for i in indices]
    return float(jnp.stack(errs).mean())


def _format_xaxis(ax, N_values):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator(N_values))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(n) for n in N_values]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def run_convergence_and_runtime(
    benchmarks: Sequence[BenchmarkSpec],
    N_values: Sequence[int],
    setup_fn: Callable,
    analytic_fn: Callable,
    error_var_indices_fn: Callable,
    *,
    name: str,
    title: str,
    data_dir: str,
    figure_dir: str,
    athenapk_npz: Optional[str] = None,
) -> dict:
    """Run the convergence + runtime sweep for every benchmark.

    Produces two figures:
      * ``{figure_dir}/{name}_convergence.svg`` — L1 error vs N
      * ``{figure_dir}/{name}_runtime.svg``     — error/runtime, runtime/N,
        time-per-iteration/N

    and one NPZ per benchmark in ``{data_dir}``. Returns a dict keyed by
    benchmark label with the captured arrays.
    """
    _ensure_dirs(data_dir, figure_dir)

    results = {}
    for spec in benchmarks:
        results[spec.label] = dict(N=[], l1=[], runtime=[], iterations=[])

    fig_err, ax_err = plt.subplots(1, 1, figsize=(8, 6))
    fig_rt, ax_rt = plt.subplots(3, 1, figsize=(8, 12))

    for spec in benchmarks:
        for N in N_values:
            result, config, params, rv, helper = _run_simulation(spec, N, setup_fn, num_gpus=1)

            indices = error_var_indices_fn(rv)
            true_state = analytic_fn(config, rv, params, helper)
            err = _mean_l1_error(result.final_state, true_state, indices)
            runtime = float(result.runtime)
            iters = int(result.num_iterations)

            results[spec.label]["N"].append(N)
            results[spec.label]["l1"].append(err)
            results[spec.label]["runtime"].append(runtime)
            results[spec.label]["iterations"].append(iters)

            print(f"[{name}] {spec.label} N={N}: L1={err:.3e}, runtime={runtime:.2f}s, iters={iters}")

        ax_err.loglog(
            N_values, results[spec.label]["l1"],
            marker="o", linewidth=2, label=spec.label,
        )
        ax_rt[0].loglog(
            results[spec.label]["runtime"], results[spec.label]["l1"],
            marker="o", linewidth=2, label=spec.label,
        )
        ax_rt[1].loglog(
            N_values, results[spec.label]["runtime"],
            marker="o", linewidth=2, label=spec.label,
        )
        time_per_iter = [
            rt / nit for rt, nit in zip(
                results[spec.label]["runtime"], results[spec.label]["iterations"]
            )
        ]
        ax_rt[2].loglog(
            N_values, time_per_iter,
            marker="o", linewidth=2, label=spec.label,
        )

        np.savez(
            os.path.join(data_dir, f"{name}_convergence_{_slug(spec.label)}.npz"),
            N_values=np.array(N_values),
            l1_errors=np.array(results[spec.label]["l1"]),
            runtimes=np.array(results[spec.label]["runtime"]),
            iterations=np.array(results[spec.label]["iterations"]),
        )

    if athenapk_npz is not None and os.path.exists(athenapk_npz):
        athena = np.load(athenapk_npz)
        ax_err.loglog(athena["N_values"], athena["l1_errors"], marker="s", linewidth=2, label="AthenaPK")
        ax_rt[0].loglog(athena["runtimes"], athena["l1_errors"], marker="s", linewidth=2, label="AthenaPK")
        ax_rt[1].loglog(athena["N_values"], athena["runtimes"], marker="s", linewidth=2, label="AthenaPK")
        ax_rt[2].loglog(
            athena["N_values"],
            [t / i for t, i in zip(athena["runtimes"], athena["iterations"])],
            marker="s", linewidth=2, label="AthenaPK",
        )

    N_arr = np.array(N_values, dtype=float)
    ref2 = (N_arr / N_arr[0]) ** -2.0
    ref5 = (N_arr / N_arr[0]) ** -5.0
    max_err_start = max(r["l1"][0] for r in results.values() if r["l1"])
    min_err_start = min(r["l1"][0] for r in results.values() if r["l1"])
    ax_err.loglog(N_arr, max_err_start * ref2, "k--", alpha=0.7, label="$O(N^{-2})$ reference")
    ax_err.loglog(N_arr, min_err_start * ref5, "k:", alpha=0.7, label="$O(N^{-5})$ reference")

    ax_err.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_err.set_ylabel("Average $L_1$ error", fontsize=12)
    ax_err.set_title(f"{title}: convergence", fontsize=14)
    _format_xaxis(ax_err, N_values)
    ax_err.legend(loc="lower left", fontsize=9)
    ax_err.grid(True, which="major", ls="-", alpha=0.2)
    fig_err.tight_layout()
    fig_err.savefig(os.path.join(figure_dir, f"{name}_convergence.svg"))

    ax_rt[0].set_xlabel("Runtime (s)", fontsize=12)
    ax_rt[0].set_ylabel("Average $L_1$ error", fontsize=12)
    ax_rt[0].set_title(f"{title}: error vs runtime", fontsize=13)
    ax_rt[1].set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_rt[1].set_ylabel("Runtime (s)", fontsize=12)
    ax_rt[1].set_title("Runtime vs N", fontsize=13)
    ax_rt[2].set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_rt[2].set_ylabel("Time per iteration (s)", fontsize=12)
    ax_rt[2].set_title("Time per iteration vs N", fontsize=13)
    for ax in (ax_rt[1], ax_rt[2]):
        _format_xaxis(ax, N_values)
    for ax in ax_rt:
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="major", ls="-", alpha=0.2)
    fig_rt.tight_layout()
    fig_rt.savefig(os.path.join(figure_dir, f"{name}_runtime.svg"))

    return results


def run_strong_scaling(
    benchmarks: Sequence[BenchmarkSpec],
    N_values: Sequence[int],
    setup_fn: Callable,
    *,
    num_gpus: int,
    name: str,
    title: str,
    data_dir: str,
    figure_dir: str,
    num_timesteps: Optional[int] = None,
    t_end: Optional[float] = None,
) -> dict:
    """Run a 1-GPU vs ``num_gpus``-GPU sweep for every benchmark.

    Produces ``{figure_dir}/{name}_strong_scaling.svg`` containing
    runtime, speedup, temporary memory and argument memory panels, plus a
    single NPZ ``{data_dir}/{name}_strong_scaling.npz`` with all raw bytes.
    Returns the captured dict keyed by benchmark label.
    """
    _ensure_dirs(data_dir, figure_dir)

    available = jax.local_device_count()
    if available < num_gpus:
        raise RuntimeError(
            f"Strong scaling needs {num_gpus} GPUs but only {available} are visible."
        )

    def _measure(spec, N, num_gpus):
        try:
            r, *_ = _run_simulation(
                spec, N, setup_fn, num_gpus=num_gpus,
                num_timesteps=num_timesteps, t_end=t_end,
            )
            return (
                float(r.runtime),
                int(r.temporary_memory_bytes),
                int(r.argument_memory_bytes),
                int(r.total_memory_bytes),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] {spec.label} N={N} on {num_gpus} GPU(s) FAILED: {exc!r}")
            return (np.nan, 0, 0, 0)

    out = {
        spec.label: dict(
            runtime_1=[], runtime_N=[],
            temp_1=[], temp_N=[],
            arg_1=[], arg_N=[],
            total_1=[], total_N=[],
        )
        for spec in benchmarks
    }

    npz_path = os.path.join(data_dir, f"{name}_strong_scaling.npz")

    def _checkpoint(n_done):
        """Persist whatever N rungs have completed so far (timeout-safe)."""
        flat = {"N_values": np.array(N_values[:n_done]), "num_gpus": num_gpus}
        for label, rec in out.items():
            slug = _slug(label)
            for k, v in rec.items():
                flat[f"{slug}__{k}"] = np.array(v)
        np.savez(npz_path, **flat)

    # N-outer / solver-inner: every rung is fully populated across all solvers
    # before moving on, so each per-rung checkpoint is rectangular and a wall
    # clock timeout still leaves a complete, plottable prefix on disk.
    for i, N in enumerate(N_values):
        for spec in benchmarks:
            rec = out[spec.label]
            t1, tmp1, arg1, tot1 = _measure(spec, N, num_gpus=1)
            tN, tmpN, argN, totN = _measure(spec, N, num_gpus=num_gpus)
            rec["runtime_1"].append(t1)
            rec["runtime_N"].append(tN)
            rec["temp_1"].append(tmp1)
            rec["temp_N"].append(tmpN)
            rec["arg_1"].append(arg1)
            rec["arg_N"].append(argN)
            rec["total_1"].append(tot1)
            rec["total_N"].append(totN)
            if np.isfinite(t1) and np.isfinite(tN) and tN > 0:
                speedup_str = f"{t1 / tN:.2f}x"
            else:
                speedup_str = "n/a"
            print(
                f"[{name}] {spec.label} N={N}: "
                f"1 GPU={t1:.2f}s, "
                f"{num_gpus} GPUs={tN:.2f}s, "
                f"speedup={speedup_str}",
                flush=True,
            )
        _checkpoint(i + 1)

    MB = 1024 ** 2
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax_t, ax_s = axes[0]
    ax_temp, ax_arg = axes[1]

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    for idx, (label, rec) in enumerate(out.items()):
        color = cycle[idx % len(cycle)]
        r1 = np.array(rec["runtime_1"])
        rN = np.array(rec["runtime_N"])
        speedup = r1 / rN

        ax_t.loglog(N_values, r1, color=color, marker="o", linestyle="-",
                    linewidth=2, label=f"{label}, 1 GPU")
        ax_t.loglog(N_values, rN, color=color, marker="s", linestyle="--",
                    linewidth=2, label=f"{label}, {num_gpus} GPUs")

        ax_s.plot(N_values, speedup, color=color, marker="o", linewidth=2, label=label)

        ax_temp.loglog(N_values, np.array(rec["temp_1"]) / MB, color=color,
                       marker="o", linestyle="-", linewidth=2, label=f"{label}, 1 GPU")
        ax_temp.loglog(N_values, np.array(rec["temp_N"]) / MB, color=color,
                       marker="s", linestyle="--", linewidth=2,
                       label=f"{label}, {num_gpus} GPUs (per device)")

        ax_arg.loglog(N_values, np.array(rec["arg_1"]) / MB, color=color,
                      marker="o", linestyle="-", linewidth=2, label=f"{label}, 1 GPU")
        ax_arg.loglog(N_values, np.array(rec["arg_N"]) / MB, color=color,
                      marker="s", linestyle="--", linewidth=2,
                      label=f"{label}, {num_gpus} GPUs (per device)")

    ax_s.axhline(num_gpus, color="k", linestyle="--", alpha=0.7,
                 label=f"ideal speedup ({num_gpus}x)")

    ax_t.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_t.set_ylabel("Runtime (s)", fontsize=12)
    ax_t.set_title(f"{title}: strong-scaling runtime", fontsize=13)
    ax_s.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_s.set_ylabel(f"Speedup ($T_1 / T_{{{num_gpus}}}$)", fontsize=12)
    ax_s.set_title("Strong-scaling speedup", fontsize=13)
    ax_temp.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_temp.set_ylabel("Temporary memory per device (MB)", fontsize=12)
    ax_temp.set_title("Compiled-step temporary memory", fontsize=13)
    ax_arg.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
    ax_arg.set_ylabel("Argument memory per device (MB)", fontsize=12)
    ax_arg.set_title("Compiled-step argument memory", fontsize=13)

    for ax in (ax_t, ax_s, ax_temp, ax_arg):
        _format_xaxis(ax, N_values)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="major", ls="-", alpha=0.2)
    ax_s.set_yscale("linear")
    speedups_seen = []
    for r in out.values():
        s = np.array(r["runtime_1"]) / np.array(r["runtime_N"])
        s = s[np.isfinite(s)]
        if s.size:
            speedups_seen.append(float(s.max()))
    max_speedup_seen = max(speedups_seen) if speedups_seen else float(num_gpus)
    ax_s.set_ylim(0, max(num_gpus, max_speedup_seen) * 1.1)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, f"{name}_strong_scaling.svg"))

    flat = {"N_values": np.array(N_values), "num_gpus": num_gpus}
    for label, rec in out.items():
        slug = _slug(label)
        for k, v in rec.items():
            if k == "N":
                continue
            flat[f"{slug}__{k}"] = np.array(v)
    np.savez(os.path.join(data_dir, f"{name}_strong_scaling.npz"), **flat)

    return out


# ---------------------------------------------------------------------------
# Scaling campaign helpers (single-GPU sweeps, block-shape sweep, weak scaling).
# All of these write a standardised (NPZ + JSON metadata) pair under
# ``pytests/scaling_results/`` so figures can be regenerated/restyled later
# without re-running the (expensive) simulations.  None of them overwrite the
# existing ``*_strong_scaling.npz`` / ``*_convergence_*.npz`` files.
# ---------------------------------------------------------------------------


def _gpu_metadata(num_gpus: int) -> dict:
    """Hardware / run metadata captured for every result file."""
    devices = jax.devices()
    kind = devices[0].device_kind if devices else "unknown"
    return dict(
        gpu_model=str(kind),
        num_gpus=int(num_gpus),
        num_processes=int(jax.process_count()),
        num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", "1")),
        partition=os.environ.get("SLURM_JOB_PARTITION", ""),
        nodelist=os.environ.get("SLURM_JOB_NODELIST", ""),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        x64=bool(jax.config.read("jax_enable_x64")),
    )


def _spec_block_shape(spec: "BenchmarkSpec"):
    bs = getattr(spec.base_config, "pallas_block_shape", None)
    return list(bs) if bs is not None else None


def _spec_time_integrator(spec: "BenchmarkSpec"):
    return int(getattr(spec.base_config, "time_integrator", -1))


def _write_results(data_dir: str, name: str, arrays: dict, metadata: dict) -> None:
    """Write ``{name}.npz`` (raw arrays) + ``{name}.json`` (metadata)."""
    _ensure_dirs(data_dir)
    np_arrays = {}
    for k, v in arrays.items():
        try:
            np_arrays[k] = np.array(v)
        except ValueError:
            np_arrays[k] = np.array(v, dtype=object)
    np.savez(os.path.join(data_dir, f"{name}.npz"), **np_arrays)
    with open(os.path.join(data_dir, f"{name}.json"), "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    print(f"[results] wrote {os.path.join(data_dir, name)}.{{npz,json}}")


def run_single_gpu_bench(
    benchmarks: Sequence[BenchmarkSpec],
    N_values: Sequence[int],
    setup_fn: Callable,
    *,
    name: str,
    setup_key: str,
    data_dir: str,
    num_timesteps: Optional[int] = None,
    t_end: Optional[float] = None,
) -> dict:
    """Single-GPU runtime + per-device memory sweep over ``N`` (grid 2N x N x N).

    One NPZ+JSON pair per benchmark spec.  Stops increasing ``N`` for a spec
    after the first failure (OOM), recording it as a NaN row so the largest
    feasible size is visible in the data.
    """
    out = {}
    for spec in benchmarks:
        rec = dict(
            N=[], grid=[], cells=[], runtime=[], iterations=[],
            temp_bytes=[], arg_bytes=[], total_bytes=[],
        )
        for N in N_values:
            gx, gy, gz = 2 * N, N, N
            try:
                r, config, *_ = _run_simulation(
                    spec, N, setup_fn, num_gpus=1,
                    num_timesteps=num_timesteps, t_end=t_end,
                )
                runtime = float(r.runtime)
                iters = int(r.num_iterations)
                temp = int(r.temporary_memory_bytes)
                arg = int(r.argument_memory_bytes)
                tot = int(r.total_memory_bytes)
                print(
                    f"[{name}] {spec.label} N={N} ({gx}x{gy}x{gz}): "
                    f"runtime={runtime:.3f}s iters={iters} "
                    f"temp={temp / 1024**2:.1f}MB total={tot / 1024**2:.1f}MB"
                )
                failed = False
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] {spec.label} N={N} FAILED: {exc!r}")
                runtime, iters, temp, arg, tot = float("nan"), 0, 0, 0, 0
                failed = True
            rec["N"].append(N)
            rec["grid"].append([gx, gy, gz])
            rec["cells"].append(gx * gy * gz)
            rec["runtime"].append(runtime)
            rec["iterations"].append(iters)
            rec["temp_bytes"].append(temp)
            rec["arg_bytes"].append(arg)
            rec["total_bytes"].append(tot)
            if failed:
                break
        meta = dict(
            _gpu_metadata(1),
            setup=setup_key,
            solver_label=spec.label,
            pallas_block_shape=_spec_block_shape(spec),
            time_integrator=_spec_time_integrator(spec),
            cfl=spec.cfl,
        )
        _write_results(data_dir, f"{name}_{_slug(spec.label)}", rec, meta)
        out[spec.label] = rec
    return out


def run_block_shape_sweep(
    base_config: SimulationConfig,
    block_shapes: Sequence[tuple],
    N: int,
    setup_fn: Callable,
    *,
    name: str,
    setup_key: str,
    data_dir: str,
    cfl: float,
    label: str = "FD (Pallas)",
    num_timesteps: Optional[int] = None,
    t_end: Optional[float] = None,
) -> dict:
    """Sweep ``pallas_block_shape`` at a fixed grid 2N x N x N on one GPU."""
    gx, gy, gz = 2 * N, N, N
    rec = dict(
        block_shape=[], runtime=[], iterations=[],
        temp_bytes=[], arg_bytes=[], total_bytes=[],
    )
    for bs in block_shapes:
        cfg = base_config._replace(pallas_block_shape=tuple(bs))
        spec = BenchmarkSpec(label=f"{label} {tuple(bs)}", base_config=cfg, cfl=cfl)
        try:
            r, *_ = _run_simulation(
                spec, N, setup_fn, num_gpus=1,
                num_timesteps=num_timesteps, t_end=t_end,
            )
            runtime = float(r.runtime)
            iters = int(r.num_iterations)
            temp = int(r.temporary_memory_bytes)
            arg = int(r.argument_memory_bytes)
            tot = int(r.total_memory_bytes)
            print(
                f"[{name}] block_shape={tuple(bs)} ({gx}x{gy}x{gz}): "
                f"runtime={runtime:.3f}s temp={temp / 1024**2:.1f}MB"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] block_shape={tuple(bs)} FAILED: {exc!r}")
            runtime, iters, temp, arg, tot = float("nan"), 0, 0, 0, 0
        rec["block_shape"].append(list(bs))
        rec["runtime"].append(runtime)
        rec["iterations"].append(iters)
        rec["temp_bytes"].append(temp)
        rec["arg_bytes"].append(arg)
        rec["total_bytes"].append(tot)
    meta = dict(
        _gpu_metadata(1),
        setup=setup_key,
        solver_label=label,
        N=N,
        grid=[gx, gy, gz],
        cfl=cfl,
        time_integrator=int(getattr(base_config, "time_integrator", -1)),
    )
    _write_results(data_dir, name, rec, meta)
    return rec


def _build_global_sharding(mesh_shape):
    """NamedSharding over ALL processes' devices for the (VAR,X,Y,Z) mesh."""
    devices = jax.devices()
    n = 1
    for m in mesh_shape:
        n *= int(m)
    if n != len(devices):
        raise RuntimeError(
            f"mesh_shape {mesh_shape} needs {n} devices but {len(devices)} are visible."
        )
    # Auto axis types: see note in _build_sharding (JAX >= 0.10 compatibility).
    mesh = jax.make_mesh(
        mesh_shape, (VARAXIS, XAXIS, YAXIS, ZAXIS),
        axis_types=(AxisType.Auto,) * 4, devices=devices,
    )
    return jax.NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))


def run_weak_scaling_point(
    state_builder: Callable,
    base_config: SimulationConfig,
    settings,
    *,
    mesh_shape,
    global_cells,
    box_size,
    cfl: float,
    dt: float,
    num_timesteps: int,
    name: str,
    data_dir: str,
    setup_key: str = "hydro_weak",
) -> dict:
    """One weak-scaling point: build the globally-sharded IC, run a fixed
    number of timesteps, record runtime + per-device memory + throughput.

    ``state_builder(config, params, sharding, settings) -> (state, config,
    params)`` must produce the initial state already globally sharded (e.g.
    :func:`astronomix.test_setups.hydrodynamics.sound_wave3D.build_sound_wave_state_sharded`).
    Only process 0 writes the result files.  ``jax.distributed.initialize``
    must already have been called by the driver.
    """
    from jax.experimental import multihost_utils as mh

    process_index = jax.process_index()
    G = jax.device_count()
    sharding = _build_global_sharding(mesh_shape)

    config = base_config._replace(
        num_cells=StaticIntVector(*global_cells),
        fixed_timestep=True,
        num_timesteps=int(num_timesteps),
        memory_analysis=True,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    params = SimulationParams(C_cfl=cfl)
    state, config, params = state_builder(config, params, sharding, settings)
    # Force a fixed, stable dt so every rung does identical per-GPU work.
    params = params._replace(t_end=float(dt) * int(num_timesteps))

    registered_variables = get_registered_variables(config)
    result = time_integration(
        state, config, params, registered_variables, sharding=sharding
    )
    if getattr(result, "final_state", None) is not None:
        result.final_state.block_until_ready()

    runtime = float(result.runtime)
    iters = int(result.num_iterations)
    temp = int(result.temporary_memory_bytes)
    arg = int(result.argument_memory_bytes)
    tot = int(result.total_memory_bytes)
    total_cells = int(global_cells[0]) * int(global_cells[1]) * int(global_cells[2])
    cells_per_s_per_gpu = (total_cells / runtime / G) if runtime > 0 else float("nan")

    # Gather each process's own measured runtime to expose load skew.
    runtimes = np.asarray(mh.process_allgather(jnp.asarray(runtime)))

    if process_index == 0:
        rec = dict(
            G=G,
            mesh_shape=list(mesh_shape),
            global_cells=list(global_cells),
            total_cells=total_cells,
            runtime=runtime,
            runtimes=runtimes,
            iterations=iters,
            temp_bytes_per_dev=temp,
            arg_bytes_per_dev=arg,
            total_bytes_per_dev=tot,
            cells_per_s_per_gpu=cells_per_s_per_gpu,
            dt=float(dt),
        )
        meta = dict(
            _gpu_metadata(G),
            setup=setup_key,
            solver_label="FD (Pallas)",
            pallas_block_shape=list(getattr(base_config, "pallas_block_shape", []) or []),
            time_integrator=int(getattr(base_config, "time_integrator", -1)),
            num_timesteps=int(num_timesteps),
            box_size=list(box_size),
        )
        _write_results(data_dir, f"{name}_G{G:03d}", rec, meta)
        print(
            f"[{name}] G={G} {global_cells} runtime={runtime:.3f}s "
            f"iters={iters} temp/dev={temp / 1024**3:.2f}GB "
            f"throughput={cells_per_s_per_gpu:.3e} cells/s/GPU"
        )

    mh.sync_global_devices("weak_scaling_point_done")
    return dict(
        G=G, runtime=runtime, total_cells=total_cells,
        cells_per_s_per_gpu=cells_per_s_per_gpu, temp_bytes_per_dev=temp,
    )
