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

# general
import json
import os
import re

# typing
from typing import Callable, NamedTuple, Optional, Sequence

# jax
import jax
import jax.numpy as jnp
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# astronomix constants
from astronomix.option_classes.simulation_config import (
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SnapshotSettings,
    SimulationParams,
    BackendConfig,
)
from astronomix.option_classes.simulation_config import StaticIntVector, get_solver_config

# astronomix functions
from astronomix import (
    get_helper_data,
    time_integration,
    get_registered_variables,
)


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


#: GPU families we name explicitly, longest-first so that e.g. "gh200" is not
#: mistaken for "h200" and "h100" is not matched inside "gh100".
_KNOWN_GPUS = (
    "gh200", "gb200", "b200", "b100",
    "h200", "h100", "a100", "v100",
    "l40s", "l40", "l4", "a40", "a30", "a10",
    "rtx6000", "a6000",
)


def node_tag() -> str:
    """Return a directory-safe tag for the GPU this process is running on.

    Benchmark runtimes are only comparable within one GPU generation, so every
    sweep stores its data under ``<data_dir>/<node_tag()>/``. Deriving the tag
    from the live device (rather than a flag) means a run physically cannot be
    filed under the wrong hardware -- the failure mode that previously mixed
    H100 runtimes into an A100 dataset.

    Recognises the datacenter parts explicitly (a100/h100/h200/...) and falls
    back to a slug of the device kind for anything unknown, so a new GPU gets
    its own folder instead of raising.
    """
    kind = jax.devices()[0].device_kind
    low = kind.lower()
    for known in _KNOWN_GPUS:
        if known in low.replace(" ", "").replace("-", ""):
            return known
    # Unknown hardware: keep going under a readable slug rather than failing a
    # long sweep at the very end when it tries to write its results.
    slug = _slug(low.replace("nvidia", ""))
    return slug or "unknown_gpu"


def _ensure_dirs(*paths: str) -> None:
    """Create every given output directory (no error if it already exists)."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _build_sharding(num_gpus: int):
    """Build the multi-device sharding for a scaling run.

    Returns ``None`` for a single-GPU run (no sharding needed); otherwise a
    ``NamedSharding`` that splits the domain across ``num_gpus`` devices along
    the x-axis while leaving the variable, y and z axes replicated.
    """
    if num_gpus <= 1:
        return None
    mesh = jax.make_mesh((1, num_gpus, 1, 1), (VARAXIS, XAXIS, YAXIS, ZAXIS))
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


#: Robust timing (``robust_timing=True``): keep re-running the integration until
#: the accumulated timed wall-clock reaches this, so a sporadic one-off cost is a
#: small fraction of at least one sample. Small grids are cheap, so this costs
#: little; large grids exceed it on the first run and are measured once.
_ROBUST_TIMING_TARGET_S = 1.0
#: Hard cap on repeats, so a pathologically fast configuration cannot spin. Sized
#: so the cheapest points (a 20-step N=8 run is ~0.02 s) can still approach the
#: target: with a cap of 5 they never got near it and kept a ~20% spread.
_ROBUST_TIMING_MAX_REPEATS = 25
#: Size of the sporadic one-off cost that contaminates a timed region (measured
#: 2026-07-17 on H100: ~0.1 s, hitting roughly one run in three).
_ROBUST_TIMING_SPIKE_S = 0.1
#: Relative error we are willing to inherit from a spike we failed to dodge.
_ROBUST_TIMING_TOL = 0.01
#: Minimum samples whenever a spike could still matter. Reaching the wall-clock
#: target is NOT sufficient on its own: a 1 s run that gets spiked is still 10%
#: wrong, which is how the 20-step N=128 sp/dp point read 71.3 ms/iter against
#: the 981-step convergence sweep's 65.4. With p(spike) ~ 1/3, 3 samples leave a
#: ~4% chance that every one is contaminated, and the min discards the rest.
_ROBUST_TIMING_MIN_SAMPLES = 3


def _run_simulation(
    spec: BenchmarkSpec,
    N: int,
    setup_fn: Callable,
    num_gpus: int = 1,
    *,
    num_timesteps: Optional[int] = None,
    t_end: Optional[float] = None,
    robust_timing: bool = False,
):
    """Build the per-N config, set up the IC, optionally shard, integrate.

    Returns ``(result, config, params, registered_variables, helper_data)``.

    ``num_timesteps`` (optional) switches the run to a fixed number of
    timesteps -- bounding walltime and making time-per-step directly
    comparable across grid sizes, which is what the scaling sweeps want.
    ``t_end`` (optional) overrides the end time set by ``setup_fn`` (with
    ``num_timesteps`` fixed, ``dt = t_end / num_timesteps``).

    ``robust_timing`` (default off, so existing callers are unchanged) makes the
    reported ``result.runtime`` trustworthy for CHEAP configurations. A single
    unrepeated run is contaminated by a sporadic ~0.1 s one-off cost that lands
    inside the timed region in roughly one run out of three (measured 2026-07-17,
    H100: the same N=8 config alternates between ~0.7 and ~2.4 ms/iter with
    clocks pinned at 1980 MHz and no thermal throttling, and the inflation hops
    between N=8 and N=16 from sweep to sweep). Because ``runtime`` is divided by
    the iteration count, that fixed cost is a ~4x error at N=8 (0.04 s of work)
    and invisible at N=128 (64 s) -- which is exactly the "smallest grid is
    slower than the next one up" artifact.

    The contamination is ADDITIVE and one-sided, so the estimator is the MINIMUM
    over repeats, not the mean: the fastest sample is the one that escaped the
    hiccup. Cheap configurations are repeated (at least ``_ROBUST_TIMING_MIN_SAMPLES``,
    and on until the accumulated wall reaches ``_ROBUST_TIMING_TARGET_S``); a run
    already long enough that a missed spike stays inside ``_ROBUST_TIMING_TOL``
    (>= 10 s) is measured exactly once, so the expensive end of a sweep costs no
    extra wall-clock.
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

    def _integrate():
        r = time_integration(
            initial_state,
            config,
            params,
            registered_variables,
            sharding=sharding,
        )
        # Make sure compute has finished before we read runtime/memory back.
        if hasattr(r, "final_state") and r.final_state is not None:
            r.final_state.block_until_ready()
        return r

    result = _integrate()

    if robust_timing and hasattr(result, "runtime"):
        # The first run also warms up anything that is lazily initialised on the
        # first launch of a freshly compiled program; taking the min over the
        # samples discards it along with any other one-off cost.
        samples = [float(result.runtime)]
        # A run long enough that even a missed spike stays within tolerance
        # (0.1 s / 0.01 = 10 s) needs no repeat -- that is the expensive end of
        # the sweep, where repeating would dominate the wall-clock budget.
        negligible_s = _ROBUST_TIMING_SPIKE_S / _ROBUST_TIMING_TOL
        if samples[0] < negligible_s:
            while len(samples) < _ROBUST_TIMING_MAX_REPEATS and (
                len(samples) < _ROBUST_TIMING_MIN_SAMPLES
                or sum(samples) < _ROBUST_TIMING_TARGET_S
            ):
                samples.append(float(_integrate().runtime))
        best = min(samples)
        if len(samples) > 1:
            spread = 100.0 * (max(samples) - best) / best if best else float("nan")
            print(
                f"[timing] N={N} {spec.label}: {len(samples)} samples, "
                f"min={best:.4f}s, max={max(samples):.4f}s, spread={spread:.0f}% "
                f"-> reporting min",
                flush=True,
            )
        result = result._replace(runtime=best)

    return result, config, params, registered_variables, helper_data


def _mean_l1_error(final_state, true_state, indices: Sequence[int]) -> float:
    """Mean over the requested variables of their per-variable mean L1 error."""
    per_variable_errors = [
        jnp.mean(jnp.abs(final_state[i] - true_state[i])) for i in indices
    ]
    return float(jnp.stack(per_variable_errors).mean())


def assert_correctness_at_resolution(
    benchmarks: Sequence[BenchmarkSpec],
    N: int,
    setup_fn: Callable,
    analytic_fn: Callable,
    error_var_indices_fn: Callable,
    *,
    name: str,
    tol: float,
):
    """Fast single-resolution correctness check against an analytic solution.

    Runs every benchmark once at the (low) resolution ``N``, measures the mean
    L1 error of the final state against ``analytic_fn``, and asserts it is below
    ``tol``. This is the quick pytest counterpart to the heavy
    ``run_convergence_and_runtime`` sweep (which lives in the examples/ paper
    generators): no figures, no runtime accounting, just correctness.
    """
    for spec in benchmarks:
        result, config, params, registered_variables, helper_data = _run_simulation(
            spec, N, setup_fn, num_gpus=1,
        )
        indices = error_var_indices_fn(registered_variables)
        true_state = analytic_fn(config, registered_variables, params, helper_data)
        err = _mean_l1_error(result.final_state, true_state, indices)
        print(f"[{name}] {spec.label} N={N}: L1={err:.3e} (tol={tol:.3e})")
        assert err < tol, (
            f"{name} / {spec.label}: L1 error {err:.3e} exceeds tol {tol:.3e} at N={N}"
        )


def _format_xaxis(ax, N_values):
    """Put the resolution axis on a log scale with one labelled tick per N.

    The default log locator would sprinkle unlabelled minor ticks between the
    sampled resolutions; pinning the major ticks to exactly ``N_values`` keeps
    the axis readable for the small, hand-picked resolution list.
    """
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
    runtime_title: Optional[str] = None,
    data_dir: str,
    figure_dir: str,
    reference_npzs: Optional[Sequence[tuple[str, str]]] = None,
) -> dict:
    """Run the convergence + runtime sweep for every benchmark.

    Produces two figures:
      * ``{figure_dir}/{name}_convergence.svg`` — L1 error vs N
      * ``{figure_dir}/{name}_runtime.svg``     — error/runtime, runtime/N,
        time-per-iteration/N

    and one NPZ per benchmark in ``{data_dir}``. ``reference_npzs`` is a list
    of ``(legend label, npz path)`` overlays (e.g. AthenaPK measurements) in
    the same NPZ format. Returns a dict keyed by benchmark label with the
    captured arrays.
    """
    _ensure_dirs(data_dir, figure_dir)

    results = {}

    # -------------------------------------------------------------
    # ====== ↓ Run the sweep and collect per-benchmark curves ↓ ===
    # -------------------------------------------------------------

    # For every benchmark and resolution we integrate the wave, measure the L1
    # error against the analytic solution plus the runtime, and persist the raw
    # arrays to an NPZ so the figures can be regenerated later without a sweep.
    for spec in benchmarks:
        results[spec.label] = dict(N=[], l1=[], runtime=[], iterations=[])
        for N in N_values:
            result, config, params, registered_variables, helper_data = _run_simulation(
                spec,
                N,
                setup_fn,
                num_gpus=1,
                # The small-N end of this sweep runs for only ~0.04-0.2 s, where a
                # sporadic ~0.1 s one-off cost inside the timed region distorts
                # ms/iter by 2-4x (see _run_simulation). Costs nothing at large N,
                # which already exceeds the repeat target on the first run.
                robust_timing=True,
            )

            indices = error_var_indices_fn(registered_variables)
            true_state = analytic_fn(config, registered_variables, params, helper_data)
            err = _mean_l1_error(result.final_state, true_state, indices)
            runtime = float(result.runtime)
            iters = int(result.num_iterations)

            results[spec.label]["N"].append(N)
            results[spec.label]["l1"].append(err)
            results[spec.label]["runtime"].append(runtime)
            results[spec.label]["iterations"].append(iters)

            print(f"[{name}] {spec.label} N={N}: L1={err:.3e}, runtime={runtime:.2f}s, iters={iters}")

        np.savez(
            os.path.join(data_dir, f"{name}_convergence_{_slug(spec.label)}.npz"),
            N_values=np.array(N_values),
            l1_errors=np.array(results[spec.label]["l1"]),
            runtimes=np.array(results[spec.label]["runtime"]),
            iterations=np.array(results[spec.label]["iterations"]),
        )

    # -------------------------------------------------------------
    # ====== ↑ Run the sweep and collect per-benchmark curves ↑ ===
    # -------------------------------------------------------------

    plot_convergence_and_runtime(
        results,
        N_values,
        name=name,
        title=title,
        runtime_title=runtime_title,
        figure_dir=figure_dir,
        reference_npzs=reference_npzs,
    )

    return results


def plot_convergence_and_runtime(
    results: dict,
    N_values: Sequence[int],
    *,
    name: str,
    title: str,
    runtime_title: Optional[str] = None,
    figure_dir: str,
    reference_npzs: Optional[Sequence[tuple[str, str]]] = None,
) -> None:
    """Render the convergence + runtime figures from captured sweep arrays.

    ``results`` maps benchmark label -> dict with ``N``, ``l1``, ``runtime``
    and ``iterations`` sequences (the return value of
    :func:`run_convergence_and_runtime`, or the per-benchmark NPZs loaded back
    from disk). Split out from the sweep so figures can be regenerated —
    e.g. with updated reference overlays — without redoing the GPU runs.

    ``runtime_title`` (default: ``title``) titles the runtime figure only.
    L1 error is a property of the scheme, not the machine — it is identical on
    every GPU — so the convergence figure should not name hardware, while the
    runtime figure must.
    """
    runtime_title = runtime_title or title
    _ensure_dirs(figure_dir)

    fig_err, ax_err = plt.subplots(1, 1, figsize=(8, 6))
    fig_rt, ax_rt = plt.subplots(3, 1, figsize=(8, 12))

    for label, rec in results.items():
        ax_err.loglog(
            N_values,
            rec["l1"],
            marker="o",
            linewidth=2,
            label=label,
        )
        ax_rt[0].loglog(
            rec["runtime"],
            rec["l1"],
            marker="o",
            linewidth=2,
            label=label,
        )
        ax_rt[1].loglog(
            N_values,
            rec["runtime"],
            marker="o",
            linewidth=2,
            label=label,
        )
        time_per_iter = [
            rt / nit for rt, nit in zip(rec["runtime"], rec["iterations"])
        ]
        ax_rt[2].loglog(
            N_values,
            time_per_iter,
            marker="o",
            linewidth=2,
            label=label,
        )

    # Overlay the reference curves (e.g. AthenaPK) for direct comparison; each
    # overlay gets its own marker so the variants stay distinguishable.
    ref_markers = ["s", "D", "^", "v"]
    for i, (ref_label, ref_npz) in enumerate(reference_npzs or []):
        if not os.path.exists(ref_npz):
            print(f"[{name}] reference NPZ missing, skipping overlay: {ref_npz}")
            continue
        ref = np.load(ref_npz)
        marker = ref_markers[i % len(ref_markers)]
        ax_err.loglog(
            ref["N_values"],
            ref["l1_errors"],
            marker=marker,
            linewidth=2,
            label=ref_label,
        )
        ax_rt[0].loglog(
            ref["runtimes"],
            ref["l1_errors"],
            marker=marker,
            linewidth=2,
            label=ref_label,
        )
        ax_rt[1].loglog(
            ref["N_values"],
            ref["runtimes"],
            marker=marker,
            linewidth=2,
            label=ref_label,
        )
        ax_rt[2].loglog(
            ref["N_values"],
            [t / i for t, i in zip(ref["runtimes"], ref["iterations"])],
            marker=marker,
            linewidth=2,
            label=ref_label,
        )

    # Reference convergence slopes: the FV scheme is expected to approach
    # second order and the FD scheme fifth order, so anchor both guide lines
    # at the coarsest-grid error for visual comparison.
    N_arr = np.array(N_values, dtype=float)
    ref2 = (N_arr / N_arr[0]) ** -2.0
    ref5 = (N_arr / N_arr[0]) ** -5.0
    max_err_start = max(r["l1"][0] for r in results.values() if len(r["l1"]))
    min_err_start = min(r["l1"][0] for r in results.values() if len(r["l1"]))
    ax_err.loglog(
        N_arr,
        max_err_start * ref2,
        "k--",
        alpha=0.7,
        label="$O(N^{-2})$ reference",
    )
    ax_err.loglog(
        N_arr,
        min_err_start * ref5,
        "k:",
        alpha=0.7,
        label="$O(N^{-5})$ reference",
    )

    # -------------------------------------------------------------
    # ============= ↓ Style and write the figures ↓ ===============
    # -------------------------------------------------------------

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
    ax_rt[0].set_title(f"{runtime_title}: error vs runtime", fontsize=13)
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

    # -------------------------------------------------------------
    # ============= ↑ Style and write the figures ↑ ===============
    # -------------------------------------------------------------


def _single_gpu_byte_budget() -> int:
    """Usable HBM (bytes) on one device, for the strong-scaling baseline guard.

    Uses the live XLA allocator limit (already scaled by
    ``XLA_PYTHON_CLIENT_MEM_FRACTION``) when available; otherwise falls back to a
    conservative 80 GiB so the guard still trips for clearly-too-big rungs.
    """
    try:
        ms = jax.devices()[0].memory_stats() or {}
        limit = ms.get("bytes_limit")
        if limit:
            return int(limit)
    except Exception:  # noqa: BLE001  (CPU / older runtimes expose no stats)
        pass
    return 80 * 1024 ** 3


# Observed peak/steady HBM ratio for the LSRK4 + FD-Pallas path: a successful
# N=640 hydro baseline reported ~31 GB steady, while N=768 needed ~118 GiB peak
# under rematerialization -> ~2.35x.  2.5 adds a little margin.  Skipping a
# baseline that would OOM is cheap (we keep the sharded number); a wedged 1-GPU
# OOM is not (it hangs the next collective in an NCCL clique rendezvous until the
# SLURM timeout), so we err toward skipping.
_PEAK_OVER_STEADY = 2.5


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
    single_gpu_byte_budget: Optional[int] = None,
    peak_over_steady: float = _PEAK_OVER_STEADY,
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

    budget = single_gpu_byte_budget or _single_gpu_byte_budget()
    # Per-solver memory of the last *successful* 1-GPU baseline: (N, steady_bytes).
    # Used to predict whether the next N's baseline will fit on one device.
    last_ok_1 = {spec.label: None for spec in benchmarks}

    def _baseline_fits(spec, N) -> bool:
        """Predict (N^3 scaling x peak factor) whether the 1-GPU baseline fits.

        Always runs the first rung (no prior data) and any rung at/below a
        known-good N.  Skipping a doomed baseline avoids the OOM->NCCL-clique
        hang that idles the whole job to the SLURM timeout.
        """
        prev = last_ok_1[spec.label]
        if prev is None:
            return True
        n_prev, steady_prev = prev
        if N <= n_prev:
            return True
        predicted_peak = steady_prev * (N / n_prev) ** 3 * peak_over_steady
        return predicted_peak <= budget

    # N-outer / solver-inner: every rung is fully populated across all solvers
    # before moving on, so each per-rung checkpoint is rectangular and a wall
    # clock timeout still leaves a complete, plottable prefix on disk.
    for i, N in enumerate(N_values):
        for spec in benchmarks:
            rec = out[spec.label]
            if _baseline_fits(spec, N):
                t1, tmp1, arg1, tot1 = _measure(spec, N, num_gpus=1)
                if np.isfinite(t1) and tot1 > 0:
                    last_ok_1[spec.label] = (N, tot1)
            else:
                gib = budget / 1024 ** 3
                print(
                    f"[{name}] {spec.label} N={N}: SKIP 1-GPU baseline "
                    f"(predicted peak > {gib:.0f} GiB device budget); "
                    f"running {num_gpus}-GPU only.",
                    flush=True,
                )
                t1, tmp1, arg1, tot1 = (np.nan, 0, 0, 0)
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
    bs = spec.base_config.backend_config.pallas_block_shape
    return list(bs) if bs is not None else None


def _solver_setting(config, name, default=-1):
    """A field of the solver container of ``config`` (``default`` if the solver has no such field)."""
    return getattr(get_solver_config(config), name, default)


def _spec_time_integrator(spec: "BenchmarkSpec"):
    return int(_solver_setting(spec.base_config, "time_integrator"))


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
        cfg = base_config._replace(backend_config=base_config.backend_config._replace(pallas_block_shape=tuple(bs)))
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
        time_integrator=int(_solver_setting(base_config, "time_integrator")),
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


def _state_extrema(state):
    """
    Per-variable minimum and maximum of a (possibly multi-process sharded)
    state, as host arrays on every process.

    Args:
        state: The ``(var, x, y, z)`` state.

    Returns:
        ``(minimum, maximum)``, each a numpy array of length ``num_vars``.
    """
    spatial_axes = tuple(range(1, state.ndim))
    minimum, maximum = jax.jit(
        lambda array: (jnp.min(array, axis=spatial_axes), jnp.max(array, axis=spatial_axes))
    )(state)
    # The reductions are replicated on every device, so each process reads its
    # own copy.
    return (
        np.asarray(minimum.addressable_shards[0].data),
        np.asarray(maximum.addressable_shards[0].data),
    )


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
    solver_label: str = "FD (Pallas)",
) -> dict:
    """One weak-scaling point: build the globally-sharded IC, run a fixed
    number of timesteps, record runtime + per-device memory + throughput.

    ``state_builder(config, params, sharding, settings) -> (state, config,
    params)`` must produce the initial state already globally sharded (e.g.
    :func:`astronomix.test_setups.hydrodynamics.sound_wave3D.build_sound_wave_state_sharded`).
    Only process 0 writes the result files.  ``jax.distributed.initialize``
    must already have been called by the driver.

    The same measurement serves a strong-scaling point: ``global_cells`` is
    then held fixed while the mesh grows.

    Args:
        state_builder: The sharding-aware initial-condition builder.
        base_config: The simulation configuration without the grid.
        settings: The problem settings handed to ``state_builder``.
        mesh_shape: The ``(var, x, y, z)`` device mesh shape.
        global_cells: The global grid ``(nx, ny, nz)``.
        box_size: The box size (recorded only; ``settings`` sets it).
        cfl: The CFL number (unused with the fixed time step, recorded).
        dt: The fixed time step.
        num_timesteps: The number of time steps per run.
        name: The result-file prefix (``{name}_G{G:03d}.{npz,json}``).
        data_dir: The result directory.
        setup_key: The setup name recorded in the metadata.
        solver_label: The solver label recorded in the metadata.

    Returns:
        A dict with the warm runtime, the cold runtime, the deduced warm-up,
        the cell count, the throughput and the per-device temporary memory.
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

    def _one_run(initial_state):
        """Integrate once and wait for the device to finish."""
        run_result = time_integration(
            initial_state, config, params, registered_variables, sharding=sharding
        )
        if getattr(run_result, "final_state", None) is not None:
            run_result.final_state.block_until_ready()
        return run_result

    result = _one_run(state)
    runtime_cold = float(result.runtime)

    # The timer inside time_integration excludes compilation (AOT compile runs
    # first), but the timed call is still the FIRST execution of the compiled
    # program: one-time costs (NCCL communicator setup, allocator growth,
    # module load) land inside the timed window and grow with node count.
    # Deduce them via T(n) = W + n*s with two points on the SAME allocation:
    # rebuild the (donated) initial state and rerun the identical executable.
    # The warm second run gives the clean per-step cost; W = cold - warm.
    #
    # Free the cold run's outputs BEFORE the warm run: holding its final state
    # alongside the fresh IC and the run's transients overflows device memory
    # on rungs that fill the GPU (the compiled footprint is sized to the card).
    del result
    del state
    state, config, params = state_builder(config, params, sharding, settings)
    params = params._replace(t_end=float(dt) * int(num_timesteps))
    result = _one_run(state)

    # Integrity check of the timed run: the benchmark waves are tiny
    # perturbations of a uniform background, so the per-variable extrema of
    # the final state must stay at the background values. A kernel that
    # computed from wrong neighbours or wrong addresses (halo exchange,
    # indexing of very large shards) would show up here.
    final_state_minimum, final_state_maximum = _state_extrema(result.final_state)

    runtime = float(result.runtime)
    warmup_s = runtime_cold - runtime
    iters = int(result.num_iterations)
    temp = int(result.temporary_memory_bytes)
    arg = int(result.argument_memory_bytes)
    tot = int(result.total_memory_bytes)
    total_cells = int(global_cells[0]) * int(global_cells[1]) * int(global_cells[2])
    # Cell *updates* per second per GPU: cells x timesteps / runtime / GPUs
    # (the convention of the paper's weak-scaling table).
    cells_per_s_per_gpu = (
        (total_cells * iters / runtime / G) if runtime > 0 else float("nan")
    )

    # Gather each process's own measured runtime to expose load skew.
    runtimes = np.asarray(mh.process_allgather(jnp.asarray(runtime)))
    runtimes_cold = np.asarray(mh.process_allgather(jnp.asarray(runtime_cold)))

    if process_index == 0:
        rec = dict(
            G=G,
            mesh_shape=list(mesh_shape),
            global_cells=list(global_cells),
            total_cells=total_cells,
            runtime=runtime,
            runtime_cold=runtime_cold,
            warmup_s=warmup_s,
            runtimes=runtimes,
            runtimes_cold=runtimes_cold,
            iterations=iters,
            temp_bytes_per_dev=temp,
            arg_bytes_per_dev=arg,
            total_bytes_per_dev=tot,
            cells_per_s_per_gpu=cells_per_s_per_gpu,
            dt=float(dt),
            final_state_min=final_state_minimum,
            final_state_max=final_state_maximum,
        )
        meta = dict(
            _gpu_metadata(G),
            setup=setup_key,
            solver_label=solver_label,
            solver_mode=int(config.solver_mode),
            weno_z=bool(_solver_setting(config, "weno_z", False)),
            riemann_solver=int(_solver_setting(config, "riemann_solver")),
            pallas_ct=bool(config.backend_config.pallas_ct),
            pallas_block_shape=list(base_config.backend_config.pallas_block_shape or []),
            time_integrator=int(_solver_setting(base_config, "time_integrator")),
            num_timesteps=int(num_timesteps),
            box_size=list(box_size),
        )
        _write_results(data_dir, f"{name}_G{G:03d}", rec, meta)
        print(
            f"[{name}] G={G} {global_cells} runtime={runtime:.3f}s "
            f"(cold={runtime_cold:.3f}s, warmup={warmup_s:.3f}s) "
            f"iters={iters} temp/dev={temp / 1024**3:.2f}GB "
            f"throughput={cells_per_s_per_gpu:.3e} cells/s/GPU",
            flush=True,
        )
        print(
            f"[{name}] final state per variable: "
            f"min={np.array2string(final_state_minimum, precision=6)} "
            f"max={np.array2string(final_state_maximum, precision=6)}",
            flush=True,
        )

    mh.sync_global_devices("weak_scaling_point_done")
    return dict(
        G=G, runtime=runtime, runtime_cold=runtime_cold, warmup_s=warmup_s,
        total_cells=total_cells,
        cells_per_s_per_gpu=cells_per_s_per_gpu, temp_bytes_per_dev=temp,
    )
