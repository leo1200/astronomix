"""
Strong scaling on JUPITER (NVIDIA GH200, 4 GPUs per node): the speedup
T(1)/T(G) versus resolution of the finite-difference (WENO-Z) and the
finite-volume (VL2) scheme, for hydrodynamics and ideal MHD -- the paper's
``fig:strong_speedup`` (``strong_speedup_hydro_mhd.svg``).

The measurement uses the weak-scaling machinery with a fixed global grid
(2N, N, N): one process per node driving all of its GPUs, the initial
condition built
already sharded (1D decomposition along x), fp32, the Pallas backend, a fixed
number of time steps, and every point executed twice on the same allocation
with the warm (second) execution quoted (``run_weak_scaling_point``). The
number of steps grows as N^-3 below N = 512, so every point does the same
work and small grids are not timed over a few milliseconds.

    # one measurement job per (scheme, physics, G), see pytests/runners/strong_scaling.sh
    srun --ntasks-per-node=1 --gres=gpu:4 --gpu-bind=none python examples/scripts/scaling/strong_scaling_speedup.py \
        --scheme fd --physics hydro --resolutions 128 192 256 384 512 640 768

    # assemble the figure from the stored points (no GPU needed)
    python examples/scripts/scaling/strong_scaling_speedup.py --plot

A 1-GPU baseline that does not fit into one GPU's memory is extrapolated with
a power law in N fitted to the largest measured baselines (time per step,
log-log); such points are drawn with open markers.
"""

# general
import argparse
import gc
import glob
import json
import math
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# distributed bootstrap and benchmark configurations (no astronomix import yet)
from _sharded_benchmarks import (  # noqa: E402
    DEFAULT_TIME_STEPS,
    SCHEME_LABELS,
    benchmark_configuration,
    benchmark_settings,
    benchmark_state_builder,
    bootstrap_distributed,
    parse_block_shape,
)

DATA_DIR = os.path.join(_HERE, "data", "strong_scaling")
FIGURE_DIR = os.path.join(_HERE, "figures")
PAPER_FIGURE_PATH = (
    "/e/project1/astronomix/storcks1/astronomix_paper/figures/strong_speedup_hydro_mhd.svg"
)

#: Default result-file tag of a measurement per scheme (FD with WENO-Z
#: weights, FV with VL2).
DEFAULT_TAGS = {"fd": "gh200z", "fv": "gh200vl2"}

#: Result-file tag of every configuration in the figure.
CAMPAIGN_TAGS = {
    ("hydro", "fd"): "gh200zpair",
    ("hydro", "fv"): "gh200vl2pair",
    ("mhd", "fd"): "gh200zctpair",
    ("mhd", "fv"): "gh200vl2pair",
}

#: Alternative single-GPU configurations whose 1-GPU points also count for the
#: baseline (the fastest one is used): FD MHD runs its multi-GPU points with the
#: Pallas constrained-transport kernels, which save inter-device communication
#: but are slower on one GPU than the default constrained transport.
SINGLE_GPU_ALTERNATIVE_TAGS = {
    ("mhd", "fd"): ("gh200znoctpair",),
}

#: Resolution above which every point runs the minimum number of steps; below
#: it the step count grows as N^-3 (equal work per point).
EQUAL_WORK_RESOLUTION = 512

#: Number of largest measured 1-GPU baselines the power-law extrapolation is
#: fitted to (the small grids do not follow the asymptotic N^3 cost).
EXTRAPOLATION_FIT_POINTS = 3

#: The colors of the schemes in all figures of the paper (``paper_style``):
#: the scheme is the identity of a line.
SCHEME_COLORS = {"fd": "tab:blue", "fv": "tab:orange"}

#: Marker and line style per GPU count (the second encoding of a line).
GPU_STYLES = {
    2: dict(marker="v", linestyle=":"),
    4: dict(marker="^", linestyle="-."),
    8: dict(marker="o", linestyle="-"),
    16: dict(marker="s", linestyle="--"),
}

PHYSICS_TITLES = {"hydro": "hydrodynamics (3D sound wave)", "mhd": "MHD (3D CP Alfvén wave)"}


# -------------------------------------------------------------
# ===================== ↓ Measurement ↓ =======================
# -------------------------------------------------------------


def number_of_steps(resolution: int, minimum_steps: int) -> int:
    """
    The fixed step count of one strong-scaling point.

    Args:
        resolution: N of the (2N, N, N) grid.
        minimum_steps: The step count at and above ``EQUAL_WORK_RESOLUTION``.

    Returns:
        ``minimum_steps * (EQUAL_WORK_RESOLUTION / N)^3``, at least
        ``minimum_steps``.
    """
    equal_work_steps = minimum_steps * (EQUAL_WORK_RESOLUTION / resolution) ** 3
    return max(minimum_steps, int(math.ceil(equal_work_steps)))


def measure(args) -> None:
    """
    Run the strong-scaling points of one (scheme, physics) on all GPUs of the
    job, one resolution after the other.

    Args:
        args: The parsed command line.
    """
    # jax.distributed.initialize() must run before astronomix is imported.
    bootstrap_distributed(single_process_gpus=args.gpus)

    # jax
    import jax

    # astronomix functions
    from astronomix.parallel.distributed import init_distributed

    pytests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "pytests")
    if pytests_dir not in sys.path:
        sys.path.insert(0, pytests_dir)
    from _benchmark_utils import run_weak_scaling_point

    info = init_distributed()
    num_gpus = info.global_device_count
    tag = args.tag or DEFAULT_TAGS[args.scheme]
    block_shape = parse_block_shape(args.block_shape, args.scheme)
    base_config = benchmark_configuration(
        args.scheme,
        args.physics,
        block_shape,
        pallas_ct=not args.default_ct,
        halo_exchange=args.halo_exchange,
    )
    state_builder = benchmark_state_builder(args.physics)

    for resolution in sorted(args.resolutions):
        global_cells = (2 * resolution, resolution, resolution)
        if global_cells[0] % num_gpus != 0:
            raise ValueError(f"2N = {global_cells[0]} does not split over {num_gpus} GPUs")
        # Box numerically equal to the grid -> h = 1, so one fixed time step
        # is stable at every resolution.
        box_size = tuple(float(extent) for extent in global_cells)
        steps = number_of_steps(resolution, args.min_steps)
        if info.process_index == 0:
            print(
                f"[strong] {SCHEME_LABELS[args.scheme]} {args.physics} N={resolution} "
                f"grid={global_cells} G={num_gpus} steps={steps} block={block_shape}",
                flush=True,
            )
        try:
            run_weak_scaling_point(
                state_builder,
                base_config,
                benchmark_settings(args.physics, box_size),
                mesh_shape=(1, num_gpus, 1, 1),
                global_cells=global_cells,
                box_size=box_size,
                cfl=1.5,
                dt=DEFAULT_TIME_STEPS[args.scheme],
                num_timesteps=steps,
                name=f"strong_{args.physics}_pallas_{tag}_N{resolution:04d}",
                data_dir=DATA_DIR,
                setup_key=f"{args.physics}_strong",
                solver_label=SCHEME_LABELS[args.scheme],
            )
        except Exception as error:  # noqa: BLE001
            # Only a single-GPU baseline may run out of memory (the larger
            # grids of the sweep then do not fit either); a failure of a
            # multi-process point would leave the other ranks waiting, so it
            # is re-raised.
            if num_gpus > 1:
                raise
            print(f"[strong] N={resolution} does not fit one GPU: {error!r}", flush=True)
            break
        finally:
            jax.clear_caches()
            gc.collect()


# -------------------------------------------------------------
# ===================== ↑ Measurement ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Analysis ↓ ========================
# -------------------------------------------------------------


def load_step_times(physics: str, tag: str):
    """
    Read the time per step of every stored point of one configuration.

    Args:
        physics: ``"hydro"`` or ``"mhd"``.
        tag: The result-file tag of the scheme.

    Returns:
        A dict ``{G: {N: (seconds per step, runtime, iterations)}}``.
    """
    # numerics
    import numpy as np

    step_times = {}
    pattern = os.path.join(DATA_DIR, f"strong_{physics}_pallas_{tag}_N*_G*.npz")
    for path in sorted(glob.glob(pattern)):
        record = np.load(path, allow_pickle=True)
        num_gpus = int(record["G"])
        resolution = int(record["global_cells"][1])
        runtime = float(record["runtime"])
        iterations = int(record["iterations"])
        step_times.setdefault(num_gpus, {})[resolution] = (runtime / iterations, runtime, iterations)
    return step_times


def single_gpu_baseline(step_times: dict, resolutions):
    """
    The 1-GPU time per step at the given resolutions, measured where available
    and otherwise extrapolated with a power law fitted to the largest measured
    baselines.

    Args:
        step_times: The output of ``load_step_times``.
        resolutions: The resolutions that need a baseline.

    Returns:
        ``(baseline, extrapolated, exponent)``: dicts of the per-step time and
        of whether it is extrapolated, and the fitted power-law exponent (None
        when nothing had to be extrapolated).
    """
    # numerics
    import numpy as np

    measured = step_times.get(1, {})
    baseline = {resolution: measured[resolution][0] for resolution in measured}
    extrapolated = {resolution: False for resolution in measured}
    missing = [resolution for resolution in resolutions if resolution not in measured]
    exponent = None
    if missing and len(measured) >= 2:
        fit_resolutions = sorted(measured)[-EXTRAPOLATION_FIT_POINTS:]
        exponent, log_prefactor = np.polyfit(
            np.log(fit_resolutions),
            np.log([measured[resolution][0] for resolution in fit_resolutions]),
            1,
        )
        for resolution in missing:
            baseline[resolution] = float(np.exp(log_prefactor + exponent * np.log(resolution)))
            extrapolated[resolution] = True
    return baseline, extrapolated, exponent


def speedups(physics: str, tag: str, baseline_tags=()):
    """
    The strong-scaling speedups of one configuration.

    Args:
        physics: ``"hydro"`` or ``"mhd"``.
        tag: The result-file tag of the scheme.
        baseline_tags: Further tags whose 1-GPU points are alternative
            single-GPU configurations of the same scheme; the baseline is the
            fastest measured one at every N.

    Returns:
        ``(curves, exponent)`` with ``curves = {G: [(N, speedup,
        extrapolated), ...]}`` for every G > 1, sorted by N, and the exponent
        of the baseline extrapolation (or None).
    """
    step_times = load_step_times(physics, tag)
    single_gpu_times = dict(step_times.get(1, {}))
    for baseline_tag in baseline_tags:
        for resolution, entry in load_step_times(physics, baseline_tag).get(1, {}).items():
            if resolution not in single_gpu_times or entry[0] < single_gpu_times[resolution][0]:
                single_gpu_times[resolution] = entry
    step_times[1] = single_gpu_times
    all_resolutions = sorted({resolution for per_gpu in step_times.values() for resolution in per_gpu})
    baseline, extrapolated, exponent = single_gpu_baseline(step_times, all_resolutions)
    curves = {}
    for num_gpus, per_resolution in sorted(step_times.items()):
        if num_gpus == 1:
            continue
        curves[num_gpus] = [
            (
                resolution,
                baseline[resolution] / per_resolution[resolution][0],
                extrapolated[resolution],
            )
            for resolution in sorted(per_resolution)
            if resolution in baseline
        ]
    return curves, exponent


# -------------------------------------------------------------
# ======================= ↑ Analysis ↑ ========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================== ↓ Figure ↓ =========================
# -------------------------------------------------------------


def plot(args) -> None:
    """
    Draw the speedup figure (hydro | MHD, FD and FV lines per GPU count),
    print the speedup table and write it to
    ``data/strong_scaling/strong_speedup_summary.json``.

    Args:
        args: The parsed command line.
    """
    # plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # astronomix constants
    from astronomix.plotting_helpers.paper_style import (
        COLUMN_WIDTH,
        GUIDE_COLOR,
        TEXT_WIDTH,
        TWO_COLUMN,
        layout_path,
    )

    # astronomix functions
    from astronomix.plotting_helpers.paper_style import (
        figure_legend,
        use_paper_style,
    )

    shown_gpu_counts = sorted(args.plot_gpus)

    use_paper_style()
    # Side by side in the single-column layout, stacked in one column (sharing
    # N) in the two-column layout.
    if TWO_COLUMN:
        figure, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 6.4), sharex=True, layout="constrained")
    else:
        figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.3), sharey=True, layout="constrained")
    summary = {}
    all_resolutions = set()

    for axis, physics in zip(axes, ("hydro", "mhd")):
        for scheme in ("fd", "fv"):
            curves, exponent = speedups(
                physics,
                CAMPAIGN_TAGS[(physics, scheme)],
                baseline_tags=SINGLE_GPU_ALTERNATIVE_TAGS.get((physics, scheme), ()),
            )
            summary[f"{physics}_{scheme}"] = dict(
                label=SCHEME_LABELS[scheme],
                baseline_extrapolation_exponent=exponent,
                speedup={
                    str(num_gpus): [
                        dict(N=resolution, speedup=value, extrapolated_baseline=flag)
                        for resolution, value, flag in curve
                    ]
                    for num_gpus, curve in curves.items()
                },
            )
            for num_gpus in shown_gpu_counts:
                curve = curves.get(num_gpus, [])
                if not curve:
                    continue
                resolutions = [point[0] for point in curve]
                values = [point[1] for point in curve]
                all_resolutions.update(resolutions)
                style = GPU_STYLES[num_gpus]
                axis.plot(
                    resolutions,
                    values,
                    color=SCHEME_COLORS[scheme],
                    linestyle=style["linestyle"],
                    zorder=2,
                )
                # Filled markers for measured baselines, open markers where the
                # 1-GPU baseline is extrapolated.
                for resolution, value, extrapolated in curve:
                    axis.plot(
                        resolution,
                        value,
                        marker=style["marker"],
                        markersize=5,
                        color=SCHEME_COLORS[scheme],
                        markerfacecolor="white" if extrapolated else SCHEME_COLORS[scheme],
                        linestyle="none",
                        zorder=3,
                    )

        for num_gpus in shown_gpu_counts:
            axis.axhline(num_gpus, color=GUIDE_COLOR, linestyle="--", linewidth=0.8, zorder=1)
        axis.set_title(PHYSICS_TITLES[physics])
        if not TWO_COLUMN or physics == "mhd":
            axis.set_xlabel(r"$N$ (grid $2N \times N \times N$)")
        axis.set_xscale("log")

    resolution_ticks = sorted(all_resolutions)
    for axis in axes:
        axis.set_xticks(resolution_ticks)
        axis.set_xticklabels([str(resolution) for resolution in resolution_ticks])
        axis.xaxis.set_minor_locator(plt.NullLocator())
        for num_gpus in shown_gpu_counts:
            # At the right end, where every curve lies well below its ideal.
            axis.annotate(
                f"ideal {num_gpus}",
                (resolution_ticks[-1], num_gpus) if resolution_ticks else (1, num_gpus),
                xytext=(0, 2),
                textcoords="offset points",
                ha="right",
                fontsize=7,
                color=GUIDE_COLOR,
            )
    for axis in axes if TWO_COLUMN else axes[:1]:
        axis.set_ylabel(r"strong-scaling speedup $T(1)/T(G)$")
        axis.set_ylim(0, max(shown_gpu_counts) * 1.08)

    legend_handles = [
        Line2D([], [], color=SCHEME_COLORS[scheme], label=SCHEME_LABELS[scheme])
        for scheme in ("fd", "fv")
    ]
    legend_handles += [
        Line2D(
            [],
            [],
            color="#52514e",
            linestyle=GPU_STYLES[num_gpus]["linestyle"],
            marker=GPU_STYLES[num_gpus]["marker"],
            label=f"{num_gpus} GPUs ({max(1, num_gpus // 4)} node{'s' if num_gpus > 4 else ''})",
        )
        for num_gpus in shown_gpu_counts
    ]
    legend_handles.append(
        Line2D(
            [],
            [],
            color="#52514e",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            label="1-GPU baseline extrapolated",
        )
    )
    figure_legend(figure, legend_handles, ncols=1 if TWO_COLUMN else 3)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    figure_stem = layout_path(os.path.join(FIGURE_DIR, "strong_speedup_hydro_mhd"))
    for extension in ("svg", "pdf", "png"):
        figure.savefig(f"{figure_stem}.{extension}")
    plt.close(figure)
    print(f"[strong] wrote {figure_stem}.{{svg,pdf,png}}")
    if args.copy_to_paper:
        shutil.copyfile(f"{figure_stem}.svg", PAPER_FIGURE_PATH)
        print(f"[strong] copied to {PAPER_FIGURE_PATH}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "strong_speedup_summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    for key, entry in summary.items():
        print(f"\n[{key}] {entry['label']} (baseline exponent {entry['baseline_extrapolation_exponent']})")
        for num_gpus, points in entry["speedup"].items():
            text = ", ".join(
                f"N={point['N']}: {point['speedup']:.2f}{'*' if point['extrapolated_baseline'] else ''}"
                for point in points
            )
            print(f"  G={num_gpus}: {text}")


# -------------------------------------------------------------
# ======================== ↑ Figure ↑ =========================
# -------------------------------------------------------------


def main():
    """Measure one configuration, or draw the figure from stored points."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="draw the figure from the stored points (no GPU)")
    parser.add_argument("--scheme", choices=["fd", "fv"], default="fd")
    parser.add_argument("--physics", choices=["hydro", "mhd"], default="hydro")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[128, 192, 256, 384, 512])
    parser.add_argument("--min-steps", type=int, default=20,
                        help="time steps at N >= 512 (more below, equal work per point)")
    parser.add_argument("--block-shape", type=str, default="",
                        help="Pallas block shape bx,by,bz (default: the scheme's tuned block)")
    parser.add_argument("--tag", type=str, default="",
                        help="result-file tag (default: gh200z for fd, gh200vl2 for fv)")
    parser.add_argument("--halo-exchange", choices=["ppermute", "pair_all_to_all"], default="pair_all_to_all",
                        help="multi-device halo exchange (BackendConfig.halo_exchange)")
    parser.add_argument("--default-ct", action="store_true",
                        help="FD MHD: the default (roll-based) constrained transport "
                             "instead of the Pallas CT kernels (faster on one GPU)")
    parser.add_argument("--gpus", type=int, default=0,
                        help="SINGLE-process mode: grab this many local GPUs via "
                             "autocvd (the 1-GPU baseline job). Ignored under srun "
                             "multi-process (SLURM_NTASKS>1).")
    parser.add_argument("--plot-gpus", type=int, nargs="+", default=[4, 8, 16],
                        help="GPU counts drawn in the figure")
    parser.add_argument("--copy-to-paper", action="store_true",
                        help="copy the SVG into the paper's figures folder")
    args = parser.parse_args()

    if args.plot:
        plot(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
