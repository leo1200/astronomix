"""
Figure: throughput of astronomix's finite-volume scheme (VL2 + PLM + HLLD +
GLM, Pallas backend) against AthenaPK on one GPU, in single and in double
precision.

Reads the results of ``benchmark.py`` (one file per precision, both codes
measured on exclusive GH200 nodes; an entry of either code can be refreshed
by a later run of that code alone) and draws the throughput (cell updates per
second; for AthenaPK its own zone-cycles per wall second) against the
resolution N of the CP Alfvén wave (grid 2N x N x N). Prints the table behind
the figure: the time per step, the AthenaPK/astronomix ratios and the
single/double speed-ups of both codes.

Usage:
    python fv_athenapk_single_precision.py [--node gh200]
        [--results-directory results] [--figure-directory <repo>/pytests/mhd/figures/<node>]

expects ``<results-directory>/benchmark_<node>_{single,double}.json`` and writes
``fv_athenapk_single_precision.{svg,pdf,png}``.
"""

# general
import argparse
import json
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# astronomix constants
from astronomix.plotting_helpers.paper_style import (
    ATHENAPK_COLOR,
    FV_COLOR,
    SINGLE_PANEL_WIDTH,
    layout_path,
)

# astronomix functions
from astronomix.plotting_helpers.paper_style import (
    use_paper_style,
    variant_style,
)

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]

ASTRONOMIX_CODE = "astronomix pallas"
ATHENAPK_CODE = "AthenaPK"
LABELS = {ASTRONOMIX_CODE: "astronomix FV (VL2, Pallas)", ATHENAPK_CODE: "AthenaPK (VL2+PLM)"}
COLORS = {ASTRONOMIX_CODE: FV_COLOR, ATHENAPK_CODE: ATHENAPK_COLOR}
MARKERS = {ASTRONOMIX_CODE: "o", ATHENAPK_CODE: "s"}


def load_benchmark(path: Path) -> dict:
    """
    The per-step times of one benchmark file.

    Args:
        path: A ``benchmark.py`` results file.

    Returns:
        A dict from code name to ``(resolutions, milliseconds_per_step,
        cell_updates_per_second)`` arrays sorted by resolution, plus ``device``.
    """
    with open(path) as file:
        entries = json.load(file)
    curves = {"device": entries[0]["device"] if entries else "unknown"}
    for code in (ASTRONOMIX_CODE, ATHENAPK_CODE):
        selected = sorted((entry for entry in entries if entry["code"] == code), key=lambda entry: entry["resolution"])
        curves[code] = (
            np.array([entry["resolution"] for entry in selected]),
            np.array([entry["milliseconds_per_step"] for entry in selected]),
            np.array([entry["zone_cycles_per_second"] for entry in selected]),
        )
    return curves


def print_table(single: dict, double: dict) -> None:
    """The numbers behind the figure, with ratios and precision speed-ups."""
    resolutions = sorted(
        set(single[ASTRONOMIX_CODE][0]) | set(single[ATHENAPK_CODE][0])
        | set(double[ASTRONOMIX_CODE][0]) | set(double[ATHENAPK_CODE][0])
    )

    def value(curves, code, resolution, column):
        resolution_values, milliseconds, throughput = curves[code]
        matches = np.nonzero(resolution_values == resolution)[0]
        if len(matches) == 0:
            return float("nan")
        return (milliseconds if column == "ms" else throughput)[matches[0]]

    print(
        f"{'N':>5} {'astronomix sp':>14} {'AthenaPK sp':>12} {'ratio sp':>9} "
        f"{'astronomix dp':>14} {'AthenaPK dp':>12} {'ratio dp':>9} {'dp/sp astronomix':>17} {'dp/sp AthenaPK':>15}"
    )
    for resolution in resolutions:
        astronomix_single = value(single, ASTRONOMIX_CODE, resolution, "ms")
        athenapk_single = value(single, ATHENAPK_CODE, resolution, "ms")
        astronomix_double = value(double, ASTRONOMIX_CODE, resolution, "ms")
        athenapk_double = value(double, ATHENAPK_CODE, resolution, "ms")
        print(
            f"{resolution:5d} {astronomix_single:11.3f} ms {athenapk_single:9.3f} ms {athenapk_single / astronomix_single:8.2f}x "
            f"{astronomix_double:11.3f} ms {athenapk_double:9.3f} ms {athenapk_double / astronomix_double:8.2f}x "
            f"{astronomix_double / astronomix_single:16.2f}x {athenapk_double / athenapk_single:14.2f}x"
        )
    print("throughput [Mcell-updates/s]:")
    for resolution in resolutions:
        print(
            f"{resolution:5d} "
            + " ".join(
                f"{label} {value(curves, code, resolution, 'throughput') / 1e6:8.1f}"
                for label, curves in (("sp", single), ("dp", double))
                for code in (ASTRONOMIX_CODE, ATHENAPK_CODE)
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node", default="gh200")
    parser.add_argument("--results-directory", default=str(HERE / "results"))
    parser.add_argument("--figure-directory", default=None)
    arguments = parser.parse_args()

    results_directory = Path(arguments.results_directory)
    single = load_benchmark(results_directory / f"benchmark_{arguments.node}_single.json")
    double = load_benchmark(results_directory / f"benchmark_{arguments.node}_double.json")
    print_table(single, double)

    # -------------------------------------------------------------
    # ==================== ↓ Throughput figure ↓ ==================
    # -------------------------------------------------------------
    # The figure shows the throughput only: it is the time per step converted
    # to cell updates per second, and unlike the time per step it shows at a
    # glance where each code saturates the GPU. The per-step times and their
    # ratios are printed above.
    use_paper_style()
    figure, axis = plt.subplots(figsize=(SINGLE_PANEL_WIDTH, 2.9), layout="constrained")
    all_resolutions = set()
    for precision_label, curves in (("single", single), ("double", double)):
        for code in (ASTRONOMIX_CODE, ATHENAPK_CODE):
            resolutions, milliseconds, throughput = curves[code]
            if len(resolutions) == 0:
                continue
            all_resolutions |= set(int(value) for value in resolutions)
            if precision_label == "single":
                style = dict(color=COLORS[code])
            else:
                style = variant_style(COLORS[code])
            axis.loglog(
                resolutions,
                throughput / 1e6,
                marker=MARKERS[code],
                label=f"{LABELS[code]}, {precision_label}",
                **style,
            )

    resolutions = sorted(all_resolutions)
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(ticker.FixedLocator(resolutions))
    axis.xaxis.set_major_formatter(ticker.FixedFormatter([str(value) for value in resolutions]))
    axis.xaxis.set_minor_locator(ticker.NullLocator())
    axis.set_xlabel(r"$N$ (grid $2N \times N \times N$)")
    axis.set_ylabel("cell updates per second (millions)")
    axis.legend(loc="lower right")

    figure_directory = Path(arguments.figure_directory or REPOSITORY / "pytests" / "mhd" / "figures" / arguments.node)
    figure_directory.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        figure.savefig(layout_path(figure_directory / f"fv_athenapk_single_precision.{extension}"))
    print(f"wrote {figure_directory}/fv_athenapk_single_precision.{{svg,pdf,png}}")
    # -------------------------------------------------------------
    # ==================== ↑ Throughput figure ↑ ==================
    # -------------------------------------------------------------


if __name__ == "__main__":
    main()
