"""
CP Alfvén wave convergence and time to solution: astronomix VL2 against AthenaPK.

For every resolution N (grid 2N x N x N), AthenaPK's GPU build
(``ATHENAPK_GPU_BIN``) runs the circularly polarized Alfvén wave for five
periods with Grete's tuned settings (single meshblock, VL2 + PLM + HLLD, CFL
0.3) and reports its L1 error and wall time. astronomix then runs the same
problem from AthenaPK's initial state on the same GPU (Pallas backend, double
and single precision), and its L1 error is evaluated with AthenaPK's formula
(``cpaw.cpp``: the mean absolute deviation of the eight conserved variables from
the analytic wave, averaged over the variables). The astronomix wall time is
that of a second run, i.e. without compilation.

Usage:
    ATHENAPK_GPU_BIN=... python alfven_convergence.py --resolutions 8 16 32 64 128
        [--output results.json] [--figure convergence.png]
"""

# general
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

# ==== GPU selection (skipped when a queue has already assigned one) ====
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =====================================================================

# numerics
import numpy as np

# jax
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import NATIVE_JAX, PALLAS

# astronomix containers
from astronomix import SnapshotSettings

# astronomix functions
from astronomix import time_integration
from astronomix._finite_volume._stage import _conserved_from_primitive

# validation helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from athenapk_runner import (  # noqa: E402
    athenapk_binary,
    athenapk_environment,
    cp_alfven_input,
    initial_output,
    performance_summary,
)
from cases import CASES_BY_NAME, astronomix_setup  # noqa: E402

BOX = (3.0, 1.5, 1.5)
END_TIME = 5.0


def analytic_conserved_state(resolution: int):
    """
    The analytic circularly polarized Alfvén wave in conserved variables at the
    cell centres, exactly as AthenaPK's ``cpaw.cpp`` evaluates it for the error
    (density 1, pressure 0.1, B_par 1, B_perp 0.1, right polarization, gamma 5/3).
    """
    density, pressure, field_parallel, field_perpendicular = 1.0, 0.1, 1.0, 0.1
    gamma_minus_one = 1.666666666666667 - 1.0
    angle_3 = math.atan(BOX[0] / BOX[1])
    angle_2 = math.atan(0.5 * (BOX[0] * math.cos(angle_3) + BOX[1] * math.sin(angle_3)) / BOX[2])
    wavelength = min(
        BOX[0] * math.cos(angle_2) * math.cos(angle_3),
        BOX[1] * math.cos(angle_2) * math.sin(angle_3),
        BOX[2] * math.sin(angle_2),
    )
    wavenumber = 2.0 * math.pi / wavelength
    velocity_perpendicular = field_perpendicular / math.sqrt(density)

    num_cells = (2 * resolution, resolution, resolution)
    centers = [(np.arange(num_cells[axis]) + 0.5) * BOX[axis] / num_cells[axis] for axis in range(3)]
    x1, x2, x3 = np.meshgrid(*centers, indexing="ij")
    along_wave = math.cos(angle_2) * (x1 * math.cos(angle_3) + x2 * math.sin(angle_3)) + x3 * math.sin(angle_2)
    sine, cosine = np.sin(wavenumber * along_wave), np.cos(wavenumber * along_wave)

    def rotate(parallel, perpendicular_1, perpendicular_2):
        return (
            parallel * math.cos(angle_2) * math.cos(angle_3)
            - perpendicular_1 * math.sin(angle_3)
            - perpendicular_2 * math.sin(angle_2) * math.cos(angle_3),
            parallel * math.cos(angle_2) * math.sin(angle_3)
            + perpendicular_1 * math.cos(angle_3)
            - perpendicular_2 * math.sin(angle_2) * math.sin(angle_3),
            parallel * math.sin(angle_2) + perpendicular_2 * math.cos(angle_2),
        )

    momentum = rotate(0.0, -density * velocity_perpendicular * sine, -density * velocity_perpendicular * cosine)
    field = rotate(field_parallel, field_perpendicular * sine, field_perpendicular * cosine)
    energy = (
        pressure / gamma_minus_one
        + 0.5 * sum(component**2 for component in momentum) / density
        + 0.5 * sum(component**2 for component in field)
    )
    return np.stack([np.full_like(x1, density), *momentum, energy, *field])


def athenapk_l1_error(conserved_state, resolution: int) -> float:
    """AthenaPK's CP Alfvén L1 error: mean |U - U_exact| per variable, averaged over the eight variables."""
    analytic = analytic_conserved_state(resolution)
    return float(np.mean(np.abs(np.asarray(conserved_state)[:8] - analytic)))


def run_athenapk(resolution: int, run_directory: str):
    """Run AthenaPK to t = 5; returns (L1 error, cycles, wall seconds) and leaves the t=0 output."""
    input_file = os.path.join(run_directory, "cpaw.in")
    with open(input_file, "w") as file:
        file.write(cp_alfven_input(resolution, compute_error=True, output_variables="prim"))
    result = subprocess.run(
        [athenapk_binary(gpu=True), "-i", input_file],
        cwd=run_directory,
        capture_output=True,
        text=True,
        env=athenapk_environment(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"AthenaPK failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    zone_cycles_per_second, _, _ = performance_summary(result.stdout)
    with open(os.path.join(run_directory, "cpaw-errors.dat")) as file:
        columns = [line.split() for line in file if not line.startswith("#")][-1]
    cycles = int(columns[3])
    l1_error = float(np.mean([float(value) for value in columns[5:13]]))
    seconds = 2 * resolution**3 * cycles / zone_cycles_per_second
    return l1_error, cycles, seconds


def run_astronomix(initial_state, backend: int, dtype):
    """Run astronomix to t = 5; returns (final conserved state, cycles, wall seconds of a second run)."""
    case = CASES_BY_NAME["cp_alfven_3d"]
    config, params, registered_variables = astronomix_setup(case, initial_state.shape, backend)
    config = config._replace(
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    params = params._replace(t_end=END_TIME)
    state = initial_state.astype(dtype)
    jax.block_until_ready(time_integration(state, config, params, registered_variables).final_state)
    start = time.perf_counter()
    result = time_integration(state, config, params, registered_variables)
    final_state = jax.block_until_ready(result.final_state)
    seconds = time.perf_counter() - start
    conserved = _conserved_from_primitive(final_state.astype(jnp.float64), case.gamma, config, registered_variables)
    return conserved, int(result.num_iterations), seconds


def plot(results, path):
    """Error against resolution and against wall time for every configuration."""
    import matplotlib.pyplot as plt

    figure, (axis_error, axis_time) = plt.subplots(1, 2, figsize=(10, 4))
    for label in sorted({entry["label"] for entry in results}):
        entries = sorted((entry for entry in results if entry["label"] == label), key=lambda entry: entry["resolution"])
        resolutions = [entry["resolution"] for entry in entries]
        errors = [entry["l1_error"] for entry in entries]
        seconds = [entry["seconds"] for entry in entries]
        if "AthenaPK" in label:
            # hollow and larger, so that the coinciding astronomix points stay visible
            style = dict(marker="s", markersize=10, markerfacecolor="none", linestyle="--", color="black", label=label)
        else:
            style = dict(marker="o", markersize=5, linestyle="-", label=label)
        axis_error.loglog(resolutions, errors, **style)
        axis_time.loglog(seconds, errors, **style)
    resolutions = np.array(sorted({entry["resolution"] for entry in results}), dtype=float)
    axis_error.loglog(resolutions, 2e-2 * (resolutions / resolutions[0]) ** -2, color="gray", linewidth=0.8, label=r"$N^{-2}$")
    axis_error.set_xlabel("N (grid 2N x N x N)")
    axis_error.set_ylabel("L1 error after five periods")
    axis_time.set_xlabel("wall time [s]")
    axis_time.set_ylabel("L1 error after five periods")
    for axis in (axis_error, axis_time):
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"3D CP Alfvén wave, VL2 + PLM + HLLD + GLM ({results[0]['device']})")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def replot(results_path: str, figure_path: str):
    """Redraw the figure from a saved results file."""
    with open(results_path) as file:
        plot(json.load(file), figure_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--native", action="store_true", help="also run the native JAX backend (double precision)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--replot", default=None, help="only redraw --figure from this results file")
    arguments = parser.parse_args()

    if arguments.replot:
        replot(arguments.replot, arguments.figure)
        return

    device = jax.devices()[0].device_kind
    results = []
    for resolution in arguments.resolutions:
        with tempfile.TemporaryDirectory() as run_directory:
            l1_error, cycles, seconds = run_athenapk(resolution, run_directory)
            results.append(dict(label="AthenaPK (double)", resolution=resolution, l1_error=l1_error, cycles=cycles, seconds=seconds, device=device))
            print(f"N={resolution:4d} AthenaPK                : L1 {l1_error:.6e}, {cycles} cycles, {seconds:8.2f} s", flush=True)
            initial_state = jnp.asarray(initial_output(run_directory, "prim")[0])

        variants = [("astronomix Pallas (double)", PALLAS, jnp.float64), ("astronomix Pallas (single)", PALLAS, jnp.float32)]
        if arguments.native:
            variants.append(("astronomix native (double)", NATIVE_JAX, jnp.float64))
        for label, backend, dtype in variants:
            conserved, cycles, seconds = run_astronomix(initial_state, backend, dtype)
            l1_error = athenapk_l1_error(conserved, resolution)
            results.append(dict(label=label, resolution=resolution, l1_error=l1_error, cycles=cycles, seconds=seconds, device=device))
            print(f"N={resolution:4d} {label:24s}: L1 {l1_error:.6e}, {cycles} cycles, {seconds:8.2f} s", flush=True)

    if arguments.output:
        with open(arguments.output, "w") as file:
            json.dump(results, file, indent=1)
    if arguments.figure:
        plot(results, arguments.figure)


if __name__ == "__main__":
    main()
