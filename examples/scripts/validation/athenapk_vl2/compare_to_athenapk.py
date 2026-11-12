"""
Compare astronomix's VL2 scheme with AthenaPK on the validation cases.

For every case, AthenaPK (double-precision CPU build, ``ATHENAPK_BIN``) is run to
the end time and astronomix is started from AthenaPK's initial state (or both
from the case's own initial condition). The final primitive states are compared
variable by variable (relative L1 and max norms) together with the number of
cycles. As the scale for "agreement to round-off", AthenaPK's GPU build
(``ATHENAPK_GPU_BIN``, optional) is compared with its CPU build in the same way:
the two AthenaPK builds differ only in floating-point details (fused
multiply-adds, library functions), so an astronomix difference of the same size
is indistinguishable from re-running AthenaPK on other hardware.

Usage:
    ATHENAPK_BIN=... [ATHENAPK_GPU_BIN=...] python compare_to_athenapk.py [case ...]
        [--backends native,pallas] [--output results.json]
"""

# general
import argparse
import json
import os
import sys
import time

# ==== GPU selection (skipped when a queue has already assigned one) ====
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
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
from athenapk_runner import final_output, initial_output, run_case  # noqa: E402
from cases import CASES, CASES_BY_NAME, astronomix_setup, initial_primitive_state  # noqa: E402

RUN_DIRECTORY = os.environ.get("VL2_VALIDATION_RUNS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))


def athenapk_layout_to_astronomix(state, case):
    """
    Select astronomix's variables from an AthenaPK state: identical for GLM-MHD,
    while astronomix's 1D/2D hydro layouts drop the inactive velocity components.
    """
    if case.mhd or case.dimensionality == 3:
        return state
    return state[[0, *range(1, case.dimensionality + 1), 4]]


def astronomix_layout_to_athenapk(state, case):
    """The inverse of :func:`athenapk_layout_to_astronomix` (zero inactive velocities)."""
    if case.mhd or case.dimensionality == 3:
        return state
    full = np.zeros((5,) + state.shape[1:])
    full[0] = state[0]
    full[1 : case.dimensionality + 1] = state[1 : case.dimensionality + 1]
    full[4] = state[-1]
    return full


def relative_differences(state, reference):
    """Per-variable relative L1 and max-norm differences of two states."""
    axes = tuple(range(1, reference.ndim))
    scale_l1 = np.maximum(np.sum(np.abs(reference), axis=axes), 1e-300)
    scale_max = np.maximum(np.max(np.abs(reference), axis=axes), 1e-300)
    difference = np.abs(np.asarray(state) - reference)
    return np.sum(difference, axis=axes) / scale_l1, np.max(difference, axis=axes) / scale_max


def run_astronomix(case, initial_state, backend):
    """Run astronomix's VL2 scheme on a case; returns (final primitive state, cycles, seconds)."""
    config, params, registered_variables = astronomix_setup(case, initial_state.shape, backend)
    config = config._replace(
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    start = time.time()
    result = time_integration(initial_state, config, params, registered_variables)
    final_state = np.asarray(result.final_state)
    return final_state, int(result.num_iterations), time.time() - start


def compare_case(case, backends):
    """Run one case with AthenaPK (CPU, optionally GPU) and astronomix; return the metrics."""
    case_directory = os.path.join(RUN_DIRECTORY, case.name)
    own_initial_state = initial_primitive_state(case)

    initial_conserved = None
    if own_initial_state is not None:
        config, params, registered_variables = astronomix_setup(case, own_initial_state.shape, NATIVE_JAX)
        initial_conserved = astronomix_layout_to_athenapk(
            np.asarray(_conserved_from_primitive(own_initial_state, case.gamma, config, registered_variables)),
            case,
        )

    run_case(case, os.path.join(case_directory, "athenapk_cpu"), initial_conserved)
    reference, _, reference_cycles = final_output(os.path.join(case_directory, "athenapk_cpu"), "prim")
    reference = athenapk_layout_to_astronomix(reference, case)

    if own_initial_state is None:
        initial_state = jnp.asarray(
            athenapk_layout_to_astronomix(initial_output(os.path.join(case_directory, "athenapk_cpu"), "prim")[0], case)
        )
    else:
        initial_state = own_initial_state

    metrics = dict(case=case.name, athenapk_cycles=reference_cycles)

    gpu_state = None
    if os.environ.get("ATHENAPK_GPU_BIN"):
        run_case(case, os.path.join(case_directory, "athenapk_gpu"), initial_conserved, gpu=True)
        gpu_state, _, gpu_cycles = final_output(os.path.join(case_directory, "athenapk_gpu"), "prim")
        gpu_state = athenapk_layout_to_astronomix(gpu_state, case)
        l1, maximum = relative_differences(gpu_state, reference)
        metrics["athenapk_gpu"] = dict(cycles=gpu_cycles, relative_l1=l1.tolist(), relative_max=maximum.tolist())

    for name in backends:
        backend = PALLAS if name == "pallas" else NATIVE_JAX
        final_state, cycles, seconds = run_astronomix(case, initial_state, backend)
        l1, maximum = relative_differences(final_state, reference)
        metrics[f"astronomix_{name}"] = dict(
            cycles=cycles,
            relative_l1=l1.tolist(),
            relative_max=maximum.tolist(),
            seconds=seconds,
            finite=bool(np.all(np.isfinite(final_state))),
        )
        if gpu_state is not None:
            l1_gpu, _ = relative_differences(final_state, gpu_state)
            metrics[f"astronomix_{name}"]["relative_l1_vs_athenapk_gpu"] = l1_gpu.tolist()
    return metrics


def markdown_table(all_metrics, backends):
    """A markdown table of the largest per-variable relative L1 differences and the cycle counts."""
    header = "| case | cycles | AthenaPK GPU vs CPU |"
    rule = "|---|---|---|"
    for name in backends:
        header += f" astronomix {name} vs AthenaPK CPU | astronomix {name} vs AthenaPK GPU |"
        rule += "---|---|"
    lines = [header, rule]
    for metrics in all_metrics:
        cycles = {str(metrics["athenapk_cycles"])}
        line = f"| {metrics['case']} |"
        gpu = metrics.get("athenapk_gpu")
        gpu_column = f" {max(gpu['relative_l1']):.1e} |" if gpu else " – |"
        if gpu:
            cycles.add(str(gpu["cycles"]))
        columns = ""
        for name in backends:
            entry = metrics[f"astronomix_{name}"]
            cycles.add(str(entry["cycles"]))
            versus_gpu = entry.get("relative_l1_vs_athenapk_gpu")
            columns += f" {max(entry['relative_l1']):.1e} | {max(versus_gpu):.1e} |" if versus_gpu else f" {max(entry['relative_l1']):.1e} | – |"
        lines.append(line + f" {' / '.join(sorted(cycles))} |" + gpu_column + columns)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="*", help="case names (default: all)")
    parser.add_argument("--backends", default="native", help="comma-separated: native,pallas")
    parser.add_argument("--output", default=None, help="write the metrics as JSON")
    arguments = parser.parse_args()

    selected = [CASES_BY_NAME[name] for name in arguments.cases] if arguments.cases else CASES
    backends = arguments.backends.split(",")
    all_metrics = []
    for case in selected:
        metrics = compare_case(case, backends)
        all_metrics.append(metrics)
        line = f"{case.name:28s} cycles AthenaPK {metrics['athenapk_cycles']:5d}"
        for key in ["athenapk_gpu"] + [f"astronomix_{name}" for name in backends]:
            if key in metrics:
                entry = metrics[key]
                line += (
                    f" | {key}: cycles {entry['cycles']:5d}, max rel L1 {max(entry['relative_l1']):.1e},"
                    f" max rel Linf {max(entry['relative_max']):.1e}"
                )
                if "relative_l1_vs_athenapk_gpu" in entry:
                    line += f" (vs AthenaPK GPU: {max(entry['relative_l1_vs_athenapk_gpu']):.1e})"
        print(line, flush=True)

    if arguments.output:
        with open(arguments.output, "w") as file:
            json.dump(all_metrics, file, indent=1)
        with open(os.path.splitext(arguments.output)[0] + ".md", "w") as file:
            file.write(markdown_table(all_metrics, backends) + "\n")


if __name__ == "__main__":
    main()
