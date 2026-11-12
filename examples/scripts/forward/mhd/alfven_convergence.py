"""
3D circularly polarized Alfvén wave: L1 convergence and single-GPU runtime of
astronomix (Pallas backend) against AthenaPK -- the paper's convergence figure
and single-GPU performance figure.

The wave (grid 2N x N x N, box 3 x 1.5 x 1.5, five periods to t = 5) is run for
N = 8 ... 128 with

* FV (Pallas): AthenaPK's scheme, VL2 + PLM + HLLD + GLM, C_cfl = 0.3,
* FD (Pallas): WENO-Z, SSPRK4, C_cfl = 1.5 (IEEE square root),

* FD in the setup of the supersonic turbulence (Pallas): WENO-Z with
  positivity-preserving WENO and the upwind dissipation of the CT edge EMFs
  (``ct_upwind_dissipation``, strength 0.03), C_cfl = 0.75; drawn in the
  convergence figure, it must keep the fifth order,

and three diagnostics that are not drawn: FD with positivity-preserving WENO at
C_cfl = 0.75 (the CFL bound of the positivity proof; it must keep the fifth
order), FD at C_cfl = 0.75 without PP (separates the effect of the CFL number
from that of PP) and FD with the approximate reciprocal square root
(``BackendConfig.use_approximate_rsqrt``).

Timing: every run is executed once cold (compilation plus first run) and then
repeated warm until at least three repetitions and one second of wall time are
reached (one repetition for runs longer than 10 s). The runtime is the fastest
warm repetition (the timer inside ``time_integration`` starts after the
ahead-of-time compilation); the compile time is stored separately as the cold
wall time minus that runtime. The L1 error is the mean over rho, v, p and B of
the mean absolute difference to the analytic state, which is evaluated with the
same discrete-curl magnetic field as the initial condition; AthenaPK's own
definition (the conserved variables against the point values of the analytic
wave, ``cpaw.cpp``) is stored alongside.

Results are stored per GPU generation (the node tag comes from the live device):

    pytests/mhd/data/astronomix/<node>/alfven_wave3D_<dp|sp>_convergence_<benchmark>.npz
    pytests/mhd/figures/<node>/alfven_wave3D_<dp|sp>_{convergence,runtime}.{svg,pdf,png}

with the AthenaPK references of the same node
(``pytests/mhd/data/athenapk/<node>/``, written by
``examples/scripts/validation/athenapk_vl2/athenapk_alfven_reference.py``)
overlaid. Usage:

    PYTHONPATH=<repo> python alfven_convergence.py [--single] [--resolutions 8 16 32 64 128]
        [--benchmarks fv_vl2 fd_wenoz ...]
    PYTHONPATH=<repo> python alfven_convergence.py --replot [--node gh200] [--single]

``--single`` runs in single precision (``jax_enable_x64`` off; a separate
process, since the flag is global). ``--replot`` redraws the figures on the CPU
from the stored data.
"""

# general
# NOTE: os/sys must come first: --replot decides whether a GPU is taken via
# autocvd or the CPU is used, and both must happen before jax is imported.
import argparse
import os
import sys
import time
from pathlib import Path

PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
PARSER.add_argument("--replot", action="store_true", help="redraw the figures from stored data (CPU)")
PARSER.add_argument("--single", action="store_true", help="single precision")
PARSER.add_argument("--node", default=None, help="node tag to replot (default: every node on disk)")
PARSER.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32, 64, 128])
PARSER.add_argument("--benchmarks", nargs="+", default=None, help="benchmark keys to run (default: all)")
ARGUMENTS = PARSER.parse_args()

# ==== GPU selection ====
if ARGUMENTS.replot:
    # Replotting touches no jax ops; keep jax off the GPUs entirely.
    os.environ["JAX_PLATFORMS"] = "cpu"
elif os.environ.get("CUDA_VISIBLE_DEVICES"):
    # A parent (a Slurm runner or generate_alfven_data.py) already picked the
    # GPU via autocvd; inherit that choice so every series is measured on it.
    pass
else:
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# jax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", not ARGUMENTS.single)

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    HLLD,
    PALLAS,
    VL2,
)
from astronomix.plotting_helpers.paper_style import (
    ATHENAPK_COLOR,
    ATHENAPK_THIRD_ORDER_COLOR,
    FD_COLOR,
    FV_COLOR,
    GUIDE_COLOR,
    COLUMN_WIDTH,
    SINGLE_PANEL_WIDTH,
    TEXT_WIDTH,
    TWO_COLUMN,
    layout_path,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    FDSolverConfig,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)
from astronomix.test_setups.mhd.alfven_wave3D import CPAlfvenWave3DSettings

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.mhd.alfven_wave3D import (
    _wave_primitive_state,
    cp_alfven_wave_solution,
    setup_cp_alfven_wave,
)
from astronomix.plotting_helpers.paper_style import (
    figure_legend,
    use_paper_style,
    variant_style,
)

# The shared benchmark helpers live under pytests/; put that directory on the
# path so this script uses the same grid, naming and error conventions.
HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY / "pytests"))

# benchmark helpers
from _benchmark_utils import (  # noqa: E402
    BenchmarkSpec,
    _mean_l1_error,
    grid_shape_3d,
    node_tag,
)

# Everything is stored per GPU generation: runtimes measured on different
# hardware must never share a file (or a figure).
DATA_ROOT = REPOSITORY / "pytests" / "mhd" / "data" / "astronomix"
FIGURE_ROOT = REPOSITORY / "pytests" / "mhd" / "figures"
ATHENAPK_ROOT = REPOSITORY / "pytests" / "mhd" / "data" / "athenapk"

PRECISION_TAG = "sp" if ARGUMENTS.single else "dp"
PRECISION_NAME = "single precision" if ARGUMENTS.single else "double precision"
NAME = f"alfven_wave3D_{PRECISION_TAG}"
TITLE = f"3D CP Alfvén wave ({PRECISION_NAME})"

# Robust timing of the warm repetitions (see _benchmark_utils._run_simulation:
# a sporadic ~0.1 s one-off cost hits roughly one run in three, so the fastest
# of several repetitions is reported; runs above 10 s are measured once).
MINIMUM_REPETITIONS = 3
TARGET_TIMED_SECONDS = 1.0
SINGLE_REPETITION_SECONDS = 10.0
MAXIMUM_REPETITIONS = 25

# Options shared by every configuration.
COMMON_OPTIONS = dict(
    mhd=True,
    dimensionality=3,
    progress_bar=False,
    memory_analysis=True,
    print_elapsed_time=True,
    return_snapshots=True,
    snapshot_settings=SnapshotSettings(return_final_state=True),
)

FV_LABEL = "FV (Pallas)"
FD_LABEL = "FD (Pallas)"
FD_PP_LABEL = "FD + PP (Pallas)"
FD_TURBULENCE_LABEL = "FD + PP + upwind EMF (Pallas)"
FD_CFL075_LABEL = "FD (Pallas), C_cfl 0.75"
FD_RSQRT_LABEL = "FD (Pallas), approx. rsqrt"

# The configurations, keyed by the name of their data files. The keys name the
# scheme (not just the backend), so the files never mix with the earlier sweeps
# (FV RK2, FD WENO-JS) stored for other nodes.
BENCHMARKS = dict(
    fv_vl2=BenchmarkSpec(
        label=FV_LABEL,
        base_config=SimulationConfig(
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLD),
            backend_config=BackendConfig(backend=PALLAS),
            **COMMON_OPTIONS,
        ),
        cfl=0.3,
    ),
    fd_wenoz=BenchmarkSpec(
        label=FD_LABEL,
        base_config=SimulationConfig(
            solver_config=FDSolverConfig(weno_z=True),
            backend_config=BackendConfig(backend=PALLAS),
            **COMMON_OPTIONS,
        ),
        cfl=1.5,
    ),
    fd_wenoz_pp=BenchmarkSpec(
        label=FD_PP_LABEL,
        base_config=SimulationConfig(
            solver_config=FDSolverConfig(weno_z=True, weno_positivity_preserving=True),
            backend_config=BackendConfig(backend=PALLAS),
            **COMMON_OPTIONS,
        ),
        cfl=0.75,
    ),
    fd_wenoz_pp_uct=BenchmarkSpec(
        label=FD_TURBULENCE_LABEL,
        base_config=SimulationConfig(
            solver_config=FDSolverConfig(
                weno_z=True,
                weno_positivity_preserving=True,
                ct_upwind_dissipation=True,
            ),
            backend_config=BackendConfig(backend=PALLAS),
            **COMMON_OPTIONS,
        ),
        cfl=0.75,
    ),
    fd_wenoz_cfl075=BenchmarkSpec(
        label=FD_CFL075_LABEL,
        base_config=SimulationConfig(
            solver_config=FDSolverConfig(weno_z=True),
            backend_config=BackendConfig(backend=PALLAS),
            **COMMON_OPTIONS,
        ),
        cfl=0.75,
    ),
    fd_wenoz_rsqrt=BenchmarkSpec(
        label=FD_RSQRT_LABEL,
        base_config=SimulationConfig(
            solver_config=FDSolverConfig(weno_z=True),
            backend_config=BackendConfig(backend=PALLAS, use_approximate_rsqrt=True),
            **COMMON_OPTIONS,
        ),
        cfl=1.5,
    ),
)

# The curves of the two paper figures; the remaining benchmarks are diagnostics.
CONVERGENCE_FIGURE_KEYS = ("fv_vl2", "fd_wenoz", "fd_wenoz_pp_uct")
RUNTIME_FIGURE_KEYS = ("fv_vl2", "fd_wenoz")

# The colours of the schemes in all figures of the paper; the options of the
# finite-difference scheme are drawn as variants of it.
COLORS = {
    FV_LABEL: FV_COLOR,
    FD_LABEL: FD_COLOR,
    FD_PP_LABEL: FD_COLOR,
    FD_TURBULENCE_LABEL: FD_COLOR,
    "AthenaPK (VL2+PLM)": ATHENAPK_COLOR,
    "AthenaPK (RK3+PPM)": ATHENAPK_THIRD_ORDER_COLOR,
}
VARIANT_LABELS = (FD_PP_LABEL, FD_TURBULENCE_LABEL)


def error_indices(registered_variables):
    """State-array indices entered into the L1 error norm (rho, v, p, B)."""
    return (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
        registered_variables.magnetic_index.x,
        registered_variables.magnetic_index.y,
        registered_variables.magnetic_index.z,
    )


def athenapk_l1_error(final_state, params, registered_variables, helper_data) -> float:
    """
    The L1 error as AthenaPK's ``cpaw.cpp`` defines it: the mean absolute
    difference of the eight conserved variables (rho, rho v, E, B) to the point
    values of the analytic wave, averaged over the variables.

    Args:
        final_state: The final primitive state.
        params: The simulation parameters (``t_end``, ``gamma``).
        registered_variables: The registered variables.
        helper_data: The helper data (cell centres).

    Returns:
        The L1 error.
    """
    centers = helper_data.geometric_centers
    analytic_primitives = _wave_primitive_state(
        centers[..., 0],
        centers[..., 1],
        centers[..., 2],
        params.t_end,
        CPAlfvenWave3DSettings(),
    )
    final_primitives = tuple(final_state[index] for index in error_indices(registered_variables))

    def conserved(density, velocity_x, velocity_y, velocity_z, pressure, field_x, field_y, field_z):
        kinetic_energy = 0.5 * density * (velocity_x**2 + velocity_y**2 + velocity_z**2)
        magnetic_energy = 0.5 * (field_x**2 + field_y**2 + field_z**2)
        total_energy = pressure / (params.gamma - 1.0) + kinetic_energy + magnetic_energy
        return (
            density,
            density * velocity_x,
            density * velocity_y,
            density * velocity_z,
            total_energy,
            field_x,
            field_y,
            field_z,
        )

    differences = [
        jnp.mean(jnp.abs(final - analytic))
        for final, analytic in zip(conserved(*final_primitives), conserved(*analytic_primitives))
    ]
    return float(jnp.mean(jnp.stack(differences)))


def run_resolution(spec: BenchmarkSpec, resolution: int) -> dict:
    """
    Run one configuration at one resolution to t = 5: once cold (compilation
    and first run), then warm repetitions for the runtime.

    Args:
        spec: The benchmark configuration.
        resolution: N (grid 2N x N x N).

    Returns:
        A dict with the L1 errors (astronomix and AthenaPK definition), the
        runtime (fastest warm repetition), the slowest warm repetition and their
        number, the cold runtime, the compile time, the iteration count and the
        temporary device memory.
    """
    # A fresh compilation per resolution, so the cold run includes the full
    # compile of this configuration.
    jax.clear_caches()
    config = spec.base_config._replace(num_cells=grid_shape_3d(resolution))
    initial_state, config, params = setup_cp_alfven_wave(config, SimulationParams(C_cfl=spec.cfl))
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    def integrate():
        result = time_integration(initial_state, config, params, registered_variables)
        jax.block_until_ready(result.final_state)
        return result

    cold_start = time.perf_counter()
    result = integrate()
    cold_wall_seconds = time.perf_counter() - cold_start
    cold_runtime = float(result.runtime)

    warm_runtimes = []
    while len(warm_runtimes) < MAXIMUM_REPETITIONS:
        warm_runtimes.append(float(integrate().runtime))
        if warm_runtimes[0] >= SINGLE_REPETITION_SECONDS:
            break
        if len(warm_runtimes) >= MINIMUM_REPETITIONS and sum(warm_runtimes) >= TARGET_TIMED_SECONDS:
            break
    runtime = min(warm_runtimes)

    true_state = cp_alfven_wave_solution(config, registered_variables, params, helper_data)
    return dict(
        l1_error=_mean_l1_error(result.final_state, true_state, error_indices(registered_variables)),
        l1_error_athenapk=athenapk_l1_error(result.final_state, params, registered_variables, helper_data),
        runtime=runtime,
        warm_runtime_max=max(warm_runtimes),
        repetitions=len(warm_runtimes),
        cold_runtime=cold_runtime,
        compile_seconds=cold_wall_seconds - runtime,
        iterations=int(result.num_iterations),
        temporary_memory_bytes=int(result.temporary_memory_bytes or -1),
    )


def data_path(node: str, key: str) -> Path:
    """The NPZ of one benchmark on one node."""
    return DATA_ROOT / node / f"{NAME}_convergence_{key}.npz"


def run_sweep(node: str, keys, resolutions) -> None:
    """
    Run every benchmark over the resolutions and store one NPZ per benchmark
    (rewritten after every resolution, so an interrupted sweep keeps its data).

    Args:
        node: The storage tag of this GPU.
        keys: The keys of the benchmarks to run.
        resolutions: The values of N.
    """
    device = jax.devices()[0].device_kind
    for key in keys:
        spec = BENCHMARKS[key]
        records = []
        for resolution in resolutions:
            record = run_resolution(spec, resolution)
            records.append(record)
            print(
                f"[alfven] {spec.label:28s} N={resolution:4d}: L1={record['l1_error']:.6e} "
                f"(AthenaPK def. {record['l1_error_athenapk']:.6e}), iterations={record['iterations']:5d}, "
                f"runtime={record['runtime']:9.4f} s ({1e3 * record['runtime'] / record['iterations']:8.4f} ms/step, "
                f"{record['repetitions']} warm runs, max {record['warm_runtime_max']:.4f} s), "
                f"cold {record['cold_runtime']:.3f} s, compile {record['compile_seconds']:6.1f} s",
                flush=True,
            )
            path = data_path(node, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                N_values=np.array(resolutions[: len(records)]),
                l1_errors=np.array([entry["l1_error"] for entry in records]),
                l1_errors_athenapk=np.array([entry["l1_error_athenapk"] for entry in records]),
                runtimes=np.array([entry["runtime"] for entry in records]),
                warm_runtime_max=np.array([entry["warm_runtime_max"] for entry in records]),
                repetitions=np.array([entry["repetitions"] for entry in records]),
                cold_runtimes=np.array([entry["cold_runtime"] for entry in records]),
                compile_seconds=np.array([entry["compile_seconds"] for entry in records]),
                iterations=np.array([entry["iterations"] for entry in records]),
                temporary_memory_bytes=np.array([entry["temporary_memory_bytes"] for entry in records]),
                cfl=spec.cfl,
                label=spec.label,
                device=device,
            )


def load_results(node: str, keys) -> dict:
    """The stored sweeps of ``keys`` on ``node`` by legend label (missing ones are skipped)."""
    results = {}
    for key in keys:
        path = data_path(node, key)
        if path.exists():
            results[BENCHMARKS[key].label] = dict(np.load(path))
        else:
            print(f"[alfven] no data for {key!r} on {node}: {path}")
    return results


def athenapk_references(node: str) -> dict:
    """
    The AthenaPK references measured on ``node`` in this precision.

    Only same-node references are overlaid: the runtime panels put astronomix
    and AthenaPK on shared axes, so mixing hardware there would invent a speed
    difference that is really a GPU difference.

    The L1 error drawn is the one in astronomix's definition (primitive
    variables against the discrete initial state, ``l1_errors_primitive``),
    so that every curve of the figure uses the same norm; references without it
    fall back to AthenaPK's own error.

    Args:
        node: The storage tag.

    Returns:
        A dict from legend label to the reference arrays, with the plotted error
        in ``l1_errors_plotted``.
    """
    suffix = "_single" if ARGUMENTS.single else ""
    candidates = {
        "AthenaPK (VL2+PLM)": ATHENAPK_ROOT / node / f"cfl03_teams256_singleblock{suffix}",
        "AthenaPK (RK3+PPM)": ATHENAPK_ROOT / node / f"cfl04_teams256_singleblock_rk3_ppm{suffix}",
    }
    references = {}
    for label, directory in candidates.items():
        path = directory / "athenapk_alfven_convergence.npz"
        if not path.exists():
            print(f"[alfven] no AthenaPK reference {label!r} on {node}: {path}")
            continue
        data = dict(np.load(path))
        data["l1_errors_plotted"] = data.get("l1_errors_primitive", data["l1_errors"])
        references[label] = data
    return references


def format_resolution_axis(axis, resolutions) -> None:
    """Log scale with one labelled tick per resolution (as _benchmark_utils._format_xaxis)."""
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(ticker.FixedLocator(resolutions))
    axis.xaxis.set_major_formatter(ticker.FixedFormatter([str(value) for value in resolutions]))
    axis.xaxis.set_minor_locator(ticker.NullLocator())


def save_figure(figure, directory: Path, stem: str) -> None:
    """Write a figure as SVG (paper), PDF and PNG."""
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        figure.savefig(layout_path(directory / f"{stem}.{extension}"))


def plot(node: str) -> None:
    """
    The convergence figure (L1 error against N) and the runtime figure (error
    against runtime, time per step against N) of one node. The AthenaPK
    curves are dashed with large open markers so that the coinciding FV (VL2)
    curve stays visible underneath. The total runtime against N is not drawn:
    it is the time per step times the number of steps, which the larger
    admissible CFL number of the finite-difference scheme reduces; the error
    against runtime already accounts for it.

    Args:
        node: The storage tag.
    """
    results = load_results(node, CONVERGENCE_FIGURE_KEYS)
    references = athenapk_references(node)
    if not results:
        raise SystemExit(f"no data for node {node!r}")
    resolutions = sorted({int(value) for data in results.values() for value in data["N_values"]})

    reference_markers = {"AthenaPK (VL2+PLM)": "s", "AthenaPK (RK3+PPM)": "D"}

    def astronomix_style(label):
        # Every astronomix run uses the Pallas backend, which the paper states
        # once; the legend leaves the backend out.
        shown = label.replace(" (Pallas)", "")
        if label in VARIANT_LABELS:
            return dict(marker="o", label=shown, **variant_style(COLORS[label]))
        return dict(marker="o", color=COLORS[label], label=shown)

    def reference_style(label):
        return dict(
            marker=reference_markers[label],
            markersize=7,
            markerfacecolor="none",
            linestyle="--",
            color=COLORS[label],
            label=label,
        )

    use_paper_style()

    # -------------------------------------------------------------
    # =============== ↓ Convergence figure ↓ ======================
    # -------------------------------------------------------------
    figure_error, axis_error = plt.subplots(figsize=(SINGLE_PANEL_WIDTH, 4.6 if TWO_COLUMN else 3.9), layout="constrained")
    for label, data in results.items():
        axis_error.loglog(data["N_values"], data["l1_errors"], **astronomix_style(label))
    for label, data in references.items():
        axis_error.loglog(data["N_values"], data["l1_errors_plotted"], **reference_style(label))

    # Guide slopes for second (FV) and fifth (FD) order, anchored at the
    # coarsest-grid errors.
    resolution_array = np.array(resolutions, dtype=float)
    first_errors = [data["l1_errors"][0] for data in results.values()]
    axis_error.loglog(
        resolution_array,
        max(first_errors) * (resolution_array / resolution_array[0]) ** -2.0,
        color=GUIDE_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=r"$\propto N^{-2}$",
    )
    axis_error.loglog(
        resolution_array,
        min(first_errors) * (resolution_array / resolution_array[0]) ** -5.0,
        color=GUIDE_COLOR,
        linestyle=":",
        linewidth=1.0,
        label=r"$\propto N^{-5}$",
    )
    axis_error.set_xlabel(r"$N$ (grid $2N \times N \times N$)")
    axis_error.set_ylabel(r"average $L_1$ error")
    format_resolution_axis(axis_error, resolutions)
    figure_legend(figure_error, ncols=1 if TWO_COLUMN else 2)
    save_figure(figure_error, FIGURE_ROOT / node, f"{NAME}_convergence")
    # -------------------------------------------------------------
    # =============== ↑ Convergence figure ↑ ======================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ Runtime figure ↓ ==========================
    # -------------------------------------------------------------
    # Side by side in the single-column layout, stacked in one column in the
    # two-column layout.
    figure_runtime, (axis_error_runtime, axis_step) = plt.subplots(
        2 if TWO_COLUMN else 1,
        1 if TWO_COLUMN else 2,
        figsize=(COLUMN_WIDTH, 5.0) if TWO_COLUMN else (TEXT_WIDTH, 2.7),
        layout="constrained",
    )
    curves = []
    for label in (BENCHMARKS[key].label for key in RUNTIME_FIGURE_KEYS):
        if label in results:
            curves.append((results[label], results[label]["l1_errors"], astronomix_style(label)))
    for label, data in references.items():
        curves.append((data, data["l1_errors_plotted"], reference_style(label)))
    for data, errors, style in curves:
        milliseconds_per_step = 1e3 * np.asarray(data["runtimes"]) / np.asarray(data["iterations"])
        axis_error_runtime.loglog(data["runtimes"], errors, **style)
        axis_step.loglog(data["N_values"], milliseconds_per_step, **style)

    axis_error_runtime.set_xlabel("runtime (s)")
    axis_error_runtime.set_ylabel(r"average $L_1$ error")
    axis_step.set_xlabel(r"$N$ (grid $2N \times N \times N$)")
    axis_step.set_ylabel("time per step (ms)")
    format_resolution_axis(axis_step, resolutions)
    figure_legend(figure_runtime, ncols=2 if TWO_COLUMN else None)
    save_figure(figure_runtime, FIGURE_ROOT / node, f"{NAME}_runtime")
    # -------------------------------------------------------------
    # =============== ↑ Runtime figure ↑ ==========================
    # -------------------------------------------------------------
    plt.close("all")
    print(f"[alfven] wrote {FIGURE_ROOT / node}/{NAME}_{{convergence,runtime}}.{{svg,pdf,png}}")


def nodes_on_disk() -> list:
    """Node tags with at least one sweep of the figure benchmarks in this precision."""
    if not DATA_ROOT.exists():
        return []
    return sorted(
        directory.name
        for directory in DATA_ROOT.iterdir()
        if directory.is_dir()
        and any(data_path(directory.name, key).exists() for key in CONVERGENCE_FIGURE_KEYS)
    )


if __name__ == "__main__":
    if ARGUMENTS.replot:
        for node in [ARGUMENTS.node] if ARGUMENTS.node else nodes_on_disk():
            plot(node)
    else:
        node = node_tag()
        selected = list(BENCHMARKS) if ARGUMENTS.benchmarks is None else ARGUMENTS.benchmarks
        print(
            f"[alfven] {PRECISION_NAME} on {jax.devices()[0].device_kind} -> node {node!r}; "
            f"benchmarks {selected}; N = {ARGUMENTS.resolutions}",
            flush=True,
        )
        run_sweep(node, selected, ARGUMENTS.resolutions)
        plot(node)
