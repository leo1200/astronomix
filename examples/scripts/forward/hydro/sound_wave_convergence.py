"""3D linear sound-wave convergence — paper figure (``sound_wave3D_convergence.svg``).

Runs the L1-convergence + runtime sweep over N = 8..128 for the pure-hydro
methods-paper configurations against the analytic linear sound-wave solution:

    FV (JAX), FV (Pallas)     the VL2 finite-volume scheme (AthenaPK's
                              van Leer predictor-corrector, PLM with van Leer
                              slopes, HLLC), C_cfl = 0.3 as AthenaPK in 3D
    FD (JAX), FD (Pallas)     the finite-difference scheme with WENO-Z
                              weights, C_cfl = 1.5

and writes

    figures/sound_wave3D_convergence.svg   (average L1 error vs N, paper figure,
                                            in the common style of the paper)
    figures/sound_wave3D_runtime.svg        (error/runtime diagnostics)

The fast correctness version (single low resolution) lives in
``pytests/hydrodynamics/sound_wave3D.py``; this script is the full sweep that
produces the paper figure. It reuses the shared benchmark harness in
``pytests/_benchmark_utils.py``.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/sound_wave_convergence.py

``--plot-only`` redraws the figures from the cached per-benchmark NPZs.
"""

# ==== GPU selection ====
import sys as _sys
if "--plot-only" not in _sys.argv:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import math
import sys
from pathlib import Path

# jax
import jax

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib import ticker

# The tiny perturbation amplitudes of a linear sound wave need double precision
# to stay above float rounding noise.
jax.config.update("jax_enable_x64", True)

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    HLLC,
    NATIVE_JAX,
    PALLAS,
    VL2,
)
from astronomix.plotting_helpers.paper_style import (
    FD_COLOR,
    FV_COLOR,
    GUIDE_COLOR,
    SINGLE_PANEL_WIDTH,
    TWO_COLUMN,
    layout_path,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SnapshotSettings,
    BackendConfig,
    FVSolverConfig,
    FDSolverConfig,
)
from astronomix.option_classes.simulation_config import StaticFloatVector

# astronomix functions
from astronomix.plotting_helpers.paper_style import (
    use_paper_style,
)
from astronomix.test_setups.hydrodynamics.sound_wave3D import (
    setup_sound_wave,
    sound_wave_solution,
)


# The shared benchmark harness lives in pytests/_benchmark_utils.py; put the
# pytests directory on the path so this example can reuse the exact sweep driver
# that the fast correctness test also builds on.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
_PYTESTS_DIR = str(_REPO / "pytests")
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)

from _benchmark_utils import (  # noqa: E402
    BenchmarkSpec,
    _slug,
    plot_convergence_and_runtime,
    run_convergence_and_runtime,
)

DATA_DIR = str(_HERE / "data" / "sound_wave3D")
FIG_DIR = str(_HERE / "figures")

# The resolution sweep for the convergence figure.
N_VALUES = [8, 16, 32, 64, 128]

# Options shared by every benchmark configuration; only the backend, solver
# mode and CFL number differ between them.
_common_kwargs = dict(
    box_size=StaticFloatVector(3.0, 1.5, 1.5),
    mhd=False,
    dimensionality=3,
    progress_bar=False,
    memory_analysis=True,
    print_elapsed_time=True,
    return_snapshots=True,
    snapshot_settings=SnapshotSettings(return_final_state=True),
)

# The finite-volume scheme is AthenaPK's VL2 with its default hydrodynamics
# Riemann solver; C_cfl = 0.3 is the value AthenaPK uses in 3D.
_finite_volume_kwargs = dict(
    solver_mode=FINITE_VOLUME,
    solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLC),
    **_common_kwargs,
)

# The finite-difference scheme with WENO-Z weights. The smooth wave needs no
# positivity preservation; it runs at C_cfl = 1.5, well inside the linear
# stability limit of SSPRK4.
_finite_difference_kwargs = dict(
    solver_config=FDSolverConfig(weno_z=True),
    **_common_kwargs,
)

BENCHMARKS = [
    BenchmarkSpec(
        label="FV (JAX)",
        base_config=SimulationConfig(
            backend_config=BackendConfig(backend=NATIVE_JAX),
            **_finite_volume_kwargs,
        ),
        cfl=0.3,
    ),
    BenchmarkSpec(
        label="FV (Pallas)",
        base_config=SimulationConfig(
            backend_config=BackendConfig(backend=PALLAS),
            **_finite_volume_kwargs,
        ),
        cfl=0.3,
    ),
    BenchmarkSpec(
        label="FD (JAX)",
        base_config=SimulationConfig(
            backend_config=BackendConfig(backend=NATIVE_JAX),
            **_finite_difference_kwargs,
        ),
        cfl=1.5,
    ),
    BenchmarkSpec(
        label="FD (Pallas)",
        base_config=SimulationConfig(
            backend_config=BackendConfig(backend=PALLAS),
            **_finite_difference_kwargs,
        ),
        cfl=1.5,
    ),
]


def _error_indices(registered_variables):
    """State indices whose L1 error the convergence sweep tracks."""
    return (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
    )


def print_convergence_orders(results, N_values):
    """
    Print the measured convergence order of every benchmark between successive
    resolutions, ``log(L1(N) / L1(2N)) / log 2``.

    Args:
        results: The sweep results keyed by benchmark label, as returned by
            ``run_convergence_and_runtime``.
        N_values: The swept resolutions.
    """
    for label, record in results.items():
        errors = record["l1"]
        orders = [
            math.log(errors[index] / errors[index + 1])
            / math.log(N_values[index + 1] / N_values[index])
            for index in range(len(N_values) - 1)
        ]
        formatted_orders = ", ".join(f"{order:.2f}" for order in orders)
        print(
            f"[sound_wave3D] {label}: orders {formatted_orders}; "
            f"L1(N={N_values[-1]}) = {errors[-1]:.3e}"
        )


def plot_paper_figure(results):
    """
    Draw the paper figure: the average L1 error of the Pallas runs of both
    schemes against N, with second- and fifth-order guide lines. The native-JAX
    runs compute the same solution (the paper states the agreement of the
    backends once) and are not drawn.

    Args:
        results: The sweep results keyed by benchmark label.
    """
    use_paper_style()
    figure, axis = plt.subplots(figsize=(SINGLE_PANEL_WIDTH, 2.9), layout="constrained")

    curve_styles = {
        "FD (Pallas)": dict(color=FD_COLOR, marker="o", label="FD (WENO-Z)"),
        "FV (Pallas)": dict(color=FV_COLOR, marker="s", label="FV (VL2)"),
    }
    for label, style in curve_styles.items():
        axis.loglog(N_VALUES, results[label]["l1"], **style)

    # The guide lines start at the coarsest-grid error of the scheme whose
    # order they show.
    resolutions = np.array(N_VALUES, dtype=float)
    second_order_start = results["FV (Pallas)"]["l1"][0]
    fifth_order_start = results["FD (Pallas)"]["l1"][0]
    axis.loglog(
        resolutions,
        second_order_start * (resolutions / resolutions[0]) ** -2.0,
        color=GUIDE_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=r"$\propto N^{-2}$",
    )
    axis.loglog(
        resolutions,
        fifth_order_start * (resolutions / resolutions[0]) ** -5.0,
        color=GUIDE_COLOR,
        linestyle=":",
        linewidth=1.0,
        label=r"$\propto N^{-5}$",
    )

    axis.set_xscale("log")
    axis.xaxis.set_major_locator(ticker.FixedLocator(N_VALUES))
    axis.xaxis.set_major_formatter(ticker.FixedFormatter([str(resolution) for resolution in N_VALUES]))
    axis.xaxis.set_minor_locator(ticker.NullLocator())
    axis.set_xlabel(r"$N$ (grid $2N \times N \times N$)")
    axis.set_ylabel(r"average $L_1$ error")
    axis.legend(loc="lower left", ncols=2)

    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    figure.savefig(layout_path(Path(FIG_DIR) / "sound_wave3D_convergence.svg"))
    plt.close(figure)


def load_results():
    """Return the cached sweep results of every benchmark, keyed by label."""
    results = {}
    for spec in BENCHMARKS:
        cached = np.load(Path(DATA_DIR) / f"sound_wave3D_convergence_{_slug(spec.label)}.npz")
        results[spec.label] = {
            "N": list(cached["N_values"]),
            "l1": list(cached["l1_errors"]),
            "runtime": list(cached["runtimes"]),
            "iterations": list(cached["iterations"]),
        }
    return results


if __name__ == "__main__" and "--plot-only" in sys.argv:
    results = load_results()
    # The overview figure is single-column only (it writes to the same file
    # name as the paper figure of the single-column layout, which follows).
    if not TWO_COLUMN:
        plot_convergence_and_runtime(
            results,
            N_VALUES,
            name="sound_wave3D",
            title="3D linear sound wave",
            figure_dir=FIG_DIR,
        )
    plot_paper_figure(results)
    print_convergence_orders(results, N_VALUES)
elif __name__ == "__main__":
    results = run_convergence_and_runtime(
        BENCHMARKS,
        N_values=N_VALUES,
        setup_fn=setup_sound_wave,
        analytic_fn=sound_wave_solution,
        error_var_indices_fn=_error_indices,
        name="sound_wave3D",
        title="3D linear sound wave",
        data_dir=DATA_DIR,
        figure_dir=FIG_DIR,
    )
    plot_paper_figure(results)
    print_convergence_orders(results, N_VALUES)
