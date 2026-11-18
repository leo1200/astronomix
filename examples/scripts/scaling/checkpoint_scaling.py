"""
Checkpoint scaling benchmark.

Sweeps ``num_checkpoints`` of the reverse-mode (BACKWARDS) time integration
with adaptive time steps and records, for each solver configuration:

  1. the runtime of one gradient evaluation (forward + backward),
  2. its compiled memory (XLA ``memory_analysis``),
  3. the theoretical number of recomputed forward steps of optimal binomial
     checkpointing, for the actual number of time steps N of the run.

The configurations are the finite-difference WENO-Z solver and the
finite-volume VL2 scheme (AthenaPK's van Leer integrator, PLM, HLLC), each on
the native-JAX and the Pallas backend. On the Pallas backend the forward
steps run the Pallas kernels while the derivatives go through the native-JAX
tangent (``diffable_pallas_call``), so the backward pass is a native-JAX one.

The differentiable testbed is the 1D linear acoustic wave of
``examples/scripts/differentiability/sensitivity.py`` (sine-wave density
perturbation in a periodic box, gradient of
``J = 0.5 * sum(rho_P^2 + v^2) * dV`` with respect to the initial
perturbations), mirrored here so the benchmark is self-contained.

Run with ``--plot-only`` to redraw the figure from ``data/astronomix/
checkpoint_scaling.json``.
"""

# general
import argparse
import json
import math
import os
from timeit import default_timer as timer

# typing
from typing import NamedTuple

# ==== GPU selection ====
import sys as _sys
if "--plot-only" not in _sys.argv:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# The linear acoustic eigenmode has tiny perturbation amplitudes, so double
# precision is required to keep the gradient signal above float rounding noise.
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    BACKWARDS,
    CARTESIAN,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    FORWARDS,
    HLLC,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    VL2,
)
from astronomix.option_classes.simulation_config import PERIODIC_ROLL
from astronomix.plotting_helpers.paper_style import (
    FD_COLOR,
    FV_COLOR,
    GUIDE_COLOR,
    REFERENCE_COLOR,
    TEXT_WIDTH,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings1D,
    FDSolverConfig,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.plotting_helpers.paper_style import (
    figure_legend,
    use_paper_style,
    variant_style,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data", "astronomix")
FIG_DIR = os.path.join(_HERE, "figures")
RESULTS_JSON = os.path.join(DATA_DIR, "checkpoint_scaling.json")
DEFAULT_CHECKPOINT_COUNTS = (2, 4, 8, 14, 22, 32, 50, 80)


# -----------------------------------------------------------------------------
# Binomial checkpointing (Griewank '92): minimal recomputations as a function
# of c (checkpoints) for a fixed forward length N.
# -----------------------------------------------------------------------------
def _beta(c: int, r: int) -> int:
    """Return the binomial reach beta(c, r) = C(c + r, c) (Griewank '92)."""
    return math.comb(c + r, c)


def _repetition_number(c: int, N: int) -> int:
    """Return the smallest repetition number ``r`` with ``beta(c, r) >= N``."""
    r = 0
    while _beta(c, r) < N:
        r += 1
    return r


def _total_forward_evals(c: int, N: int) -> int:
    """Return the total forward evaluations of optimal binomial checkpointing."""
    r = _repetition_number(c, N)
    return r * N - math.comb(c + r, c + 1)


def recomputations(c: int, N: int) -> int:
    """Extra forward evaluations beyond the original N forward steps."""
    return max(0, _total_forward_evals(c, N) - N)


def _repetition_threshold(N: int, r_target: int, c_max: int):
    """Smallest ``c`` such that ``beta(c, r_target) >= N`` in ``[1, c_max]``."""
    for c in range(1, c_max + 1):
        if _beta(c, r_target) >= N:
            return c
    return None


# -----------------------------------------------------------------------------
# Linear-wave testbed (1D periodic acoustic eigenmode).
# -----------------------------------------------------------------------------
class _Problem(NamedTuple):
    """The 1D linear acoustic eigenmode used as the differentiable testbed.

    Attributes:
        rho_B: The background density.
        c_s: The background sound speed.
        gamma: The adiabatic index.
        eps: The perturbation amplitude of the density eigenmode.
        L: The box length.
        t_end: The integration end time.
        N: The number of cells.
    """
    rho_B: float = 1.0
    c_s: float = 2.0
    gamma: float = 5.0 / 3.0
    eps: float = 1.0e-6
    L: float = 1.0
    t_end: float = 1.5
    N: int = 64

    @property
    def P_B(self) -> float:
        """The background pressure implied by the sound speed and density."""
        return (self.c_s**2) * self.rho_B / self.gamma


class CheckpointSpec(NamedTuple):
    """A single solver configuration in the checkpoint sweep.

    Attributes:
        label: The human-readable label used in prints and plot legends.
        solver_mode: The solver mode (finite-volume or finite-difference).
        backend: The compute backend (native JAX or Pallas).
    """
    label: str
    solver_mode: int
    backend: int = NATIVE_JAX


DEFAULT_CONFIGURATIONS = (
    CheckpointSpec("FD WENO-Z (JAX)", FINITE_DIFFERENCE, NATIVE_JAX),
    CheckpointSpec("FD WENO-Z (Pallas)", FINITE_DIFFERENCE, PALLAS),
    CheckpointSpec("FV VL2 (JAX)", FINITE_VOLUME, NATIVE_JAX),
    CheckpointSpec("FV VL2 (Pallas)", FINITE_VOLUME, PALLAS),
)


def _base_config(problem: _Problem, spec: CheckpointSpec):
    """Build the simulation config and params for the linear-wave testbed.

    Args:
        problem: The linear acoustic eigenmode specification.
        spec: The solver configuration (solver mode and backend).

    Returns:
        A ``(config, params)`` tuple for the periodic 1D acoustic problem.
    """
    if spec.solver_mode == FINITE_VOLUME:
        solver_config = FVSolverConfig(time_integrator=VL2, riemann_solver=HLLC)
    else:
        solver_config = FDSolverConfig(weno_z=True)

    config = SimulationConfig(
        solver_mode=spec.solver_mode,
        geometry=CARTESIAN,
        progress_bar=False,
        boundary_handling=PERIODIC_ROLL,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=1,
        box_size=problem.L,
        num_cells=problem.N,
        boundary_settings=BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY,
            right_boundary=PERIODIC_BOUNDARY,
        ),
        return_snapshots=False,
        backend_config=BackendConfig(backend=spec.backend),
        solver_config=solver_config,
    )
    # In 1D the finite-difference and the VL2 time steps are both
    # C_cfl * dx / max(|v| + c_s), so the two solvers take the same number of
    # adaptive steps and share one recomputation curve.
    params = SimulationParams(
        C_cfl=0.4,
        t_end=problem.t_end,
        gamma=problem.gamma,
        gravitational_constant=0.0,
    )
    return config, params


def _wave_ic(helper_data, problem: _Problem):
    """Return the initial density and velocity perturbations of the eigenmode.

    Args:
        helper_data: The helper data holding the geometric cell centers.
        problem: The linear acoustic eigenmode specification.

    Returns:
        A ``(rho_P0, v_P0)`` tuple of the density and velocity perturbations.
    """
    x = jnp.squeeze(helper_data.geometric_centers)
    k = 2.0 * jnp.pi * 2.0 / problem.L
    rho_P0 = problem.eps * jnp.sin(k * x)
    v_P0 = jnp.zeros_like(rho_P0)
    return rho_P0, v_P0


def _initial_state(rho_P0, v_P0, config, registered_variables, problem: _Problem):
    """Superimpose the perturbations on the uniform background state.

    The pressure perturbation follows the acoustic relation p' = c_s^2 rho'.

    Args:
        rho_P0: The initial density perturbation.
        v_P0: The initial velocity perturbation.
        config: The simulation configuration.
        registered_variables: The registered variables.
        problem: The linear acoustic eigenmode specification.

    Returns:
        The primitive initial state.
    """
    return construct_primitive_state(
        config,
        registered_variables,
        density=problem.rho_B + rho_P0,
        velocity_x=v_P0,
        gas_pressure=problem.P_B + (problem.c_s**2) * rho_P0,
    )


def _forward_and_cost(rho_P0, v_P0, config, params, problem: _Problem):
    """Integrate the wave forward and return the scalar cost ``J``.

    Args:
        rho_P0: The initial density perturbation.
        v_P0: The initial velocity perturbation.
        config: The simulation configuration.
        params: The simulation parameters.
        problem: The linear acoustic eigenmode specification.

    Returns:
        The scalar cost ``J = 0.5 * sum(rho_P^2 + v^2) * dV`` at ``t_end``.
    """
    registered_variables = get_registered_variables(config)
    dV = problem.L / problem.N
    initial_state = _initial_state(rho_P0, v_P0, config, registered_variables, problem)
    config_final = finalize_config(config, initial_state.shape)
    final = time_integration(initial_state, config_final, params, registered_variables)

    final_rho_P = final[registered_variables.density_index] - problem.rho_B
    final_v = final[registered_variables.velocity_index]
    return 0.5 * jnp.sum(final_rho_P**2 + final_v**2) * dV


def _time_call(function, arguments, repeats: int) -> float:
    """Mean wall time of ``function(*arguments)`` after one warm-up call.

    Args:
        function: The compiled function.
        arguments: The tuple of its arguments.
        repeats: The number of timed calls.

    Returns:
        The mean runtime in seconds.
    """
    jax.block_until_ready(function(*arguments))
    times = []
    for _ in range(repeats):
        start_time = timer()
        jax.block_until_ready(function(*arguments))
        times.append(timer() - start_time)
    return float(np.mean(times))


# -----------------------------------------------------------------------------
# Single forward run: the actual number of timesteps N and the forward runtime.
# -----------------------------------------------------------------------------
def _measure_forward(spec: CheckpointSpec, problem: _Problem, repeats: int):
    """Run the forward integration, return its step count and runtime.

    Args:
        spec: The solver configuration.
        problem: The linear acoustic eigenmode specification.
        repeats: The number of timed forward runs to average over.

    Returns:
        A ``(num_iterations, mean_runtime_s)`` tuple of the plain
        (non-differentiated, FORWARDS-mode) forward run.
    """
    jax.clear_caches()
    config, params = _base_config(problem, spec)
    config = config._replace(
        differentiation_mode=FORWARDS,
        return_snapshots=True,
        num_snapshots=1,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    rho_P0, v_P0 = _wave_ic(helper_data, problem)
    initial_state = _initial_state(rho_P0, v_P0, config, registered_variables, problem)
    config = finalize_config(config, initial_state.shape)

    forward = jax.jit(lambda state: time_integration(state, config, params, registered_variables))
    num_iterations = int(forward(initial_state).num_iterations)
    runtime = _time_call(forward, (initial_state,), repeats)
    return num_iterations, runtime


# -----------------------------------------------------------------------------
# Per-(config, c) gradient measurement.
# -----------------------------------------------------------------------------
def _measure_grad(spec: CheckpointSpec, problem: _Problem, num_checkpoints: int, repeats: int):
    """Measure the compiled memory and runtime of one gradient pass.

    Args:
        spec: The solver configuration.
        problem: The linear acoustic eigenmode specification.
        num_checkpoints: The number of binomial checkpoints for the backward
            pass.
        repeats: The number of timed gradient evaluations to average over.

    Returns:
        A ``(memory_MB, mean_runtime_s)`` tuple for the gradient pass.
    """
    # Clear the caches so that every configuration is compiled and measured
    # independently of the previous ones.
    jax.clear_caches()
    config, params = _base_config(problem, spec)
    config = config._replace(
        differentiation_mode=BACKWARDS,
        num_checkpoints=num_checkpoints,
    )
    helper_data = get_helper_data(config)
    rho_P0, v_P0 = _wave_ic(helper_data, problem)

    def cost(rho_P0_in, v_P0_in):
        return _forward_and_cost(rho_P0_in, v_P0_in, config, params, problem)

    grad_fn = jax.jit(jax.grad(cost, argnums=(0, 1)))

    # Compile first and read the compiled memory footprint before timing, so the
    # reported memory reflects the same executable that the timed runs use.
    compiled = grad_fn.lower(rho_P0, v_P0).compile()
    stats = compiled.memory_analysis()
    if stats is not None:
        total_bytes = (
            stats.temp_size_in_bytes
            + stats.argument_size_in_bytes
            + stats.output_size_in_bytes
            - stats.alias_size_in_bytes
        )
    else:
        total_bytes = 0
    memory_MB = total_bytes / (1024**2)

    runtime = _time_call(compiled, (rho_P0, v_P0), repeats)
    return memory_MB, runtime


# -----------------------------------------------------------------------------
# Sweep + plotting.
# -----------------------------------------------------------------------------
def run_sweep(
    checkpoint_counts,
    problem: _Problem = _Problem(),
    configurations=DEFAULT_CONFIGURATIONS,
    repeats: int = 3,
) -> dict:
    """Sweep the checkpoint count for every configuration and collect results.

    Args:
        checkpoint_counts: The checkpoint counts ``c`` to sweep over.
        problem: The linear acoustic eigenmode specification.
        configurations: The solver configurations to benchmark.
        repeats: The number of timed evaluations to average over.

    Returns:
        A dict mapping each configuration label to its recorded forward step
        count, forward runtime, checkpoint counts, gradient runtimes and
        memory footprints.
    """
    results = {}
    for spec in configurations:
        print(f"\n=== {spec.label} (solver_mode={spec.solver_mode}, backend={spec.backend}) ===")
        N_iters, forward_runtime = _measure_forward(spec, problem, repeats)
        print(f"[{spec.label}] forward N = {N_iters} timesteps, forward runtime = {forward_runtime:.4f}s")
        runtime_list, memory_list = [], []
        for c in checkpoint_counts:
            memory, runtime = _measure_grad(spec, problem, int(c), repeats)
            print(f"[{spec.label}] c={int(c):3d}: memory={memory:8.2f} MB, runtime={runtime:6.3f}s")
            runtime_list.append(runtime)
            memory_list.append(memory)
        results[spec.label] = dict(
            N_iters=N_iters,
            forward_runtime_s=forward_runtime,
            checkpoints=list(map(int, checkpoint_counts)),
            runtime_s=runtime_list,
            memory_MB=memory_list,
        )
    return results


def _curve_style(label):
    """The colours of the schemes in all figures of the paper: the Pallas runs
    as filled markers on solid lines on top of the native-JAX runs, which are
    the dashed, hollow variant, so that both stay visible where they overlap.
    """
    color = FD_COLOR if label.startswith("FD") else FV_COLOR
    marker = "o" if label.startswith("FD") else "s"
    if "Pallas" in label:
        return dict(color=color, linestyle="-", marker=marker, zorder=3)
    return dict(marker=marker, markersize=7, zorder=2, **variant_style(color))


def plot_results(results, figure_path):
    """Plot runtime, recomputation cost and memory versus checkpoint count.

    Args:
        results: The per-configuration results dict from :func:`run_sweep`.
        figure_path: The path to write the resulting SVG figure to.
    """
    use_paper_style()
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 3.0), layout="constrained")
    ax_rt, ax_recomp, ax_mem = axs

    # -------------------------------------------------------------
    # ============== ↓ Measured runtime and memory ↓ ==============
    # -------------------------------------------------------------

    c_max_global = 0
    for label, data in results.items():
        checkpoint_array = np.array(data["checkpoints"])
        c_max_global = max(c_max_global, int(checkpoint_array.max()))
        style = _curve_style(label)
        ax_rt.plot(checkpoint_array, data["runtime_s"], label=label, **style)
        ax_mem.plot(checkpoint_array, data["memory_MB"], label=label, **style)

    # -------------------------------------------------------------
    # ============== ↑ Measured runtime and memory ↑ ==============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =========== ↓ Theoretical recomputation curve ↓ =============
    # -------------------------------------------------------------

    # One recomputation curve per distinct forward length N (the solvers
    # normally share one; a differing N gets its own, labelled curve).
    N_values = sorted({data["N_iters"] for data in results.values()})
    c_grid = np.arange(1, c_max_global + 1)
    for N in N_values:
        recomp = np.array([recomputations(int(c), N) for c in c_grid])
        ax_recomp.plot(
            c_grid,
            recomp,
            color=REFERENCE_COLOR,
            marker="o",
            markersize=2,
            linewidth=1.0,
            label="minimal recomputations" + (f" (N = {N})" if len(N_values) > 1 else ""),
        )

    # Mark where the repetition number drops to r <= 3 and r <= 2, the regimes
    # where binomial checkpointing becomes cheap.
    N = N_values[0]
    c_r3 = _repetition_threshold(N, 3, c_max_global)
    c_r2 = _repetition_threshold(N, 2, c_max_global)
    if c_r3 is not None:
        ax_recomp.axvline(
            c_r3,
            color=GUIDE_COLOR,
            linestyle=":",
            linewidth=1.0,
            label=f"repetition number $r \\leq 3$ ($c \\geq {c_r3}$)",
        )
    if c_r2 is not None:
        ax_recomp.axvline(
            c_r2,
            color=GUIDE_COLOR,
            linestyle="-.",
            linewidth=1.0,
            label=f"repetition number $r \\leq 2$ ($c \\geq {c_r2}$)",
        )

    # -------------------------------------------------------------
    # =========== ↑ Theoretical recomputation curve ↑ =============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================== ↓ Axis formatting ↓ ======================
    # -------------------------------------------------------------

    ax_rt.set_yscale("log")
    ax_rt.set_xlabel("number of checkpoints $c$")
    ax_rt.set_ylabel("gradient runtime (s)")
    ax_rt.set_title("backward-pass runtime")

    ax_recomp.set_yscale("log")
    ax_recomp.set_xlabel("number of checkpoints $c$")
    ax_recomp.set_ylabel("recomputed forward steps")
    ax_recomp.set_title(f"recomputations ($N = {', '.join(str(value) for value in N_values)}$)")

    ax_mem.set_xlabel("number of checkpoints $c$")
    ax_mem.set_ylabel("gradient memory (MB)")
    ax_mem.set_title("backward-pass memory")

    figure_legend(fig, ncols=3)
    fig.savefig(figure_path)
    plt.close(fig)
    print(f"\nSaved {figure_path}")

    # -------------------------------------------------------------
    # ================== ↑ Axis formatting ↑ ======================
    # -------------------------------------------------------------


def _flatten_for_npz(results):
    """Flatten the nested results dict into ``label_key`` arrays for ``savez``."""
    flat = {}
    for label, data in results.items():
        for k, v in data.items():
            flat[f"{label}_{k}"] = np.array(v)
    return flat


def _print_backend_ratios(results):
    """Print the Pallas-over-JAX ratios of runtime and memory per solver.

    Args:
        results: The per-configuration results dict from :func:`run_sweep`.
    """
    for solver_prefix in ("FD WENO-Z", "FV VL2"):
        jax_label, pallas_label = f"{solver_prefix} (JAX)", f"{solver_prefix} (Pallas)"
        if jax_label not in results or pallas_label not in results:
            continue
        runtime_ratio = np.array(results[jax_label]["runtime_s"]) / np.array(results[pallas_label]["runtime_s"])
        memory_ratio = np.array(results[jax_label]["memory_MB"]) / np.array(results[pallas_label]["memory_MB"])
        forward_ratio = results[jax_label]["forward_runtime_s"] / results[pallas_label]["forward_runtime_s"]
        print(
            f"{solver_prefix}: JAX / Pallas gradient runtime {runtime_ratio.min():.2f} - {runtime_ratio.max():.2f}, "
            f"memory {memory_ratio.min():.2f} - {memory_ratio.max():.2f}, forward runtime {forward_ratio:.2f}"
        )
    for label, data in results.items():
        gradient_over_forward = np.array(data["runtime_s"]) / data["forward_runtime_s"]
        print(
            f"{label}: gradient / forward runtime {gradient_over_forward.min():.1f} - "
            f"{gradient_over_forward.max():.1f}"
        )


def run_checkpoint_scaling(checkpoint_counts=DEFAULT_CHECKPOINT_COUNTS, plot_only=False):
    """Run the checkpoint sweep, plot the results and save the raw data.

    Args:
        checkpoint_counts: The checkpoint counts ``c`` to sweep over.
        plot_only: Redraw the figure from the saved JSON instead of measuring.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    if plot_only:
        with open(RESULTS_JSON) as results_file:
            results = json.load(results_file)
    else:
        results = run_sweep(checkpoint_counts, problem=_Problem())
        with open(RESULTS_JSON, "w") as results_file:
            json.dump(results, results_file, indent=1)
        np.savez(
            os.path.join(DATA_DIR, "checkpoint_scaling.npz"),
            **_flatten_for_npz(results),
        )
    _print_backend_ratios(results)
    plot_results(results, os.path.join(FIG_DIR, "checkpoint_scaling.svg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_COUNTS))
    parser.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved JSON")
    args = parser.parse_args()
    run_checkpoint_scaling(tuple(args.checkpoints), plot_only=args.plot_only)
