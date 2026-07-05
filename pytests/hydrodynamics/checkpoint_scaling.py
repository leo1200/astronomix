"""
Checkpoint scaling benchmark.

Sweeps ``num_checkpoints`` for the backwards-differentiable time integration
and records, for each solver configuration (FV and FD by default):

  1. Gradient-pass runtime
  2. Gradient-pass compiled memory (XLA ``memory_analysis``)
  3. Theoretical extra forward evaluations from binomial checkpointing,
     using the actual number of timesteps ``N`` from the simulation.

The differentiable testbed is the 1D linear acoustic wave from
``tests/sensitivity/sensitivity.py`` (sine wave initial condition,
periodic box, gradient of ``J = 0.5 * sum(rho_P^2 + v^2) * dV`` with
respect to the initial primitive perturbations). The setup is mirrored
here so the script doesn't depend on the ``tests/sensitivity`` package
being importable.
"""

import math
import os
from timeit import default_timer as timer
from typing import NamedTuple

# ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from astronomix import (
    CARTESIAN,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    FORWARDS,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    BoundarySettings1D,
    SimulationConfig,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams


_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data", "astronomix")
FIG_DIR = os.path.join(_HERE, "figures")


# -----------------------------------------------------------------------------
# Binomial checkpointing (Griewank '92): minimal recomputations as a function
# of c (checkpoints) for a fixed forward length N.
# -----------------------------------------------------------------------------
def _beta(c: int, r: int) -> int:
    return math.comb(c + r, c)


def _repetition_number(c: int, N: int) -> int:
    r = 0
    while _beta(c, r) < N:
        r += 1
    return r


def _total_forward_evals(c: int, N: int) -> int:
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
    rho_B: float = 1.0
    c_s: float = 2.0
    gamma: float = 5.0 / 3.0
    eps: float = 1.0e-6
    L: float = 1.0
    t_end: float = 1.5
    N: int = 64

    @property
    def P_B(self) -> float:
        return (self.c_s**2) * self.rho_B / self.gamma


def _base_config(problem: _Problem, solver_mode: int, backend: int = NATIVE_JAX):
    config = SimulationConfig(
        solver_mode=solver_mode,
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
        backend=backend,
        pallas_use_triton=True,
        pallas_interpret=False,
    )
    params = SimulationParams(
        C_cfl=0.4,
        t_end=problem.t_end,
        gamma=problem.gamma,
        gravitational_constant=0.0,
    )
    return config, params


def _wave_ic(helper_data, problem: _Problem):
    x = jnp.squeeze(helper_data.geometric_centers)
    k = 2.0 * jnp.pi * 2.0 / problem.L
    rho_P0 = problem.eps * jnp.sin(k * x)
    v_P0 = jnp.zeros_like(rho_P0)
    return rho_P0, v_P0


def _forward_and_cost(rho_P0, v_P0, config, params, problem: _Problem):
    registered_variables = get_registered_variables(config)
    dV = problem.L / problem.N

    rho = problem.rho_B + rho_P0
    p = problem.P_B + (problem.c_s**2) * rho_P0

    initial_state = construct_primitive_state(
        config,
        registered_variables,
        density=rho,
        velocity_x=v_P0,
        gas_pressure=p,
    )
    config_final = finalize_config(config, initial_state.shape)
    final = time_integration(initial_state, config_final, params, registered_variables)

    final_rho_P = final[registered_variables.density_index] - problem.rho_B
    final_v = final[registered_variables.velocity_index]
    return 0.5 * jnp.sum(final_rho_P**2 + final_v**2) * dV


# -----------------------------------------------------------------------------
# Single forward run to discover N (the actual number of timesteps).
# -----------------------------------------------------------------------------
def _measure_num_iterations(solver_mode: int, problem: _Problem,
                            backend: int = NATIVE_JAX) -> int:
    config, params = _base_config(problem, solver_mode, backend)
    config = config._replace(
        differentiation_mode=FORWARDS,
        return_snapshots=True,
        num_snapshots=1,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    rho_P0, v_P0 = _wave_ic(helper_data, problem)

    rho = problem.rho_B + rho_P0
    p = problem.P_B + (problem.c_s**2) * rho_P0
    initial_state = construct_primitive_state(
        config,
        registered_variables,
        density=rho,
        velocity_x=v_P0,
        gas_pressure=p,
    )
    config = finalize_config(config, initial_state.shape)
    result = time_integration(initial_state, config, params, registered_variables)
    return int(result.num_iterations)


# -----------------------------------------------------------------------------
# Per-(config, c) gradient measurement.
# -----------------------------------------------------------------------------
def _measure_grad(
    solver_mode: int, problem: _Problem, num_checkpoints: int, repeats: int,
    backend: int = NATIVE_JAX,
):
    jax.clear_caches()
    config, params = _base_config(problem, solver_mode, backend)
    config = config._replace(
        differentiation_mode=BACKWARDS,
        num_checkpoints=num_checkpoints,
    )
    helper_data = get_helper_data(config)
    rho_P0, v_P0 = _wave_ic(helper_data, problem)

    def cost(rho_P0_in, v_P0_in):
        return _forward_and_cost(rho_P0_in, v_P0_in, config, params, problem)

    grad_fn = jax.jit(jax.grad(cost, argnums=(0, 1)))

    lowered = grad_fn.lower(rho_P0, v_P0)
    compiled = lowered.compile()
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

    out = grad_fn(rho_P0, v_P0)
    jax.block_until_ready(out)

    times = []
    for _ in range(repeats):
        t0 = timer()
        out = grad_fn(rho_P0, v_P0)
        jax.block_until_ready(out)
        t1 = timer()
        times.append(t1 - t0)
    return memory_MB, float(np.mean(times))


# -----------------------------------------------------------------------------
# Sweep + plotting.
# -----------------------------------------------------------------------------
class CheckpointSpec(NamedTuple):
    label: str
    solver_mode: int
    backend: int = NATIVE_JAX


# Compare backends on the backward (reverse-mode AD) pass.  In BACKWARDS mode
# the FD hydro WENO flux routes its backward through the native Pallas adjoint
# kernel (see astronomix/_pallas_helpers.pallas_vjp_call), so FD (Pallas) runs
# the whole reverse pass on the GPU instead of the slow native VJP.  FV (Pallas)
# is left out of the comparison (the FV evolve backward is not the focus here).
DEFAULT_CONFIGURATIONS = (
    CheckpointSpec("FD (JAX)", FINITE_DIFFERENCE, NATIVE_JAX),
    CheckpointSpec("FD (Pallas)", FINITE_DIFFERENCE, PALLAS),
    CheckpointSpec("FV (JAX)", FINITE_VOLUME, NATIVE_JAX),
)


def run_sweep(
    checkpoint_counts,
    problem: _Problem = _Problem(),
    configurations=DEFAULT_CONFIGURATIONS,
    repeats: int = 3,
) -> dict:
    results = {}
    for spec in configurations:
        print(f"\n=== {spec.label} (solver_mode={spec.solver_mode}, "
              f"backend={spec.backend}) ===")
        N_iters = _measure_num_iterations(spec.solver_mode, problem, spec.backend)
        print(f"[{spec.label}] forward N = {N_iters} timesteps")
        rt_list, mem_list = [], []
        for c in checkpoint_counts:
            mem, rt = _measure_grad(spec.solver_mode, problem, int(c),
                                    repeats=repeats, backend=spec.backend)
            print(f"[{spec.label}] c={int(c):3d}: memory={mem:8.2f} MB, runtime={rt:6.3f}s")
            rt_list.append(rt)
            mem_list.append(mem)
        results[spec.label] = dict(
            N_iters=N_iters,
            checkpoints=list(map(int, checkpoint_counts)),
            runtime_s=rt_list,
            memory_MB=mem_list,
        )
    return results


def plot_results(results, figure_path):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    ax_rt, ax_recomp, ax_mem = axs

    # Consistent backend colours across all figures: FD (JAX) light blue,
    # FD (Pallas) violet, FV (JAX) orange.  FD (JAX) is a thick solid line / 'o',
    # FD (Pallas) a thinner dashed line / 'x' drawn on top, so the two FD curves
    # stay clearly distinguishable where they overlap.
    def _style(label):
        if label.startswith("FD"):
            if "Pallas" in label:
                return dict(color="darkviolet", linestyle="--", linewidth=1.8,
                            marker="x", markersize=7, zorder=3)
            return dict(color="cornflowerblue", linestyle="-", linewidth=3.0,
                        marker="o", markersize=6, zorder=2)
        return dict(color="tab:orange", linestyle="-", linewidth=2.2,
                    marker="o", markersize=6, zorder=2)

    c_max_global = 0
    for label, data in results.items():
        c_arr = np.array(data["checkpoints"])
        c_max_global = max(c_max_global, int(c_arr.max()))
        st = _style(label)
        ax_rt.plot(c_arr, data["runtime_s"], label=label, **st)
        ax_mem.plot(c_arr, data["memory_MB"], label=label, **st)

    N_values = {data["N_iters"] for data in results.values()}
    if len(N_values) != 1:
        raise ValueError(
            f"Expected a single forward N across configurations, got {N_values}. "
            "The theoretical recomputation curve assumes one N."
        )
    N = N_values.pop()

    c_grid = np.arange(1, c_max_global + 1)
    recomp = np.array([recomputations(int(c), N) for c in c_grid])
    ax_recomp.plot(
        c_grid, recomp, marker="o", markersize=3, linewidth=1.5,
        label="minimal recomputations",
    )

    c_r3 = _repetition_threshold(N, 3, c_max_global)
    c_r2 = _repetition_threshold(N, 2, c_max_global)
    if c_r3 is not None:
        ax_recomp.axvline(
            c_r3, linestyle=":", linewidth=1.2,
            label=f"threshold where repetition number r ≤ 3 (at c = {c_r3})",
        )
    if c_r2 is not None:
        ax_recomp.axvline(
            c_r2, linestyle="-.", linewidth=1.2,
            label=f"threshold where repetition number r ≤ 2 (at c = {c_r2})",
        )

    ax_rt.set_yscale("log")
    ax_rt.set_xlabel("number of checkpoints c")
    ax_rt.set_ylabel("gradient runtime [s]")
    ax_rt.set_title("Backward-pass runtime")
    ax_rt.grid(True, which="both", linewidth=0.5, alpha=0.5)
    ax_rt.legend()

    ax_recomp.set_yscale("log")
    ax_recomp.set_xlabel("number of checkpoints c")
    ax_recomp.set_ylabel("extra forward steps / recomputations")
    ax_recomp.set_title(f"Checkpointing recomputation cost for N = {N}")
    ax_recomp.grid(True, which="both", linewidth=0.5, alpha=0.5)
    ax_recomp.legend()

    ax_mem.set_xlabel("number of checkpoints c")
    ax_mem.set_ylabel("compiled gradient memory [MB]")
    ax_mem.set_title("Backward-pass memory")
    ax_mem.grid(True, which="both", linewidth=0.5, alpha=0.5)
    ax_mem.legend()

    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {figure_path}")


def _flatten_for_npz(results):
    flat = {}
    for label, data in results.items():
        for k, v in data.items():
            flat[f"{label}_{k}"] = np.array(v)
    return flat


def test_checkpoint_scaling():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    problem = _Problem()
    checkpoint_counts = (2, 4, 8, 14, 22, 32, 50, 80)
    results = run_sweep(checkpoint_counts, problem=problem)
    plot_results(results, os.path.join(FIG_DIR, "checkpoint_scaling.svg"))
    np.savez(
        os.path.join(DATA_DIR, "checkpoint_scaling.npz"),
        **_flatten_for_npz(results),
    )


if __name__ == "__main__":
    test_checkpoint_scaling()
