"""1D stress tests for the finite-difference WENO solver.

A small battery of Riemann problems chosen to exercise the regimes in which the
production runs need post-hoc stabilisation: near-vacuum (double
rarefactions), huge pressure and density ratios, and cold high-Mach flow with
density contrast. Every problem is run with *no* positivity machinery at all,
so a failure here is a failure of the bare scheme.

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/riemann1d.py --variant baseline
"""

# general
import argparse
import os
import sys
from typing import NamedTuple

# numerics
import numpy as np

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs, weno_variant_name


class RiemannProblem(NamedTuple):
    """One 1D Riemann problem on [0, box_size] with diaphragm at x0."""

    name: str
    gamma: float
    rho_L: float
    u_L: float
    p_L: float
    rho_R: float
    u_R: float
    p_R: float
    t_end: float
    box_size: float = 1.0
    x0: float = 0.5
    exact: bool = True


PROBLEMS = [
    RiemannProblem("sod", 1.4, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 0.2),
    RiemannProblem("einfeldt123", 1.4, 1.0, -2.0, 0.4, 1.0, 2.0, 0.4, 0.15),
    RiemannProblem("near_vacuum", 1.4, 1.0, -3.5, 0.4, 1.0, 3.5, 0.4, 0.1),
    RiemannProblem("toro3_blast", 1.4, 1.0, 0.0, 1000.0, 1.0, 0.0, 0.01, 0.012),
    RiemannProblem(
        "leblanc", 5.0 / 3.0, 1.0, 0.0, (2.0 / 3.0) * 1e-1,
        1e-3, 0.0, (2.0 / 3.0) * 1e-10, 6.0, box_size=9.0, x0=3.0,
    ),
    # cold dense slab rammed at Mach ~800 (dense side) into tenuous gas at rest
    RiemannProblem("cold_ram", 5.0 / 3.0, 100.0, 1.0, 1e-4, 1.0, 0.0, 1e-4, 0.3),
    # cold colliding flows, Mach ~ 80
    RiemannProblem("cold_collide", 5.0 / 3.0, 1.0, 1.0, 1e-4, 1.0, -1.0, 1e-4, 0.3),
    # tenuous gas rammed into a dense slab at rest, Mach ~ 80
    # the Evrard cloud edge (e0 = 0.05 cloud vs floored ambient), dx = 0.125 at N=32
    RiemannProblem("evrard_edge", 5.0 / 3.0, 0.16, 0.0, 5.3e-3, 1e-4, 0.0, 3.3e-6, 0.4, box_size=4.0, x0=2.0),
    RiemannProblem("cold_ram_rev", 5.0 / 3.0, 1.0, 1.0, 1e-4, 100.0, 0.0, 1e-4, 0.3),
]


def run_problem(problem, num_cells, precision, variant, cfl):
    """Run one problem; return (status, L1 density error, min rho, min p)."""

    import jax.numpy as jnp
    from astronomix import (
        SimulationConfig,
        SimulationParams,
        get_helper_data,
        get_registered_variables,
        time_integration,
        construct_primitive_state,
        finalize_config,
    )
    from astronomix.option_classes.simulation_config import (
        BoundarySettings1D,
        FDSolverConfig,
        OPEN_BOUNDARY,
    )
    from astronomix.test_setups.reference_solutions.riemann_solver import (
        _exact_riemann_ideal_gas,
    )

    config = SimulationConfig(
        dimensionality=1,
        num_cells=num_cells,
        box_size=problem.box_size,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
        ),
        solver_config=FDSolverConfig(**weno_variant_kwargs()),
        **variant_config_kwargs(variant),
    )
    params = SimulationParams(
        t_end=problem.t_end,
        gamma=problem.gamma,
        C_cfl=cfl,
        minimum_density=1e-14,
        minimum_pressure=1e-14,
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers
    left = x < problem.x0
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.where(left, problem.rho_L, problem.rho_R),
        velocity_x=jnp.where(left, problem.u_L, problem.u_R),
        gas_pressure=jnp.where(left, problem.p_L, problem.p_R),
    )
    config = finalize_config(config, state.shape)
    final = time_integration(state, config, params, registered_variables)
    final = np.asarray(final, dtype=np.float64)
    rho = final[registered_variables.density_index]
    pressure = final[registered_variables.pressure_index]

    finite = np.all(np.isfinite(final))
    l1 = np.nan
    if finite and problem.exact:
        rho_exact, _, _ = _exact_riemann_ideal_gas(
            problem.rho_L, problem.u_L, problem.p_L,
            problem.rho_R, problem.u_R, problem.p_R,
            problem.gamma, x, problem.t_end, problem.x0,
        )
        l1 = float(np.mean(np.abs(rho - np.asarray(rho_exact))) * problem.box_size)
    tag = os.environ.get("LAB_TAG", variant)
    os.makedirs("examples/scripts/validation/weno_stability/out", exist_ok=True)
    np.savez(
        f"examples/scripts/validation/weno_stability/out/{problem.name}_{tag}_x{precision}_n{num_cells}.npz",
        x=np.asarray(x), state=final,
    )
    status = "ok" if finite and rho.min() > 0 and pressure.min() > 0 else "FAIL"
    if not finite:
        status = "NaN"
    return status, l1, float(np.nanmin(rho)), float(np.nanmin(pressure))


def variant_config_kwargs(variant):
    """Map a variant name onto SimulationConfig keyword arguments."""
    if variant == "baseline":
        return {}
    raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--precision", type=int, default=64)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    if args.precision == 64:
        os.environ.setdefault("JAX_ENABLE_X64", "1")

    for problem in PROBLEMS:
        if args.only and problem.name not in args.only.split(","):
            continue
        try:
            status, l1, rho_min, p_min = run_problem(
                problem, args.n, args.precision, args.variant, args.cfl
            )
        except Exception as error:  # noqa: BLE001 - report and continue
            status, l1, rho_min, p_min = f"ERR {type(error).__name__}", np.nan, np.nan, np.nan
            print(error, file=sys.stderr)
        tag = os.environ.get("LAB_TAG", args.variant)
        print(
            f"{tag:>12s} x{args.precision} N={args.n:5d}  {problem.name:14s} "
            f"{status:8s} L1(rho)={l1:10.3e}  min rho={rho_min:10.3e}  min p={p_min:10.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
