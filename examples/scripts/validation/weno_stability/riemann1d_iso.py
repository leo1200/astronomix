"""1D isothermal stress tests: deep rarefactions and strong shocks.

Isothermal rarefactions are exponentially deep (rho* = rho exp(-du/2c) for a
symmetric expansion), so a velocity jump of 20 c produces a physical density of
~1e-9 — the regime of the voids in Mach-10 isothermal turbulence.

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/riemann1d_iso.py
"""

# general
import argparse
import os
import sys
from typing import NamedTuple

# numerics
import numpy as np
from scipy.optimize import brentq


BX = float(os.environ.get("LAB_BX", "0.0"))
BY = float(os.environ.get("LAB_BY", "0.0"))


class IsoProblem(NamedTuple):
    name: str
    rho_L: float
    u_L: float
    rho_R: float
    u_R: float
    t_end: float
    c: float = 1.0
    box_size: float = 1.0
    x0: float = 0.5


PROBLEMS = [
    IsoProblem("iso_rare5", 1.0, -5.0, 1.0, 5.0, 0.05),
    IsoProblem("iso_rare10", 1.0, -10.0, 1.0, 10.0, 0.03),
    IsoProblem("iso_rare20", 1.0, -20.0, 1.0, 20.0, 0.015),
    IsoProblem("iso_collide50", 1.0, 50.0, 1.0, -50.0, 0.004),
    IsoProblem("iso_ram", 100.0, 20.0, 1.0, 0.0, 0.01),
    IsoProblem("iso_contrast_rare", 1.0, -10.0, 100.0, 10.0, 0.02),
]


def exact_isothermal(problem, x, t):
    """Exact solution of the isothermal Riemann problem (no vacuum case)."""
    c = problem.c

    def wave(rho_star, rho_k):
        if rho_star > rho_k:
            return c * (rho_star - rho_k) / np.sqrt(rho_star * rho_k)
        return c * np.log(rho_star / rho_k)

    def residual(log_rho):
        rho_star = np.exp(log_rho)
        return wave(rho_star, problem.rho_L) + wave(rho_star, problem.rho_R) + problem.u_R - problem.u_L

    log_rho_star = brentq(residual, -200.0, 50.0, xtol=1e-14)
    rho_star = np.exp(log_rho_star)
    u_star = problem.u_L - wave(rho_star, problem.rho_L)

    xi = (np.asarray(x) - problem.x0) / t
    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    for k, s in enumerate(xi):
        if s <= u_star:
            if rho_star > problem.rho_L:
                speed = problem.u_L - c * np.sqrt(rho_star / problem.rho_L)
                rho[k], u[k] = (problem.rho_L, problem.u_L) if s < speed else (rho_star, u_star)
            else:
                head, tail = problem.u_L - c, u_star - c
                if s < head:
                    rho[k], u[k] = problem.rho_L, problem.u_L
                elif s > tail:
                    rho[k], u[k] = rho_star, u_star
                else:
                    u[k] = s + c
                    rho[k] = problem.rho_L * np.exp((problem.u_L - u[k]) / c)
        else:
            if rho_star > problem.rho_R:
                speed = problem.u_R + c * np.sqrt(rho_star / problem.rho_R)
                rho[k], u[k] = (problem.rho_R, problem.u_R) if s > speed else (rho_star, u_star)
            else:
                head, tail = problem.u_R + c, u_star + c
                if s > head:
                    rho[k], u[k] = problem.rho_R, problem.u_R
                elif s < tail:
                    rho[k], u[k] = rho_star, u_star
                else:
                    u[k] = s - c
                    rho[k] = problem.rho_R * np.exp((u[k] - problem.u_R) / c)
    return rho, u, rho_star


def run_problem(problem, num_cells, precision, cfl, rho_floor):
    import jax.numpy as jnp
    from astronomix import (
        SimulationConfig,
        SimulationParams,
        construct_primitive_state,
        finalize_config,
        get_helper_data,
        get_registered_variables,
        time_integration,
    )
    from astronomix.option_classes.simulation_config import (
        ISOTHERMAL,
        OPEN_BOUNDARY,
        BoundarySettings1D,
    )

    config = SimulationConfig(
        dimensionality=1,
        num_cells=num_cells,
        box_size=problem.box_size,
        equation_of_state=ISOTHERMAL,
        mhd=True,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
        ),
    )
    params = SimulationParams(
        t_end=problem.t_end,
        C_cfl=cfl,
        isothermal_sound_speed=problem.c,
        minimum_density=rho_floor,
        minimum_pressure=rho_floor,
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
        velocity_y=jnp.zeros_like(x),
        velocity_z=jnp.zeros_like(x),
        magnetic_field_x=jnp.full_like(x, BX),
        magnetic_field_y=jnp.full_like(x, BY),
        magnetic_field_z=jnp.zeros_like(x),
    )
    config = finalize_config(config, state.shape)
    final = np.asarray(time_integration(state, config, params, registered_variables), dtype=np.float64)
    rho = final[registered_variables.density_index]
    velocity = final[registered_variables.velocity_index.x]
    rho_exact, u_exact, rho_star = exact_isothermal(problem, np.asarray(x), problem.t_end)
    finite = np.all(np.isfinite(final))
    l1_log_rho = float(np.mean(np.abs(np.log10(np.maximum(rho, 1e-300)) - np.log10(rho_exact)))) if finite else np.nan
    l1_u = float(np.mean(np.abs(velocity - u_exact))) if finite else np.nan
    tag = os.environ.get("LAB_TAG", "run")
    os.makedirs("examples/scripts/validation/weno_stability/out", exist_ok=True)
    np.savez(
        f"examples/scripts/validation/weno_stability/out/{problem.name}_{tag}_x{precision}_n{num_cells}.npz",
        x=np.asarray(x), state=final, rho_exact=rho_exact, u_exact=u_exact,
    )
    status = "ok" if finite and rho.min() > 0 else ("NaN" if not finite else "FAIL")
    return status, rho_star, l1_log_rho, l1_u, float(np.nanmin(rho)), float(np.nanmax(np.abs(velocity)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--precision", type=int, default=64)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--rho-floor", type=float, default=1e-14)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    if args.precision == 64:
        os.environ.setdefault("JAX_ENABLE_X64", "1")
    tag = os.environ.get("LAB_TAG", "run")
    for problem in PROBLEMS:
        if args.only and problem.name not in args.only.split(","):
            continue
        try:
            status, rho_star, l1_log_rho, l1_u, rho_min, v_max = run_problem(
                problem, args.n, args.precision, args.cfl, args.rho_floor
            )
        except Exception as error:  # noqa: BLE001
            print(error, file=sys.stderr)
            status, rho_star, l1_log_rho, l1_u, rho_min, v_max = "ERR", np.nan, np.nan, np.nan, np.nan, np.nan
        print(
            f"{tag:>10s} x{args.precision} N={args.n:5d} {problem.name:18s} {status:5s} "
            f"rho*={rho_star:9.2e} L1(log rho)={l1_log_rho:9.3e} L1(u)={l1_u:9.3e} "
            f"min rho={rho_min:9.2e} max|u|={v_max:9.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
