"""Correctness check for AD through the Pallas FD backend.

The Pallas WENO/SSPRK kernels are wrapped by ``diffable_pallas_call``: the
primal runs the (fast, aliased) Pallas branch while the tangent is routed
through the equivalent native-JAX branch. This script verifies that

  * forward-mode (``jax.jvp``) and reverse-mode (``jax.grad``) both run
    through the Pallas backend without error, and
  * the resulting gradients match the pure-NATIVE_JAX backend to tight
    tolerance (the Pallas primal is bit-identical, so the gradient obtained
    by transposing the native tangent at the Pallas-evaluated state must
    agree with the fully-native gradient).

Tested on the 1D acoustic-wave sensitivity testbed (same as
``checkpoint_scaling.py``) and a 3D ideal-gas hydro forward map.
"""
import os
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
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
    FORWARDS,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams


# ---------------------------------------------------------------- 1D testbed
def cost_1d(backend, N=64, t_end=1.5, diff_mode=BACKWARDS):
    rho_B, c_s, gamma, eps, L = 1.0, 2.0, 5.0 / 3.0, 1e-6, 1.0
    P_B = c_s**2 * rho_B / gamma
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        geometry=CARTESIAN,
        progress_bar=False,
        boundary_handling=PERIODIC_ROLL,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=1,
        box_size=L,
        num_cells=N,
        boundary_settings=BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        return_snapshots=False,
        backend=backend,
        differentiation_mode=diff_mode,
        num_checkpoints=32,
    )
    params = SimulationParams(C_cfl=0.4, t_end=t_end, gamma=gamma,
                              gravitational_constant=0.0)
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    x = jnp.squeeze(helper_data.geometric_centers)
    k = 2.0 * jnp.pi * 2.0 / L
    rho_P0 = eps * jnp.sin(k * x)
    v_P0 = jnp.zeros_like(rho_P0)
    dV = L / N

    def cost(rho_P0_in, v_P0_in):
        rho = rho_B + rho_P0_in
        p = P_B + c_s**2 * rho_P0_in
        s0 = construct_primitive_state(config, rv, density=rho,
                                       velocity_x=v_P0_in, gas_pressure=p)
        cfg = finalize_config(config, s0.shape)
        final = time_integration(s0, cfg, params, rv)
        frho = final[rv.density_index] - rho_B
        fv = final[rv.velocity_index]
        return 0.5 * jnp.sum(frho**2 + fv**2) * dV

    return cost, (rho_P0, v_P0)


# ---------------------------------------------------------------- 3D testbed
def cost_3d(backend, N=16, t_end=0.15, diff_mode=BACKWARDS):
    gamma, L = 5.0 / 3.0, 1.0
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        geometry=CARTESIAN,
        progress_bar=False,
        boundary_handling=PERIODIC_ROLL,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=3,
        box_size=L,
        num_cells=N,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        return_snapshots=False,
        backend=backend,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
        differentiation_mode=diff_mode,
        num_checkpoints=16,
    )
    params = SimulationParams(C_cfl=0.4, t_end=t_end, gamma=gamma,
                              gravitational_constant=0.0)
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    c = helper_data.geometric_centers
    x, y, z = c[..., 0], c[..., 1], c[..., 2]
    k = 2.0 * jnp.pi / L

    def cost(amp):
        rho = 1.0 + amp * jnp.sin(k * x) * jnp.cos(k * y)
        p = 1.0 + 0.5 * amp * jnp.cos(k * z)
        vx = 0.1 * amp * jnp.sin(k * y)
        s0 = construct_primitive_state(config, rv, density=rho, velocity_x=vx,
                                       velocity_y=jnp.zeros_like(rho),
                                       velocity_z=jnp.zeros_like(rho),
                                       gas_pressure=p)
        cfg = finalize_config(config, s0.shape)
        final = time_integration(s0, cfg, params, rv)
        return jnp.sum(final[rv.density_index] ** 2) * (L / N) ** 3

    return cost, (jnp.float64(0.2),)


def report(name, gn, gp):
    """Compare two gradients with combined relative+absolute tolerance so a
    component that is ~0 in both (e.g. a symmetry-zero gradient at float noise
    level) is not flagged as a relative mismatch."""
    gn = np.asarray(gn, dtype=np.float64).ravel()
    gp = np.asarray(gp, dtype=np.float64).ravel()
    scale = max(np.abs(gn).max(), np.abs(gp).max(), 1e-30)
    abs_diff = np.abs(gn - gp).max()
    rel = abs_diff / scale  # rel to the gradient's own magnitude scale
    ok = np.allclose(gn, gp, rtol=1e-6, atol=1e-9 * scale)
    print(f"  {name}: |max|={scale:.6e}  max abs diff={abs_diff:.3e}  "
          f"rel={rel:.3e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    print("=== 1D acoustic-wave testbed (FD) ===")
    # reverse-mode (grad) uses BACKWARDS (checkpointed loop)
    cost_n, args = cost_1d(NATIVE_JAX, diff_mode=BACKWARDS)
    cost_p, _ = cost_1d(PALLAS, diff_mode=BACKWARDS)
    vn = float(cost_n(*args)); vp = float(cost_p(*args))
    print(f"  primal: native={vn:.10e}  pallas={vp:.10e}  "
          f"reldiff={abs(vn - vp) / max(abs(vn), 1e-30):.2e}")
    gn = jax.grad(cost_n, argnums=(0, 1))(*args)
    gp = jax.grad(cost_p, argnums=(0, 1))(*args)
    report("grad d/drhoP0", gn[0], gp[0])
    report("grad d/dvP0", gn[1], gp[1])
    # forward-mode (jvp) needs FORWARDS (plain while loop)
    cost_nf, _ = cost_1d(NATIVE_JAX, diff_mode=FORWARDS)
    cost_pf, _ = cost_1d(PALLAS, diff_mode=FORWARDS)
    tan = (jnp.ones_like(args[0]), jnp.ones_like(args[1]))
    _, jn = jax.jvp(cost_nf, args, tan)
    _, jp = jax.jvp(cost_pf, args, tan)
    print(f"  jvp: native={float(jn):.6e}  pallas={float(jp):.6e}  "
          f"reldiff={abs(float(jn) - float(jp)) / max(abs(float(jn)), 1e-30):.2e}")

    print("\n=== 3D ideal-gas hydro testbed (FD) ===")
    c3n, a3 = cost_3d(NATIVE_JAX, diff_mode=BACKWARDS)
    c3p, _ = cost_3d(PALLAS, diff_mode=BACKWARDS)
    v3n = float(c3n(*a3)); v3p = float(c3p(*a3))
    print(f"  primal: native={v3n:.10e}  pallas={v3p:.10e}  "
          f"reldiff={abs(v3n - v3p) / max(abs(v3n), 1e-30):.2e}")
    g3n = jax.grad(c3n)(*a3); g3p = jax.grad(c3p)(*a3)
    print(f"  grad d/damp: native={float(g3n):.6e}  pallas={float(g3p):.6e}  "
          f"reldiff={abs(float(g3n) - float(g3p)) / max(abs(float(g3n)), 1e-30):.2e}")
    c3nf, _ = cost_3d(NATIVE_JAX, diff_mode=FORWARDS)
    c3pf, _ = cost_3d(PALLAS, diff_mode=FORWARDS)
    _, j3n = jax.jvp(c3nf, a3, (jnp.float64(1.0),))
    _, j3p = jax.jvp(c3pf, a3, (jnp.float64(1.0),))
    print(f"  jvp: native={float(j3n):.6e}  pallas={float(j3p):.6e}  "
          f"reldiff={abs(float(j3n) - float(j3p)) / max(abs(float(j3n)), 1e-30):.2e}")


if __name__ == "__main__":
    main()
