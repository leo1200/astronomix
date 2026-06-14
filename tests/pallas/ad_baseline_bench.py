"""Baseline measurement for the differentiable-Pallas evaluation.

Quantifies, on a small 3D FD hydro sound-wave setup, what reverse-mode AD
through ``time_integration`` costs today under the three relevant configs:

  1. NATIVE_JAX forward + grad           (reference)
  2. PALLAS forward (no AD)              (the fast path we want to keep)
  3. PALLAS forward + grad (current      (custom_jvp with native tangent —
     ``diffable_pallas_call`` design)     forward runs Pallas AND native
                                          residuals, backward is native)

Reports compile (cold) and warm wall time plus max|grad_PALLAS −
grad_NATIVE| as the correctness anchor.

Run with the repo root on PYTHONPATH (astronomix installed non-editable):

    PYTHONPATH=/export/home/lstorcks/agent-home/astronomix \
        python tests/pallas/ad_baseline_bench.py [N] [T_END]
"""

import os
import sys
import time

# ==== device selection: one free GPU via autocvd ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# ====================================================

import jax
import jax.numpy as jnp

from astronomix import get_helper_data, get_registered_variables, time_integration
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    CARTESIAN,
    FINITE_DIFFERENCE,
    NATIVE_JAX,
    PALLAS,
    PALLAS_AD_JVP_NATIVE,
    PALLAS_AD_VJP_PALLAS,
    PALLAS_AD_VJP_REMAT,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
T_END = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
L = 1.0
GAMMA = 5.0 / 3.0
RHO_B = 1.0
C_S = 2.0
P_B = (C_S**2) * RHO_B / GAMMA
NUM_CHECKPOINTS = int(os.environ.get("AD_NUM_CHECKPOINTS", "100"))


def make_config(backend, ad_mode=PALLAS_AD_JVP_NATIVE):
    pb = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    return SimulationConfig(
        backend=backend,
        pallas_block_shape=(4, 4, 8),
        pallas_ad_mode=ad_mode,
        solver_mode=FINITE_DIFFERENCE,
        geometry=CARTESIAN,
        progress_bar=False,
        self_gravity=False,
        boundary_handling=PERIODIC_ROLL,
        differentiation_mode=BACKWARDS,
        num_checkpoints=NUM_CHECKPOINTS,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=3,
        box_size=StaticFloatVector(x=L, y=L, z=L),
        num_cells=StaticIntVector(x=N, y=N, z=N),
        boundary_settings=BoundarySettings(pb, pb, pb),
        return_snapshots=False,
    )


def build(backend, ad_mode=PALLAS_AD_JVP_NATIVE):
    config = make_config(backend, ad_mode)
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    centers = jnp.transpose(helper_data.geometric_centers, (3, 0, 1, 2))
    X = centers[0]
    rho_p = 1e-3 * jnp.sin(2 * jnp.pi * 2 * X / L)
    zeros = jnp.zeros_like(rho_p)

    def loss(rho_pert):
        initial_state = construct_primitive_state(
            config,
            registered_variables,
            density=RHO_B + rho_pert,
            gas_pressure=P_B + (C_S**2) * rho_pert,
            velocity_x=zeros,
            velocity_y=zeros,
            velocity_z=zeros,
        )
        config_final = finalize_config(config, initial_state.shape)
        params = SimulationParams(t_end=T_END, C_cfl=0.4, gamma=GAMMA)
        final_state = time_integration(
            initial_state, config_final, params, registered_variables
        )
        return jnp.sum(final_state**2)

    return loss, rho_p


def timed(name, fn, arg):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(arg))
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(arg))
    t_warm = time.perf_counter() - t0
    print(f"[{name:30s}] cold={t_cold:7.2f}s warm={t_warm:7.3f}s", flush=True)
    return out


if __name__ == "__main__":
    print(f"N={N} t_end={T_END} checkpoints={NUM_CHECKPOINTS} "
          f"devices={jax.devices()}", flush=True)

    loss_nat, rho_p = build(NATIVE_JAX)
    loss_pal, _ = build(PALLAS)

    j_nat = timed("NATIVE fwd loss", jax.jit(loss_nat), rho_p)
    j_pal = timed("PALLAS fwd loss", jax.jit(loss_pal), rho_p)
    print(f"  |J_PALLAS - J_NATIVE| = {abs(float(j_pal) - float(j_nat)):.3e} "
          f"(J={float(j_nat):.6e})", flush=True)

    g_nat = timed("NATIVE grad", jax.jit(jax.grad(loss_nat)), rho_p)

    def report(name, g):
        diff = float(jnp.max(jnp.abs(g - g_nat)))
        denom = float(jnp.max(jnp.abs(g_nat)))
        print(f"  {name}: max|grad - NATIVE| = {diff:.3e}  "
              f"(rel {diff / denom:.3e})", flush=True)

    g_pal = timed("PALLAS grad JVP_NATIVE", jax.jit(jax.grad(loss_pal)), rho_p)
    report("JVP_NATIVE", g_pal)

    loss_remat, _ = build(PALLAS, PALLAS_AD_VJP_REMAT)
    g_remat = timed("PALLAS grad VJP_REMAT", jax.jit(jax.grad(loss_remat)), rho_p)
    report("VJP_REMAT ", g_remat)

    loss_adj, _ = build(PALLAS, PALLAS_AD_VJP_PALLAS)
    g_adj = timed("PALLAS grad VJP_PALLAS", jax.jit(jax.grad(loss_adj)), rho_p)
    report("VJP_PALLAS", g_adj)
