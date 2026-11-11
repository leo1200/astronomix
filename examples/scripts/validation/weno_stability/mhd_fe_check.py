"""One forward-Euler step of the actual MHD-CT stage machinery from a dumped state.

``ssprk4`` is replaced by a single forward-Euler step (pre_stage, rhs at dt,
post_stage, finalize), so the step is exactly one SSP building block of the
production integrator. The positivity proof covers C_FE <= 1/2 of the code's
sum-of-axes CFL (C_cfl <= 0.754 for the full SSPRK(5,4) step).

    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python examples/scripts/validation/weno_stability/mhd_fe_check.py out/blast_pp_n50_lastgood.npz
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs

from astronomix import (
    BackendConfig, BoundarySettings, BoundarySettings1D, NATIVE_JAX, PERIODIC_BOUNDARY, PositivityConfig,
    SimulationConfig, SimulationParams, finalize_config, get_helper_data, get_registered_variables,
)
from astronomix._fluid_equations._equations_mhd import conserved_state_from_primitive_mhd
from astronomix._finite_difference._time_integrators import _ssprk as ssprk_module
from astronomix._finite_difference._timestep_estimation._timestep_estimator import _cfl_time_step_fd
from astronomix.option_classes.simulation_config import IDEAL_GAS

GAMMA = 1.4
dump = np.load(sys.argv[1])
index = int(sys.argv[2]) if len(sys.argv) > 2 else -1
padded = jnp.asarray(dump["states"][index])
n_padded = padded.shape[-1]
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)


def fe_step(u0, dt, *, rhs, pre_stage=lambda u: u, post_stage=lambda u: u, finalize=lambda u: u):
    u = pre_stage(u0)
    du = rhs(u, dt)
    return finalize(post_stage(jax.tree_util.tree_map(lambda a, b: a + b, u, du)))


ssprk_module.ssprk4 = fe_step

config = SimulationConfig(
    equation_of_state=IDEAL_GAS, dimensionality=2, num_cells=n_padded - 8, box_size=1.0, mhd=True,
    backend_config=BackendConfig(backend=NATIVE_JAX),
    boundary_settings=BoundarySettings(periodic, periodic, periodic),
    positivity_config=PositivityConfig(clamp_in_estimates=False), **weno_variant_kwargs(),
)
config = finalize_config(config, (padded.shape[0], n_padded - 8, n_padded - 8))
rv = get_registered_variables(config)
params = SimulationParams(C_cfl=0.75, gamma=GAMMA, minimum_density=1e-30, minimum_pressure=1e-30)
helper_data = get_helper_data(config)
print("num_ghost_cells", config.num_ghost_cells, "grid_spacing", config.grid_spacing, "padded", padded.shape)

pressure_index = rv.pressure_index
dt_cfl = float(_cfl_time_step_fd(padded, config.grid_spacing, jnp.inf, GAMMA, config, params, rv, 1.0))
print(f"min p (state) = {float(padded[pressure_index].min()):.3e}; dt at C=1: {dt_cfl:.3e}")

q = conserved_state_from_primitive_mhd(padded[:-3], GAMMA, rv)
bx, by, bz = padded[-3], padded[-2], padded[-1]
inner = (slice(4, -4), slice(4, -4))
for courant in (0.125, 0.25, 0.5, 0.75, 1.0):
    dt = courant * dt_cfl
    new_q, *_ = ssprk_module._ssprk4_with_ct(jnp.array(q, copy=True), jnp.array(bx, copy=True), jnp.array(by, copy=True), jnp.array(bz, copy=True), GAMMA, config.grid_spacing, dt, params,
                                             helper_data, config, rv)
    kinetic = 0.5 * sum(new_q[k] ** 2 for k in rv.momentum_index) / new_q[rv.density_index]
    magnetic = 0.5 * sum(new_q[k] ** 2 for k in rv.magnetic_index)
    p_new = (GAMMA - 1.0) * (new_q[rv.energy_index] - kinetic - magnetic)
    core = p_new[inner]
    worst = np.unravel_index(int(jnp.argmin(core)), core.shape)
    print(f"FE C_FE={courant:5.3f}: min p = {float(core.min()):+.3e} at {worst}, cells p<0: {int((core < 0).sum())}, "
          f"min rho = {float(new_q[rv.density_index][inner].min()):.3e}", flush=True)
