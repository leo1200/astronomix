"""Step the real SSPRK(5,4) + CT from a dumped state and report min p per stage.

    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python examples/scripts/validation/weno_stability/mhd_step_check.py DUMP.npz [steps] [C_cfl]
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
from astronomix._fluid_equations._equations_mhd import conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from astronomix._finite_difference._time_integrators import _ssprk as ssprk_module
from astronomix._integrators import _explicit_rk
from astronomix._finite_difference._timestep_estimation._timestep_estimator import _cfl_time_step_fd
from astronomix.option_classes.simulation_config import IDEAL_GAS

GAMMA = 1.4
dump = np.load(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
courant = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
padded = jnp.asarray(dump["states"][-1])
n_padded = padded.shape[-1]
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
config = SimulationConfig(
    equation_of_state=IDEAL_GAS, dimensionality=2, num_cells=n_padded - 8, box_size=1.0, mhd=True,
    backend_config=BackendConfig(backend=NATIVE_JAX),
    boundary_settings=BoundarySettings(periodic, periodic, periodic),
    positivity_config=PositivityConfig(clamp_in_estimates=False), **weno_variant_kwargs(),
)
config = finalize_config(config, (padded.shape[0], n_padded - 8, n_padded - 8))
rv = get_registered_variables(config)
params = SimulationParams(C_cfl=courant, gamma=GAMMA, minimum_density=1e-30, minimum_pressure=1e-30)
helper_data = get_helper_data(config)
inner = (slice(4, -4), slice(4, -4))


def pressure_of(q):
    kinetic = 0.5 * sum(q[k] ** 2 for k in rv.momentum_index) / q[rv.density_index]
    magnetic = 0.5 * sum(q[k] ** 2 for k in rv.magnetic_index)
    return (GAMMA - 1.0) * (q[rv.energy_index] - kinetic - magnetic)


stage_log = []
_original_ssprk4 = _explicit_rk.ssprk4


def logging_ssprk4(u0, dt, *, rhs, pre_stage, post_stage, finalize):
    def logged_post(u):
        out = post_stage(u)
        jax.debug.callback(lambda a, b: stage_log.append((float(a), float(b))),
                           jnp.min(pressure_of(u[0])[inner]), jnp.min(pressure_of(out[0])[inner]))
        return out
    return _original_ssprk4(u0, dt, rhs=rhs, pre_stage=pre_stage, post_stage=logged_post, finalize=finalize)


ssprk_module.ssprk4 = logging_ssprk4
primitive = padded
for step in range(steps):
    previous = primitive
    dt = float(_cfl_time_step_fd(primitive, config.grid_spacing, jnp.inf, GAMMA, config, params, rv, courant))
    q = conserved_state_from_primitive_mhd(primitive[:-3], GAMMA, rv)
    stage_log.clear()
    new_q, bx, by, bz = ssprk_module._ssprk4_with_ct(
        jnp.array(q, copy=True), *(jnp.array(primitive[k], copy=True) for k in (-3, -2, -1)),
        GAMMA, config.grid_spacing, dt, params, helper_data, config, rv)
    jax.effects_barrier()
    stages = " ".join(f"{a:+.1e}/{b:+.1e}" for a, b in stage_log)
    p_new = pressure_of(new_q)[inner]
    print(f"step {step}: dt={dt:.3e} min p in={float(primitive[rv.pressure_index][inner].min()):+.3e} "
          f"out={float(p_new.min()):+.3e} | stages {stages}", flush=True)
    primitive = jnp.concatenate([
        primitive_state_from_conserved_mhd(new_q, 1e-30, 1e-30, GAMMA, config, rv), bx[None], by[None], bz[None]], axis=0)
    if float(p_new.min()) < 0:
        np.savez(sys.argv[1].replace(".npz", "_prefail.npz"), states=np.asarray(previous)[None])
        break
