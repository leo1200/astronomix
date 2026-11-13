"""Audit of the assumptions behind the PP-WENO positivity theorem on a real state.

The forward-Euler update of a cell is

    q^{n+1} = (1 - lambda sum_d S_d) q + (lambda sum_d S_d / 2) (own + in),

and is admissible when (i) lambda sum_d S_d <= 1 locally, (ii) the own and
inflow states are admissible. The limiter enforces (ii) given that the
axis-summed first-order inflow B_i is admissible. Measured here:

* B_i: its pressure relative to the cell's, over all cells;
* the local Courant number lambda_FE sum_d S_d of the stiffest SSPRK stage
  (lambda_FE = dt / (1.508 dx)) for a given C_cfl, against the global one;
* for one SSPRK step with the production scheme: cells whose specific entropy
  p / rho^gamma falls below the minimum over their 27-cell neighbourhood
  (the minimum entropy principle, a stronger invariant domain than p > 0).

    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python examples/scripts/validation/weno_stability/invariant_domain_audit.py STATE.npy
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("state")
parser.add_argument("--cfl", type=float, default=1.5)
parser.add_argument("--step", type=int, default=1)
args = parser.parse_args()
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs

from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    finalize_config,
    get_helper_data,
    get_registered_variables,
)
from astronomix._fluid_equations._equations_mhd import conserved_state_from_primitive_mhd
from astronomix._finite_difference._interface_fluxes._weno_positivity import mhd_inflow_reference
from astronomix._finite_difference._magnetic_update._constrained_transport import update_cell_center_fields
from astronomix._finite_difference._time_integrators import _ssprk as ssprk_module
from astronomix._finite_difference._timestep import _cfl_time_step_fd
from astronomix.option_classes.simulation_config import IDEAL_GAS

GAMMA = 5.0 / 3.0
loaded = np.load(args.state)
primitive = jnp.asarray((loaded["state"] if args.state.endswith(".npz") else loaded).astype(np.float64))
n = primitive.shape[-1]
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
config = SimulationConfig(
    equation_of_state=IDEAL_GAS,
    dimensionality=3,
    num_cells=n,
    box_size=1.0,
    mhd=True,
    backend_config=BackendConfig(backend=NATIVE_JAX),
    boundary_settings=BoundarySettings(periodic, periodic, periodic),
    positivity_config=PositivityConfig(clamp_in_estimates=False),
    solver_config=FDSolverConfig(**weno_variant_kwargs()),
)
config = finalize_config(config, primitive.shape)
rv = get_registered_variables(config)
params = SimulationParams(C_cfl=args.cfl, gamma=GAMMA, minimum_density=1e-30, minimum_pressure=1e-30)
dx = config.grid_spacing


def pressure(q):
    kinetic = 0.5 * sum(q[k] ** 2 for k in rv.momentum_index) / q[rv.density_index]
    magnetic = 0.5 * sum(q[k] ** 2 for k in rv.magnetic_index)
    return (GAMMA - 1.0) * (q[rv.energy_index] - kinetic - magnetic)


q = conserved_state_from_primitive_mhd(primitive[:-3], GAMMA, rv)
q = update_cell_center_fields(q, primitive[-3], primitive[-2], primitive[-1], config, rv)
p = pressure(q)

# (1) the axis-summed first-order inflow
reference, speed_sum = jax.jit(lambda s: mhd_inflow_reference(s, params, config, rv))(q)
ratio = np.asarray(pressure(reference) / p)
print(f"B_i: min p(B)/p = {ratio.min():+.3e}; cells with p(B) <= 0: {(ratio <= 0).sum()}; "
      f"quantiles p(B)/p (0.001%, 0.01%, 1%): {np.quantile(ratio, [1e-5, 1e-4, 1e-2])}")

# (2) local Courant numbers of the stiffest forward-Euler substep
dt = float(_cfl_time_step_fd(primitive, dx, jnp.inf, GAMMA, config, params, rv, args.cfl))
local = np.asarray(dt / 1.508 / dx * speed_sum)
print(f"C_cfl={args.cfl}: dt={dt:.3e}; local lambda_FE sum S: max {local.max():.3f}, "
      f"cells > 1: {(local > 1).sum()} ({100 * (local > 1).mean():.2f} %)")
low = np.asarray(p) < np.quantile(np.asarray(p), 1e-3)
print(f"  in the 0.1 % lowest-pressure cells: max {local[low].max():.3f}, median {np.median(local[low]):.3f}")

# (3) one production step: minimum entropy principle
helper_data = get_helper_data(config)
entropy = np.asarray(p / q[rv.density_index] ** GAMMA)
neighbourhood_min = entropy.copy()
for shift in [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]:
    neighbourhood_min = np.minimum(neighbourhood_min, np.roll(entropy, shift, axis=(0, 1, 2)))
state = primitive
for step in range(args.step):
    faces = [jnp.array(state[k], copy=True) for k in (-3, -2, -1)]
    qs = conserved_state_from_primitive_mhd(state[:-3], GAMMA, rv)
    new_q, bx, by, bz = ssprk_module._ssprk_with_ct(jnp.array(qs, copy=True), *faces, GAMMA, dx, dt, params,
                                                      helper_data, config, rv)
new_entropy = np.asarray(pressure(new_q) / new_q[rv.density_index] ** GAMMA)
undershoot = new_entropy / neighbourhood_min
print(f"one step: cells with s < local min s^n: {(undershoot < 1).sum()} ({100 * (undershoot < 1).mean():.2f} %); "
      f"min s/s_min = {undershoot.min():.3e}; quantiles (0.01%, 0.1%, 1%): {np.quantile(undershoot, [1e-4, 1e-3, 1e-2])}")
lowest = np.argsort(np.asarray(p).ravel())[:5]
print("  at the 5 lowest-pressure cells, s^{n+1}/s_min:", " ".join(f"{undershoot.ravel()[i]:.3f}" for i in lowest))
