"""Temporal convergence of the energy error (fixed dt), per WENO / coupling variant.

Mirrors examples/scripts/forward/self_gravity/evrard_timestep_convergence.py
(mild Evrard, e0 = 0.2, 32^3, float64, fourth-order conservative coupling):
the conservative coupling conserves energy exactly in space, so its energy
error is the time integrator's (~dt^4).

    WENO_VARIANT=pp PYTHONPATH=. python examples/scripts/validation/weno_stability/evrard_dt.py --limit-work 0
"""

# general
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=32)
parser.add_argument("--e0", type=float, default=0.2)
parser.add_argument("--steps", default="250,500,1000,2000,4000")
parser.add_argument("--coupling", default="fourth", choices=["second", "fourth"])
parser.add_argument("--fct", type=int, default=0)
parser.add_argument("--tag", default="")
args = parser.parse_args()

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs, weno_variant_name

from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
)

GAMMA = 5.0 / 3.0
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
gravity_kwargs = {}
if args.fct:
    gravity_kwargs["work_flux_correction"] = True

results = []
for num_timesteps in [int(s) for s in args.steps.split(",")]:
    config = SimulationConfig(
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=FOURTH_ORDER_CONSERVATIVE if args.coupling == "fourth" else SECOND_ORDER_CONSERVATIVE,
            poisson_manual_open_boundaries=True,
            **gravity_kwargs,
        ),
        dimensionality=3,
        box_size=4.0,
        num_cells=args.n,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        positivity_config=PositivityConfig(),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        fixed_timestep=True,
        num_timesteps=num_timesteps,
        return_snapshots=True,
        num_snapshots=2,
        snapshot_settings=SnapshotSettings(
            return_states=False, return_final_state=True, return_total_energy=True,
        ),
        **weno_variant_kwargs(),
    )
    params = SimulationParams(t_end=1.2, dt_max=jnp.inf, minimum_density=1e-5, minimum_pressure=3e-6)
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    radius = helper_data.r
    rho = jnp.where(radius <= 1.0, 1.0 / (2 * jnp.pi * radius), 1e-4)
    pressure = jnp.maximum((GAMMA - 1) * rho * args.e0, params.minimum_pressure)
    zero = jnp.zeros_like(rho)
    state = construct_primitive_state(
        config=config, registered_variables=registered_variables, density=rho,
        velocity_x=zero, velocity_y=zero, velocity_z=zero, gas_pressure=pressure,
    )
    config = finalize_config(config, state.shape)
    snapshots = jax.block_until_ready(time_integration(state, config, params, registered_variables))
    total = np.asarray(snapshots.total_energy)
    final = np.asarray(snapshots.final_state)
    error = float(abs(total[-1] - total[0]) / abs(total[0]))
    results.append((num_timesteps, error, float(final[registered_variables.pressure_index].min())))
    print(f"[{weno_variant_name()}{args.tag} fct={args.fct}] steps={num_timesteps:6d} "
          f"dE/E={error:.3e} min p={results[-1][2]:.3e}", flush=True)

orders = [np.log2(results[k][1] / results[k + 1][1]) for k in range(len(results) - 1)]
print(f"DT-ORDER [{weno_variant_name()}{args.tag} fct={args.fct}] "
      + " ".join(f"{o:.2f}" for o in orders), flush=True)
