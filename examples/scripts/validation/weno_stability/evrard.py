"""Cold Evrard collapse with NO positivity machinery — the self-gravity stress test.

The conservative FD gravity coupling NaNs on this problem below 128^3 in both
precisions (tests/gravity_stability/FINDINGS.md, deleted in f2caa99). Here it is
run with the bare scheme so any stabilisation must come from the WENO kernel.

    PYTHONPATH=. python examples/scripts/validation/weno_stability/evrard.py --n 32 --precision 32
"""

# general
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=32)
parser.add_argument("--precision", type=int, default=32)
parser.add_argument("--t-end", type=float, default=1.2)
parser.add_argument("--e0", type=float, default=0.05)
parser.add_argument("--gravity", default="fourth", choices=["simple", "second", "fourth"])
parser.add_argument("--tag", default="")
parser.add_argument("--save-states", action="store_true")
parser.add_argument("--num-snapshots", type=int, default=25)
parser.add_argument("--dual", type=int, default=0)
parser.add_argument("--fct", type=int, default=0)
args = parser.parse_args()

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
if args.precision == 64:
    os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
# numerics
import numpy as np

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs, weno_variant_name

# jax
import jax
import jax.numpy as jnp

# astronomix
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
    SIMPLE_SOURCE,
)

GRAVITY = dict(
    simple=SIMPLE_SOURCE, second=SECOND_ORDER_CONSERVATIVE, fourth=FOURTH_ORDER_CONSERVATIVE
)
GAMMA = 5.0 / 3.0

config = SimulationConfig(
    gravity_config=GravityConfig(
        self_gravity=True,
        self_gravity_version=GRAVITY[args.gravity],
        poisson_manual_open_boundaries=True,
        work_flux_correction=bool(args.fct),
    ),
    dimensionality=3,
    box_size=4.0,
    num_cells=args.n,
    backend_config=BackendConfig(backend=NATIVE_JAX),
    positivity_config=PositivityConfig(clamp_in_estimates=False),
    dual_energy=bool(args.dual),
    **weno_variant_kwargs(),
    boundary_settings=BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    return_snapshots=True,
    num_snapshots=args.num_snapshots,
    snapshot_settings=SnapshotSettings(
        return_states=args.save_states,
        return_final_state=True,
        return_total_energy=True,
        return_internal_energy=True,
        return_kinetic_energy=True,
        return_gravitational_energy=True,
    ),
)
params = SimulationParams(
    t_end=args.t_end,
    C_cfl=0.4,
    minimum_density=1e-5,
    minimum_pressure=3e-6,
)
helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

radius = helper_data.r
rho = jnp.where(radius <= 1.0, 1.0 / (2 * jnp.pi * radius), 1e-4)
pressure = jnp.maximum((GAMMA - 1) * rho * args.e0, params.minimum_pressure)
zero = jnp.zeros_like(rho)
state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=zero,
    velocity_y=zero,
    velocity_z=zero,
    gas_pressure=pressure,
)
config = finalize_config(config, state.shape)

start = time.time()
snapshots = jax.block_until_ready(
    time_integration(state, config, params, registered_variables)
)
elapsed = time.time() - start

times = np.asarray(snapshots.time_points)
total = np.asarray(snapshots.total_energy)
internal = np.asarray(snapshots.internal_energy)
final = np.asarray(snapshots.final_state)
finite_mask = np.isfinite(total) & (times > 0)
t_reached = float(times[finite_mask].max()) if finite_mask.any() else 0.0
ok = bool(np.all(np.isfinite(final))) and t_reached >= 0.999 * args.t_end
energy_drift = float(np.nanmax(np.abs(total[finite_mask] - total[0])) / np.abs(total[0])) if finite_mask.any() else np.nan
face = weno_variant_name()
print(
    f"EVRARD face={face} tag={args.tag} n={args.n} x{args.precision} grav={args.gravity} "
    f"dual={args.dual} fct={args.fct} lab={os.environ.get('ASTX_LAB', '')} e0={args.e0}: {'COMPLETE' if ok else 'FAILED'} "
    f"t_reached={t_reached:.3f} max|dE|/|E0|={energy_drift:.3e} "
    f"min rho={np.nanmin(final[0]):.3e} min p={np.nanmin(final[registered_variables.pressure_index]):.3e} "
    f"wall={elapsed:.0f}s",
    flush=True,
)
os.makedirs("examples/scripts/validation/weno_stability/out", exist_ok=True)
np.savez(
    f"examples/scripts/validation/weno_stability/out/evrard_{face}{args.tag}_n{args.n}_x{args.precision}_{args.gravity}.npz",
    times=times, total=total, internal=internal,
    kinetic=np.asarray(snapshots.kinetic_energy),
    gravitational=np.asarray(snapshots.gravitational_energy),
    **(dict(states=np.asarray(snapshots.states)) if args.save_states else {}),
)
