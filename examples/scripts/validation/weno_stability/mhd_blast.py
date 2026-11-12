"""2D low-beta MHD blast wave: the standard positivity test for ideal MHD.

Balsara & Spicer (1999) set-up as used by Christlieb et al. (2015, SIAM J. Sci.
Comput. 37, A1825) and Wu & Shu (2018): rho = 1, v = 0, B = (100/sqrt(4 pi), 0, 0),
p = 1000 for r < 0.1 and 0.1 outside, gamma = 1.4, [-0.5, 0.5]^2, t = 0.01.
The ambient plasma beta is 2.5e-4. Without positivity preservation the
pressure goes negative within a few steps.

    WENO_VARIANT=pp PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/mhd_blast.py --n 200 --cfl 0.75
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--cfl", type=float, default=0.75)
parser.add_argument("--field", type=float, default=100.0 / (4.0 * 3.141592653589793) ** 0.5)
parser.add_argument("--t-end", type=float, default=0.01)
parser.add_argument("--precision", type=int, default=64)
parser.add_argument("--nsnap", type=int, default=20)
parser.add_argument("--tag", default="")
args = parser.parse_args()

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
if args.precision == 64:
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
    FDSolverConfig,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    initialize_interface_fields,
    time_integration,
)
from astronomix.option_classes.simulation_config import IDEAL_GAS

GAMMA = 1.4
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
config = SimulationConfig(
    equation_of_state=IDEAL_GAS,
    dimensionality=2,
    num_cells=args.n,
    box_size=1.0,
    mhd=True,
    backend_config=BackendConfig(backend=NATIVE_JAX),
    boundary_settings=BoundarySettings(periodic, periodic, periodic),
    positivity_config=PositivityConfig(clamp_in_estimates=False),
    return_snapshots=False,
    activate_snapshot_callback=True,
    num_snapshots=args.nsnap,
    solver_config=FDSolverConfig(**weno_variant_kwargs()),
)
params = SimulationParams(C_cfl=args.cfl, gamma=GAMMA, t_end=args.t_end, minimum_density=1e-30, minimum_pressure=1e-30)
registered_variables = get_registered_variables(config)

centres = (jnp.arange(args.n) + 0.5) / args.n - 0.5
x, y = jnp.meshgrid(centres, centres, indexing="ij")
radius = jnp.sqrt(x**2 + y**2)
density = jnp.ones_like(x)
pressure = jnp.where(radius < 0.1, 1000.0, 0.1)
zero = jnp.zeros_like(x)
field_x = jnp.full_like(x, args.field)
bx_face, by_face, bz_face = initialize_interface_fields(field_x, zero, zero, dimensionality=2)
state = construct_primitive_state(
    config=config, registered_variables=registered_variables, density=density,
    velocity_x=zero, velocity_y=zero, gas_pressure=pressure,
    magnetic_field_x=field_x, magnetic_field_y=zero, magnetic_field_z=zero,
    interface_magnetic_field_x=bx_face, interface_magnetic_field_y=by_face, interface_magnetic_field_z=bz_face,
)
config = finalize_config(config, state.shape)
records = []
good_states = []


def diagnostics(time, state, registered_variables):
    rho = state[registered_variables.density_index]
    p = state[registered_variables.pressure_index]
    stats = jnp.stack([time, jnp.min(rho), jnp.min(p), jnp.max(p),
                       jnp.any(~jnp.isfinite(state)).astype(jnp.float64 if args.precision == 64 else jnp.float32)])
    jax.debug.callback(lambda s: records.append([float(v) for v in np.asarray(s)]), stats)
    if os.environ.get("BLAST_DUMP"):
        def keep(s, full):
            if np.isfinite(np.asarray(full)).all() and float(np.asarray(full)[registered_variables.pressure_index].min()) > 0:
                good_states.append((float(np.asarray(s)[0]), np.asarray(full)))
                del good_states[:-3]
        jax.debug.callback(keep, stats, state)
    if os.environ.get("BLAST_VERBOSE"):
        jax.debug.print("t={t:.6f} min rho={r:.3e} min p={p:.3e} max p={q:.3e} bad={b}", t=time, r=stats[1], p=stats[2], q=stats[3], b=stats[4])


time_integration(state, config, params, registered_variables, diagnostics)
records = np.array(sorted(records))
if os.environ.get("BLAST_DUMP"):
    np.savez(os.environ["BLAST_DUMP"], states=np.stack([g[1] for g in good_states]),
             times=np.array([g[0] for g in good_states]))
beta = 2 * 0.1 / args.field**2
bad = records[:, 4] > 0
t_last = records[~bad, 0].max() if (~bad).any() else 0.0
verdict = "COMPLETE" if (not bad.any() and t_last >= 0.999 * args.t_end) else f"FAILED at t={t_last:.4g}"
print(f"BLAST [{weno_variant_name()}{args.tag}] N={args.n} cfl={args.cfl} beta_ambient={beta:.2e}: {verdict}; "
      f"min rho={np.nanmin(records[:, 1]):.3e} min p={np.nanmin(records[:, 2]):.3e} max p={np.nanmax(records[:, 3]):.3e}",
      flush=True)
