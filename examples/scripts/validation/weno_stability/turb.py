"""Driven supersonic turbulence (HOW-MHD ISM case) with every protection switchable.

Mirrors ``examples/scripts/forward/mhd/turbulence/paper_turbulence.py`` (OU
forcing, v_rms ~ 1 normalisation, a = 1 / M_turb, B along z), but defaults to
NO stabilisation at all and records min(rho), max|v| and the first NaN through
a snapshot callback. The last finite state before a blow-up is written to disk
for forensics.

    PYTHONPATH=. python examples/scripts/validation/weno_stability/turb.py --N 64 --tag bare
"""

# general
import argparse
import os
import time as walltime

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=64)
parser.add_argument("--mturb", type=float, default=10.0)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--eos", choices=["iso", "adiabatic"], default="iso")
parser.add_argument("--mhd", type=int, default=1)
parser.add_argument("--cfl", type=float, default=1.5)
parser.add_argument("--rhomin", type=float, default=1e-10)
parser.add_argument("--tcross", type=float, default=5.0)
parser.add_argument("--nsnap", type=int, default=100)
parser.add_argument("--backend", choices=["native", "pallas"], default="native")
parser.add_argument("--precision", type=int, default=32)
parser.add_argument("--step", choices=["none", "floor"], default="none")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--clamp", type=int, default=1,
                    help="PositivityConfig.clamp_in_estimates (0 = no read-only or step-end clamps)")
parser.add_argument("--pmin", type=float, default=1e-10)
parser.add_argument("--save-every", type=int, default=0,
                    help="dump the full state every this many snapshots (0 = never)")
parser.add_argument("--state-dir", default="/export/data/lstorcks/weno_stability")
parser.add_argument("--gpus", type=int, default=1, help="devices to shard the x axis over")
parser.add_argument("--donate", type=int, default=0, help="config.donate_state")
parser.add_argument("--stop-on-negative-p", type=int, default=0,
                    help="save the last positive and the first negative-pressure state, then exit")
parser.add_argument("--tag", required=True)
args = parser.parse_args()

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=args.gpus)
# NVLS multicast hangs collectives on these nodes (see casa multi-GPU notes)
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
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
    NATIVE_JAX,
    PALLAS,
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
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    ISOTHERMAL,
    POSITIVITY_HARD_FLOOR,
    POSITIVITY_NONE,
)

POSITIVITY_MODES = dict(none=POSITIVITY_NONE, floor=POSITIVITY_HARD_FLOOR)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "turb")
os.makedirs(OUT_DIR, exist_ok=True)

rho0 = 1.0
gamma = 5.0 / 3.0
sound_speed = 1.0 / args.mturb
adiabatic = args.eos == "adiabatic"
pressure0 = rho0 * sound_speed**2 / gamma if adiabatic else None
thermal_pressure = pressure0 if adiabatic else sound_speed**2 * rho0
magnetic_field0 = float(np.sqrt(2.0 * thermal_pressure / args.beta))
periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)

config = SimulationConfig(
    equation_of_state=IDEAL_GAS if adiabatic else ISOTHERMAL,
    dimensionality=3,
    num_cells=args.N,
    box_size=1.0,
    mhd=bool(args.mhd),
    random_seed=args.seed,
    backend_config=BackendConfig(backend=NATIVE_JAX if args.backend == "native" else PALLAS),
    boundary_settings=BoundarySettings(periodic, periodic, periodic),
    turbulent_forcing_config=TurbulentForcingConfig(
        turbulent_forcing=True, ou_forcing=True,
    ),
    positivity_config=PositivityConfig(
        per_step_mode=POSITIVITY_MODES[args.step],
        clamp_in_estimates=bool(args.clamp),
    ),
    return_snapshots=False,
    activate_snapshot_callback=True,
    num_snapshots=args.nsnap,
    donate_state=bool(args.donate),
    **weno_variant_kwargs(),
)
t_cross = 0.5
t_c_value = t_cross
params = SimulationParams(
    C_cfl=args.cfl,
    gamma=gamma,
    isothermal_sound_speed=sound_speed,
    t_end=args.tcross * t_cross,
    turbulent_forcing_params=TurbulentForcingParams(
        forcing_amplitude=3.5,
        correlation_time=0.5,
        forcing_wavenumber=3.0 * np.pi,
    ),
    minimum_density=args.rhomin,
    minimum_pressure=args.pmin,
)

registered_variables = get_registered_variables(config)
density = jnp.full((args.N,) * 3, rho0, dtype=jnp.float64 if args.precision == 64 else jnp.float32)
zero = jnp.zeros_like(density)
initial = dict(
    config=config, registered_variables=registered_variables, density=density,
    velocity_x=zero, velocity_y=zero, velocity_z=zero,
)
if args.mhd:
    magnetic_z = jnp.full_like(density, magnetic_field0)
    bx_face, by_face, bz_face = initialize_interface_fields(zero, zero, magnetic_z)
    initial.update(
        magnetic_field_x=zero, magnetic_field_y=zero, magnetic_field_z=magnetic_z,
        interface_magnetic_field_x=bx_face, interface_magnetic_field_y=by_face,
        interface_magnetic_field_z=bz_face,
    )
if adiabatic:
    initial["gas_pressure"] = jnp.full_like(density, pressure0)
initial_state = construct_primitive_state(**initial)
config = finalize_config(config, initial_state.shape)
sharding = None
if args.gpus > 1:
    from jax.sharding import AxisType, PartitionSpec
    # Auto axes: the solver's with_sharding_constraint rejects jax 0.10's Explicit default
    mesh = jax.make_mesh((args.gpus,), ("x",), axis_types=(AxisType.Auto,))
    sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(None, "x"))
    jax.config.update("jax_use_shardy_partitioner", False)
    initial_state = jax.device_put(initial_state, sharding)

density_index = registered_variables.density_index
vx, vy, vz = (registered_variables.velocity_index.x, registered_variables.velocity_index.y,
              registered_variables.velocity_index.z)
face = weno_variant_name()
lab = os.environ.get("ASTX_LAB", "")
label = f"{args.tag} face={face} lab={lab}"
records = []
last_good = {"state": None, "t": None}
first_nan = {"t": None}


def diagnostics(time, state, registered_variables):
    rho = state[density_index]
    speed = jnp.sqrt(state[vx] ** 2 + state[vy] ** 2 + state[vz] ** 2)
    if adiabatic:
        pressure_field = state[registered_variables.pressure_index]
    else:
        pressure_field = rho * sound_speed**2
    stats = jnp.stack([
        time, jnp.min(rho), jnp.max(rho), jnp.max(speed),
        jnp.sqrt(jnp.mean(speed**2)),
        jnp.any(~jnp.isfinite(state)).astype(jnp.float32),
        jnp.min(pressure_field),
        # cells sitting at (or below) a floor: zero means no floor ever acted
        jnp.sum(rho <= 1.0001 * args.rhomin).astype(jnp.float32),
        jnp.sum(pressure_field <= 1.0001 * args.pmin).astype(jnp.float32) if adiabatic else jnp.float32(0.0),
    ])

    def host(stats_array, full_state):
        values = [float(x) for x in np.asarray(stats_array)]
        t, rho_min, rho_max, v_max, v_rms, nan = values[:6]
        p_min, n_rho_floor, n_p_floor = values[6:]
        records.append((t, rho_min, rho_max, v_max, v_rms, nan, p_min, n_rho_floor, n_p_floor))
        if nan > 0 or rho_min <= 0:
            if first_nan["t"] is None:
                first_nan["t"] = t
        elif args.stop_on_negative_p and adiabatic and p_min < 0:
            os.makedirs(args.state_dir, exist_ok=True)
            np.save(os.path.join(args.state_dir, f"{args.tag}_lastpositive.npy"), last_good["state"])
            np.save(os.path.join(args.state_dir, f"{args.tag}_firstnegative.npy"), np.asarray(full_state))
            print(f"NEGATIVE-P [{label}] first at t/tc={t/t_c_value:.4f} (last positive t/tc="
                  f"{last_good['t']/t_c_value:.4f}), min p={p_min:.3e}; states saved", flush=True)
            os._exit(0)
        else:
            last_good["state"] = np.asarray(full_state)
            last_good["t"] = t
            snapshot_index = len(records) - 1
            if args.save_every and snapshot_index % args.save_every == 0:
                os.makedirs(args.state_dir, exist_ok=True)
                np.save(os.path.join(args.state_dir, f"{args.tag}_snap{snapshot_index:03d}.npy"),
                        last_good["state"].astype(np.float32))
        print(f"[{label}] t/tc={t/t_cross:.3f} min_rho={rho_min:.3e} max_rho={rho_max:.3e} "
              f"max|v|={v_max:.3f} v_rms={v_rms:.3f} min_p={p_min:.3e} "
              f"at_floor(rho,p)=({int(n_rho_floor)},{int(n_p_floor)}) bad={int(nan > 0 or rho_min <= 0)}", flush=True)

    jax.debug.callback(host, stats, state)


start = walltime.time()
time_integration(initial_state, config, params, registered_variables, diagnostics, sharding=sharding)
elapsed = walltime.time() - start
records = np.array(sorted(records))
np.savetxt(os.path.join(OUT_DIR, f"diag_{args.tag}.txt"), records,
           header="t min_rho max_rho max_v v_rms bad min_p n_rho_at_floor n_p_at_floor")
t_last = float(np.nanmax(records[:, 0])) if len(records) else 0.0
if first_nan["t"] is not None:
    verdict = f"FAILED (NaN) after t/tc={t_last / t_cross:.3f}"
elif t_last < 0.999 * params.t_end:
    verdict = f"ABORTED (dt collapse) at t/tc={t_last / t_cross:.3f}"
else:
    verdict = "COMPLETE"
print(f"RESULT [{label}] N={args.N} M={args.mturb} beta={args.beta} eos={args.eos} cfl={args.cfl} "
      f"rhomin={args.rhomin}: {verdict}; min rho over run={np.nanmin(records[:,1]):.3e} "
      f"max|v|={np.nanmax(records[:,3]):.2f} min p over run={np.nanmin(records[:,6]):.3e} "
      f"floor hits (rho,p)=({int(np.nansum(records[:,7]))},{int(np.nansum(records[:,8]))}) wall={elapsed:.0f}s", flush=True)
if first_nan["t"] is not None and last_good["state"] is not None:
    np.savez(os.path.join(OUT_DIR, f"lastgood_{args.tag}.npz"), state=last_good["state"],
             t=last_good["t"])
