"""Restart a turbulence forensic state WITHOUT forcing and watch it for a short time.

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/turb_restart.py out/turb/lastgood_bare64_current.npz
"""

# general
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("state_file")
parser.add_argument("--dt-total", type=float, default=0.05)
parser.add_argument("--nsnap", type=int, default=50)
parser.add_argument("--mturb", type=float, default=10.0)
parser.add_argument("--mhd", type=int, default=1)
parser.add_argument("--cfl", type=float, default=1.5)
parser.add_argument("--rhomin", type=float, default=1e-10)
parser.add_argument("--precision", type=int, default=32)
parser.add_argument("--backend", choices=["native", "pallas"], default="native")
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
if args.precision == 64:
    os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import numpy as np

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs, weno_variant_name
import jax
import jax.numpy as jnp

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
    SnapshotSettings,
    finalize_config,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    ISOTHERMAL,
)

data = np.load(args.state_file)
state = jnp.asarray(data["state"], dtype=jnp.float64 if args.precision == 64 else jnp.float32)
t0 = float(data["t"])
n = state.shape[-1]
periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
config = SimulationConfig(
    equation_of_state=ISOTHERMAL,
    dimensionality=3,
    num_cells=n,
    box_size=1.0,
    mhd=bool(args.mhd),
    backend_config=BackendConfig(backend=NATIVE_JAX if args.backend == "native" else PALLAS),
    boundary_settings=BoundarySettings(periodic, periodic, periodic),
    positivity_config=PositivityConfig(clamp_in_estimates=False),
    return_snapshots=True,
    num_snapshots=args.nsnap,
    **weno_variant_kwargs(),
    snapshot_settings=SnapshotSettings(return_states=True),
)
params = SimulationParams(
    C_cfl=args.cfl,
    isothermal_sound_speed=1.0 / args.mturb,
    t_end=args.dt_total,
    minimum_density=args.rhomin,
    minimum_pressure=1e-10,
)
registered_variables = get_registered_variables(config)
config = finalize_config(config, state.shape)
result = time_integration(state, config, params, registered_variables)
states = np.asarray(result.states)
times = np.asarray(result.time_points)
rho = states[:, 0]
speed = np.sqrt((states[:, 1:4] ** 2).sum(1))
for k in range(len(times)):
    bad = not np.all(np.isfinite(states[k]))
    print(f"t0+{times[k]:.4f} min rho={np.nanmin(rho[k]):.3e} max|v|={np.nanmax(speed[k]):.3f} "
          f"argmin={np.unravel_index(np.nanargmin(rho[k]), rho[k].shape) if not bad else '-'} "
          f"{'NaN' if bad else ''}", flush=True)
    if bad:
        break
tag = "_" + weno_variant_name()
finite = [k for k in range(len(times)) if np.all(np.isfinite(states[k])) and times[k] > 0 or k == 0]
last = max(finite)
# keep only the snapshots around the end (full 128^3 series are tens of GB)
keep = slice(max(0, last - 3), min(last + 2, len(times)))
np.savez(args.state_file.replace(".npz", f"_restart{tag}.npz"), states=states[keep], times=times[keep])
for k in range(max(0, last - 3), min(last + 2, len(times))):
    s = states[k]
    if not np.all(np.isfinite(s)):
        bad = np.argwhere(~np.isfinite(s).all(axis=0))
        print(f"snapshot {k}: {len(bad)} non-finite cells, first at {tuple(bad[0])}")
        continue
    rho = s[0]; speed = np.sqrt((s[1:4] ** 2).sum(0)); b = np.sqrt((s[4:7] ** 2).sum(0))
    i = np.unravel_index(np.argmax(speed), speed.shape)
    j = np.unravel_index(np.argmin(rho), rho.shape)
    print(f"snapshot {k} t0+{times[k]:.5f}: max|v|={speed.max():.3f} at {i} (rho={rho[i]:.3e}, |B|={b[i]:.3f}); "
          f"min rho={rho.min():.3e} at {j} (|v|={speed[j]:.3f}); max|B|={b.max():.3f}")
