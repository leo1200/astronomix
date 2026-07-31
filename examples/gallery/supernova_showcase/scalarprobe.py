"""Track the passive-scalar range through a REAL run (adaptive dt, cooling,
the full iteration-level update chain) -- the conditions the low-CFL direct
_evolve_state_fd probe failed to reproduce."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import casa_orlando as CO
from astronomix import time_integration, SnapshotSettings

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--t-end", type=float, default=0.02)
ap.add_argument("--nsnap", type=int, default=11)
ap.add_argument("--no-cooling", action="store_true")
ap.add_argument("--no-pistons", action="store_true")
ap.add_argument("--cfl", type=float, default=None)
ap.add_argument("--no-bounds", action="store_true")
ap.add_argument("--no-subcycle", action="store_true")
a = ap.parse_args()

args = argparse.Namespace(profile='casa_1d_map150.npz', n=a.n, age=350.0,
    nsnap=a.nsnap, gpus=1, cooling=not a.no_cooling, limiter_alpha=None,
    clump_sigma=1.0, csm_sigma=0.4, shell=True, shell_radius=1.5,
    shell_density=20.0, pistons=not a.no_pistons, composition=True,
    save_state=None, ic_only=True)
state, config, params, rv, cu, hd, meta = CO.build(args)
if a.no_bounds:
    config = config._replace(passive_scalar_bounds=())
if a.no_subcycle:
    config = config._replace(max_passive_scalar_substeps=1)
config = config._replace(num_snapshots=a.nsnap,
    snapshot_settings=SnapshotSettings(return_states=True, return_final_state=True))
params = params._replace(t_end=a.t_end)
if a.cfl is not None:
    params = params._replace(C_cfl=a.cfl)
print(f"bounds={bool(config.passive_scalar_bounds)} maxsub={config.max_passive_scalar_substeps} n={a.n} cooling={not a.no_cooling} pistons={not a.no_pistons} "
      f"C_cfl={params.C_cfl} t_end={a.t_end} dtype={state.dtype}", flush=True)

snaps = time_integration(state, config, params, rv)
states = np.asarray(snaps.states); tp = np.asarray(snaps.time_points)
i0 = rv.passive_scalar_index
names = list(CO.SCALAR_NAMES) + ["entropy_initial","shocked_fraction",
                                 "time_since_shock","density_time"]
for k in range(states.shape[0]):
    cej = states[k][i0]; dtm = states[k][i0 + rv.num_passive_scalars - 1]
    print(f"  t={tp[k]:.5f}  C_ej [{np.nanmin(cej):+.4e},{np.nanmax(cej):+.4e}]  "
          f"density_time max {np.nanmax(dtm):+.4e}", flush=True)
