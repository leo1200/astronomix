import os, sys
pass  # GPU assigned by pq
pass
sys.path.insert(0, '/export/home/lstorcks/jf1uids/examples/gallery/supernova_showcase')
os.chdir('/export/home/lstorcks/jf1uids/examples/gallery/supernova_showcase')
import numpy as np, jax, jax.numpy as jnp
import argparse
import casa_orlando as CO
from astronomix._finite_difference._state_evolution._evolve_state import _evolve_state_fd

args = argparse.Namespace(profile='casa_1d_map150.npz', n=48, age=350.0, nsnap=2,
    gpus=1, cooling=False, limiter_alpha=None, clump_sigma=1.0, csm_sigma=0.4,
    shell=True, shell_radius=1.5, shell_density=20.0, pistons=True,
    composition=True, save_state=None, ic_only=True)
state, config, params, rv, cu, hd, meta = CO.build(args)
print("dtype", state.dtype, "num_vars", rv.num_vars, "scalar idx", rv.passive_scalar_index)
s = np.asarray(state)
print("IC nan:", np.isnan(s).sum())
names = list(CO.SCALAR_NAMES) + ["entropy_initial","shocked_fraction","time_since_shock","density_time"]
i0 = rv.passive_scalar_index
for k,nm in enumerate(names):
    a = s[i0+k]; print(f"  IC {nm:18s} [{a.min():+.4e},{a.max():+.4e}]")

st = state
dt = jnp.asarray(1e-4, dtype=state.dtype)
for step in range(1, 41):
    st = _evolve_state_fd(st, dt, params.gamma, config, params, hd, rv)
    a = np.asarray(st)
    if np.isnan(a).any():
        bad = [names[k] for k in range(len(names)) if np.isnan(a[i0+k]).any()]
        hydro_bad = np.isnan(a[:5]).any()
        print(f"STEP {step}: FIRST NaN. hydro_nan={hydro_bad} scalars_nan={bad}")
        for k,nm in enumerate(names):
            f = a[i0+k]; n=np.isnan(f).sum()
            print(f"   {nm:18s} nan={n:8d} finite range [{np.nanmin(f):+.4e},{np.nanmax(f):+.4e}]")
        break
    if step % 10 == 0:
        print(f"  step {step} ok; C_ej [{a[i0].min():+.3e},{a[i0].max():+.3e}] "
              f"rho [{a[0].min():.3e},{a[0].max():.3e}]")
else:
    print("no NaN in 40 steps")
