"""Cost of the passive-scalar sub-cycling: steps/s with scalars off, on with the
substep count forced to 1, and on with it derived from the flow."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import casa_orlando as CO
from astronomix import time_integration, SnapshotSettings

def run(composition, max_sub, t_end=0.004, n=128):
    args = argparse.Namespace(profile='casa_1d_map150.npz', n=n, age=350.0, nsnap=2,
        gpus=1, cooling=True, limiter_alpha=None, clump_sigma=1.0, csm_sigma=0.4,
        shell=True, shell_radius=1.5, shell_density=20.0, pistons=True,
        composition=composition, save_state=None, ic_only=True)
    state, config, params, rv, cu, hd, meta = CO.build(args)
    config = config._replace(max_passive_scalar_substeps=max_sub)
    params = params._replace(t_end=t_end)
    snaps = time_integration(state, config, params, rv)   # warm up + compile
    jax.block_until_ready(snaps)
    t0 = time.perf_counter()
    snaps = time_integration(state, config, params, rv)
    jax.block_until_ready(snaps)
    return time.perf_counter() - t0

for label, comp, mx in (("no scalars", False, 1), ("scalars, no subcycle", True, 1),
                        ("scalars, subcycle<=8", True, 8)):
    dtv = run(comp, mx)
    print(f"{label:24s} {dtv:7.2f} s", flush=True)
