# ↓───────────────────────────────────────────────────────────────────────↓
# Forensics on the gate_hotbub_g1.npz blow-up state (isobaric bubbles +
# knots + cooling limiter, aborted t=0.0228): locate the runaway cells and
# test them against the Ni-bubble geometry and the jet axis.
# ↑───────────────────────────────────────────────────────────────────────↑
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import jax
import jax.numpy as jnp

BOX_SIZE = 7.0
EJECTA_RADIUS = 1.5
SEED = 7
JET_AXIS = np.array([-1.0, 1.0, 0.25])
JET_AXIS /= np.linalg.norm(JET_AXIS)

d = np.load("gate_hotbub_g1.npz")
rho = d["rho"]; press = d["press"]
N = rho.shape[0]
dx = BOX_SIZE / N
x = (np.arange(N) + 0.5) * dx - BOX_SIZE / 2

# ↓ runaway cells: density far beyond any physical value in this problem ↓
thr = 1e4
idx = np.argwhere(rho > thr)
print(f"N={N} rho[{rho.min():.3e},{rho.max():.3e}] press[{press.min():.3e},{press.max():.3e}]")
print(f"runaway cells (rho>{thr:g}): {len(idx)}")
if len(idx):
    pos = np.stack([x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]], axis=1)
    r = np.linalg.norm(pos, axis=1)
    cosang = (pos @ JET_AXIS) / np.maximum(r, 1e-9)
    ang = np.degrees(np.arccos(np.clip(np.abs(cosang), 0, 1)))
    print(f"  r: {r.min():.2f}-{r.max():.2f} pc  (median {np.median(r):.2f})")
    print(f"  angle to jet axis: {ang.min():.0f}-{ang.max():.0f} deg (median {np.median(ang):.0f})")
    print(f"  ix range: {idx[:,0].min()}-{idx[:,0].max()} (4-way shard seams at 128/256/384)")
    com = pos.mean(axis=0)
    print(f"  centroid: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})  r={np.linalg.norm(com):.2f}")
    spread = np.linalg.norm(pos - com, axis=1).max()
    print(f"  max spread from centroid: {spread:.3f} pc  ({spread/dx:.1f} cells)")

# ↓ bubble geometry: same key sequence as cassiopeia_realistic.main ↓
keys = jax.random.split(jax.random.PRNGKey(SEED), 4)
bkeys = jax.random.split(keys[3], 3)
centers = np.asarray(jax.random.ball(bkeys[0], d=3, shape=(5,)) * 0.6 * EJECTA_RADIUS)
lo, hi = 0.15, 0.35
radii = np.asarray((jax.random.uniform(bkeys[1], (5,)) * (hi - lo) + lo) * EJECTA_RADIUS)
print("\nbubbles (center, radius, |center| + radius = wall apex r):")
for c, rad in zip(centers, radii):
    print(f"  ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})  R={rad:.2f}  wall spans r "
          f"{max(np.linalg.norm(c)-rad,0):.2f}-{np.linalg.norm(c)+rad:.2f}")
if len(idx):
    dists = np.linalg.norm(pos[:, None, :] - centers[None, :, :], axis=2)
    wall_off = np.abs(dists - radii[None, :])
    nearest = wall_off.min(axis=1)
    which = wall_off.argmin(axis=1)
    print(f"\nrunaway distance to nearest bubble WALL: "
          f"{nearest.min():.3f}-{nearest.max():.3f} pc (median {np.median(nearest):.3f}; "
          f"wall sigma = 0.3*R ≈ {0.3*radii.mean():.2f})")
    print(f"nearest bubble index counts: {np.bincount(which, minlength=5)}")

# ↓ context: where is cold dense gas overall (crush fodder)? ↓
Tproxy = press / np.maximum(rho, 1e-30)
sel = rho > 100
print(f"\ncells rho>100: {sel.sum()}  their T-proxy p/rho: "
      f"{Tproxy[sel].min():.3e}-{Tproxy[sel].max():.3e}" if sel.sum() else "\nno rho>100 cells")
