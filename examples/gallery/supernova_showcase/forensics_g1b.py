# ↓───────────────────────────────────────────────────────────────────────↓
# Deeper forensics: velocity / thermodynamic profile through the g1
# runaway centroid, and the IC clump/knot/bubble multiplier at that spot.
# ↑───────────────────────────────────────────────────────────────────────↑
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np

BOX_SIZE = 7.0
N = 512
dx = BOX_SIZE / N
x = (np.arange(N) + 0.5) * dx - BOX_SIZE / 2

d = np.load("gate_hotbub_g1.npz")
rho = d["rho"]; press = d["press"]
vx = d["vx"]; vy = d["vy"]; vz = d["vz"]

idx = np.argwhere(rho > 1e4)
ic, jc, kc = np.round(idx.mean(axis=0)).astype(int)
print(f"centroid cell ({ic},{jc},{kc}) pos ({x[ic]:+.2f},{x[jc]:+.2f},{x[kc]:+.2f})")

# velocity in/around the runaway: homologous free expansion would be
# v ≈ r / t_age with v(1.5 pc) ≈ 11,344 km/s  → v(0.92) ≈ 7000 km/s
CODE_V_KMS = None
sl = np.s_[ic - 8:ic + 9, jc, kc]
r_line = np.sqrt(x[ic - 8:ic + 9] ** 2 + x[jc] ** 2 + x[kc] ** 2)
vr_line = (vx[sl] * x[ic - 8:ic + 9] + vy[sl] * x[jc] + vz[sl] * x[kc]) / np.maximum(r_line, 1e-9)
print("\nx-line through centroid (i, r, rho, p, p/rho, v_r[code]):")
for m, i in enumerate(range(ic - 8, ic + 9)):
    print(f"  {i:4d} r={r_line[m]:.3f}  rho={rho[i,jc,kc]:.3e}  p={press[i,jc,kc]:.3e}  "
          f"T~{press[i,jc,kc]/rho[i,jc,kc]:.3e}  vr={vr_line[m]:+.4f}")

vmag = np.sqrt(vx[idx[:,0],idx[:,1],idx[:,2]]**2 + vy[idx[:,0],idx[:,1],idx[:,2]]**2
               + vz[idx[:,0],idx[:,1],idx[:,2]]**2)
print(f"\nrunaway |v| code units: {vmag.min():.4f}-{vmag.max():.4f}")
# reference: max ejecta speed in code units at IC
vall = np.sqrt(vx**2 + vy**2 + vz**2)
print(f"global |v| max now: {vall.max():.4f}")

# how many cells are at/below tiny pressures (floored) near the clump?
box = np.s_[ic-16:ic+17, jc-16:jc+17, kc-16:kc+17]
pb = press[box]; rb = rho[box]
print(f"\n33^3 box around clump: rho[{rb.min():.2e},{rb.max():.2e}] "
      f"p[{pb.min():.2e},{pb.max():.2e}]")
print(f"  cells with T~p/rho < 1e-9: {(pb/rb < 1e-9).sum()} of {pb.size}")
