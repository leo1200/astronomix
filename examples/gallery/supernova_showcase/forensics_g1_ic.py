# ↓───────────────────────────────────────────────────────────────────────↓
# Reconstruct the G1 initial condition on CPU and inspect the runaway
# location (cell 321,254,241 / r=0.92 pc): was the converging-velocity
# cold ridge already seeded at t=0 (ambient-drag velocity deficit in
# knot-scale voids), and how dense/cold was the seed?
# ↑───────────────────────────────────────────────────────────────────────↑
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

import cassiopeia_realistic as cr

state, config, params, rv, cu, age = cr.build(
    512, 0.08, cooling=True, dual_energy=True, jet=True,
    ni_bubbles=True, knot_sigma=0.25, limiter_alpha=4.0,
)
s = np.asarray(state)
rho = s[rv.density_index]
p = s[rv.pressure_index]
vx = s[rv.velocity_index.x]
vy = s[rv.velocity_index.y]
vz = s[rv.velocity_index.z]

N = 512
dx = 7.0 / N
x = (np.arange(N) + 0.5) * dx - 3.5
ic, jc, kc = 321, 254, 241

print("IC x-line through the G1 runaway centroid (i, r, rho, T~p/rho, vr, vr/r):")
ii = np.arange(ic - 8, ic + 9)
r_line = np.sqrt(x[ii] ** 2 + x[jc] ** 2 + x[kc] ** 2)
for m, i in enumerate(ii):
    vr = (vx[i, jc, kc] * x[i] + vy[i, jc, kc] * x[jc] + vz[i, jc, kc] * x[kc]) / max(r_line[m], 1e-9)
    print(f"  {i:4d} r={r_line[m]:.3f}  rho={rho[i,jc,kc]:.3e}  T~{p[i,jc,kc]/rho[i,jc,kc]:.3e}"
          f"  vr={vr:+.4f}  vr/r={vr/r_line[m]:.4f}")

# global scale of the seeded velocity-shear: vr/r spread inside the ejecta
r3 = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
inner = (r3 > 0.3) & (r3 < 1.2)
vr3 = (vx * x[:, None, None] + vy * x[None, :, None] + vz * x[None, None, :]) / np.maximum(r3, 1e-9)
hom = vr3 / np.maximum(r3, 1e-9)
print(f"\nvr/r inside ejecta (0.3<r<1.2): median {np.median(hom[inner]):.4f}  "
      f"p5 {np.percentile(hom[inner],5):.4f}  p95 {np.percentile(hom[inner],95):.4f}  "
      f"min {hom[inner].min():.4f}")
# how big is the ambient there vs ejecta?
print(f"rho at centroid {rho[ic,jc,kc]:.3e}; 17-cell-line min {rho[ii,jc,kc].min():.3e} "
      f"max {rho[ii,jc,kc].max():.3e}")
