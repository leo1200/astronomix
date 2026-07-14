"""
One-figure summary of the TRML temperature-PDF study (N=128 fiducial).

2x2 panels:
  (1) Measured Lagrangian transition kernel p(dT/dt | T) with drift A(T) and
      the diffusion band A +/- sqrt(D/dt)  -- what the tracers give us.
  (2) The temperature PDF: Eulerian mass-weighted, Lagrangian tracers, Fokker-
      Planck P_hat, flux-driven transfer matrix, and the closed (J=0) transfer
      matrix -- the models reproduce the marginal ONLY with the flux.
  (3) The one-step kernel is non-Gaussian: row total-variation between the
      empirical kernel and the FP Gaussian built from the same A, D -- the
      dominant reason FP is not exact.
  (4) Chapman-Kolmogorov test vs a parametric Markov null: T carries memory
      (non-Markov), the second reason.

Run trml_tracers.py, fokker_planck_test.py, transfer_matrix_test.py first (any
preset); this reads their saved data/*.npz.  Figure -> figures/key_insights*.png
"""

import os
from math import erf

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PRESET = os.environ.get("TRML_PRESET", "fiducial")
SUFFIX = os.environ.get(
    "TRML_SUFFIX", {"quick": "_quick", "long64": "_long64", "fiducial": ""}[PRESET]
)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
data = np.load(f"data/trml_tracers{SUFFIX}.npz")
fp = np.load(f"data/fp_result{SUFFIX}.npz")
tm = np.load(f"data/tm_result{SUFFIX}.npz")

time_points = data["time_points"]
tracer_T = data["tracer_temperature"]
tracer_pos = data["tracer_position"]
tracer_generation = data["tracer_generation"] if "tracer_generation" in data else None
logT_edges = data["logT_bin_edges"]
t_sh = float(data["t_sh"])
L_x = float(data["L_x"]); L_y = float(data["L_y"]); L_z = float(data["L_z"])
steady_t_sh = float(data["steady_state_t_sh"])

logT_centers = 0.5 * (logT_edges[:-1] + logT_edges[1:])
T_centers = 10.0 ** logT_centers
dlogT = logT_edges[1] - logT_edges[0]

T_support_lo = float(fp["T_support_lo"]); T_support_hi = float(fp["T_support_hi"])
check0_l1 = float(fp["check0_l1"]); fp_l1 = float(fp["fp_l1"])
best_dt_tsh = float(fp["best_dt_tsh"])
eul_mass_pdf = fp["eul_mass_pdf"]; lag_pdf = fp["lag_pdf"]; recon = fp["recon_dMdlogT"]

# ---------------------------------------------------------------------------
# steady window + teleport masking (same logic as the analyses)
# ---------------------------------------------------------------------------
steady_idx = np.where(time_points >= steady_t_sh * t_sh)[0]
Ts = tracer_T[steady_idx]
ts = time_points[steady_idx]
xs = tracer_pos[steady_idx][:, :, 0]
ys = tracer_pos[steady_idx][:, :, 1]
zs = tracer_pos[steady_idx][:, :, 2]
M_snap, Np = Ts.shape
dt_snap = float(np.median(np.diff(ts)))


def _mi(c, L):
    raw = np.abs(c[1:] - c[:-1]); return np.minimum(raw, L - raw)


step_tel = np.zeros_like(zs, dtype=np.int32)
step_tel[1:] = ((_mi(xs, L_x) > 0.3 * L_x) | (_mi(ys, L_y) > 0.3 * L_y)
                | (np.abs(zs[1:] - zs[:-1]) > 0.3 * L_z)).astype(np.int32)
if tracer_generation is not None:
    gen = tracer_generation[steady_idx]
    step_tel[1:] = np.maximum(step_tel[1:], (gen[1:] != gen[:-1]).astype(np.int32))
cum_tel = np.cumsum(step_tel, axis=0)

# ---------------------------------------------------------------------------
# Panel 1 pieces: transition density + A(T), D(T) at the FP best lag
# ---------------------------------------------------------------------------
m_best = max(1, int(round(best_dt_tsh / (dt_snap / t_sh))))
dt_best = m_best * dt_snap
T0 = Ts[:-m_best].ravel()
rate = ((Ts[m_best:] - Ts[:-m_best]) / dt_best).ravel()
valid = ((cum_tel[m_best:] - cum_tel[:-m_best]) == 0).ravel() & np.isfinite(rate) & (T0 > 0)

# connected conditional moments -> A(T), D(T)
logT0 = np.log10(T0[valid]); dTk = (rate * dt_best).ravel()[valid]
cnt, _ = np.histogram(logT0, bins=logT_edges)
s1, _ = np.histogram(logT0, bins=logT_edges, weights=dTk)
s2, _ = np.histogram(logT0, bins=logT_edges, weights=dTk ** 2)
cs = np.where(cnt > 0, cnt, 1)
mean_dT = s1 / cs; var_dT = s2 / cs - mean_dT ** 2
A = np.where(cnt >= 30, mean_dT / dt_best, np.nan)
D = np.where(cnt >= 30, var_dT / dt_best, np.nan)
support = (T_centers >= T_support_lo) & (T_centers <= T_support_hi)
std_rate = np.sqrt(np.maximum(D, 0.0) / dt_best)
frame = np.isfinite(A) & np.isfinite(std_rate) & support
rate_max = float(np.nanmax(np.abs(A[frame]) + 3.5 * std_rate[frame]))
rate_edges = np.linspace(-rate_max, rate_max, 200)
counts2d, xe, ye = np.histogram2d(np.log10(T0[valid]), rate[valid],
                                  bins=[logT_edges, rate_edges])
colsum = counts2d.sum(axis=1, keepdims=True)
cond = counts2d / np.where(colsum > 0, colsum, 1.0)

# ---------------------------------------------------------------------------
# Panel 3 pieces: empirical vs FP-Gaussian one-step kernel (coarse bins)
# ---------------------------------------------------------------------------
NB = int(tm["NB"]); m0 = int(tm["m0"])
A_bin = tm["A_bin"]; D_bin = tm["D_bin"]; tm_T = tm["tm_T"]
tm_edges = np.linspace(np.log10(T_support_lo), np.log10(T_support_hi), NB + 1)
tm_dlogT = tm_edges[1] - tm_edges[0]


def _to_bin(T):
    return np.clip(np.floor((np.log10(T) - tm_edges[0]) / tm_dlogT).astype(int), 0, NB - 1)


Ta = Ts[:-m0]; Tb = Ts[m0:]
vv = ((cum_tel[m0:] - cum_tel[:-m0]) == 0) & np.isfinite(Ta) & np.isfinite(Tb) & (Ta > 0) & (Tb > 0)
ci, cj = _to_bin(Ta[vv]), _to_bin(Tb[vv])
Kc = np.zeros((NB, NB)); np.add.at(Kc, (ci, cj), 1.0)
rc = Kc.sum(axis=1)
M_emp = Kc / np.where(rc > 0, rc, 1.0)[:, None]
dt0 = m0 * dt_snap
edges_T = 10.0 ** tm_edges
M_fp = np.zeros((NB, NB))
for b in range(NB):
    if not np.isfinite(A_bin[b]) or not np.isfinite(D_bin[b]) or D_bin[b] <= 0:
        M_fp[b] = M_emp[b]; continue
    mu = tm_T[b] + A_bin[b] * dt0; sig = np.sqrt(D_bin[b] * dt0)
    cdf = 0.5 * (1 + np.array([erf((e - mu) / (np.sqrt(2) * sig)) for e in edges_T]))
    p = np.clip(np.diff(cdf), 0, None); s = p.sum()
    M_fp[b] = p / s if s > 0 else M_emp[b]
kernel_tv = np.where(rc >= 30, 0.5 * np.abs(M_emp - M_fp).sum(axis=1), np.nan)

# transfer-matrix marginals (coarse)
tm_dlogT_c = np.log10(tm_T[1]) - np.log10(tm_T[0])
tm_flux = tm["tm_pi_flux"] / tm_dlogT_c
tm_closed = tm["tm_pi_closed"] / tm_dlogT_c

# ---------------------------------------------------------------------------
# assemble the figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
(axK, axP), (axG, axC) = axes

# (1) transition kernel
mesh = axK.pcolormesh(10.0 ** xe, ye, cond.T, cmap="magma", shading="auto",
                      norm=LogNorm(vmin=1e-3, vmax=max(cond.max(), 1e-2)))
axK.plot(T_centers, A, "c-", lw=2.2, label=r"drift $A(T)=\langle\delta T\rangle/\Delta t$")
axK.plot(T_centers, A + std_rate, "c--", lw=1.2, label=r"$A(T)\pm\sqrt{D(T)/\Delta t}$ (diffusion)")
axK.plot(T_centers, A - std_rate, "c--", lw=1.2)
axK.axhline(0, color="w", ls=":", lw=0.7)
axK.set_xscale("log"); axK.set_xlim(T_support_lo, T_support_hi); axK.set_ylim(-rate_max, rate_max)
axK.set_xlabel(r"$T(t)$"); axK.set_ylabel(r"$\delta T/\Delta t$ (Lagrangian rate)")
axK.set_title(fr"(1) Measured transition kernel $p(\delta T/\Delta t\,|\,T)$"
              fr"  ($\Delta t/t_{{sh}}={dt_best/t_sh:.3f}$)")
axK.legend(fontsize=8, loc="lower left", framealpha=0.6)
fig.colorbar(mesh, ax=axK, label=r"$p(\delta T/\Delta t\,|\,T)$")

# (2) marginals
axP.plot(T_centers, eul_mass_pdf, lw=3, color="tab:blue", label="Eulerian mass-weighted")
axP.plot(T_centers, lag_pdf, "--", lw=2, color="tab:orange", label="Lagrangian tracers")
axP.plot(T_centers, recon, ":", lw=2.6, color="black", label=fr"Fokker-Planck $\widehat P$ (L1={fp_l1:.3f})")
axP.plot(tm_T, tm_flux, "-o", lw=1.8, ms=3, color="tab:green", label="transfer matrix, flux-driven")
axP.plot(tm_T, tm_closed, "-s", lw=1.1, ms=3, color="tab:red", alpha=0.6,
         label=r"transfer matrix, closed ($J{=}0$): collapses")
axP.set_xscale("log"); axP.set_yscale("log")
axP.set_xlim(T_support_lo, T_support_hi)
axP.set_ylim(bottom=max(1e-4, eul_mass_pdf[eul_mass_pdf > 0].min() * 0.5))
axP.set_xlabel("T"); axP.set_ylabel(r"$dM/d\log T$ (normalized)")
axP.set_title(f"(2) Temperature PDF: models reproduce it -- but only with flux "
              f"(Check0 L1={check0_l1:.3f})")
axP.legend(fontsize=8)

# (3) kernel non-Gaussianity
axG.plot(tm_T, kernel_tv, "o-", color="tab:purple")
axG.axhline(float(tm["mean_kernel_tv"]), color="0.5", ls="--",
            label=fr"mean = {float(tm['mean_kernel_tv']):.3f}")
axG.set_xscale("log"); axG.set_xlim(T_support_lo, T_support_hi); axG.set_ylim(bottom=0)
axG.set_xlabel(r"$T(t)$")
axG.set_ylabel(r"row TV$(M_{\rm empirical},\,M_{\rm FP\ Gaussian})$")
axG.set_title("(3) The one-step kernel is non-Gaussian\n(dominant reason FP is not exact)")
axG.legend(fontsize=9)

# (4) Chapman-Kolmogorov
axC.plot(tm["ck_lags_tsh"], tm["ck_signal"], "o-", lw=2, color="tab:red",
         label=r"measured $TV(M(\Delta t)^k, M(k\Delta t))$")
axC.plot(tm["ck_lags_tsh"], tm["ck_null"], "s--", lw=2, color="0.5",
         label="Markov null (parametric bootstrap)")
axC.set_xlabel(r"total lag $k\,\Delta t/t_{sh}$")
axC.set_ylabel("stationary-weighted row TV")
axC.set_title("(4) Temperature is non-Markov\n(signal >> null; the second reason)")
axC.set_ylim(bottom=0); axC.legend(fontsize=9)

fig.suptitle("TRML temperature PDF from Lagrangian tracers: Fokker-Planck vs transfer matrix "
             f"(N={'128' if SUFFIX=='' else SUFFIX.strip('_')})", fontsize=14, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = f"figures/key_insights{SUFFIX}.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
