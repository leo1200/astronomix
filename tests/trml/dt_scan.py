"""
How the diffusion/Markov picture depends on the measurement lag dt.

Everything the FP/transfer-matrix model assumes is a statement measured at a lag
dt, and each should improve as dt coarse-grains past the turbulent correlation
time tau_c (ballistic -> diffusive; memory washed out; increments Gaussianize):

  (1) drift A(T;dt) and diffusion D(T;dt) -- A should be ~lag-independent
      (robust estimation), D should rise from 0 (ballistic) to a plateau.
  (2) kernel non-Gaussianity: mean row-TV( empirical M(dt), Gaussian(A,D) )
      -- should DECREASE with dt (CLT: sums of sub-increments Gaussianize).
  (3) non-Markovianity: CK divergence TV( M(dt)^2, M(2dt) ) vs a per-dt
      parametric Markov null -- the memory (signal above null) should DECREASE
      with dt as dt passes tau_c.
  (4) the FP reconstruction error L1(dt) (loaded) -- its minimum is the
      "validity window"; it should sit where (2) and (3) are smallest.

Reads the saved fiducial tracer data + fp_result.  Figure -> figures/dt_scan*.
"""
import os
from math import erf

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRESET = os.environ.get("TRML_PRESET", "fiducial")
SUFFIX = os.environ.get(
    "TRML_SUFFIX", {"quick": "_quick", "long64": "_long64", "fiducial": ""}[PRESET]
)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)

data = np.load(f"data/trml_tracers{SUFFIX}.npz")
fp = np.load(f"data/fp_result{SUFFIX}.npz")
tp = data["time_points"]; T = data["tracer_temperature"]; pos = data["tracer_position"]
gen = data["tracer_generation"] if "tracer_generation" in data else None
t_sh = float(data["t_sh"]); Lx = float(data["L_x"]); Ly = float(data["L_y"]); Lz = float(data["L_z"])
st = float(data["steady_state_t_sh"])
xi = float(data["xi"]) if "xi" in data else float("nan")
lo = float(fp["T_support_lo"]); hi = float(fp["T_support_hi"])

idx = np.where(tp >= st * t_sh)[0]
Ts = T[idx]; ts = tp[idx]
xs = pos[idx][:, :, 0]; ys = pos[idx][:, :, 1]; zs = pos[idx][:, :, 2]
M_snap, Np = Ts.shape
dt_snap_tsh = float(np.median(np.diff(ts))) / t_sh


def mi(c, L): r = np.abs(c[1:] - c[:-1]); return np.minimum(r, L - r)
tel = np.zeros_like(zs, dtype=np.int32)
tel[1:] = ((mi(xs, Lx) > 0.3 * Lx) | (mi(ys, Ly) > 0.3 * Ly)
           | (np.abs(zs[1:] - zs[:-1]) > 0.3 * Lz)).astype(np.int32)
if gen is not None:
    g = gen[idx]; tel[1:] = np.maximum(tel[1:], (g[1:] != g[:-1]).astype(np.int32))
cum = np.cumsum(tel, axis=0)

NB = 40
edges = np.linspace(np.log10(lo), np.log10(hi), NB + 1)
tm_T = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
dl = edges[1] - edges[0]


def tob(x): return np.clip(np.floor((np.log10(x) - edges[0]) / dl).astype(int), 0, NB - 1)


def build(m):
    """Coarse transfer matrix + per-bin drift/diffusion at lag m."""
    Ta = Ts[:-m]; Tb = Ts[m:]
    v = ((cum[m:] - cum[:-m]) == 0) & np.isfinite(Ta) & np.isfinite(Tb) & (Ta > 0) & (Tb > 0)
    i = tob(Ta[v]); j = tob(Tb[v]); dT = Tb[v] - Ta[v]
    K = np.zeros((NB, NB)); np.add.at(K, (i, j), 1.0); rc = K.sum(1)
    M = K / np.where(rc > 0, rc, 1.0)[:, None]; M[rc == 0] = 1.0 / NB
    dt = m * dt_snap_tsh * t_sh
    A = np.full(NB, np.nan); D = np.full(NB, np.nan)
    for b in range(NB):
        s = i == b
        if s.sum() >= 30:
            d = dT[s]; A[b] = d.mean() / dt; D[b] = d.var() / dt
    return M, rc, A, D


def gaussian_kernel(A, D, dt):
    eT = 10.0 ** edges; Mg = np.zeros((NB, NB))
    for b in range(NB):
        if not np.isfinite(A[b]) or not np.isfinite(D[b]) or D[b] <= 0:
            Mg[b, b] = 1.0; continue
        mu = tm_T[b] + A[b] * dt; sg = np.sqrt(D[b] * dt)
        cdf = 0.5 * (1 + np.array([erf((e - mu) / (np.sqrt(2) * sg)) for e in eT]))
        p = np.clip(np.diff(cdf), 0, None); s = p.sum()
        Mg[b] = p / s if s > 0 else np.eye(NB)[b]
    return Mg


def stationary(M):
    w, v = np.linalg.eig(M.T); k = int(np.argmin(np.abs(w - 1)))
    pi = np.real(v[:, k]);  pi = -pi if pi.sum() < 0 else pi
    pi = np.where(pi < 0, 0, pi); s = pi.sum()
    return pi / s if s > 0 else np.full(NB, 1.0 / NB)


def row_tv(A, B, w):
    tv = 0.5 * np.abs(A - B).sum(1); w = w / w.sum(); return float((w * tv).sum())


def simulate(M, pi0, ntr, nst):
    cdf = np.cumsum(M, 1); cdf[:, -1] = 1
    s = np.empty((nst + 1, ntr), dtype=np.int64)
    s[0] = rng.choice(NB, ntr, p=pi0 / pi0.sum())
    for t in range(nst):
        u = rng.random(ntr); s[t + 1] = (u[:, None] >= cdf[s[t]]).sum(1).clip(0, NB - 1)
    return s


def M_from(states, m):
    K = np.zeros((NB, NB)); np.add.at(K, (states[:-m].ravel(), states[m:].ravel()), 1.0)
    rc = K.sum(1); M = K / np.where(rc > 0, rc, 1.0)[:, None]; M[rc == 0] = 1.0 / NB
    return M, rc


# lag scan (2m must stay well under the regeneration timescale ~0.2 t_sh)
ms = [m for m in [2, 3, 4, 6, 8, 11, 16, 22] if 2 * m < M_snap - 1
      and 2 * m * dt_snap_tsh < 0.16]
dt_tsh = np.array([m * dt_snap_tsh for m in ms])

nonG = []; nm_sig = []; nm_null = []
Amix = []; Dmix = []
b_mix = int(np.argmin(np.abs(tm_T - np.sqrt(0.03 * 0.7))))   # representative mixing T
for m in ms:
    M1, rc1, A, D = build(m)
    M2, rc2, _, _ = build(2 * m)
    dt = m * dt_snap_tsh * t_sh
    Mg = gaussian_kernel(A, D, dt)
    tvg = np.where(rc1 >= 30, 0.5 * np.abs(M1 - Mg).sum(1), np.nan)
    nonG.append(float(np.nanmean(tvg)))
    Amix.append(A[b_mix]); Dmix.append(D[b_mix])
    pi = stationary(M1)
    w = np.where((rc1 > 0) & (rc2 > 0), pi, 0.0); w = w if w.sum() else np.ones(NB)
    sig = row_tv(np.linalg.matrix_power(M1, 2), M2, w)
    sim = simulate(M1, pi, min(Np, 8000), M_snap - 1)
    M1s, _ = M_from(sim, m); M2s, r2s = M_from(sim, 2 * m)
    ws = np.where(r2s > 0, pi, 0.0); ws = ws if ws.sum() else np.ones(NB)
    null = row_tv(np.linalg.matrix_power(M1s, 2), M2s, ws)
    nm_sig.append(sig); nm_null.append(null)
    print(f"dt/t_sh={m*dt_snap_tsh:.3f}  nonGauss={nonG[-1]:.3f}  "
          f"CK_signal={sig:.4f}  CK_null={null:.4f}  ratio={sig/null if null>0 else np.inf:.1f}  "
          f"A_mix={A[b_mix]:.2f}  D_mix={D[b_mix]:.3f}")

# FP reconstruction error vs lag (loaded)
fp_dt = fp["dt_over_tsh"]; fp_l1 = fp["l1_of_lag"]

# ---- figure ---------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))
(aAD, aG), (aM, aL) = ax

aAD2 = aAD.twinx()
l1 = aAD.plot(dt_tsh, -np.array(Amix), "o-", color="tab:blue",
              label=r"$-A(T_{\rm mix};\Delta t)$ (drift, left)")
l2 = aAD2.plot(dt_tsh, Dmix, "s-", color="tab:red",
               label=r"$D(T_{\rm mix};\Delta t)$ (diffusion, right)")
aAD.set_xscale("log"); aAD.set_xlabel(r"$\Delta t/t_{sh}$")
aAD.set_ylabel(r"$-A$", color="tab:blue"); aAD2.set_ylabel(r"$D$", color="tab:red")
aAD.set_ylim(bottom=0); aAD2.set_ylim(bottom=0)
aAD.set_title(f"(1) Both |drift| and D FALL with $\\Delta t$ "
              f"(finite-lag bias; drift-dominated, no plateau)")
aAD.legend(l1 + l2, [x.get_label() for x in l1 + l2], fontsize=8); aAD.grid(alpha=0.3)

aG.plot(dt_tsh, nonG, "o-", color="tab:purple")
aG.set_xscale("log"); aG.set_xlabel(r"$\Delta t/t_{sh}$")
aG.set_ylabel("mean kernel non-Gaussianity (row TV)")
aG.set_title("(2) Kernel gets MORE non-Gaussian with $\\Delta t$\n(nonlinear drift + bounded domain, not CLT)")
aG.set_ylim(bottom=0); aG.grid(alpha=0.3)

aM.plot(dt_tsh, nm_sig, "o-", color="tab:red", label="CK signal (memory)")
aM.plot(dt_tsh, nm_null, "s--", color="0.5", label="Markov null")
aM.set_xscale("log"); aM.set_xlabel(r"$\Delta t/t_{sh}$")
aM.set_ylabel(r"$TV(M(\Delta t)^2, M(2\Delta t))$")
aM.set_title("(3) Non-Markovianity GROWS with $\\Delta t$ (memory accumulates)")
aM.set_ylim(bottom=0); aM.legend(fontsize=9); aM.grid(alpha=0.3)

aL.plot(fp_dt, fp_l1, "o-", color="tab:green")
aL.set_xscale("log"); aL.set_xlabel(r"$\Delta t/t_{sh}$")
aL.set_ylabel(r"FP reconstruction $L1(\widehat P,\ {\rm hist})$")
aL.set_title("(4) FP error smallest at the SMALLEST $\\Delta t$ (no scale-separated window)")
aL.set_xlim(dt_tsh.min() * 0.7, max(fp_dt.max(), dt_tsh.max()))
aL.grid(alpha=0.3)

fig.suptitle(rf"Diffusion picture vs measurement lag $\Delta t$  "
             rf"($\xi=$Damk$\ddot{{o}}$hler$={xi:.0f}$).  "
             rf"A window needs a minimum in (2),(3),(4) at intermediate $\Delta t$.", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"figures/dt_scan{SUFFIX}.png", dpi=140)
print(f"\nwrote figures/dt_scan{SUFFIX}.png")
