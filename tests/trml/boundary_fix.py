"""
Fixing the transfer-matrix boundary/flux discretization.

The flux-driven transfer matrix had a full-support vs interior L1 gap (~0.106 vs
~0.065) caused by a point-source: injecting the whole flux J into the edge bins
piles probability there (the hottest bin is a boundary-maintained reservoir, not
part of the cooling cascade, so it should NOT accumulate the injected flux).

Fix: a Dirichlet hot boundary -- pin the top bin(s) to the MEASURED hot-phase
occupation (the inflow reservoir set by the boundary condition) and solve the
interior stationarity with a cold sink. This is the discrete analog of how the
FP march borrows the cold boundary value P(T_cold): one boundary value, not a
free shape. We compare it to the point-source version and to a finer bin grid.
"""
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
data = np.load("data/trml_tracers.npz"); fp = np.load("data/fp_result.npz"); tm = np.load("data/tm_result.npz")
tp = data["time_points"]; T = data["tracer_temperature"]; pos = data["tracer_position"]
gen = data["tracer_generation"] if "tracer_generation" in data else None
t_sh = float(data["t_sh"]); Lx = float(data["L_x"]); Ly = float(data["L_y"]); Lz = float(data["L_z"])
st = float(data["steady_state_t_sh"]); lo = float(fp["T_support_lo"]); hi = float(fp["T_support_hi"])
m0 = int(tm["m0"]); J = float(tm["J_per_step"])

idx = np.where(tp >= st * t_sh)[0]; Ts = T[idx]
xs = pos[idx][:, :, 0]; ys = pos[idx][:, :, 1]; zs = pos[idx][:, :, 2]
def mi(c, L): r = np.abs(c[1:] - c[:-1]); return np.minimum(r, L - r)
tel = np.zeros_like(zs, dtype=np.int32)
tel[1:] = ((mi(xs, Lx) > 0.3 * Lx) | (mi(ys, Ly) > 0.3 * Ly) | (np.abs(zs[1:] - zs[:-1]) > 0.3 * Lz)).astype(np.int32)
if gen is not None:
    g = gen[idx]; tel[1:] = np.maximum(tel[1:], (g[1:] != g[:-1]).astype(np.int32))
cum = np.cumsum(tel, axis=0)

# Eulerian target (fine) to rebin
ef = fp["eul_mass_pdf"]; Tf = fp["T_centers"]; lf = np.log10(Tf); dlf = lf[1] - lf[0]


def run(NB):
    edges = np.linspace(np.log10(lo), np.log10(hi), NB + 1); dl = edges[1] - edges[0]
    def tob(x): return np.clip(np.floor((np.log10(x) - edges[0]) / dl).astype(int), 0, NB - 1)
    Ta = Ts[:-m0]; Tb = Ts[m0:]
    v = ((cum[m0:] - cum[:-m0]) == 0) & np.isfinite(Ta) & np.isfinite(Tb) & (Ta > 0) & (Tb > 0)
    K = np.zeros((NB, NB)); np.add.at(K, (tob(Ta[v]), tob(Tb[v])), 1.0); rc = K.sum(1)
    M = K / np.where(rc > 0, rc, 1.0)[:, None]; M[rc == 0] = 1.0 / NB
    hot = int(np.where(rc > 0)[0][-1]); cold = int(np.where(rc > 0)[0][0])
    # coarse Eulerian mass per bin
    m = np.zeros(NB)
    for b in range(NB):
        s = (np.floor((lf - edges[0]) / dl).astype(int) == b)
        if s.any(): m[b] = np.sum(ef[s]) * dlf
    m = m / m.sum()

    def L1(pi, mask=None):
        d = np.abs(pi - m); return 0.5 * np.sum(d[mask] if mask is not None else d)
    interior = np.ones(NB, bool); interior[[cold, cold + 1, hot - 1, hot]] = False

    # (a) point-source injection (current scheme)
    A = np.vstack([np.eye(NB) - M.T, np.ones(NB)]); src = np.zeros(NB)
    for b in (hot, hot - 1): src[b] += 0.5 * J
    for b in (cold, cold + 1): src[b] -= 0.5 * J
    pi_inj, *_ = np.linalg.lstsq(A, np.concatenate([src, [1.0]]), rcond=None)
    pi_inj = np.where(pi_inj < 0, 0, pi_inj); pi_inj /= pi_inj.sum()

    # (b) Dirichlet hot boundary: pin top-K bins to measured, solve interior + cold sink
    K_pin = max(1, NB // 20)
    Fset = list(range(hot - K_pin + 1, hot + 1))
    piF = m[Fset]
    U = [b for b in range(NB) if b not in Fset]
    rows = []; rhs = []
    for j in U:
        if j == cold:
            continue  # cold bin is the sink: drop its stationarity
        row = np.zeros(len(U))
        for a, i in enumerate(U):
            row[a] = M[i, j] - (1.0 if i == j else 0.0)
        rows.append(row); rhs.append(-sum(piF[k] * M[Fset[k], j] for k in range(len(Fset))))
    rows.append(np.ones(len(U))); rhs.append(1.0 - piF.sum())  # normalization
    pu, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    pi_dir = np.zeros(NB)
    for a, i in enumerate(U): pi_dir[i] = pu[a]
    for k, b in enumerate(Fset): pi_dir[b] = piF[k]
    pi_dir = np.where(pi_dir < 0, 0, pi_dir); pi_dir /= pi_dir.sum()

    return dict(NB=NB, inj_full=L1(pi_inj), inj_int=L1(pi_inj, interior),
                dir_full=L1(pi_dir), dir_int=L1(pi_dir, interior))


print(f"FP support-L1 (reference)          : {float(fp['fp_l1']):.4f}\n")
print(f"{'scheme':<34}{'full-support L1':>16}{'interior L1':>14}")
for NB in (40, 80):
    r = run(NB)
    print(f"NB={NB:<3} point-source injection        {r['inj_full']:>14.4f}{r['inj_int']:>14.4f}")
    print(f"NB={NB:<3} Dirichlet hot boundary (fix)  {r['dir_full']:>14.4f}{r['dir_int']:>14.4f}")
