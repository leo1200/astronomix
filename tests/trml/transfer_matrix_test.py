"""
Transfer-matrix (discrete Markov) model for the TRML temperature PDF, as an
alternative to the Fokker-Planck (FP) diffusion model.

Motivation
----------
The FP test (``fokker_planck_test.py``) models the Lagrangian temperature as an
Ito diffusion ``dT = A(T) dt + sqrt(D(T)) dW`` -- a *second-order* (Gaussian)
truncation of the transition kernel. On this TRML it reproduces the equilibrium
PDF only to ~0.25-0.3 dex. The natural more-general object is the full empirical
transition matrix

    M_ij(dt) = P( T(t+dt) in bin j | T(t) in bin i ),

estimated directly from tracer temperature increments -- NO Gaussian / small-dt
assumption. Its stationary left eigenvector pi (pi M = pi) is a candidate for the
marginal P(T).

Confirmation-bias warnings (read before trusting any "alignment")
-----------------------------------------------------------------
1. The stationary eigenvector pi of the *empirically measured* one-step matrix is
   ~equal to the empirical occupation measure (the measured marginal) BY
   CONSTRUCTION -- in a stationary series, transitions in = transitions out, so
   the measured marginal is an approximate fixed point of M. "pi matches P(T)" is
   therefore near-self-fulfilling and is NOT evidence that a Markov-in-T model
   holds. (Same self-fulfilling trap the note flags for the inverse FP
   construction.) We show it, but do not count it as validation.

2. The genuine, falsifiable test of the underlying assumption (that T alone is
   Markov) is Chapman-Kolmogorov (CK):  M(dt)^k  vs  M(k*dt) measured directly.
   If T is Markov these agree up to sampling noise; if T has memory (finite
   turbulent correlation time) they diverge. To judge "small vs large" HONESTLY
   we build the Markov NULL by a parametric bootstrap: simulate a synthetic
   Markov chain with exactly the measured M(dt), same sample sizes, and measure
   its CK distance. The real CK distance is only meaningful relative to that null.

3. We also compare the empirical one-step kernel to the FP Gaussian kernel built
   from the same A(T), D(T): where they differ is exactly what the diffusion
   truncation misses (skew / fat tails / jumps).

Run ``trml_tracers.py`` then ``fokker_planck_test.py`` (for the FP P_hat overlay)
first. Figures -> ``figures/tm_*``.
"""

import os
import sys

import numpy as np

np.seterr(divide="ignore", invalid="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PRESET = os.environ.get("TRML_PRESET", "long64")
SUFFIX = os.environ.get(
    "TRML_SUFFIX", {"quick": "_quick", "long64": "_long64", "fiducial": ""}[PRESET]
)

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("figures", exist_ok=True)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# load (mirror fokker_planck_test.py)
# ---------------------------------------------------------------------------
data = np.load(f"data/trml_tracers{SUFFIX}.npz")
time_points = data["time_points"]
tracer_T = data["tracer_temperature"]                 # (S, Np)
tracer_pos = data["tracer_position"]                  # (S, Np, 3)
mass_pdf_snaps = data["mass_temperature_pdf"]
vol_pdf_snaps = data["temperature_pdf"]
tracer_generation = data["tracer_generation"] if "tracer_generation" in data else None
logT_edges = data["logT_bin_edges"]
t_sh = float(data["t_sh"])
xi = float(data["xi"]) if "xi" in data else float("nan")
L_x = float(data["L_x"]) if "L_x" in data else float(data["L_z"]) / 1.5
L_y = float(data["L_y"]) if "L_y" in data else float(data["L_z"]) / 1.5
L_z = float(data["L_z"])
steady_t_sh = float(data["steady_state_t_sh"])

logT_centers = 0.5 * (logT_edges[:-1] + logT_edges[1:])
T_centers = 10.0 ** logT_centers
dlogT = logT_edges[1] - logT_edges[0]
T_lo_full = float(10.0 ** logT_edges[0])
T_hi_full = float(10.0 ** logT_edges[-1])


def normalize_dlogT(hist):
    integral = np.sum(hist) * dlogT
    return hist / integral if integral > 0 else hist


# steady window + Eulerian marginals
steady = time_points >= steady_t_sh * t_sh
eul_mass_pdf = normalize_dlogT(mass_pdf_snaps[steady].mean(axis=0))   # dM/dlogT
eul_vol_pdf = normalize_dlogT(vol_pdf_snaps[steady].mean(axis=0))

# support of the marginal (trim empty tails via cumulative mass)
_cum = np.cumsum(eul_mass_pdf) * dlogT
lo_idx = int(np.where(_cum > 1e-3)[0][0])
hi_idx = int(np.where(_cum < 1.0 - 1e-4)[0][-1])
T_support_lo = float(T_centers[lo_idx])
T_support_hi = float(T_centers[hi_idx])

# Lagrangian marginal (mass-seeded tracers -> count proportional to dM/dlogT)
lag_T = tracer_T[steady].ravel()
lag_hist, _ = np.histogram(
    np.log10(np.clip(lag_T, T_lo_full, T_hi_full)), bins=logT_edges
)
lag_pdf = normalize_dlogT(lag_hist.astype(float))

# ---------------------------------------------------------------------------
# teleport / regeneration masking (identical logic to the FP test)
# ---------------------------------------------------------------------------
steady_idx = np.where(steady)[0]
Ts = tracer_T[steady_idx]              # (M, Np)
ts = time_points[steady_idx]
xs = tracer_pos[steady_idx][:, :, 0]
ys = tracer_pos[steady_idx][:, :, 1]
zs = tracer_pos[steady_idx][:, :, 2]
M_snap, Np = Ts.shape


def _minimum_image_jump(coordinate, box_length):
    raw = np.abs(coordinate[1:] - coordinate[:-1])
    return np.minimum(raw, box_length - raw)


step_teleport = np.zeros_like(zs, dtype=np.int32)
step_teleport[1:] = (
    (_minimum_image_jump(xs, L_x) > 0.3 * L_x)
    | (_minimum_image_jump(ys, L_y) > 0.3 * L_y)
    | (np.abs(zs[1:] - zs[:-1]) > 0.3 * L_z)
).astype(np.int32)
if tracer_generation is not None:
    gen = tracer_generation[steady_idx]
    step_teleport[1:] = np.maximum(
        step_teleport[1:], (gen[1:] != gen[:-1]).astype(np.int32)
    )
cum_teleport = np.cumsum(step_teleport, axis=0)   # (M, Np)

# ---------------------------------------------------------------------------
# transfer-matrix bins: coarser than the 150 PDF bins so each M row is well
# sampled. Defined over the support; endpoints outside are clamped into the edge
# bins (they carry little mass). A separate coarse grid is honest about the
# resolution the transition statistics actually support.
# ---------------------------------------------------------------------------
NB = 40
tm_edges = np.linspace(np.log10(T_support_lo), np.log10(T_support_hi), NB + 1)
tm_centers_log = 0.5 * (tm_edges[:-1] + tm_edges[1:])
tm_T = 10.0 ** tm_centers_log
tm_dlogT = tm_edges[1] - tm_edges[0]


def _to_bin(T):
    """Map temperatures to coarse-grid bin indices (clamped to [0, NB-1])."""
    idx = np.floor((np.log10(T) - tm_edges[0]) / tm_dlogT).astype(np.int64)
    return np.clip(idx, 0, NB - 1)


def build_transfer_matrix(m, idx_subset=None):
    """Empirical row-stochastic transfer matrix at lag ``m`` snapshots.

    Counts transitions (bin_i at snapshot s) -> (bin_j at snapshot s+m) over all
    valid pairs (no teleport / regeneration in between). ``idx_subset`` restricts
    the tracer columns (used for the split-half noise floor).

    Returns (M, row_counts): M is (NB, NB) row-stochastic on rows with counts,
    rows with no data are left as a uniform row (irrelevant: pi weights them ~0).
    """
    if m >= M_snap:
        return None, None
    T0 = Ts[:-m]
    T1 = Ts[m:]
    valid = (cum_teleport[m:] - cum_teleport[:-m]) == 0
    valid &= np.isfinite(T0) & np.isfinite(T1) & (T0 > 0) & (T1 > 0)
    if idx_subset is not None:
        col = np.zeros(Np, dtype=bool)
        col[idx_subset] = True
        valid &= col[None, :]

    i = _to_bin(T0[valid])
    j = _to_bin(T1[valid])
    counts = np.zeros((NB, NB), dtype=np.float64)
    np.add.at(counts, (i, j), 1.0)
    row_counts = counts.sum(axis=1)
    safe = np.where(row_counts > 0, row_counts, 1.0)
    Mmat = counts / safe[:, None]
    # empty rows -> uniform (so matrix_power stays stochastic); pi ~0 there
    empty = row_counts == 0
    Mmat[empty] = 1.0 / NB
    return Mmat, row_counts


def stationary(Mmat):
    """Left eigenvector of a row-stochastic matrix (pi M = pi), >=0, sum 1."""
    vals, vecs = np.linalg.eig(Mmat.T)
    k = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, k])
    pi = np.where(pi < 0, 0.0, pi)   # kill tiny negative numerical dust
    s = pi.sum()
    return pi / s if s > 0 else pi


def row_tv(A, B, weights):
    """Stationary-weighted mean total-variation distance between matrix rows."""
    tv = 0.5 * np.abs(A - B).sum(axis=1)
    w = weights / weights.sum()
    return float((w * tv).sum())


# ---------------------------------------------------------------------------
# base lag and CK lags
# ---------------------------------------------------------------------------
dt_snap = float(np.median(np.diff(ts)))
dt_snap_tsh = dt_snap / t_sh
# base lag ~ diffusive-window scale (~0.02 t_sh), but at least 1 snapshot
m0 = max(1, int(round(0.02 / dt_snap_tsh)))
print(f"snapshot spacing dt_snap/t_sh = {dt_snap_tsh:.4f}; base lag m0 = {m0} "
      f"(= {m0 * dt_snap_tsh:.4f} t_sh)")

# CK factors k: keep k*m0 well under the regeneration timescale so pairs survive
K_MAX = 6
ck_factors = [k for k in range(2, K_MAX + 1) if (k * m0) < M_snap - 1]

M1, rc1 = build_transfer_matrix(m0)
pi1 = stationary(M1)          # CLOSED (conservative) stationary -> the J=0 analog
# stationary-weight for CK averaging (down-weights empty/edge bins)
pi_weight = np.where(rc1 > 0, pi1, 0.0)
if pi_weight.sum() == 0:
    pi_weight = np.ones(NB)


# ---------------------------------------------------------------------------
# Flux-driven stationary distribution from the empirical kernel.
# ---------------------------------------------------------------------------
# The closed M is row-stochastic => conservative => zero net probability
# current => its stationary pi is the J=0 solution, which (like FP with J=0)
# drains to the cold peak. The real TRML marginal is maintained by a constant
# hot->cold probability current J. The DISCRETE analog of the FP constant-flux
# first integral, using the FULL empirical (non-Gaussian) kernel: find pi such
# that the net current across every internal bin boundary equals a constant J.
#
#   current across boundary b (between bins <b and >=b), per step, for a walker
#   distributed as pi and advanced once by M:
#     C_b(pi) = sum_{i<b} sum_{j>=b} pi_i M_ij   (up-crossings b-1 -> b, hot->? )
#             - sum_{i>=b} sum_{j<b} pi_i M_ij   (down-crossings)
#   Require C_b(pi) = -J for all internal b (J>0 = net DOWN/cooling current),
#   plus normalization. Linear in pi -> least squares. This uses the measured,
#   possibly non-Gaussian M; no Gaussian truncation and no free shape parameter
#   (J is measured independently below).
def flux_driven_stationary(Mmat, J_per_step, hot_bin, cold_bin):
    """Source-driven steady state of the empirical (non-Gaussian) kernel.

    A conservative (row-stochastic) M cannot carry a net current -- its only
    steady state is the zero-flux one that drains to cold. To reproduce the
    flux-maintained TRML marginal we add an explicit boundary source: probability
    is injected in the hot bin and removed from the cold bin at the measured rate
    J per step. The discrete stationary balance is then

        (I - M^T) pi = J * (e_hot - e_cold),

    the direct discrete analog of the FP constant-flux first integral but with
    the FULL measured kernel. (I - M^T) is singular (its null space is the closed
    pi), so we solve in least squares with the normalization row appended and fix
    the residual null-space component by that normalization.
    """
    A = np.vstack([np.eye(NB) - Mmat.T, np.ones(NB)])
    src = np.zeros(NB)
    # Spread injection/removal uniformly over the two edge-most populated bins
    # rather than a single delta -- a point source spikes the boundary bin (a
    # discretization artifact); a 2-bin uniform spread borrows no target info.
    for b in (hot_bin, hot_bin - 1):
        src[b] += 0.5 * J_per_step
    for b in (cold_bin, cold_bin + 1):
        src[b] -= 0.5 * J_per_step
    rhs = np.concatenate([src, [1.0]])
    pi, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    pi = np.where(pi < 0, 0.0, pi)
    s = pi.sum()
    return pi / s if s > 0 else pi


# measure J_per_step: net downward crossing per tracer per step (lag m0),
# averaged over interior reference boundaries (constant in steady state).
def _net_down_crossing_per_step(m):
    T0 = Ts[:-m]
    T1 = Ts[m:]
    valid = (cum_teleport[m:] - cum_teleport[:-m]) == 0
    valid &= np.isfinite(T0) & np.isfinite(T1) & (T0 > 0) & (T1 > 0)
    T0v, T1v = T0[valid], T1[valid]
    n_pairs = T0v.size
    rates = []
    for b in range(NB // 4, 3 * NB // 4):     # interior boundaries only
        T_ref = 10.0 ** tm_edges[b]
        down = np.sum((T0v >= T_ref) & (T1v < T_ref))
        up = np.sum((T0v < T_ref) & (T1v >= T_ref))
        rates.append((down - up) / n_pairs)
    return float(np.median(rates))


J_per_step = _net_down_crossing_per_step(m0)
# hot source / cold sink bins = the populated support edges
hot_bin = int(np.where(rc1 > 0)[0][-1])
cold_bin = int(np.where(rc1 > 0)[0][0])
pi_flux = flux_driven_stationary(M1, J_per_step, hot_bin, cold_bin)
print(f"measured net down-crossing J_per_step (lag m0) = {J_per_step:.3e} "
      f"(hot_bin={hot_bin}, cold_bin={cold_bin})")

# ---------------------------------------------------------------------------
# Figure 1: PDF alignment (Eulerian, Lagrangian, FP P_hat, TM closed + flux)
# ---------------------------------------------------------------------------
tm_pi_closed_dMdlogT = pi1 / tm_dlogT            # closed (J=0): collapses to cold
tm_pi_flux_dMdlogT = pi_flux / tm_dlogT          # flux-driven (measured J)

# FP reconstruction (load if fokker_planck_test.py produced it)
fp_T = fp_pdf = None
fp_path = f"data/fp_result{SUFFIX}.npz"
if os.path.exists(fp_path):
    fp = np.load(fp_path)
    fp_T = fp["T_centers"]
    fp_pdf = fp["recon_dMdlogT"]
    fp_l1 = float(fp["fp_l1"])
else:
    print(f"(note: {fp_path} not found -- run fokker_planck_test.py for the FP overlay)")

# re-bin the measured Eulerian mass PDF onto the coarse grid for an L1 vs pi
eul_on_coarse = np.zeros(NB)
fine_bin_of = np.clip(
    np.floor((logT_centers - tm_edges[0]) / tm_dlogT).astype(int), 0, NB - 1
)
for b in range(NB):
    sel = fine_bin_of == b
    if sel.any():
        eul_on_coarse[b] = np.sum(eul_mass_pdf[sel]) * dlogT   # mass in coarse bin
eul_on_coarse_density = eul_on_coarse / tm_dlogT


def _l1_coarse(pi, mask=None):
    d = np.abs(pi / tm_dlogT - eul_on_coarse_density)
    if mask is not None:
        d = d[mask]
    return 0.5 * np.sum(d) * tm_dlogT


# interior metric excludes the source/sink injection bins (their boundary
# treatment is a point-source artifact, not a model property) -- analogous to
# how FP borrows the cold boundary value; here we simply don't score those bins.
interior = np.ones(NB, dtype=bool)
interior[[cold_bin, cold_bin + 1, hot_bin - 1, hot_bin]] = False
tm_l1_closed = _l1_coarse(pi1)
tm_l1_flux = _l1_coarse(pi_flux)
tm_l1_flux_interior = _l1_coarse(pi_flux, interior)

fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.plot(T_centers, eul_mass_pdf, lw=3.0, color="tab:blue",
        label="Eulerian mass-weighted (target)")
ax.plot(T_centers, lag_pdf, "--", lw=2.0, color="tab:orange",
        label="Lagrangian tracers")
if fp_T is not None:
    ax.plot(fp_T, fp_pdf, ":", lw=2.4, color="black",
            label=fr"Fokker-Planck $\widehat{{P}}$, meas. $J$ (L1={fp_l1:.3f})")
ax.plot(tm_T, tm_pi_flux_dMdlogT, "-o", lw=2.0, ms=3, color="tab:green",
        label=(fr"Transfer-matrix, flux-driven (meas. $J$) "
               fr"(L1$_{{\rm int}}$={tm_l1_flux_interior:.3f})"))
ax.plot(tm_T, tm_pi_closed_dMdlogT, "-s", lw=1.3, ms=3, color="tab:red", alpha=0.7,
        label=fr"Transfer-matrix, closed ($J{{=}}0$) (L1={tm_l1_closed:.3f})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(T_support_lo, T_support_hi)
ax.set_ylim(bottom=max(1e-4, eul_mass_pdf[eul_mass_pdf > 0].min() * 0.5))
ax.set_xlabel("T"); ax.set_ylabel(r"$dM/d\log T$ (normalized)")
ax.set_title("Temperature PDF alignment\n"
             "closed transfer-matrix collapses to cold (no flux); "
             "flux-driven one uses the true kernel")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"figures/tm_four_pdfs{SUFFIX}.png", dpi=150)
plt.close(fig)
tm_l1 = tm_l1_flux
print(f"\nTransfer-matrix stationary vs Eulerian mass PDF (coarse-grid L1):")
print(f"  closed (J=0):              L1 = {tm_l1_closed:.4f}  (collapses to cold)")
print(f"  flux-driven (meas J):      L1 = {tm_l1_flux:.4f}  (full support, incl. injection bins)")
print(f"  flux-driven interior:      L1 = {tm_l1_flux_interior:.4f}  (compare to FP "
      f"{fp_l1 if fp_T is not None else float('nan'):.4f})")

# ---------------------------------------------------------------------------
# Figure 2: Chapman-Kolmogorov test with a parametric Markov null
# ---------------------------------------------------------------------------
def simulate_markov(Mmat, pi0, n_tracers, n_steps):
    """Simulate a synthetic Markov chain with transition ``Mmat``.

    Returns an (n_steps+1, n_tracers) array of bin indices; the Markov NULL for
    the CK distance (finite-sample M^k vs M(k) divergence a true Markov process
    would show with these sample sizes).
    """
    cdf = np.cumsum(Mmat, axis=1)
    cdf[:, -1] = 1.0
    states = np.empty((n_steps + 1, n_tracers), dtype=np.int64)
    states[0] = rng.choice(NB, size=n_tracers, p=pi0 / pi0.sum())
    for s in range(n_steps):
        u = rng.random(n_tracers)
        cur = states[s]
        states[s + 1] = (u[:, None] >= cdf[cur]).sum(axis=1).clip(0, NB - 1)
    return states


def transfer_from_states(states, m):
    """Transfer matrix at lag ``m`` from a synthetic (n_steps+1, n_tracers) run."""
    T0 = states[:-m]
    T1 = states[m:]
    counts = np.zeros((NB, NB))
    np.add.at(counts, (T0.ravel(), T1.ravel()), 1.0)
    rc = counts.sum(axis=1)
    Mmat = counts / np.where(rc > 0, rc, 1.0)[:, None]
    Mmat[rc == 0] = 1.0 / NB
    return Mmat, rc

# real CK signal: M1^k vs directly-measured M(k*m0)
ck_signal = []
ck_null_mean = []
ck_null_hi = []
n_sim_tracers = min(Np, 12000)
n_sim_steps = M_snap - 1
# one synthetic Markov run reused for all k (its own M1_sim rebuilt for fairness)
sim_states = simulate_markov(M1, pi1, n_sim_tracers, n_sim_steps)
M1_sim, _ = transfer_from_states(sim_states, m0)

for k in ck_factors:
    Mk, rck = build_transfer_matrix(k * m0)
    if Mk is None:
        continue
    M1_pow = np.linalg.matrix_power(M1, k)
    w = np.where((rc1 > 0) & (rck > 0), pi1, 0.0)
    if w.sum() == 0:
        w = np.ones(NB)
    signal = row_tv(M1_pow, Mk, w)

    # Markov null: same construction on the synthetic Markov data
    Mk_sim, rck_sim = transfer_from_states(sim_states, k * m0)
    M1sim_pow = np.linalg.matrix_power(M1_sim, k)
    w_sim = np.where((rck_sim > 0), pi1, 0.0)
    if w_sim.sum() == 0:
        w_sim = np.ones(NB)
    null = row_tv(M1sim_pow, Mk_sim, w_sim)

    ck_signal.append(signal)
    ck_null_mean.append(null)
    print(f"  CK  k={k}  (lag {k*m0*dt_snap_tsh:.3f} t_sh):  "
          f"TV(M1^k, M_k) = {signal:.4f}   Markov-null = {null:.4f}   "
          f"ratio = {signal / null if null > 0 else np.inf:.2f}")

ck_lags_tsh = [k * m0 * dt_snap_tsh for k in ck_factors[: len(ck_signal)]]

fig, ax = plt.subplots(figsize=(6.6, 4.7))
ax.plot(ck_lags_tsh, ck_signal, "o-", lw=2, color="tab:red",
        label=r"measured  $TV(M(\Delta t)^k,\, M(k\Delta t))$")
ax.plot(ck_lags_tsh, ck_null_mean, "s--", lw=2, color="0.5",
        label="Markov null (parametric bootstrap)")
ax.set_xlabel(r"total lag $k\,\Delta t / t_{sh}$")
ax.set_ylabel("stationary-weighted row TV distance")
ax.set_title("Chapman-Kolmogorov test: is T alone Markov?\n"
             "signal >> null  =>  memory (non-Markov in T)")
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(f"figures/tm_chapman_kolmogorov{SUFFIX}.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: empirical one-step kernel vs the FP Gaussian kernel
# ---------------------------------------------------------------------------
# Build the FP Gaussian kernel from the SAME data: per start-bin i, mean shift
# A(T_i)*dt and variance D(T_i)*dt from connected conditional moments at lag m0.
T0 = Ts[:-m0]
T1 = Ts[m0:]
valid = (cum_teleport[m0:] - cum_teleport[:-m0]) == 0
valid &= np.isfinite(T0) & np.isfinite(T1) & (T0 > 0) & (T1 > 0)
i_all = _to_bin(T0[valid])
dT_all = (T1[valid] - T0[valid])
dt_here = m0 * dt_snap

A_bin = np.full(NB, np.nan)
D_bin = np.full(NB, np.nan)
for b in range(NB):
    sel = i_all == b
    if sel.sum() >= 30:
        d = dT_all[sel]
        A_bin[b] = d.mean() / dt_here
        D_bin[b] = d.var() / dt_here

# FP Gaussian transition matrix on the same bins
M_fp = np.zeros((NB, NB))
for b in range(NB):
    if not np.isfinite(A_bin[b]) or not np.isfinite(D_bin[b]) or D_bin[b] <= 0:
        M_fp[b] = M1[b]  # fall back to empirical where we cannot build it
        continue
    mean = tm_centers_log[b] * 0.0  # work in T, not logT, for the kernel mean
    T_start = tm_T[b]
    mu = T_start + A_bin[b] * dt_here
    sigma = np.sqrt(D_bin[b] * dt_here)
    # probability mass into each bin via the Gaussian CDF across bin edges in T
    edges_T = 10.0 ** tm_edges
    from math import erf
    cdf = 0.5 * (1.0 + np.array([erf((e - mu) / (np.sqrt(2) * sigma)) for e in edges_T]))
    p = np.diff(cdf)
    p = np.clip(p, 0, None)
    s = p.sum()
    M_fp[b] = p / s if s > 0 else M1[b]

# kernel discrepancy per start bin (row TV), weighted display
kernel_tv = 0.5 * np.abs(M1 - M_fp).sum(axis=1)
kernel_tv_masked = np.where(rc1 >= 30, kernel_tv, np.nan)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.0))
vmax = max(M1.max(), M_fp.max())
im0 = axA.imshow(M1.T, origin="lower", aspect="auto", cmap="magma",
                 extent=[tm_edges[0], tm_edges[-1], tm_edges[0], tm_edges[-1]],
                 norm=LogNorm(vmin=1e-3, vmax=vmax))
axA.set_title(fr"empirical kernel $M(\Delta t)$ ($\Delta t/t_{{sh}}={m0*dt_snap_tsh:.3f}$)")
axA.set_xlabel(r"$\log_{10} T(t)$"); axA.set_ylabel(r"$\log_{10} T(t+\Delta t)$")
fig.colorbar(im0, ax=axA)
axB.plot(tm_T, kernel_tv_masked, "o-", color="tab:purple")
axB.set_xscale("log")
axB.set_xlabel("T(t)")
axB.set_ylabel(r"row TV$(M_{\rm emp}, M_{\rm FP\ Gaussian})$")
axB.set_title("where the Gaussian (FP) kernel misfits the empirical kernel")
axB.set_xlim(T_support_lo, T_support_hi)
axB.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(f"figures/tm_kernel_vs_fp{SUFFIX}.png", dpi=150)
plt.close(fig)

mean_kernel_tv = float(np.nanmean(kernel_tv_masked))
print(f"\nmean empirical-vs-FP-Gaussian kernel TV over populated bins: "
      f"{mean_kernel_tv:.4f}")

# ---------------------------------------------------------------------------
# save a compact result for cross-run comparison / the write-up
# ---------------------------------------------------------------------------
np.savez(
    f"data/tm_result{SUFFIX}.npz",
    xi=xi,
    NB=NB,
    m0=m0,
    dt_snap_tsh=dt_snap_tsh,
    J_per_step=J_per_step,
    tm_l1_closed=tm_l1_closed,
    tm_l1_flux=tm_l1_flux,
    tm_l1_flux_interior=tm_l1_flux_interior,
    fp_l1=fp_l1 if fp_T is not None else np.nan,
    ck_lags_tsh=np.array(ck_lags_tsh),
    ck_signal=np.array(ck_signal),
    ck_null=np.array(ck_null_mean),
    mean_kernel_tv=mean_kernel_tv,
    tm_T=tm_T,
    tm_pi_closed=pi1,
    tm_pi_flux=pi_flux,
    A_bin=A_bin,
    D_bin=D_bin,
)
print(f"\nfigures written to {os.path.join(HERE, 'figures')}/tm_*{SUFFIX}.png")
print("data/tm_result written.")
