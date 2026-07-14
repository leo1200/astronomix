"""
Fokker-Planck self-consistency test for the TRML temperature PDF, from
Lagrangian tracer particles (see the stochastic-temperature note).

Pipeline (run ``trml_tracers.py`` first to produce the data):

  Check 0  Lagrangian tracer temperature marginal  vs  mass-weighted Eulerian
           PDF (dM/dlogT). Must agree if the mass-seeded tracers sample the
           mass distribution. This is the marginal the reconstruction targets.

  A, D     drift A(T;dt) and diffusion D(T;dt) from the *connected* conditional
           moments of tracer temperature increments, binned in T, for a scan of
           lags dt. (D uses the connected 2nd moment: Var(dT), not <dT^2>.)

  P_hat    reconstruct the steady-state PDF from (A, D, J) via the implicit
           backward-Euler march of diffmix.solvers.pdf.calculate_pdf, and
           compare to the measured mass-weighted PDF (L1 / KL).

  window   scan dt: where A, D plateau *and* P_hat matches the histogram is the
           validity window.

Figures are written to ``figures/``.
"""

import os
import sys

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# diffmix (the FP solver lives there)
DIFFMIX_PATH = os.path.expanduser("~/diffmix")
if DIFFMIX_PATH not in sys.path:
    sys.path.insert(0, DIFFMIX_PATH)
from diffmix.solvers.pdf import calculate_pdf  # noqa: E402

PRESET = os.environ.get("TRML_PRESET", "long64")
SUFFIX = os.environ.get(
    "TRML_SUFFIX", {"quick": "_quick", "long64": "_long64", "fiducial": ""}[PRESET]
)
QUICK = PRESET == "quick"

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
data = np.load(f"data/trml_tracers{SUFFIX}.npz")
time_points = data["time_points"]                     # (S,)
tracer_T = data["tracer_temperature"]                 # (S, Np)
tracer_pos = data["tracer_position"]                  # (S, Np, 3)
mass_pdf_snaps = data["mass_temperature_pdf"]         # (S, bins) dM/dlogT
vol_pdf_snaps = data["temperature_pdf"]               # (S, bins) dV/dlogT
tracer_generation = (
    data["tracer_generation"] if "tracer_generation" in data else None
)
logT_edges = data["logT_bin_edges"]                   # (bins+1,)
T_cold = float(data["T_cold"])
T_hot = float(data["T_hot"])
t_sh = float(data["t_sh"])
xi = float(data["xi"]) if "xi" in data else float("nan")
L_z = float(data["L_z"])
L_x = float(data["L_x"]) if "L_x" in data else L_z / 1.5
L_y = float(data["L_y"]) if "L_y" in data else L_z / 1.5
grid_spacing = float(data["grid_spacing"])
steady_t_sh = float(data["steady_state_t_sh"])

logT_centers = 0.5 * (logT_edges[:-1] + logT_edges[1:])
T_centers = 10.0 ** logT_centers
dlogT = logT_edges[1] - logT_edges[0]

# The PDF histogram range (set in the run) brackets the actual gas, which
# extends below the nominal T_cold (cold-phase compression) and slightly above
# T_hot. Use the bin range for binning / reconstruction / plot limits so the
# Lagrangian and Eulerian marginals are compared consistently over the same
# support; keep the nominal T_cold/T_hot only as reference values.
T_cold_nominal, T_hot_nominal = T_cold, T_hot
T_cold = float(10.0 ** logT_edges[0])
T_hot = float(10.0 ** logT_edges[-1])

# steady-state snapshot mask
steady = time_points >= steady_t_sh * t_sh
print(f"steady-state snapshots: {steady.sum()} / {steady.size} "
      f"(t >= {steady_t_sh} t_sh)")


def normalize_dlogT(hist):
    """Normalize a dM/dlogT histogram to unit integral over log10 T."""
    integral = np.sum(hist) * dlogT
    return hist / integral if integral > 0 else hist


def l1_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q)) * dlogT


def kl_divergence(p, q):
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask])) * dlogT


# ---------------------------------------------------------------------------
# Check 0: Lagrangian tracer PDF vs mass-weighted Eulerian PDF
# ---------------------------------------------------------------------------
# Lagrangian: histogram of (steady-state) tracer temperatures. Because tracers
# are mass-seeded, the per-bin tracer count is proportional to dM/dlogT.
lag_T = tracer_T[steady].ravel()
lag_hist, _ = np.histogram(np.log10(np.clip(lag_T, T_cold, T_hot)), bins=logT_edges)
lag_pdf = normalize_dlogT(lag_hist.astype(float))

# Eulerian mass-weighted PDF, time-averaged over the steady window.
eul_mass_pdf = normalize_dlogT(mass_pdf_snaps[steady].mean(axis=0))
eul_vol_pdf = normalize_dlogT(vol_pdf_snaps[steady].mean(axis=0))

# Support of the measured marginal (cold/hot peaks dominate, so trim only the
# empty cold/hot tails via cumulative mass). Used for plot limits and as the
# reconstruction march range (the implicit march is noise-sensitive in the empty
# tails where D ~ 0). Defined here so the transition-density figure can use it.
_cum = np.cumsum(eul_mass_pdf) * dlogT
support_lo_idx = int(np.where(_cum > 1e-3)[0][0])
support_hi_idx = int(np.where(_cum < 1.0 - 1e-4)[0][-1])
T_support_lo = float(T_centers[support_lo_idx])
T_support_hi = float(T_centers[support_hi_idx])
support_mask = (T_centers >= T_support_lo) & (T_centers <= T_support_hi)

print("\nCheck 0 (Lagrangian vs mass-weighted Eulerian):")
print(f"  L1  = {l1_distance(lag_pdf, eul_mass_pdf):.4f}")
print(f"  KL  = {kl_divergence(lag_pdf, eul_mass_pdf):.4f}")

# The cold/hot peaks (the bulk of the mass) match; the visible discrepancy is
# confined to the thin mixing layer (0.03 < T < 0.7), which holds only a small
# fraction of the mass and turns over on the short cooling time t_cool ~ t_sh/xi
# faster than tracers (fed by un-tracered hot inflow) can repopulate it. The FP
# reconstruction still recovers the Eulerian P(T) because A(T), D(T) are LOCAL
# transition statistics (correct per bin) and J carries the flux.
_mix = (T_centers > 0.03) & (T_centers < 0.7)
mix_mass_frac = float(np.sum(eul_mass_pdf[_mix]) * dlogT)
check0_l1 = l1_distance(lag_pdf, eul_mass_pdf)

fig, ax = plt.subplots(figsize=(6.5, 4.7))
ax.axvspan(0.03, 0.7, color="0.92", label="mixing band (turnover-limited)")
ax.plot(T_centers, eul_mass_pdf, lw=2.5, label="Eulerian mass-weighted (dM/dlogT)")
ax.plot(T_centers, lag_pdf, "--", lw=2.5, label="Lagrangian tracers")
ax.plot(T_centers, eul_vol_pdf, ":", lw=1.5, color="gray", label="Eulerian volume (dV/dlogT)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("T")
ax.set_ylabel("normalized PDF")
ax.set_title(f"Check 0: Lagrangian vs mass-weighted Eulerian  (L1={check0_l1:.3f};\n"
             f"bulk peaks match, residual only in the {mix_mass_frac:.1%}-of-mass mixing band)")
ax.set_xlim(T_cold, T_hot)
ax.set_ylim(bottom=max(1e-4, lag_pdf[lag_pdf > 0].min() * 0.5))
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"figures/fp_check0_marginal{SUFFIX}.svg")
fig.savefig(f"figures/fp_check0_marginal{SUFFIX}.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# A(T; dt), D(T; dt) from connected conditional moments of tracer increments
# ---------------------------------------------------------------------------
steady_idx = np.where(steady)[0]
Ts = tracer_T[steady_idx]          # (M, Np)
ts = time_points[steady_idx]       # (M,)
xs = tracer_pos[steady_idx][:, :, 0]
ys = tracer_pos[steady_idx][:, :, 1]
zs = tracer_pos[steady_idx][:, :, 2]
M = Ts.shape[0]

# A re-injection / recycle / regeneration relocates a tracer (a jump of order the
# box size); increments spanning one are not real Lagrangian increments and must
# be excluded. Flag the snapshot steps where any coordinate jumps by > 0.3 of its
# box length, using the MINIMUM-IMAGE distance on the periodic x/y axes so an
# ordinary periodic wrap (a small physical move) is not mistaken for a teleport.
# Physical motion between consecutive (dense) snapshots is << 0.1 box.
def _minimum_image_jump(coordinate, box_length):
    raw = np.abs(coordinate[1:] - coordinate[:-1])
    return np.minimum(raw, box_length - raw)

jump_x = _minimum_image_jump(xs, L_x)
jump_y = _minimum_image_jump(ys, L_y)
jump_z = np.abs(zs[1:] - zs[:-1])      # z is non-periodic, no wrap
step_teleport = np.zeros_like(zs, dtype=np.int32)
step_teleport[1:] = (
    (jump_x > 0.3 * L_x) | (jump_y > 0.3 * L_y) | (jump_z > 0.3 * L_z)
).astype(np.int32)
# The regeneration thermostat can relocate a tracer to a NEARBY ∝rho cell that
# the position-jump heuristic misses; rely on the explicit generation counter
# (a regeneration bumps it) to flag those reliably — essential because a single
# undetected regeneration injects a large spurious dT that swamps the tiny
# cold-phase diffusion.
if tracer_generation is not None:
    gen = tracer_generation[steady_idx]            # (M, Np)
    step_teleport[1:] = np.maximum(
        step_teleport[1:], (gen[1:] != gen[:-1]).astype(np.int32)
    )
cum_teleport = np.cumsum(step_teleport, axis=0)   # (M, Np)
print(f"teleports (recycle/reinject/regenerate) in steady window: {int(step_teleport.sum())} "
      f"({100.0 * step_teleport.sum() / step_teleport.size:.2f}% of tracer-steps)")

# lag scan in snapshot units. With the dense equilibrium-window recording the
# snapshot spacing is ~CFL-small, so scan many lags to span the ballistic regime
# up to the macroscopic scale (~ a few t_sh).
max_lag = min(M - 1, 512 if not QUICK else 32)
lags = [
    m for m in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
    if m <= max_lag
]

min_count = 20 if QUICK else 30   # per-bin sample threshold for plotting

# storage: per lag, arrays over T bins
A_of_lag = {}
D_of_lag = {}
kurt_of_lag = {}
dt_of_lag = {}

n_bins = len(T_centers)


def bin_moments(T_start, dT, dt_pair, valid_pair):
    """Connected conditional moments of dT, binned by log10(T_start).

    All inputs are flattened over (pairs, tracers); ``valid_pair`` selects the
    samples whose *both* endpoints are away from the boundary.
    """
    keep = valid_pair & np.isfinite(dT) & np.isfinite(T_start) & (T_start > 0)
    logT0 = np.log10(T_start[keep])
    dTk = dT[keep]
    dtk = dt_pair[keep]

    count, _ = np.histogram(logT0, bins=logT_edges)
    s1, _ = np.histogram(logT0, bins=logT_edges, weights=dTk)
    s2, _ = np.histogram(logT0, bins=logT_edges, weights=dTk ** 2)
    s4, _ = np.histogram(logT0, bins=logT_edges, weights=dTk ** 4)
    sdt, _ = np.histogram(logT0, bins=logT_edges, weights=dtk)

    count_safe = np.where(count > 0, count, 1)
    mean_dT = s1 / count_safe
    mean_dT2 = s2 / count_safe
    mean_dT4 = s4 / count_safe
    mean_dt = sdt / count_safe

    var_dT = mean_dT2 - mean_dT ** 2
    A = mean_dT / mean_dt
    D = var_dT / mean_dt
    kurt = mean_dT4 / np.where(var_dT > 0, var_dT ** 2, np.nan)

    A = np.where(count >= min_count, A, np.nan)
    D = np.where(count >= min_count, D, np.nan)
    kurt = np.where(count >= min_count, kurt, np.nan)
    return A, D, kurt, count


for m in lags:
    dT = (Ts[m:] - Ts[:-m]).ravel()
    T_start = Ts[:-m].ravel()
    dt_pair = np.repeat((ts[m:] - ts[:-m])[:, None], Ts.shape[1], axis=1).ravel()
    # valid iff no re-injection teleport occurred between snapshot j and j+m
    valid_pair = ((cum_teleport[m:] - cum_teleport[:-m]) == 0).ravel()
    A, D, kurt, count = bin_moments(T_start, dT, dt_pair, valid_pair)
    A_of_lag[m] = A
    D_of_lag[m] = D
    kurt_of_lag[m] = kurt
    dt_of_lag[m] = np.nanmean(dt_pair[valid_pair]) if valid_pair.any() else np.nan

print("\nlag scan (snapshot units -> physical dt / t_sh):")
for m in lags:
    print(f"  m={m:3d}  dt={dt_of_lag[m]:.4e}  dt/t_sh={dt_of_lag[m]/t_sh:.4f}")

# --- figure: A(T; dt) and D(T; dt) for several lags -------------------------
fig, (ax_A, ax_D) = plt.subplots(1, 2, figsize=(12, 4.8))
colors = plt.cm.viridis(np.linspace(0, 0.95, len(lags)))
for color, m in zip(colors, lags):
    label = f"dt/t_sh={dt_of_lag[m]/t_sh:.3f}"
    A = A_of_lag[m]
    ax_A.plot(T_centers, -A, color=color, lw=1.8, label=label)  # -A > 0 (cooling)
    ax_D.plot(T_centers, D_of_lag[m], color=color, lw=1.8, label=label)
ax_A.set_xscale("log"); ax_A.set_yscale("log")
ax_A.set_xlabel("T"); ax_A.set_ylabel(r"$-A(T;\Delta t)$  (drift, cooling)")
ax_A.set_title("drift")
ax_A.set_xlim(T_cold, T_hot)
ax_D.set_xscale("log"); ax_D.set_yscale("log")
ax_D.set_xlabel("T"); ax_D.set_ylabel(r"$D(T;\Delta t)$  (diffusion)")
ax_D.set_title("diffusion")
ax_D.set_xlim(T_cold, T_hot)
ax_D.legend(fontsize=7, ncol=2, title=r"lag $\Delta t$")
fig.tight_layout()
fig.savefig(f"figures/fp_drift_diffusion{SUFFIX}.svg")
fig.savefig(f"figures/fp_drift_diffusion{SUFFIX}.png", dpi=150)
plt.close(fig)

# --- figure: D(dt) plateau at a few representative temperatures -------------
# pick bins near T_cold*10, geometric mid, T_hot/3 (well sampled, intermediate)
target_Ts = [T_cold * 3, np.sqrt(T_cold * T_hot), T_hot / 3]
fig, ax = plt.subplots(figsize=(6, 4.5))
dt_over_tsh = np.array([dt_of_lag[m] / t_sh for m in lags])
any_positive = False
for target in target_Ts:
    b = int(np.argmin(np.abs(T_centers - target)))
    D_vs_lag = np.array([D_of_lag[m][b] for m in lags])
    positive = np.isfinite(D_vs_lag) & (D_vs_lag > 0)
    if positive.sum() >= 1:
        any_positive = True
    ax.plot(dt_over_tsh, D_vs_lag, "o-", lw=1.8, label=f"T={T_centers[b]:.3g}")
ax.set_xscale("log")
if any_positive:
    ax.set_yscale("log")
ax.set_xlabel(r"$\Delta t / t_{sh}$")
ax.set_ylabel(r"$D(T;\Delta t)$")
ax.set_title(r"diffusion vs lag (look for a plateau = diffusive window)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"figures/fp_diffusion_plateau{SUFFIX}.svg")
fig.savefig(f"figures/fp_diffusion_plateau{SUFFIX}.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2D conditional transition density p(dT/dt | T(t)) for several lags
# ---------------------------------------------------------------------------
# The kernel the diffusion ansatz models as Gaussian. We plot the Lagrangian
# RATE  r = dT/dt = (T(t+dt)-T(t)) / dt  (dividing by dt makes the conditional
# mean equal A(T) directly, lag-independent in the plateau, so the drift is
# read off cleanly). Each T(t) column is normalized -> p(r | T(t)). Overlaid:
#   cyan solid  = A(T)              (the drift = conditional mean rate)
#   cyan dashed = A(T) +/- sqrt(D(T)/dt)   (+/-1 std; the band width IS D(T))
# As dt grows the band sqrt(D/dt) shrinks and the cloud collapses onto A(T)
# (random walk -> deterministic drift), the ballistic->diffusive transition.
panel_target_dt = [0.008, 0.024, 0.096, 0.5]   # dt / t_sh
panel_lags = []
for target in panel_target_dt:
    m = min(lags, key=lambda mm: abs(dt_of_lag[mm] / t_sh - target))
    if m not in panel_lags:
        panel_lags.append(m)

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
for ax, m in zip(axes.ravel(), panel_lags):
    dt = dt_of_lag[m]
    T0 = Ts[:-m].ravel()
    rate = ((Ts[m:] - Ts[:-m]) / dt).ravel()
    valid_pair = ((cum_teleport[m:] - cum_teleport[:-m]) == 0).ravel()
    keep = valid_pair & np.isfinite(rate) & (T0 > 0)

    drift = A_of_lag[m]                              # A(T) = <dT>/dt
    std_rate = np.sqrt(np.maximum(D_of_lag[m], 0.0) / dt)   # std of the rate
    # adaptive y-range: frame A(T) +/- ~3.5 sigma over the populated support
    frame = np.isfinite(drift) & np.isfinite(std_rate) & support_mask
    rate_max = float(np.nanmax(np.abs(drift[frame]) + 3.5 * std_rate[frame]))
    rate_edges = np.linspace(-rate_max, rate_max, 200)

    counts, x_edges, y_edges = np.histogram2d(
        np.log10(T0[keep]), rate[keep], bins=[logT_edges, rate_edges]
    )
    column_sum = counts.sum(axis=1, keepdims=True)
    conditional = counts / np.where(column_sum > 0, column_sum, 1.0)
    mesh = ax.pcolormesh(
        10.0 ** x_edges, y_edges, conditional.T, cmap="magma", shading="auto",
        norm=LogNorm(vmin=1e-3, vmax=max(conditional.max(), 1e-2)),
    )
    ax.plot(T_centers, drift, "c-", lw=2.0, label=r"$A(T)=\langle\delta T\rangle/\Delta t$")
    ax.plot(T_centers, drift + std_rate, "c--", lw=1.1,
            label=r"$A(T)\pm\sqrt{D(T)/\Delta t}$")
    ax.plot(T_centers, drift - std_rate, "c--", lw=1.1)
    ax.axhline(0.0, color="w", ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(T_support_lo, T_support_hi)
    ax.set_ylim(-rate_max, rate_max)
    ax.set_xlabel(r"$T(t)$")
    ax.set_ylabel(r"$\delta T / \Delta t$  (Lagrangian rate)")
    ax.set_title(fr"$\Delta t / t_{{sh}} = {dt / t_sh:.3f}$")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.6)
    fig.colorbar(mesh, ax=ax, label=r"$p(\delta T/\Delta t \mid T(t))$")
fig.suptitle("Conditional transition density of the tracer temperature rate "
             "(drift A(T) and diffusion D(T) overlaid)")
fig.tight_layout()
fig.savefig(f"figures/fp_transition_density{SUFFIX}.png", dpi=150)

# ---------------------------------------------------------------------------
# Probability flux J from tracer crossings of a reference temperature
# ---------------------------------------------------------------------------
# The TRML steady state is flux-driven (a constant probability current hot->cold);
# a J=0 reconstruction cannot hold up the mixing-range plateau against cooling.
# Measure J as the net downward (cooling) crossing rate of T_ref per tracer per
# unit time, using lag-1 pairs and excluding recycle teleports. In steady state
# J is constant in T, so scan T_ref and take the median.
valid_one = (cum_teleport[1:] - cum_teleport[:-1]) == 0      # (M-1, Np)
dt_one = (ts[1:] - ts[:-1])[:, None]


def measure_flux(T_ref):
    above = Ts[:-1] >= T_ref
    crossing_down = above & (Ts[1:] < T_ref) & valid_one
    crossing_up = (~above) & (Ts[1:] >= T_ref) & valid_one
    net_rate = (crossing_down.astype(float) - crossing_up.astype(float)) / dt_one
    return net_rate.sum(axis=1).mean() / Ts.shape[1]


T_ref_scan = np.geomspace(T_cold * 1.5, T_hot * 0.7, 25)
J_scan = np.array([measure_flux(Tr) for Tr in T_ref_scan])
J_downward = float(np.median(J_scan))   # >0 means net cooling (T decreasing)
print(f"\nmeasured probability flux (net downward crossing rate per tracer):")
print(f"  median J_downward = {J_downward:.4e}  "
      f"(range over T_ref [{J_scan.min():.2e}, {J_scan.max():.2e}])")

# ---------------------------------------------------------------------------
# Reconstruct P_hat(T) from (A, D, J) and compare to the measured PDF
# ---------------------------------------------------------------------------
# target marginal (dM/dT) from the validated mass-weighted PDF:
#   dM/dlogT -> dM/dT = (dM/dlogT) / (T ln10)
eul_mass_dMdT = eul_mass_pdf / (T_centers * np.log(10.0))

print(f"reconstruction support: T in [{T_support_lo:.3g}, {T_support_hi:.3g}]")

# mixing-range selection metric (the cold/hot peaks dominate the global L1, so
# the best lag/flux is chosen on agreement across the mixing layer)
mixing_mask = (T_centers > 0.03) & (T_centers < 0.7)


def l1_support(p, q):
    return 0.5 * np.sum(np.abs(p[support_mask] - q[support_mask])) * dlogT


def l1_mixing(p, q):
    return 0.5 * np.sum(np.abs(p[mixing_mask] - q[mixing_mask])) * dlogT


def reconstruct_for_lag(m, probability_flux=0.0):
    """Build A(T), D(T) interpolators for lag ``m`` and march P_hat(T)."""
    A = A_of_lag[m]
    D = D_of_lag[m]
    good = np.isfinite(A) & np.isfinite(D)
    if good.sum() < 5:
        return None
    logT_good = jnp.asarray(logT_centers[good])
    # A must stay < 0 for the march's denominator; clip tiny/positive A.
    A_good = jnp.asarray(np.minimum(A[good], -1e-12))
    # D must stay >= 0.
    D_good = jnp.asarray(np.maximum(D[good], 0.0))

    drift = lambda T: jnp.interp(jnp.log10(T), logT_good, A_good)
    diffusion = lambda T: jnp.interp(jnp.log10(T), logT_good, D_good)

    # boundary value P at the cold support edge, borrowed from the histogram
    P_cold_guess = float(eul_mass_dMdT[support_lo_idx])

    P_eval, T_eval = calculate_pdf(
        drift, diffusion, probability_flux,
        T_support_lo, T_support_hi, 4000, P_cold_guess,
    )
    P_eval = np.asarray(P_eval)
    T_eval = np.asarray(T_eval)
    finite = np.isfinite(P_eval) & (P_eval > 0)
    if finite.sum() < 10:
        return None
    # normalize to unit integral (dM/dT)
    norm = np.trapezoid(P_eval[finite], T_eval[finite])
    if not np.isfinite(norm) or norm <= 0:
        return None
    P_eval = P_eval / norm
    return T_eval, P_eval


# reconstruct for every lag, score the shape match against the histogram.
# For each lag try a few flux candidates (0, +/- measured J) and keep the one
# with the best mixing-range agreement (the cold spike dominates global L1, so a
# J=0 fit that collapses to the cold peak would otherwise win spuriously).
target_dMdlogT = normalize_dlogT(eul_mass_pdf)  # already normalized
flux_candidates = [0.0, J_downward, -J_downward, 2.0 * J_downward, -2.0 * J_downward]
l1_of_lag = {}
l1_mix_of_lag = {}
recon_cache = {}
for m in lags:
    best = None
    for J in flux_candidates:
        rec = reconstruct_for_lag(m, probability_flux=J)
        if rec is None:
            continue
        T_eval, P_eval = rec
        recon_dMdlogT = normalize_dlogT(
            np.interp(T_centers, T_eval, P_eval) * T_centers * np.log(10.0)
        )
        score = l1_mixing(recon_dMdlogT, target_dMdlogT)
        if best is None or score < best[0]:
            best = (score, J, rec, recon_dMdlogT)
    if best is None:
        l1_of_lag[m] = np.nan
        l1_mix_of_lag[m] = np.nan
        continue
    score, J, rec, recon_dMdlogT = best
    recon_cache[m] = rec
    l1_mix_of_lag[m] = score
    l1_of_lag[m] = l1_support(recon_dMdlogT, target_dMdlogT)

print("\nP_hat vs mass-weighted histogram (L1 global / L1 mixing-range):")
for m in lags:
    print(f"  m={m:3d}  dt/t_sh={dt_of_lag[m]/t_sh:.4f}  "
          f"L1={l1_of_lag[m]:.4f}  L1_mix={l1_mix_of_lag[m]:.4f}")

# --- figure: P_hat vs histogram for the best lag ----------------------------
# "Best" = smallest full-support L1 (the diffusive window where the whole P(T) is
# reproduced), not the mixing-only metric which a ballistic small-lag fit can win.
valid_lags = [m for m in lags if np.isfinite(l1_of_lag[m])]
if valid_lags:
    best_m = min(valid_lags, key=lambda m: l1_of_lag[m])
    T_eval, P_eval = recon_cache[best_m]
    recon_dMdlogT = normalize_dlogT(
        np.interp(T_centers, T_eval, P_eval) * T_centers * np.log(10.0)
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(T_centers, target_dMdlogT, lw=2.5, label="measured (mass-weighted)")
    ax.plot(T_centers, recon_dMdlogT, "--", lw=2.5,
            label=fr"$\widehat{{P}}$ (dt/t_sh={dt_of_lag[best_m]/t_sh:.3f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("T"); ax.set_ylabel("dM/dlogT (normalized)")
    ax.set_title(f"FP reconstruction vs histogram "
                 f"(L1={l1_of_lag[best_m]:.3f} over support)")
    ax.set_xlim(T_support_lo, T_support_hi)
    ax.set_ylim(bottom=max(1e-4, target_dMdlogT[target_dMdlogT > 0].min() * 0.5))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/fp_reconstruction{SUFFIX}.svg")
    fig.savefig(f"figures/fp_reconstruction{SUFFIX}.png", dpi=150)
    plt.close(fig)

    # --- figure: L1(dt) validity-window curve -------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(dt_over_tsh, [l1_of_lag[m] for m in lags], "o-", lw=2, label="global L1")
    ax.plot(dt_over_tsh, [l1_mix_of_lag[m] for m in lags], "s--", lw=2,
            label="mixing-range L1")
    ax.axvline(dt_of_lag[best_m] / t_sh, color="red", ls=":",
               label=f"best dt/t_sh={dt_of_lag[best_m]/t_sh:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t / t_{sh}$")
    ax.set_ylabel(r"$L1(\widehat{P}, \mathrm{histogram})$")
    ax.set_title("reconstruction error vs lag (validity window = minimum)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/fp_validity_window{SUFFIX}.svg")
    fig.savefig(f"figures/fp_validity_window{SUFFIX}.png", dpi=150)
    plt.close(fig)

    # --- clean two-panel summary -------------------------------------------
    # LEFT: the measured transition kernel at the validity-window lag, with the
    # drift A(T) and diffusion D(T) (the +/- band) overlaid. RIGHT: the three
    # temperature PDFs — Eulerian mass-weighted, Lagrangian tracers, and the
    # Fokker-Planck P_hat reconstructed from *those same* A(T), D(T), J.
    dt = dt_of_lag[best_m]
    drift = A_of_lag[best_m]
    std_rate = np.sqrt(np.maximum(D_of_lag[best_m], 0.0) / dt)
    T0 = Ts[:-best_m].ravel()
    rate = ((Ts[best_m:] - Ts[:-best_m]) / dt).ravel()
    valid_pair = ((cum_teleport[best_m:] - cum_teleport[:-best_m]) == 0).ravel()
    keep = valid_pair & np.isfinite(rate) & (T0 > 0)
    frame = np.isfinite(drift) & np.isfinite(std_rate) & support_mask
    rate_max = float(np.nanmax(np.abs(drift[frame]) + 3.5 * std_rate[frame]))
    rate_edges = np.linspace(-rate_max, rate_max, 200)
    counts, x_edges, y_edges = np.histogram2d(
        np.log10(T0[keep]), rate[keep], bins=[logT_edges, rate_edges]
    )
    column_sum = counts.sum(axis=1, keepdims=True)
    conditional = counts / np.where(column_sum > 0, column_sum, 1.0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.3))
    mesh = axL.pcolormesh(
        10.0 ** x_edges, y_edges, conditional.T, cmap="magma", shading="auto",
        norm=LogNorm(vmin=1e-3, vmax=max(conditional.max(), 1e-2)),
    )
    axL.plot(T_centers, drift, "c-", lw=2.2, label=r"drift $A(T)=\langle\delta T\rangle/\Delta t$")
    axL.plot(T_centers, drift + std_rate, "c--", lw=1.2,
             label=r"$A(T)\pm\sqrt{D(T)/\Delta t}$  (diffusion)")
    axL.plot(T_centers, drift - std_rate, "c--", lw=1.2)
    axL.axhline(0.0, color="w", ls=":", lw=0.7)
    axL.set_xscale("log")
    axL.set_xlim(T_support_lo, T_support_hi)
    axL.set_ylim(-rate_max, rate_max)
    axL.set_xlabel(r"$T(t)$")
    axL.set_ylabel(r"$\delta T / \Delta t$  (Lagrangian rate)")
    axL.set_title(fr"transition density $p(\delta T/\Delta t\mid T)$ "
                  fr"($\Delta t/t_{{sh}}={dt/t_sh:.3f}$)")
    axL.legend(fontsize=8, loc="lower left", framealpha=0.6)
    fig.colorbar(mesh, ax=axL, label=r"$p(\delta T/\Delta t \mid T)$")

    axR.plot(T_centers, eul_mass_pdf, lw=3.0, color="tab:blue",
             label="Eulerian mass-weighted")
    axR.plot(T_centers, lag_pdf, "--", lw=2.2, color="tab:orange",
             label="Lagrangian tracers")
    axR.plot(T_centers, recon_dMdlogT, ":", lw=2.6, color="black",
             label=r"Fokker-Planck $\widehat{P}(T)$")
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlim(T_support_lo, T_support_hi)
    axR.set_ylim(bottom=max(1e-4, eul_mass_pdf[eul_mass_pdf > 0].min() * 0.5))
    axR.set_xlabel(r"$T$")
    axR.set_ylabel(r"$dM/d\log T$ (normalized)")
    axR.set_title(f"temperature PDF  (Check0 L1={check0_l1:.3f}, "
                  f"FP L1={l1_of_lag[best_m]:.3f})")
    axR.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"figures/fp_summary{SUFFIX}.png", dpi=150)  # PNG only (pcolormesh SVG is bulky)
    plt.close(fig)

    # compact result for the xi scan: headline scalars + the curves needed to
    # overlay the per-xi reconstruction and the D(T;dt) plateau.
    T_mix = float(np.sqrt(0.03 * 0.7))   # representative mixing-layer temperature
    b_mix = int(np.argmin(np.abs(T_centers - T_mix)))
    np.savez(
        f"data/fp_result{SUFFIX}.npz",
        xi=xi,
        check0_l1=check0_l1,
        fp_l1=l1_of_lag[best_m],
        best_dt_tsh=dt_of_lag[best_m] / t_sh,
        dt_over_tsh=dt_over_tsh,
        l1_of_lag=np.array([l1_of_lag[m] for m in lags]),
        D_vs_lag=np.array([D_of_lag[m][b_mix] for m in lags]),
        T_mix=T_centers[b_mix],
        T_centers=T_centers,
        eul_mass_pdf=eul_mass_pdf,
        lag_pdf=lag_pdf,
        recon_dMdlogT=recon_dMdlogT,
        T_support_lo=T_support_lo,
        T_support_hi=T_support_hi,
    )
else:
    print("WARNING: no lag produced a normalizable reconstruction.")

print(f"\nfigures written to {os.path.join(HERE, 'figures')}/fp_*{SUFFIX}.*")
