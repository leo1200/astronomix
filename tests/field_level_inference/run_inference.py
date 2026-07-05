#!/usr/bin/env python
"""MONEY PLOT at fine=128³ / coarse=64³ (matches the 128³ headline 2×2 figure).
Terminal logo-matching loss (128³ z-projection MSE) vs optimisation wall-time for:
  naive       : optimise ALL Fourier modes at 128³ from step 1 (full mask).
  k-windowing : k-expansion schedule (8->16->32->64) entirely at 128³.
  multigrid   : SAME schedule, low-k bands on the cheap 64³ grid (prolongation-free
                transfer), high-k bands on 128³ -- our warm-start method.

128³ backward is ~18x the 64³ cost, so the coarse stage is where multigrid pays off.
Each method is given an equal wall-time BUDGET; the multigrid's 64³ time IS counted;
the 128³ loss during the 64³ stage is a forward-only diagnostic (excluded from time).
All start from the SAME turbulent velocity.  Periodic npz saves -> plot any time.
Use --table-only to print the per-k-stage parameter count (no GPU).  PALLAS, astx/GPU.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--fine-N", type=int, default=128)
ap.add_argument("--coarse-N", type=int, default=64)
ap.add_argument("--bands", type=str, default="8:8,16:8,32:10,64:12",
                help="k_cut:steps schedule (fine Nyquist = fine-N/2).")
ap.add_argument("--coarse-cut", type=int, default=16,
                help="(legacy two-grid mode, unused when --ladder).")
ap.add_argument("--ladder", action="store_true", default=True,
                help="multigrid: assign each band to the coarsest grid with a Nyquist gap.")
ap.add_argument("--gap-ratio", type=float, default=4.0,
                help="ladder: band k_cut needs grid N >= gap_ratio*k_cut (4 = half-Nyquist gap).")
ap.add_argument("--min-grid", type=int, default=32, help="ladder: coarsest grid allowed.")
ap.add_argument("--budget", type=float, default=14400.0,
                help="per-method optimisation wall-time budget [s] (0 = run full schedule).")
ap.add_argument("--diag-every", type=int, default=1,
                help="evaluate the fine-grid diagnostic loss every N steps during coarse stage.")
ap.add_argument("--num-checkpoints", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--save-every", type=int, default=2, help="save npz every N logged points.")
ap.add_argument("--out", type=str, default="data/bench_money128.npz")
ap.add_argument("--warm-state", type=str, default="",
                help="continue: load the velocity from a saved full state as the start (theta0).")
ap.add_argument("--state-out", type=str, default="data/best_state_full128.npy",
                help="save the best (lowest-loss) reconstructed full state here (for the 2x2 panels).")
ap.add_argument("--only", choices=["naive", "k-windowing", "multigrid", "all"], default="all")
ap.add_argument("--table-only", action="store_true", help="print param-count table and exit (no GPU).")
args = ap.parse_args()

SCHED = [(int(a.split(":")[0]), int(a.split(":")[1])) for a in args.bands.split(",")]
import numpy as np


def mode_count(N, kcut):
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    return int((np.sqrt(KX**2 + KY**2 + KZ**2) < kcut).sum())


def _pow2_ceil(x):
    p = 1
    while p < x:
        p *= 2
    return p


def grid_for_band(kc):
    """Coarsest power-of-two grid resolving band k_cut with a Nyquist gap
    (N >= gap_ratio*k_cut), floored at --min-grid and capped at the target."""
    if not args.ladder:
        return args.coarse_N if kc <= args.coarse_cut else args.fine_N
    g = _pow2_ceil(int(np.ceil(args.gap_ratio * kc)))
    return int(min(max(g, args.min_grid), args.fine_N))


def param_table():
    Nf = args.fine_N
    full = 3 * Nf**3
    print(f"\n=== optimised parameters per k-stage (velocity Fourier DOF = 3 x modes) ===")
    print(f"  fine={Nf}³  ladder gap_ratio={args.gap_ratio} (N>={args.gap_ratio}·k_cut, "
          f"min {args.min_grid}³)\n")
    hdr = f"  {'stage':5s} {'k_cut':5s} {'grid(mg)':8s} {'steps':5s} " \
          f"{'#modes':>9s} {'#params(3·m)':>13s} {f'% of {Nf}³ DOF':>14s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for i, (kc, st) in enumerate(SCHED, 1):
        grid = grid_for_band(kc)
        gap = grid // 2 - kc
        m = mode_count(grid, kc)
        npar = 3 * m
        rows.append((i, kc, grid, st, m, npar, 100 * npar / full))
        print(f"  {i:<5d} {kc:<5d} {str(grid)+'³':8s} {st:<5d} "
              f"{m:>9d} {npar:>13d} {100*npar/full:>13.3f}%   (gap to Nyq: {gap})")
    print(f"\n  naive optimises the FULL fine grid every step: "
          f"3·{Nf}³ = {full:,} params ({100.0:.1f}% of {Nf}³ DOF).")
    print(f"  k-windowing uses the same per-stage #modes but ALWAYS on the {Nf}³ grid.")
    return rows


print(f"=== MONEY128  bands={SCHED}  fine={args.fine_N}  ladder={args.ladder} "
      f"gap_ratio={args.gap_ratio} min_grid={args.min_grid}  lr={args.lr} "
      f"budget={args.budget:.0f}s ===", flush=True)
param_table()
if args.table_only:
    raise SystemExit(0)

from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
import jax
_CACHE = os.path.expanduser("~/.cache/astronomix_jax_cache")
jax.config.update("jax_compilation_cache_dir", _CACHE)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 30)
import jax.numpy as jnp
from PIL import Image

from astronomix import (SimulationConfig, SimulationParams, get_registered_variables,
                        construct_primitive_state, time_integration, CodeUnits)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    PALLAS, BACKWARDS, FORWARDS, finalize_config, GravityConfig, PositivityConfig,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astropy import units as u
import astropy.constants as c
from _spectral_ops import prolong, restrict

CU = CodeUnits(3 * u.parsec, 100 * u.M_sun, 100 * u.km / u.s)
RHO0 = (2 * c.m_p / u.cm**3).to(CU.code_density).value
P0 = (3e4 * u.K / u.cm**3 * c.k_B).to(CU.code_pressure).value
B0 = (13.5 * u.microgauss / c.mu0**0.5).to(CU.code_magnetic_field).value
DEDT = (4.3e34 * u.erg / u.s).to(CU.code_energy / CU.code_time).value
NF, NC = args.fine_N, args.coarse_N


def make_cfg(N, forcing, backward):
    return SimulationConfig(
        positivity_config=PositivityConfig(default_positivity_protection=True),
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        donate_state=False, dimensionality=3,
        box_size=1.0, num_cells=N,
        differentiation_mode=BACKWARDS if backward else FORWARDS,
        num_checkpoints=args.num_checkpoints,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=forcing),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY) for _ in range(3)]),
        backend=PALLAS, pallas_block_shape=(4, 4, 8), pallas_use_triton=True)


def build(cfg, rv, rho, v, p, B):
    bxb, byb, bzb = initialize_interface_fields(B[0], B[1], B[2])
    return construct_primitive_state(
        config=cfg, registered_variables=rv, density=rho, gas_pressure=p,
        velocity_x=v[0], velocity_y=v[1], velocity_z=v[2],
        magnetic_field_x=B[0], magnetic_field_y=B[1], magnetic_field_z=B[2],
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb, interface_magnetic_field_z=bzb)


def load_image(path, N):
    img = jnp.array(Image.open(path).convert("L")); img = 1.0 - img / 255.0
    h, w = img.shape; ph, pw = (-h) % N, (-w) % N
    img = jnp.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    hp, wp = img.shape
    r = img.reshape(N, hp // N, N, wp // N).mean(axis=(1, 3))
    return 1.0 + (r - 0.01) / 0.99 * 0.01


def kmask(N, kcut):
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    return jnp.asarray((np.sqrt(KX**2 + KY**2 + KZ**2) < kcut).astype(np.float64))


def lowpass(theta, mask):
    return jnp.fft.ifftn(jnp.fft.fftn(theta, axes=(-3, -2, -1)) * mask, axes=(-3, -2, -1)).real


def main():
    cfgg = make_cfg(NF, True, False); rv = get_registered_variables(cfgg)
    params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5 / 3,
                              minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
                              turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))
    sh = (NF,) * 3; z = jnp.zeros(sh)
    s0 = build(finalize_config(cfgg, (11,) + sh), rv, jnp.ones(sh) * RHO0, (z, z, z),
               jnp.ones(sh) * P0, (jnp.ones(sh) * B0, z, z))
    cfgg = finalize_config(cfgg, s0.shape)
    turb = jax.block_until_ready(time_integration(s0, cfgg, params._replace(
        t_end=(24 * 1e4 * u.yr).to(CU.code_time).value), rv))
    di, vi, pi, mi = rv.density_index, rv.velocity_index, rv.pressure_index, rv.magnetic_index
    rms = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    t_end = 0.5 / max(rms, 1e-12)
    inv = params._replace(t_end=t_end)
    print(f"turb gen done  rms={rms:.4f}  t_end={t_end:.4f}", flush=True)

    GRIDS = sorted({grid_for_band(kc) for kc, _ in SCHED} | {NF})   # full ladder + target
    print(f"resolution ladder: {GRIDS}", flush=True)
    cfgB = {N: finalize_config(make_cfg(N, False, True), (11,) + (N,) * 3) for N in GRIDS}
    cfgF = finalize_config(make_cfg(NF, False, False), turb.shape)
    rvN = {N: get_registered_variables(make_cfg(N, False, True)) for N in GRIDS}

    def bg(N):
        rs = lambda x: x if x.shape[-1] == N else restrict(x, N)
        return (rs(turb[di][None])[0], rs(turb[pi][None])[0],
                rs(jnp.stack([turb[mi.x], turb[mi.y], turb[mi.z]])))
    BG = {N: bg(N) for N in GRIDS}
    TGT = {N: (lambda t: t / jnp.sum(t) * jnp.sum(BG[N][0]))(load_image("logo.png", N))
           for N in GRIDS}

    def make_vg(N):
        rho, p, B = BG[N]; r = rvN[N]; tg = TGT[N]
        def loss(theta, mask):
            v = lowpass(theta, mask)
            s = build(cfgB[N], r, rho, (v[0], v[1], v[2]), p, (B[0], B[1], B[2]))
            proj = jnp.sum(time_integration(s, cfgB[N], inv, r)[di], axis=2)
            return jnp.mean((proj - tg) ** 2)
        return jax.jit(jax.value_and_grad(loss))
    VG = {N: make_vg(N) for N in GRIDS}

    rhoF, pF, BF = BG[NF]
    def lossF_fwd(theta, mask):
        v = lowpass(theta, mask)
        s = build(cfgF, rv, rhoF, (v[0], v[1], v[2]), pF, (BF[0], BF[1], BF[2]))
        proj = jnp.sum(time_integration(s, cfgF, inv, rv)[di], axis=2)
        return jnp.mean((proj - TGT[NF]) ** 2)
    FLF = jax.jit(lossF_fwd)

    theta0 = jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]])
    if args.warm_state:                                  # continue from a saved reconstruction
        ws = jnp.asarray(np.load(args.warm_state))
        theta0 = jnp.stack([ws[vi.x], ws[vi.y], ws[vi.z]])
        print(f"warm-started theta0 from {args.warm_state}", flush=True)
    FULL = NF                                            # kcut covering all fine modes
    TOTAL = sum(s for _, s in SCHED)

    only = args.only
    tw = time.time()
    if only in ("all", "naive", "k-windowing"):
        jax.block_until_ready(VG[NF](theta0, kmask(NF, FULL)))
    if only in ("all", "multigrid"):
        for N in GRIDS:                       # ladder grids + diagnostic forward
            jax.block_until_ready(VG[N](restrict(theta0, N) if N != NF else theta0,
                                        kmask(N, min(N // 4, 8))))
        jax.block_until_ready(FLF(theta0, kmask(NF, 8)))
    print(f"warmup/compile done ({time.time()-tw:.0f}s)", flush=True)

    def stages_for(method):
        if method == "naive":
            return [(FULL, TOTAL, NF)]
        if method == "multigrid":
            return [(kc, st, grid_for_band(kc)) for (kc, st) in SCHED]
        return [(kc, st, NF) for (kc, st) in SCHED]   # k-windowing: all at target

    def run(method, store):
        stages = stages_for(method)
        N0 = stages[0][2]
        theta = restrict(theta0, N0) if N0 != NF else theta0
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, args.lr
        m = jnp.zeros_like(theta); vv = jnp.zeros_like(theta)
        cum = 0.0; tstep = 0; rec = []
        best_loss = float("inf"); best_theta = theta; best_kc = stages[0][0]
        for (kc, steps, N) in stages:
            if theta.shape[-1] != N:
                theta = prolong(theta, N); m = prolong(m, N); vv = prolong(vv, N)
            mask = kmask(N, kc)
            for s in range(steps):
                t0 = time.time()
                l, g = VG[N](theta, mask)
                tstep += 1
                m = b1 * m + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
                theta = theta - lr * (m / (1 - b1 ** tstep)) / (jnp.sqrt(vv / (1 - b2 ** tstep)) + eps)
                cum += time.time() - t0
                if N == NF:
                    lossF = float(l)
                elif s % args.diag_every == 0 or s == steps - 1:
                    lossF = float(FLF(prolong(theta, NF), kmask(NF, kc)))
                else:
                    lossF = None
                if lossF is not None:
                    rec.append((cum, lossF))
                    if lossF < best_loss and N == NF:   # best fine-grid iterate (for the figure)
                        best_loss = lossF; best_theta = theta; best_kc = kc
                    print(f"   [{method:11s}] kc={kc:2d}(N={N}) step {tstep:3d}: "
                          f"lossF={lossF:.4e} cum={cum:.0f}s", flush=True)
                    if len(rec) % args.save_every == 0:
                        store[method.replace("-", "_")] = np.array(rec)
                        np.savez(args.out, sched=np.array(SCHED), coarse_cut=args.coarse_cut,
                                 fine_N=NF, coarse_N=NC, **store)
                        if method == "naive" and N == NF:   # periodic best full state for the 2x2
                            rho_, p_, B_ = BG[NF]; r_ = rvN[NF]
                            vb_ = lowpass(best_theta, kmask(NF, best_kc))
                            sf_ = build(cfgB[NF], r_, rho_, (vb_[0], vb_[1], vb_[2]), p_, (B_[0], B_[1], B_[2]))
                            np.save(args.state_out, np.asarray(sf_))
                if args.budget and cum >= args.budget:
                    print(f"   [{method}] budget {args.budget:.0f}s reached -> stop", flush=True)
                    return np.array(rec), theta, best_theta, best_loss, best_kc
        return np.array(rec), theta, best_theta, best_loss, best_kc

    methods = ("naive", "k-windowing", "multigrid") if args.only == "all" else (args.only,)
    store = {}
    if os.path.exists(args.out):           # accumulate across --only invocations
        try:
            store = {k: v for k, v in np.load(args.out).items()
                     if k in ("naive", "k_windowing", "multigrid")}
        except Exception:
            store = {}
    for method in methods:
        print(f"--- {method} ---", flush=True)
        rec, theta, best_theta, best_loss, best_kc = run(method, store)
        key = method.replace("-", "_")
        store[key] = rec
        np.savez(args.out, sched=np.array(SCHED), coarse_cut=args.coarse_cut,
                 fine_N=NF, coarse_N=NC, **store)
        if method == "multigrid":
            np.save("data/best_theta_money_mg128.npy", np.asarray(theta))
        # reconstruct + save the best full (11,N,N,N) initial state for the 2x2 panels
        rho, p, B = BG[NF]; r = rvN[NF]
        vbest = lowpass(best_theta, kmask(NF, best_kc))
        s_full = build(cfgB[NF], r, rho, (vbest[0], vbest[1], vbest[2]), p, (B[0], B[1], B[2]))
        out_state = args.state_out if method == "naive" else f"data/best_state_{key}128.npy"
        np.save(out_state, np.asarray(s_full))
        print(f"  saved best full state -> {out_state} (best lossF={best_loss:.4e} "
              f"@ k<{best_kc})", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    for k, v in store.items():
        if len(v):
            print(f"  {k:12s} final lossF={v[-1,1]:.4e} @ {v[-1,0]:.0f}s ({len(v)} pts)", flush=True)


if __name__ == "__main__":
    main()
