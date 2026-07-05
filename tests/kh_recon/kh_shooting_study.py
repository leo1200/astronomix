r"""
Single vs multiple shooting for Kelvin-Helmholtz initial-condition reconstruction
=================================================================================

A fair, Adam-only comparison of two optimisation *formulations* for recovering the
unknown initial perturbation of a 2D Kelvin-Helmholtz flow from its terminal
velocity field.  The forward model is the differentiable astronomix finite-difference
hydro solver, run with the **Pallas WENO backend and its reverse-mode adjoint**.

Problem.  A fixed tanh shear base flow plus an unknown broadband transverse-velocity
seed parametrised by a 10-D mode-space control `c` (cos/sin amplitudes of streamwise
wavenumbers kx in [2,6], localised on the shear layer).  The forward map integrates
to a horizon T and we observe the final velocity field `y`.  We reconstruct `c` from
`y` (noiseless twin experiment, inviscid, N=256).

The two formulations (identical Adam optimiser -- only the formulation differs):

  * SINGLE shooting -- optimise `c` directly through ONE integration over the whole
    horizon T:                 J(c) = 1/2 || H(Phi_T(s0(c))) - y ||^2
    The gradient is the product of all M segment Jacobians; at strong nonlinear
    folding this is hypersensitive ("garbage") and Adam stalls / diverges.

  * MULTIPLE shooting (best version) -- lift M-1 independent interior states and
    impose continuity by an AUGMENTED LAGRANGIAN (lambda-update + rho-ramp), Adam on
    the inner merit at fixed (lambda, rho):
        L = 1/2||H(Phi_h(s_{M-1})) - y||^2 + sum_j[ <lam_j, r_j> + rho/2 ||r_j||^2 ],
        r_j = Phi_h(s_j) - s_{j+1},   s_0 = s0(c),   h = T/M.
    Each gradient flows through only ONE short segment -> near-linear, well-conditioned
    gradients even when the full-horizon gradient is useless.  The schedule closes the
    defects while staying optimizable (a fixed penalty cannot: small mu -> open
    defects, large mu -> data-fit stalls).

Both methods use Adam; single gets a cosine-decay LR, MS gets a constant inner LR with
the rho-ramp providing the schedule.  Both are run for the same total Adam-step budget
so the convergence curves are directly comparable.

Headline finding (N=256, M=8, 16 cold inits each, paired):

    short horizon  T=20 t_g (x~15 fold):  SINGLE wins -- recovers exactly
        (median IC error 0.002, recovers 16/16); MS good but less accurate (~0.02).
    long  horizon  T=60 t_g (x~40 fold):  MULTIPLE wins -- single DIVERGES
        (median IC error ~2.5, 0/16 recovered) while MS recovers (median ~0.08,
        9/16 < 0.1, terminal loss ~3 orders lower).

So the multiple-shooting advantage is real but regime-specific: it is an *optimisation*
advantage that appears only once the single-shooting horizon is too chaotic to
differentiate through.  The 2x3 figure (rows = horizon, cols = J / IC error / defect)
shows this crossover.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
Run one (horizon, init) cell on one GPU (writes data/study_N256_T<hor>_i<init>.npz,
resumable -- skips a cell already at the target step budget):

    PYTHONPATH=<repo-root> CUDA_VISIBLE_DEVICES=0 \
        python kh_shooting_study.py run --horizon 60 --init 0

Fan the whole campaign across GPUs (e.g. inits 0..15 over GPUs 0,1,4), for both
horizons -- plain shell, no extra script needed:

    for T in 20 60; do for i in $(seq 0 15); do
        g=$(( i % 3 )); g=$([ $g -eq 2 ] && echo 4 || echo $g)
        CUDA_VISIBLE_DEVICES=$g PYTHONPATH=<repo-root> \
            python kh_shooting_study.py run --horizon $T --init $i &
    done; wait; done

Aggregate paired statistics (Wilson 68% CIs) for one horizon:

    python kh_shooting_study.py agg --horizon 60

Generate the 2x3 figure (no GPU needed):

    python kh_shooting_study.py plot --horizons 20 60 --out figures/study_2x3_N256.png

Run against the refactor worktree where the Pallas reverse adjoint lives
(`PYTHONPATH=.../astronomix-refactor-port`); env = jax 0.6.2 + optax.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from functools import partial
from typing import NamedTuple

# ----------------------------------------------------------------------------
# defaults (the converged campaign settings)
# ----------------------------------------------------------------------------
N_DEFAULT = 256
M_DEFAULT = 8
SNITER_DEFAULT = 1980      # single-shooting Adam steps (matched to MS budget)
INNER_DEFAULT = 180        # AL inner Adam steps per outer
OUTER_DEFAULT = 11         # AL outer iterations  (INNER*OUTER total steps)
LR_DEFAULT = 3e-3
RHO0_DEFAULT = 1.0
RHOFAC_DEFAULT = 1.6
KMIN_DEFAULT, KMAX_DEFAULT = 2, 6
SEED_DEFAULT = 0
INITSCALE_DEFAULT = 1e-2
REC_DEFAULT = 10           # record metrics every REC Adam steps
PALLAS_BLOCK = (4, 4, 4)
DATA_DIR = "data"


def cell_path(n, horizon, init, data_dir=DATA_DIR):
    return os.path.join(data_dir, f"study_N{n}_T{horizon:.0f}_i{init}.npz")


# ============================================================================
#  KH problem (forward operator, broadband seed, observation) -- self-contained
# ============================================================================
class KHParams(NamedTuple):
    n: int = N_DEFAULT
    box: float = 1.0
    yc: float = 0.5            # shear-layer centre
    delta: float = 0.02        # shear-layer thickness
    dV: float = 1.0            # velocity jump across the layer
    rho0: float = 1.0
    mach: float = 0.3          # dV / c_s (low -> ~incompressible)
    gamma: float = 5.0 / 3.0
    reynolds: float = 2000.0
    c_cfl: float = 1.0
    num_checkpoints: int = 800
    k_min: int = KMIN_DEFAULT
    k_max: int = KMAX_DEFAULT
    seed_amp: float = 1e-2
    env_width: float = 0.04    # Gaussian envelope half-width in y around yc

    @property
    def pressure(self):
        c_s = self.dV / self.mach
        return c_s ** 2 * self.rho0 / self.gamma

    @property
    def viscosity(self):
        return self.rho0 * self.dV * self.delta / self.reynolds

    @property
    def growth_time(self):
        return self.delta / self.dV


def _build_backend(khp, horizon, pallas):
    """Config + params for the backward (reverse-mode) inviscid KH solver."""
    import jax.numpy as jnp  # noqa: F401  (kept local so `plot`/`agg` need no jax)
    from astronomix import SimulationConfig, SimulationParams
    from astronomix.option_classes.simulation_config import (
        BACKWARDS, DYNAMIC_VISCOSITY, FINITE_DIFFERENCE, OPEN_BOUNDARY, PALLAS,
        PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D, SnapshotSettings,
    )

    T = horizon * khp.growth_time
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, dimensionality=2, num_cells=khp.n,
        box_size=khp.box, progress_bar=False, diffusion=False,
        viscosity_type=DYNAMIC_VISCOSITY, differentiation_mode=BACKWARDS,
        num_checkpoints=khp.num_checkpoints,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)),
    )
    if pallas:
        config = config._replace(backend=PALLAS, pallas_block_shape=PALLAS_BLOCK,
                                 pallas_use_triton=True)
    params = SimulationParams(viscosity=khp.viscosity, t_end=T, C_cfl=khp.c_cfl, gamma=khp.gamma)
    return config, params, T


def _make_problem(khp, horizon, pallas):
    """Returns the closures the optimisers need (seed_of, s0_of, Phi, PhiT, H), plus
    the truth seed, the terminal observation y, and an ic_err metric."""
    import jax
    import jax.numpy as jnp
    from astronomix import (get_helper_data, get_registered_variables, time_integration)
    from astronomix.initial_condition_generation.construct_primitive_state import (
        construct_primitive_state)
    from astronomix.option_classes.simulation_config import finalize_config

    config, params, T = _build_backend(khp, horizon, pallas)
    M = M_DEFAULT
    h = T / M
    rv = get_registered_variables(config)
    centers = get_helper_data(config).geometric_centers
    X, Y = centers[..., 0], centers[..., 1]

    ks = jnp.arange(khp.k_min, khp.k_max + 1)
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    cosx = jnp.cos(2 * jnp.pi * ks[None, None, :] * X[..., None])
    sinx = jnp.sin(2 * jnp.pi * ks[None, None, :] * X[..., None])

    def base_state(seed):
        vx = (khp.dV / 2.0) * jnp.tanh((Y - khp.yc) / khp.delta)
        return construct_primitive_state(
            config=config, registered_variables=rv,
            density=khp.rho0 * jnp.ones_like(vx),
            velocity_x=vx + seed[0], velocity_y=seed[1],
            gas_pressure=khp.pressure * jnp.ones_like(vx))

    config = finalize_config(config, base_state(jnp.zeros((2, khp.n, khp.n))).shape)

    def seed_of(c):
        svy = env * jnp.sum(c[:, 0] * cosx + c[:, 1] * sinx, axis=-1)
        return jnp.stack([jnp.zeros_like(svy), svy])

    s0_of = lambda c: base_state(seed_of(c))
    Phi = jax.jit(lambda s: time_integration(s, config, params._replace(t_end=h), rv))
    PhiT = jax.jit(lambda s: time_integration(s, config, params._replace(t_end=T), rv))
    H = lambda s: jnp.stack([s[rv.velocity_index.x], s[rv.velocity_index.y]])

    # truth seed (broadband, rms-normalised) and the M-segment terminal observation
    def random_seed(key):
        ka, kp = jax.random.split(key)
        amps = jax.random.normal(ka, ks.shape)
        phis = jax.random.uniform(kp, ks.shape, minval=0.0, maxval=2 * jnp.pi)
        phases = 2 * jnp.pi * ks[None, None, :] * X[..., None] + phis[None, None, :]
        svy = jnp.sum(amps[None, None, :] * jnp.cos(phases), axis=-1) * env
        svy = khp.seed_amp * khp.dV * svy / jnp.sqrt(jnp.mean(svy ** 2) + 1e-30)
        return jnp.stack([jnp.zeros_like(svy), svy])

    truth = random_seed(jax.random.PRNGKey(SEED_DEFAULT))
    s = base_state(truth)
    for _ in range(M):
        s = Phi(s)
    y = H(s)
    tnorm = float(jnp.linalg.norm(truth))
    ic_err = lambda c: float(jnp.linalg.norm(seed_of(c) - truth) / tnorm)
    return dict(seed_of=seed_of, s0_of=s0_of, base_state=base_state, Phi=Phi, PhiT=PhiT,
                H=H, y=y, truth=truth, ic_err=ic_err, n_modes=len(ks), M=M,
                box=khp.box, n=khp.n)


# ============================================================================
#  run one (horizon, init) cell: best single + best AL-MS
# ============================================================================
def run_cell(horizon, init, *, n=N_DEFAULT, sniter=SNITER_DEFAULT, inner=INNER_DEFAULT,
             outer=OUTER_DEFAULT, lr=LR_DEFAULT, rho0=RHO0_DEFAULT, rhofac=RHOFAC_DEFAULT,
             rec=REC_DEFAULT, pallas=True, out=None):
    import time

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    out = out or cell_path(n, horizon, init)
    if os.path.exists(out):
        d = np.load(out)
        if int(d["single_step"].max()) >= sniter and int(d["ms_step"].max()) >= inner * outer:
            print(f"[skip] {out} already complete"); return out

    khp = KHParams(n=n)
    P = _make_problem(khp, horizon, pallas)
    seed_of, s0_of, Phi, PhiT, H, y, ic_err, M = (
        P["seed_of"], P["s0_of"], P["Phi"], P["PhiT"], P["H"], P["y"], P["ic_err"], P["M"])
    c0 = INITSCALE_DEFAULT * jax.random.normal(jax.random.PRNGKey(100 + init), (P["n_modes"], 2))
    print(f"N={n} T={horizon} M={M} init={init} | single {sniter} | AL {inner}x{outer}="
          f"{inner * outer} | pallas={pallas}", flush=True)

    # ---- best single: Adam + cosine-decay LR over the full budget ----
    loss_single = jax.jit(lambda c: 0.5 * jnp.sum((H(PhiT(s0_of(c))) - y) ** 2))
    opt_s = optax.adam(optax.cosine_decay_schedule(lr, sniter, alpha=0.05))

    @jax.jit
    def step_s(c, st):
        g = jax.grad(loss_single)(c); u, st = opt_s.update(g, st)
        return optax.apply_updates(c, u), st

    c = c0; st = opt_s.init(c)
    s_step, s_J, s_ic = [], [], []
    t0 = time.time()
    for it in range(sniter + 1):
        if it % rec == 0:
            s_step.append(it); s_J.append(float(loss_single(c))); s_ic.append(ic_err(c))
        if it < sniter:
            c, st = step_s(c, st)
    c_single = np.asarray(c)
    print(f"  single: ic {s_ic[0]:.2f}->{s_ic[-1]:.2f}  J {s_J[0]:.2e}->{s_J[-1]:.2e}  "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # ---- best multiple shooting: augmented Lagrangian, Adam inner ----
    vPhi = jax.vmap(Phi)

    def residuals(c, S):
        starts = jnp.concatenate([s0_of(c)[None], S], axis=0)
        finals = vPhi(starts)
        return finals[:-1] - starts[1:], H(finals[-1]) - y     # defects, data residual

    def merit(p, lam, rho):
        rc, rd = residuals(p["c"], p["S"])
        return 0.5 * jnp.sum(rd ** 2) + jnp.sum(lam * rc) + 0.5 * rho * jnp.sum(rc ** 2)

    @jax.jit
    def jdef(p):
        rc, rd = residuals(p["c"], p["S"])
        return 0.5 * jnp.sum(rd ** 2), jnp.sum(rc ** 2)

    opt_m = optax.adam(lr)

    @jax.jit
    def step_m(p, st, lam, rho):
        g = jax.grad(merit)(p, lam, rho); u, st = opt_m.update(g, st)
        return optax.apply_updates(p, u), st

    S = []; sj = s0_of(c0)                       # feasible interiors: forward-propagate cold IC
    for _ in range(M - 1):
        sj = Phi(sj); S.append(sj)
    p = {"c": c0, "S": jnp.stack(S)}
    lam = jnp.zeros_like(p["S"]); rho = rho0
    st = opt_m.init(p)
    m_step, m_J, m_def, m_ic = [], [], [], []
    gstep = 0; prev_def = None
    jj, dd = jdef(p)
    m_step.append(0); m_J.append(float(jj)); m_def.append(float(dd)); m_ic.append(ic_err(p["c"]))
    t0 = time.time()
    for _ in range(outer):
        for _ in range(inner):
            p, st = step_m(p, st, lam, rho)
            gstep += 1
            if gstep % rec == 0:
                jj, dd = jdef(p)
                m_step.append(gstep); m_J.append(float(jj))
                m_def.append(float(dd)); m_ic.append(ic_err(p["c"]))
        rc, _ = residuals(p["c"], p["S"])
        lam = lam + rho * rc
        _, defect = jdef(p)
        if prev_def is not None and float(defect) > 0.7 * prev_def:
            rho = min(rho * rhofac, 1e6)
        prev_def = float(defect)
    print(f"  AL-MS:  ic {m_ic[0]:.2f}->{m_ic[-1]:.2f}  J {m_J[0]:.2e}->{m_J[-1]:.2e}  "
          f"def {m_def[0]:.2e}->{m_def[-1]:.2e}  rho={rho:.1e}  [{time.time() - t0:.0f}s]", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, N=n, T=horizon, M=M, init=init, lr=lr,
             single_step=np.asarray(s_step), single_J=np.asarray(s_J), single_ic=np.asarray(s_ic),
             ms_step=np.asarray(m_step), ms_J=np.asarray(m_J), ms_def=np.asarray(m_def), ms_ic=np.asarray(m_ic),
             single_final_J=s_J[-1], single_final_ic=s_ic[-1],
             ms_final_J=m_J[-1], ms_final_ic=m_ic[-1], ms_final_def=m_def[-1],
             c_single=c_single, c_ms=np.asarray(p["c"]))
    print(f"-> {out}", flush=True)
    return out


# ============================================================================
#  aggregate + figure  (numpy/matplotlib only -- no jax / no GPU)
# ============================================================================
def _load_cells(n, horizon, data_dir=DATA_DIR):
    import numpy as np
    files = sorted(glob.glob(os.path.join(data_dir, f"study_N{n}_T{horizon:.0f}_i*.npz")),
                   key=lambda f: int(f.split("_i")[-1].replace(".npz", "")))
    return files, [np.load(f) for f in files]


def _wilson(k, num, z=1.0):
    if num == 0:
        return (0.0, 0.0, 0.0)
    p = k / num; d = 1 + z * z / num
    c = (p + z * z / (2 * num)) / d
    h = z * math.sqrt(p * (1 - p) / num + z * z / (4 * num * num)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def aggregate(horizon, *, n=N_DEFAULT, data_dir=DATA_DIR):
    import numpy as np
    files, runs = _load_cells(n, horizon, data_dir)
    if not runs:
        print(f"no data for N={n} T={horizon:.0f}"); return
    sIC = np.array([float(r["single_final_ic"]) for r in runs])
    mIC = np.array([float(r["ms_final_ic"]) for r in runs])
    sJ = np.array([float(r["single_final_J"]) for r in runs])
    mJ = np.array([float(r["ms_final_J"]) for r in runs])
    mDef = np.array([float(r["ms_final_def"]) for r in runs])
    num = len(runs)
    print(f"=== N={n} T={horizon:.0f} M={int(runs[0]['M'])}  (n={num}) ===")
    for name, sv, mv in (("IC error", sIC, mIC), ("terminal J", sJ, mJ)):
        wins = int(np.sum(mv < sv)); p, lo, hi = _wilson(wins, num)
        ratio = np.median(mv) / np.median(sv) if np.median(sv) else float("nan")
        print(f"  {name}: MS-better {wins}/{num} = {p:.2f} [{lo:.2f},{hi:.2f}] | "
              f"median single {np.median(sv):.3e}  MS {np.median(mv):.3e}  ratio {ratio:.3f}")
    for name, v in (("single", sIC), ("MS", mIC)):
        k = int(np.sum(v < 0.1)); p, lo, hi = _wilson(k, num)
        print(f"  recovery(ic<0.1) {name}: {k}/{num} = {p:.2f} [{lo:.2f},{hi:.2f}]")
    print(f"  MS defect: median {np.median(mDef):.2e}  max {np.max(mDef):.2e}")


def make_figure(horizons, out, *, n=N_DEFAULT, data_dir=DATA_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    C_S, C_M = "#1f77b4", "#d62728"   # single, multiple
    rows = [(T, _load_cells(n, T, data_dir)[1]) for T in horizons]
    for T, runs in rows:
        if not runs:
            raise SystemExit(f"no data for N={n} T={T:.0f}")

    def mean_curve(runs, stepkey, key):       # robust to mixed-length traces
        L = max(len(r[key]) for r in runs)
        sel = [r[key] for r in runs if len(r[key]) == L]
        steps = next(r[stepkey] for r in runs if len(r[key]) == L)
        return steps, np.mean(sel, axis=0)

    cols = [("single_J", "ms_J", r"terminal loss $J$", True, False),
            ("single_ic", "ms_ic", r"IC error $\|c-c^\star\|/\|c^\star\|$", True, False),
            (None, "ms_def", "continuity defect (MS)", True, True)]
    fig, ax = plt.subplots(len(rows), 3, figsize=(15, 4.2 * len(rows)), squeeze=False)
    for ri, (T, runs) in enumerate(rows):
        for ci, (ks_, km, ylab, logy, ms_only) in enumerate(cols):
            a = ax[ri][ci]
            if not ms_only:
                for r in runs:
                    a.plot(r["single_step"], r[ks_], color=C_S, alpha=0.2, lw=0.7)
                a.plot(*mean_curve(runs, "single_step", ks_), color=C_S, lw=2.3, label="single shooting")
            for r in runs:
                a.plot(r["ms_step"], r[km], color=C_M, alpha=0.2, lw=0.7)
            a.plot(*mean_curve(runs, "ms_step", km), color=C_M, lw=2.3, label="multiple shooting")
            if logy:
                a.set_yscale("log")
            a.set_ylabel(ylab); a.grid(alpha=0.25)
            if ri == len(rows) - 1:
                a.set_xlabel("total Adam step")
            if ri == 0 and ci == 0:
                a.legend(fontsize=9, loc="best")
        ax[ri][0].text(-0.30, 0.5, f"$T={T:.0f}\\,t_g$  (n={len(runs)})",
                       transform=ax[ri][0].transAxes, rotation=90, va="center",
                       ha="center", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> {out}  (" + ", ".join(f"T{T:.0f} n={len(r)}" for T, r in rows) + ")")


# ============================================================================
#  2x4 reconstruction-example figure
#  cols: true IC | optimized IC (both v_y') ; observed final | reconstructed final (both omega_z)
#  rows: one successful reconstruction per horizon (the winning method at that horizon)
# ============================================================================
def _best_cell(horizon, n, data_dir):
    """Best SUCCESSFUL (ic<0.1) reconstruction at this horizon, across both methods:
    returns (method, init, c_rec, ic_err).  Naturally picks single at short horizon,
    MS at long horizon."""
    import numpy as np
    files, runs = _load_cells(n, horizon, data_dir)
    best = None
    for f, r in zip(files, runs):
        init = int(f.split("_i")[-1].replace(".npz", ""))
        for method, ickey, ckey in (("single", "single_final_ic", "c_single"),
                                     ("multiple", "ms_final_ic", "c_ms")):
            ic = float(r[ickey])
            if ic < 0.1 and (best is None or ic < best[3]):
                best = (method, init, np.asarray(r[ckey]), ic)
    if best is None:
        raise SystemExit(f"no successful (ic<0.1) reconstruction found at T={horizon:.0f}")
    return best


def _vorticity(v, dx):
    """omega_z = dvy/dx - dvx/dy, central differences, periodic in x (v: (2,Nx,Ny))."""
    import numpy as np
    dvydx = (np.roll(v[1], -1, 0) - np.roll(v[1], 1, 0)) / (2 * dx)
    dvxdy = (np.roll(v[0], -1, 1) - np.roll(v[0], 1, 1)) / (2 * dx)
    return dvydx - dvxdy


def _recon_fields(horizons, n, data_dir, pallas):
    """Forward-integrate truth and recovered ICs for each horizon (needs a GPU)."""
    import jax
    import numpy as np
    rows = []
    for T in horizons:
        method, init, c_rec, ic = _best_cell(T, n, data_dir)
        P = _make_problem(KHParams(n=n), T, pallas)
        seed_of, base_state, PhiT, H = P["seed_of"], P["base_state"], P["PhiT"], P["H"]
        dx = P["box"] / P["n"]
        truth = P["truth"]
        c0 = INITSCALE_DEFAULT * jax.random.normal(jax.random.PRNGKey(100 + init), (P["n_modes"], 2))
        ic_true = np.asarray(truth[1])                       # v_y' of the truth seed
        ic_init = np.asarray(seed_of(c0)[1])                 # v_y' of the cold start
        ic_opt = np.asarray(seed_of(c_rec)[1])               # v_y' of the optimized seed
        w_obs = _vorticity(np.asarray(H(PhiT(base_state(truth)))), dx)       # S_T(truth)
        w_init = _vorticity(np.asarray(H(PhiT(P["s0_of"](c0)))), dx)         # S_T(cold start)
        w_rec = _vorticity(np.asarray(H(PhiT(P["s0_of"](c_rec)))), dx)       # S_T(recovered)
        rows.append(dict(T=float(T), method=method, init=int(init), ic_err=float(ic),
                         ic_true=ic_true, ic_init=ic_init, ic_opt=ic_opt,
                         w_obs=w_obs, w_init=w_init, w_rec=w_rec))
        print(f"  T={T:.0f}: {method} init {init}  ic_err={ic:.4f}", flush=True)
    return rows, P["box"]


def make_recon_figure(out, horizons=(20.0, 60.0), *, n=N_DEFAULT, data_dir=DATA_DIR,
                      pallas=True, cache=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cache = cache or os.path.join(data_dir, "recon_fields.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        rows, box = list(z["rows"]), float(z["box"])
    else:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            from autocvd import autocvd
            autocvd(num_gpus=1)
        rows, box = _recon_fields(horizons, n, data_dir, pallas)
        np.savez(cache, rows=np.array(rows, dtype=object), box=box)
        print(f"  cached -> {cache}")

    # 4x3 grid: columns = (truth, initialization, optimized); per horizon two rows,
    # an initial-state row (v_y') above a final-state row (omega_z).
    ylo, yhi = 0.28, 0.72                          # zoom onto the shear layer
    ext = [0, box, ylo, yhi]
    col_titles = ["truth", "initialization", "optimized"]
    lab = {"single": "single shooting", "multiple": "multiple shooting"}

    plot_rows = []   # (triple of fields, kind, ylabel)
    for row in rows:
        plot_rows.append(([row["ic_true"], row["ic_init"], row["ic_opt"]], "ic",
                          f"$T={row['T']:.0f}\\,t_g$ ({lab[row['method']]})\n"
                          f"initial state $v_y'$\nIC error $= {row['ic_err']:.3f}$"))
        plot_rows.append(([row["w_obs"], row["w_init"], row["w_rec"]], "w",
                          f"$T={row['T']:.0f}\\,t_g$\nfinal state $\\omega_z$"))

    fig, ax = plt.subplots(len(plot_rows), 3, figsize=(12, 2.7 * len(plot_rows)),
                           constrained_layout=True)
    ax = np.atleast_2d(ax)
    for ri, (fields, kind, ylabel) in enumerate(plot_rows):
        m = fields[0].shape[0]
        j0, j1 = int(ylo * m), int(yhi * m)
        crop = [f[:, j0:j1] for f in fields]
        if kind == "ic":
            vlim = float(np.max(np.abs(crop)))
        else:
            vlim = float(np.percentile(np.abs(np.stack(crop)), 99.5))
        ims = []
        for ci, fld in enumerate(crop):
            a = ax[ri][ci]
            im = a.imshow(fld.T, origin="lower", extent=ext, cmap="RdBu_r",
                          vmin=-vlim, vmax=vlim, aspect="auto", interpolation="bilinear",
                          rasterized=True)
            ims.append(im)
            if ri == 0:
                a.set_title(col_titles[ci], fontsize=12, pad=5)
            a.set_xticks([]); a.set_yticks([])     # axes span the box; ticks/numbers carry no info
        ax[ri][0].set_ylabel(ylabel, fontsize=9.5)
        fig.colorbar(ims[2], ax=ax[ri][:], shrink=0.85, aspect=30, location="right",
                     label=(r"$v_y'$" if kind == "ic" else r"$\omega_z$"), pad=0.01)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    svg = os.path.splitext(out)[0] + ".svg"   # imshow pixels rasterized, text/axes vector
    fig.savefig(svg, dpi=160, bbox_inches="tight")
    print(f"-> {out}\n-> {svg}")


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Single vs multiple shooting (Adam) for KH IC reconstruction.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one (horizon, init) cell on the current GPU")
    r.add_argument("--horizon", type=float, required=True, help="final time in KH growth times")
    r.add_argument("--init", type=int, required=True, help="cold-init index")
    r.add_argument("--n", type=int, default=N_DEFAULT)
    r.add_argument("--sniter", type=int, default=SNITER_DEFAULT)
    r.add_argument("--inner", type=int, default=INNER_DEFAULT)
    r.add_argument("--outer", type=int, default=OUTER_DEFAULT)
    r.add_argument("--no-pallas", action="store_true", help="use native JAX backend instead of Pallas")
    r.add_argument("--out", default=None)

    a = sub.add_parser("agg", help="print paired statistics for one horizon")
    a.add_argument("--horizon", type=float, required=True)
    a.add_argument("--n", type=int, default=N_DEFAULT)

    p = sub.add_parser("plot", help="render the 2x3 convergence figure (no GPU)")
    p.add_argument("--horizons", type=float, nargs="+", default=[20.0, 60.0])
    p.add_argument("--n", type=int, default=N_DEFAULT)
    p.add_argument("--out", default="figures/study_2x3_N256.png")

    rc = sub.add_parser("recon", help="render the 2x4 reconstruction-example figure "
                                      "(forward-sims truth + recovered IC; needs a GPU unless cached)")
    rc.add_argument("--horizons", type=float, nargs="+", default=[20.0, 60.0])
    rc.add_argument("--n", type=int, default=N_DEFAULT)
    rc.add_argument("--out", default="figures/recon_4x3_N256.png")
    rc.add_argument("--no-pallas", action="store_true")

    args = ap.parse_args()
    if args.cmd == "run":
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            from autocvd import autocvd
            autocvd(num_gpus=1)
        run_cell(args.horizon, args.init, n=args.n, sniter=args.sniter,
                 inner=args.inner, outer=args.outer, pallas=not args.no_pallas, out=args.out)
    elif args.cmd == "agg":
        aggregate(args.horizon, n=args.n)
    elif args.cmd == "plot":
        make_figure(args.horizons, args.out, n=args.n)
    elif args.cmd == "recon":
        make_recon_figure(args.out, horizons=tuple(args.horizons), n=args.n,
                          pallas=not args.no_pallas)


if __name__ == "__main__":
    main()
