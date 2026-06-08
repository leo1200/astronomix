"""
THE BIG TEST: full-space CONSTRAINED Gauss-Newton multiple shooting (Bock),
condensed solve. Variables theta=(p, s_1..s_{M-1}): p = mode-space IC coeffs,
s_j = FULL interior states. Continuity Phi(s_j)-s_{j+1}=0 are HARD constraints
(linearized each step); the data misfit H(Phi(s_{M-1}))-y is the GN least-squares
objective. One simultaneous Newton step over all states via condensing:

  Delta s_{j+1} = G_j Delta s_j + r_c^j,  Delta s_0 = B Delta p
  => Delta s_{M-1} = P Delta p + w,  P = (prod G_j) B,  w = accumulated defects
  minimize ||r_d + H' G_{M-1}(P dp + w)||^2  (LM-damped, np x np)  -> back-substitute.

All matrix-free via forward-mode jvp through ONE segment (length T/M) -- so the
per-segment sensitivity is e^{lambda T/M}, never the e^{lambda T} monodromy.

KEY: with FEASIBLE cold interiors (forward-prop of cold IC, defects=0) this is
provably identical to single-shooting GN. The genuine MS test is INFEASIBLE
interiors (KH_INTERIOR=base: interiors=unperturbed reference, no truth info).

Env: KH_TREC, KH_M, KH_RE, KH_KMIN, KH_KMAX, KH_SEED(truth), KH_INIT, KH_INITSCALE,
     KH_NOISE, KH_ITERS, KH_INTERIOR in {feasible,base}, KH_RHO, KH_OUT.
"""
# ==== GPU selection ====
import os, sys
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd
    autocvd(num_gpus=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ruff: noqa: E402
# =======================
import time
import jax, jax.numpy as jnp, numpy as np
import problem as P
from astronomix import time_integration

N = int(os.environ.get("KH_N", 64))
RE = float(os.environ.get("KH_RE", 2000))
TREC = float(os.environ.get("KH_TREC", 80))
M_ = int(os.environ.get("KH_M", 4))
KMIN = int(os.environ.get("KH_KMIN", 2)); KMAX = int(os.environ.get("KH_KMAX", 6))
TRUTH_SEED = int(os.environ.get("KH_SEED", 0))
INIT = int(os.environ.get("KH_INIT", 0)); INITSCALE = float(os.environ.get("KH_INITSCALE", 1e-2))
NOISE = float(os.environ.get("KH_NOISE", 1e-2))
ITERS = int(os.environ.get("KH_ITERS", 40))
INTERIOR = os.environ.get("KH_INTERIOR", "base")     # feasible | base
RHO = float(os.environ.get("KH_RHO", 1.0))            # continuity weight in merit
OUT = os.environ.get("KH_OUT", f"data/msgn_{INTERIOR}_M{M_}_T{TREC:.0f}_i{INIT}.npz")


def main():
    khp = P.KHParams(n=N, reynolds=RE, k_min=KMIN, k_max=KMAX)
    T = TREC * khp.growth_time; h = T / M_
    cfg, par = P.make_config_params(khp, h, snapshots=0, backward=False)  # forward-mode
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0p = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0p.shape)
    ks = jnp.arange(KMIN, KMAX + 1)
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    cosx = jnp.cos(2 * jnp.pi * ks[None, None, :] * X[..., None])
    sinx = jnp.sin(2 * jnp.pi * ks[None, None, :] * X[..., None])

    def seed_of(c):
        svy = env * jnp.sum(c[:, 0] * cosx + c[:, 1] * sinx, axis=-1)
        return jnp.stack([jnp.zeros_like(svy), svy])

    def s0_of(c):
        return P.build_state(seed_of(c), khp, cfg, rv, X, Y)

    Phi = jax.jit(lambda s: time_integration(s, cfg, par._replace(t_end=h), rv))
    H = lambda s: P.velocity_of(s, rv)
    base_state = s0_of(jnp.zeros((len(ks), 2)))

    @jax.jit
    def Gjvp(s, v):                                   # (Phi(s), dPhi/ds . v)
        return jax.jvp(Phi, (s,), (v,))

    @jax.jit
    def s0tan(c, dc):                                 # (s0, B dc)  (B constant: linear)
        return jax.jvp(s0_of, (c,), (dc,))

    @jax.jit
    def Gjvp_batch(s, V):                             # G(s) applied to a batch of tangents
        return jax.vmap(lambda v: jax.jvp(Phi, (s,), (v,))[1])(V)

    @jax.jit
    def s0tan_batch(c, E):                            # B applied to a batch of dc
        return jax.vmap(lambda e: jax.jvp(s0_of, (c,), (e,))[1])(E)

    truth = P.random_broadband_seed(jax.random.PRNGKey(TRUTH_SEED), khp, X, Y)
    s = P.build_state(truth, khp, cfg, rv, X, Y)
    for _ in range(M_):
        s = Phi(s)
    obs_clean = H(s)
    b = base_state
    for _ in range(M_):
        b = Phi(b)
    pert_rms = float(jnp.sqrt(jnp.mean((obs_clean - H(b)) ** 2)))
    nsd = NOISE * pert_rms
    y = obs_clean + nsd * jax.random.normal(jax.random.PRNGKey(12345), obs_clean.shape)
    floor = 0.5 * nsd ** 2 * y.size
    print(f"  MSGN M={M_} T={TREC} interior={INTERIOR}: pert_rms={pert_rms:.3e} nsd={nsd:.3e}")

    def ic_err(c):
        return float(jnp.linalg.norm(seed_of(c) - truth) / jnp.linalg.norm(truth))

    # ---- init ----
    c = INITSCALE * jax.random.normal(jax.random.PRNGKey(100 + INIT), (len(ks), 2))
    s0 = s0_of(c)
    if INTERIOR == "feasible":                        # forward-prop cold IC (defects=0)
        S = []; sj = s0
        for _ in range(M_ - 1):
            sj = Phi(sj); S.append(sj)
    else:                                             # infeasible: unperturbed reference
        S = [base_state for _ in range(M_ - 1)]
    S = [jnp.asarray(x) for x in S]

    def residuals(c, S):
        s0 = s0_of(c)
        starts = [s0] + S
        finals = [Phi(st) for st in starts]           # f_j = Phi(s_j)
        rc = [finals[j] - starts[j + 1] for j in range(M_ - 1)]
        rd = (H(finals[-1]) - y)
        return starts, finals, rc, rd

    def merit(c, S):
        _, finals, rc, rd = residuals(c, S)
        m = 0.5 * jnp.sum(rd ** 2)
        for r in rc:
            m = m + 0.5 * RHO * jnp.sum(r ** 2)
        return float(m), 0.5 * float(jnp.sum(rd ** 2))

    lam = 1e-2
    t0 = time.time()
    m, dcost = merit(c, S)
    print(f"  it=0: merit={m:.3e} data={dcost:.3e} ic_err={ic_err(c):.3f}")
    npv = c.size
    for it in range(ITERS):
        starts, finals, rc, rd = residuals(c, S)
        sML = starts[-1]                              # s_{M-1}
        # P columns (vmapped over the np basis tangents): propagate B E through
        # G_0..G_{M-2} (at current s_j), then G_{M-1}
        E = jnp.eye(npv).reshape((npv,) + c.shape)    # (np, nk, 2) basis
        V = s0tan_batch(c, E)                          # (np, *state) = B E
        for j in range(M_ - 1):
            V = Gjvp_batch(starts[j], V)               # G_j V
        V = Gjvp_batch(sML, V)                         # G_{M-1} V
        A = np.asarray(jax.vmap(H)(V)).reshape(npv, -1).T   # (nobs, np)
        # w: accumulate defects u_{j+1}=G_j u_j + r_c^j, u_0=0
        u = jnp.zeros_like(sML)
        for j in range(M_ - 1):
            _, gu = Gjvp(starts[j], u)
            u = gu + rc[j]
        _, gw = Gjvp(sML, u)
        bb = np.asarray((rd + H(gw))).ravel()         # r_d + H' G_{M-1} w
        # LM solve for dp
        g = A.T @ bb; Hn = A.T @ A
        for _ in range(25):
            dp = np.linalg.solve(Hn + lam * np.eye(npv), -g).reshape(c.shape)
            dpj = jnp.asarray(dp)
            # back-substitute interiors: ds_0=B dp; ds_{j+1}=G_j ds_j + r_c^j
            _, ds = s0tan(c, dpj)
            dS = []
            for j in range(M_ - 1):
                _, gds = Gjvp(starts[j], ds)
                ds = gds + rc[j]; dS.append(ds)
            c_new = c + dpj; S_new = [S[j] + dS[j] for j in range(M_ - 1)]
            m_new, dcost_new = merit(c_new, S_new)
            if m_new < m:
                c, S, m, dcost = c_new, S_new, m_new, dcost_new
                lam = max(lam * 0.5, 1e-9); break
            lam *= 4.0
        else:
            print(f"  LM stuck at it={it}"); break
        if it % 5 == 0 or it == ITERS - 1:
            cdef = float(sum(jnp.sum(r ** 2) for r in rc))
            print(f"  it={it+1}: merit={m:.3e} data={dcost:.3e} defect={cdef:.2e} "
                  f"ic_err={ic_err(c):.3f} lam={lam:.1e}")
        if dcost <= 1.1 * floor and m < 2 * floor:
            print(f"  converged (data+feasible) at it={it+1}"); break
    err = ic_err(c); rt = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez(OUT, mode="msgn", interior=INTERIOR, M=M_, t_g=TREC, re=RE, init=INIT,
             ic_err=err, final_data=dcost, floor=float(floor), runtime=rt,
             c_rec=np.asarray(c))
    print(f"[done] MSGN {INTERIOR} M={M_} T={TREC:.0f} i{INIT}: ic_err={err:.3f} "
          f"data={dcost:.2e} (floor {float(floor):.2e}) {rt:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
