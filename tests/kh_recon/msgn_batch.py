"""
Fully-JAX, on-GPU, vmapped-over-inits CONSTRAINED Gauss-Newton MULTIPLE SHOOTING.
Same condensed LM step as recover_msgn.py, but everything (segment integrations,
jvp sensitivity propagation, npxnp LM solve, accept/reject, outer loop) is JAX with
vmap-safe control flow (jnp.where, lax.scan), so the WHOLE solve vmaps over a batch
of cold inits -> all inits in one compiled call. Feasible interiors (forward-prop
of cold IC). M=1 reduces to single shooting.

Env: KH_N, KH_RE, KH_TREC, KH_M, KH_KMIN, KH_KMAX, KH_SEED(truth), KH_NB(batch),
     KH_INITSCALE, KH_NOISE, KH_NITER, KH_RHO, KH_OUT.
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

N = int(os.environ.get("KH_N", 64)); RE = float(os.environ.get("KH_RE", 2000))
TREC = float(os.environ.get("KH_TREC", 80)); M_ = int(os.environ.get("KH_M", 4))
KMIN = int(os.environ.get("KH_KMIN", 2)); KMAX = int(os.environ.get("KH_KMAX", 6))
TRUTH_SEED = int(os.environ.get("KH_SEED", 0))
NB = int(os.environ.get("KH_NB", 100)); INITSCALE = float(os.environ.get("KH_INITSCALE", 1e-2))
NOISE = float(os.environ.get("KH_NOISE", 1e-2)); NITER = int(os.environ.get("KH_NITER", 120))
RHO = float(os.environ.get("KH_RHO", 1.0))
OUT = os.environ.get("KH_OUT", f"data/msgnbatch_M{M_}_T{TREC:.0f}.npz")


def main():
    khp = P.KHParams(n=N, reynolds=RE, k_min=KMIN, k_max=KMAX)
    T = TREC * khp.growth_time; h = T / M_
    cfg, par = P.make_config_params(khp, h, snapshots=0, backward=False)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0p = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0p.shape)
    ks = jnp.arange(KMIN, KMAX + 1)
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    cosx = jnp.cos(2 * jnp.pi * ks[None, None, :] * X[..., None])
    sinx = jnp.sin(2 * jnp.pi * ks[None, None, :] * X[..., None])
    npar = 2 * len(ks)

    def seed_of(c):
        svy = env * jnp.sum(c[:, 0] * cosx + c[:, 1] * sinx, axis=-1)
        return jnp.stack([jnp.zeros_like(svy), svy])

    def s0_of(c):
        return P.build_state(seed_of(c), khp, cfg, rv, X, Y)

    Phi = lambda s: time_integration(s, cfg, par._replace(t_end=h), rv)
    H = lambda s: P.velocity_of(s, rv)
    base_state = s0_of(jnp.zeros((len(ks), 2)))

    def Gj(s, v):
        return jax.jvp(Phi, (s,), (v,))[1]

    def s0t(c, dc):
        return jax.jvp(s0_of, (c,), (dc,))[1]

    truth = P.random_broadband_seed(jax.random.PRNGKey(TRUTH_SEED), khp, X, Y)
    s = P.build_state(truth, khp, cfg, rv, X, Y)
    for _ in range(M_):
        s = Phi(s)
    obs_clean = H(s)
    b = base_state
    for _ in range(M_):
        b = Phi(b)
    nsd = NOISE * float(jnp.sqrt(jnp.mean((obs_clean - H(b)) ** 2)))
    y = obs_clean + nsd * jax.random.normal(jax.random.PRNGKey(12345), obs_clean.shape)
    floor = 0.5 * nsd ** 2 * y.size
    tnorm = float(jnp.linalg.norm(truth))
    Ebasis = jnp.eye(npar).reshape((npar,) + (len(ks), 2))    # (npar, nk, 2)

    def _prop1(s, _):                         # one segment: carry=Phi(s), emit Phi(s)
        s2 = Phi(s)
        return s2, s2

    def interior_feasible(c):                 # forward-prop cold IC -> (M-1) states
        _, S = jax.lax.scan(_prop1, s0_of(c), None, length=M_ - 1)
        return S                               # (M-1, *state); empty if M_==1

    def starts_finals(c, S):
        s0 = s0_of(c)
        starts = jnp.concatenate([s0[None], S], axis=0)   # (M, *state)
        # scan (not vmap) over segments: avoids a 2nd primal vmap dim on top of the
        # outer inits-vmap (which would push the ghost-cell pad to nd=5)
        _, finals = jax.lax.scan(lambda _, s: (None, Phi(s)), None, starts)
        return starts, finals

    def merit_of(c, S):
        starts, finals = starts_finals(c, S)
        rd = H(finals[-1]) - y
        rc = finals[:-1] - starts[1:]                     # (M-1, *state)
        return 0.5 * jnp.sum(rd ** 2) + 0.5 * RHO * jnp.sum(rc ** 2)

    def lm_step(state, _):
        c, S, lam, mer = state
        starts, finals = starts_finals(c, S)                 # (M,*state) each
        rc = finals[:-1] - starts[1:]                        # (M-1, *state)
        rd = H(finals[-1]) - y
        # A = H' (prod_{j=0}^{M-1} G_j) B : scan G over ALL M starts (vmapped over npar)
        V0 = jax.vmap(lambda e: s0t(c, e))(Ebasis)           # (npar, *state)
        VA, _ = jax.lax.scan(lambda V, s: (jax.vmap(lambda v: Gj(s, v))(V), None),
                             V0, starts)
        A = jax.vmap(H)(VA).reshape(npar, -1).T              # (nobs, npar)
        # w: u_{j+1}=G_j u_j + r_c^j over starts[0..M-2]; then G at starts[M-1]
        u0 = jnp.zeros_like(starts[0])
        uw, _ = jax.lax.scan(lambda u, sr: (Gj(sr[0], u) + sr[1], None),
                             u0, (starts[:-1], rc))
        bb = (rd + H(Gj(starts[-1], uw))).reshape(-1)
        g = A.T @ bb
        Hn = A.T @ A + lam * jnp.eye(npar)
        dp = jnp.linalg.solve(Hn, -g).reshape(c.shape)
        # back-substitute interiors: ds_{j+1}=G_j ds_j + r_c^j ; emit ds_1..ds_{M-1}
        ds0 = s0t(c, dp)
        _, dS = jax.lax.scan(lambda ds, sr: ((lambda d: (d, d))(Gj(sr[0], ds) + sr[1])),
                             ds0, (starts[:-1], rc))          # dS: (M-1, *state)
        c_try = c + dp; S_try = S + dS
        mer_try = merit_of(c_try, S_try)
        acc = mer_try < mer
        c = jnp.where(acc, c_try, c)
        S = jnp.where(acc, S_try, S)
        lam = jnp.where(acc, jnp.maximum(lam * 0.5, 1e-9), lam * 4.0)
        mer = jnp.where(acc, mer_try, mer)
        return (c, S, lam, mer), mer

    LAM0 = float(os.environ.get("KH_LAM0", 1.0))
    def solve_one(c0):
        S0 = interior_feasible(c0)
        st0 = (c0, S0, LAM0, merit_of(c0, S0))
        (c, S, lam, mer), _ = jax.lax.scan(lm_step, st0, None, length=NITER)
        return jnp.linalg.norm(seed_of(c) - truth) / tnorm, mer

    solve_batch = jax.jit(jax.vmap(solve_one))
    C0 = INITSCALE * jax.random.normal(jax.random.PRNGKey(0), (NB, len(ks), 2))
    t0 = time.time()
    icerrs, mers = solve_batch(C0)
    icerrs = np.asarray(jax.block_until_ready(icerrs)); mers = np.asarray(mers)
    rt = time.time() - t0
    rec = int((icerrs < 0.1).sum())
    print(f"[msgn_batch] M={M_} T={TREC:.0f} NB={NB}: recovered {rec}/{NB} (<0.1), "
          f"best={icerrs.min():.3f}, median={np.median(icerrs):.3f}; {rt:.0f}s ({rt/NB:.1f}s/init)")
    np.savez(OUT, M=M_, t_g=TREC, nb=NB, niter=NITER, ic_err=icerrs, merit=mers,
             floor=float(floor), runtime=rt)
    print(f"  sorted ic_err[:12]: {np.array2string(np.sort(icerrs)[:12], precision=2)} -> {OUT}")


if __name__ == "__main__":
    main()
