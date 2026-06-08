"""
Fully-JAX, on-GPU, vmapped-over-inits single-shooting Gauss-Newton (Levenberg-
Marquardt). Everything -- forward integration, jacfwd Jacobian, np x np LM solve,
accept/reject, outer loop -- lives in JAX, so we can `vmap` the WHOLE solve over a
batch of cold inits and recover all of them in ONE compiled call (far better GPU
utilization than B separate processes). Control flow is vmap-safe: fixed NITER
outer steps, per-init lambda via jnp.where (no Python branching, no host transfer).

Env: KH_N, KH_RE, KH_TREC, KH_KMIN, KH_KMAX, KH_SEED(truth), KH_NB(batch=#inits),
     KH_INITSCALE, KH_NOISE, KH_NITER, KH_OUT.
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
TREC = float(os.environ.get("KH_TREC", 80))
KMIN = int(os.environ.get("KH_KMIN", 2)); KMAX = int(os.environ.get("KH_KMAX", 6))
TRUTH_SEED = int(os.environ.get("KH_SEED", 0))
NB = int(os.environ.get("KH_NB", 100)); INITSCALE = float(os.environ.get("KH_INITSCALE", 1e-2))
NOISE = float(os.environ.get("KH_NOISE", 1e-2)); NITER = int(os.environ.get("KH_NITER", 120))
OUT = os.environ.get("KH_OUT", f"data/gnbatch_T{TREC:.0f}.npz")


def main():
    khp = P.KHParams(n=N, reynolds=RE, k_min=KMIN, k_max=KMAX)
    T = TREC * khp.growth_time
    cfg, par = P.make_config_params(khp, T, snapshots=0, backward=False)
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

    def fwd(c):
        s = P.build_state(seed_of(c), khp, cfg, rv, X, Y)
        return P.velocity_of(time_integration(s, cfg, par._replace(t_end=T), rv), rv).ravel()

    truth = P.random_broadband_seed(jax.random.PRNGKey(TRUTH_SEED), khp, X, Y)
    obs_clean = P.velocity_of(time_integration(P.build_state(truth, khp, cfg, rv, X, Y),
                              cfg, par._replace(t_end=T), rv), rv).ravel()
    base = P.velocity_of(time_integration(P.build_state(jnp.zeros_like(truth), khp, cfg, rv, X, Y),
                         cfg, par._replace(t_end=T), rv), rv).ravel()
    nsd = NOISE * float(jnp.sqrt(jnp.mean((obs_clean - base) ** 2)))
    y = obs_clean + nsd * jax.random.normal(jax.random.PRNGKey(12345), obs_clean.shape)
    floor = 0.5 * nsd ** 2 * y.size
    tnorm = float(jnp.linalg.norm(truth))

    def resid(c):
        return fwd(c) - y

    def cost(c):
        r = resid(c); return 0.5 * jnp.sum(r ** 2)

    jacf = jax.jacfwd(resid)

    def lm_step(state, _):
        c, lam, cst = state
        r = resid(c)
        J = jacf(c).reshape(r.shape[0], npar)
        g = J.T @ r
        Hn = J.T @ J + lam * jnp.eye(npar)
        dp = jnp.linalg.solve(Hn, -g).reshape(c.shape)
        c_try = c + dp
        cst_try = cost(c_try)
        acc = cst_try < cst
        c = jnp.where(acc, c_try, c)
        cst = jnp.where(acc, cst_try, cst)
        lam = jnp.where(acc, jnp.maximum(lam * 0.5, 1e-9), lam * 4.0)
        return (c, lam, cst), cst

    def solve_one(c0):
        state0 = (c0, 1e-3, cost(c0))
        (c, lam, cst), _ = jax.lax.scan(lm_step, state0, None, length=NITER)
        icerr = jnp.linalg.norm(seed_of(c) - truth) / tnorm
        return c, cst, icerr

    solve_batch = jax.jit(jax.vmap(solve_one))

    key = jax.random.PRNGKey(0)
    C0 = INITSCALE * jax.random.normal(key, (NB, len(ks), 2))
    t0 = time.time()
    C, csts, icerrs = solve_batch(C0)
    C.block_until_ready()
    rt = time.time() - t0
    icerrs = np.asarray(icerrs); csts = np.asarray(csts)
    rec = int((icerrs < 0.1).sum())
    print(f"[gn_batch] T={TREC:.0f} NB={NB} NITER={NITER}: recovered {rec}/{NB} "
          f"(<0.1), best={icerrs.min():.3f}, median={np.median(icerrs):.3f}")
    print(f"  floor={floor:.2e}; {rt:.0f}s total = {rt/NB:.1f}s/init (batched, 1 GPU)")
    np.savez(OUT, t_g=TREC, nb=NB, niter=NITER, ic_err=icerrs, final_cost=csts,
             floor=float(floor), runtime=rt)
    print(f"  ic_err: {np.array2string(np.sort(icerrs)[:12], precision=2)} ... -> {OUT}")


if __name__ == "__main__":
    main()
