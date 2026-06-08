"""
Single-shooting GAUSS-NEWTON (Levenberg-Marquardt) recovery -- the second-order
control for the mode-2 basin test. The IC is low-D (mode-space, kx in [KMIN,KMAX]),
so the GN Jacobian J=du_T/dp (nobs x np) is formed by np forward-mode passes
(jax.jacfwd) -- no adjoint. LM-damped normal equations: (J^T J + lam I) dp = -J^T r.

Question: do the multimodal basins that first-order Adam single shooting got stuck
in (ic_err scatter 0.05->1.0 at T=45 kx2-6) DISSOLVE under a second-order optimizer?
If yes, the "multimodality" was largely an Adam artifact; if GN still scatters, the
basins are real (and, by the condensing equivalence, terminal-only MS-GN inherits
them). Multi-restart via KH_INIT (decoupled from truth KH_SEED), amplitude-scaled.

Env: KH_N, KH_RE, KH_TREC, KH_KMIN, KH_KMAX, KH_SEED (truth), KH_INIT (restart),
     KH_INITSCALE, KH_NOISE, KH_ITERS, KH_OUT.
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
TREC = float(os.environ.get("KH_TREC", 45))
KMIN = int(os.environ.get("KH_KMIN", 2))
KMAX = int(os.environ.get("KH_KMAX", 6))
TRUTH_SEED = int(os.environ.get("KH_SEED", 0))
INIT = int(os.environ.get("KH_INIT", 0))
INITSCALE = float(os.environ.get("KH_INITSCALE", 1e-2))
NOISE = float(os.environ.get("KH_NOISE", 1e-2))
ITERS = int(os.environ.get("KH_ITERS", 40))
OUT = os.environ.get("KH_OUT", f"data/gn_single_i{INIT}.npz")


def main():
    khp = P.KHParams(n=N, reynolds=RE, k_min=KMIN, k_max=KMAX)
    T = TREC * khp.growth_time
    cfg, par = P.make_config_params(khp, T, snapshots=0, backward=False)  # forward-mode AD
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0p = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0p.shape)
    ks = jnp.arange(KMIN, KMAX + 1)
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    cosx = jnp.cos(2 * jnp.pi * ks[None, None, :] * X[..., None])
    sinx = jnp.sin(2 * jnp.pi * ks[None, None, :] * X[..., None])

    def seed_of(c):                                  # (nk,2)->(2,Nx,Ny)
        svy = env * jnp.sum(c[:, 0] * cosx + c[:, 1] * sinx, axis=-1)
        return jnp.stack([jnp.zeros_like(svy), svy])

    truth_seed = P.random_broadband_seed(jax.random.PRNGKey(TRUTH_SEED), khp, X, Y)

    def fwd(c):
        s = P.build_state(seed_of(c), khp, cfg, rv, X, Y)
        return P.velocity_of(time_integration(s, cfg, par._replace(t_end=T), rv), rv)

    # truth obs + perturbation-relative noise
    obs_clean = P.velocity_of(
        time_integration(P.build_state(truth_seed, khp, cfg, rv, X, Y),
                         cfg, par._replace(t_end=T), rv), rv)
    base = P.velocity_of(time_integration(P.build_state(jnp.zeros_like(truth_seed), khp, cfg, rv, X, Y),
                                          cfg, par._replace(t_end=T), rv), rv)
    pert_rms = float(jnp.sqrt(jnp.mean((obs_clean - base) ** 2)))
    nsd = NOISE * pert_rms
    obs = obs_clean + nsd * jax.random.normal(jax.random.PRNGKey(12345), obs_clean.shape)
    J_floor = 0.5 * nsd ** 2 * obs.size       # sum-of-squares floor (cost uses sum, not mean)

    def resid(c):
        return (fwd(c) - obs).ravel()

    def ic_err(c):
        seed = seed_of(c)
        return float(jnp.linalg.norm(seed - truth_seed) / jnp.linalg.norm(truth_seed))

    jac = jax.jit(jax.jacfwd(resid))
    rfun = jax.jit(resid)

    # init: cold random, OR warm from a previous solution (horizon continuation =
    # the MS basin-enlargement principle without the MS machinery)
    WARMNPZ = os.environ.get("KH_WARMNPZ", "")
    if WARMNPZ and os.path.exists(WARMNPZ):
        c = jnp.asarray(np.load(WARMNPZ, allow_pickle=True)["c_rec"])
        print(f"  warm-start from {WARMNPZ}")
    else:
        c = INITSCALE * jax.random.normal(jax.random.PRNGKey(100 + INIT), (len(ks), 2))
    shp = c.shape
    lam = 1e-3
    t0 = time.time()
    r = np.asarray(rfun(c)); cost = 0.5 * float(r @ r)
    print(f"  GN i{INIT} it=0: cost={cost:.3e} ic_err={ic_err(c):.3f}")
    for it in range(ITERS):
        J = np.asarray(jac(c)).reshape(-1, c.size)        # (nobs, np)
        g = J.T @ r                                       # np
        H = J.T @ J                                       # np x np
        # LM with simple adaptation
        for _ in range(20):
            dp = np.linalg.solve(H + lam * np.eye(c.size), -g)
            c_new = c + jnp.asarray(dp.reshape(shp))
            r_new = np.asarray(rfun(c_new)); cost_new = 0.5 * float(r_new @ r_new)
            if cost_new < cost:
                c, r, cost = c_new, r_new, cost_new; lam = max(lam * 0.5, 1e-9); break
            lam *= 4.0
        else:
            break                                         # cannot decrease -> converged/stuck
        if it % 5 == 0 or it == ITERS - 1:
            print(f"  GN i{INIT} it={it+1}: cost={cost:.3e} ic_err={ic_err(c):.3f} lam={lam:.1e}")
        if cost <= 1.1 * J_floor:
            print(f"  GN i{INIT} converged to floor at it={it+1}"); break
    err = ic_err(c); rt = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez(OUT, init=INIT, ic_err=err, final_cost=cost, j_floor=float(J_floor),
             c_rec=np.asarray(c), runtime=rt, t_g=TREC, kmin=KMIN, kmax=KMAX)
    print(f"[done] GN i{INIT} T={TREC:.0f}: ic_err={err:.3f} cost={cost:.3e} "
          f"(floor {float(J_floor):.2e}) {rt:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
