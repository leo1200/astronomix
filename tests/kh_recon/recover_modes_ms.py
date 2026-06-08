"""
The decisive multiple-shooting test, done RIGHT: observable-subspace IC control
(so over-parametrization is NOT the confound) + M-segment shooting. Run at a
horizon where well-posed single-shooting mode-space recovery fails by MODE 2
(T=40 t_g, Re=2000: single lowk~0.76, yet SVD k_rec=22>6 -> optimization-, not
information-limited). Question: does segmenting the trajectory (taming the
long-horizon multimodality) recover what single shooting cannot?

Only the IC is mode-space (~10 DOF, what we score). Interior segment-start states
are full fields, pinned by the data + continuity defect, so they do NOT reintroduce
the IC null space. M=1 reduces to single-shooting mode-space (the baseline).

Env: KH_TREC, KH_RE, KH_M, KH_MSMODE in {single,hard,soft}, KH_MU, KH_KCUT,
     KH_N, KH_STEPS, KH_LR, KH_KMAX, KH_NOISE, KH_SEED, KH_OUT.
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
import jax, jax.numpy as jnp, numpy as np, optax
import problem as P, metrics as M
from astronomix import time_integration

N = int(os.environ.get("KH_N", 64))
TREC = float(os.environ.get("KH_TREC", 40))
RE = float(os.environ.get("KH_RE", 2000))
MROOT = int(os.environ.get("KH_M", 4))
MSMODE = os.environ.get("KH_MSMODE", "soft")
MU = float(os.environ.get("KH_MU", 30.0))
KCUT = float(os.environ.get("KH_KCUT", 6))
STEPS = int(os.environ.get("KH_STEPS", 200))
LR = float(os.environ.get("KH_LR", 3e-3))
KMAX = int(os.environ.get("KH_KMAX", 6))
KMIN = int(os.environ.get("KH_KMIN", 2))
TRUTH_SEED = int(os.environ.get("KH_SEED", 0))
NOISE = float(os.environ.get("KH_NOISE", 1e-2))
OUT = os.environ.get("KH_OUT", f"data/modems_{MSMODE}_M{MROOT}_T{TREC:.0f}_Re{RE:.0f}.npz")


def main():
    khp = P.KHParams(n=N, reynolds=RE, k_min=KMIN, k_max=KMAX)
    M_ = 1 if MSMODE == "single" else MROOT
    T = TREC * khp.growth_time
    h = T / M_
    cfg, par = P.make_config_params(khp, h, snapshots=0, backward=True)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0_probe = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0_probe.shape)
    y = jnp.asarray(np.asarray(Y[0])); slab = M.shear_layer_slab(y, khp.yc, 0.12)

    ks = jnp.arange(KMIN, KMAX + 1)
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    cosx = jnp.cos(2 * jnp.pi * ks[None, None, :] * X[..., None])
    sinx = jnp.sin(2 * jnp.pi * ks[None, None, :] * X[..., None])

    def seed_of(coeffs):                       # (nk,2) -> (2,Nx,Ny)
        svy = env * jnp.sum(coeffs[:, 0] * cosx + coeffs[:, 1] * sinx, axis=-1)
        return jnp.stack([jnp.zeros_like(svy), svy])

    truth_seed = P.random_broadband_seed(jax.random.PRNGKey(TRUTH_SEED), khp, X, Y)

    def seg(state):
        return time_integration(state, cfg, par._replace(t_end=h), rv)

    # truth path + observation (built from M true segments)
    s = P.build_state(truth_seed, khp, cfg, rv, X, Y)
    true_starts = [s]
    for _ in range(M_ - 1):
        s = seg(s); true_starts.append(s)
    obs_clean = P.velocity_of(seg(true_starts[-1]), rv)
    b = P.build_state(jnp.zeros_like(truth_seed), khp, cfg, rv, X, Y)
    for _ in range(M_):
        b = seg(b)
    pert_rms = float(jnp.sqrt(jnp.mean((obs_clean - P.velocity_of(b, rv)) ** 2)))
    nsd = NOISE * pert_rms
    obs = obs_clean + nsd * jax.random.normal(jax.random.PRNGKey(999 + TRUTH_SEED), obs_clean.shape)
    J_floor = 0.5 * nsd ** 2
    print(f"  {MSMODE} M={M_} T={TREC} Re={RE} mu={MU}: pert_rms={pert_rms:.3e} nsd={nsd:.3e}")

    def defect(finals, starts):
        d = 0.0
        for j in range(M_ - 1):
            diff = finals[j] - starts[j + 1]
            if MSMODE == "soft":
                diff = M.lowpass_x(diff, KCUT)
            d = d + jnp.mean(diff ** 2)
        return d

    def loss(theta):
        s0 = P.build_state(seed_of(theta["modes"]), khp, cfg, rv, X, Y)
        starts = [s0] + [theta["seg"][j] for j in range(M_ - 1)]
        finals = [seg(st) for st in starts]
        data = 0.5 * jnp.mean((P.velocity_of(finals[-1], rv) - obs) ** 2)
        total = data + (0.5 * MU * defect(finals, starts) if M_ > 1 else 0.0)
        return total, data

    def lowk_err(coeffs):
        seed = seed_of(coeffs)
        return float(jnp.linalg.norm(M.lowpass_x(seed - truth_seed, KMAX))
                     / (jnp.linalg.norm(M.lowpass_x(truth_seed, KMAX)) + 1e-30))

    # init decoupled from truth: KH_INIT seeds the cold start, KH_INITSCALE sets its
    # amplitude. For a basin (multimodality) test use INITSCALE ~ seed amplitude so
    # restarts genuinely start in DIFFERENT regions of parameter space.
    INIT = int(os.environ.get("KH_INIT", TRUTH_SEED))
    INITSCALE = float(os.environ.get("KH_INITSCALE", 1e-4))
    coeffs0 = INITSCALE * jax.random.normal(jax.random.PRNGKey(100 + INIT), (len(ks), 2))
    s = P.build_state(seed_of(coeffs0), khp, cfg, rv, X, Y); interior = []
    for _ in range(M_ - 1):
        s = seg(s); interior.append(s)
    theta = {"modes": coeffs0,
             "seg": jnp.stack(interior) if interior else jnp.zeros((0, *s0_probe.shape))}

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    st = opt.init(theta); vg = jax.value_and_grad(loss, has_aux=True)

    @jax.jit
    def step(theta, st):
        (val, data), g = vg(theta); upd, st = opt.update(g, st)
        return optax.apply_updates(theta, upd), st, val, data

    t0 = time.time(); stopped = STEPS; best = lowk_err(theta["modes"])
    for it in range(STEPS):
        theta, st, val, data = step(theta, st)
        lk = lowk_err(theta["modes"]); best = min(best, lk)
        if it % 25 == 0 or it == STEPS - 1:
            print(f"  {MSMODE} M={M_} it={it:3d}: J={float(val):.3e} data={float(data):.3e} lowk={lk:.3f}")
        if float(data) <= 1.1 * J_floor:
            stopped = it; print(f"  early stop it={it}"); break
    lk = lowk_err(theta["modes"])
    rt = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez(OUT, mode=MSMODE, M=M_, mu=MU, t_g=TREC, re=RE, n=N, lowk_err=lk,
             best_lowk=best, stopped_at=stopped, runtime=rt, init=INIT,
             final_data=float(data), modes_rec=np.asarray(theta["modes"]),
             rec=np.asarray(seed_of(theta["modes"])), truth=np.asarray(truth_seed))
    print(f"[done] {MSMODE} M={M_} mu={MU} T={TREC:.0f} Re={RE:.0f}: lowk={lk:.3f} "
          f"best={best:.3f} stop@{stopped} {rt:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
