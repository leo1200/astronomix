"""
KH reconstruction -- Stage 4: multiple shooting (hard vs SOFT continuity).

Split [0,T] into M segments with free interior segment-start states. Continuity
between segments is penalised either on the FULL field (hard) or only on the
LOW-PASS (large-scale) streamwise components (soft) -- the latter tolerates
small-scale mismatch, which is the right move when only the large scales are
recoverable (mode-3 frontier). M=1 reduces to single shooting.

One (M, mode, mu) configuration per run (parallelise across GPUs / configs):
  Env: KH_M, KH_MSMODE in {single,hard,soft}, KH_MU, KH_KCUT, KH_TREC, KH_N,
       KH_STEPS, KH_SEED, KH_OUT.
"""

# ==== GPU selection ====
import os, sys
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not _cvd:
    from autocvd import autocvd
    autocvd(num_gpus=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ruff: noqa: E402
# =======================

import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

import problem as P
import metrics as M
from astronomix import time_integration

N = int(os.environ.get("KH_N", 192))
MROOT = int(os.environ.get("KH_M", 4))
MSMODE = os.environ.get("KH_MSMODE", "soft")        # single | hard | soft
MU = float(os.environ.get("KH_MU", 30.0))
KCUT = float(os.environ.get("KH_KCUT", 4))
TREC = float(os.environ.get("KH_TREC", 80))
STEPS = int(os.environ.get("KH_STEPS", 150))
TRUTH_SEED = int(os.environ.get("KH_SEED", 0))
NOISE = float(os.environ.get("KH_NOISE", 1e-2))   # obs noise (rel. to field rms)
KCTRL = float(os.environ.get("KH_KCTRL", 0))      # band-limit control+truth to |kx|<=KCTRL (prior); 0=off
RE = float(os.environ.get("KH_RE", 2000))
OUT = os.environ.get("KH_OUT", f"data/ms_{MSMODE}_M{MROOT}_T{TREC:.0f}_s{TRUTH_SEED}.npz")


def main():
    # band-limited control = built-in prior (the under-determined high-k cannot
    # overfit the noise); band-limit the truth to the same band so the inverse
    # is well-posed and the recoverable target is unambiguous.
    kmax = int(KCTRL) if KCTRL > 0 else 32
    khp = P.KHParams(n=N, reynolds=RE, k_max=kmax)
    M_ = 1 if MSMODE == "single" else MROOT
    T = TREC * khp.growth_time
    h = T / M_
    cfg, par = P.make_config_params(khp, h, snapshots=0, backward=True)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0_probe = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0_probe.shape)
    y = jnp.asarray(np.asarray(Y[0])); slab = M.shear_layer_slab(y, khp.yc, 0.12)

    truth_seed = P.random_broadband_seed(jax.random.PRNGKey(TRUTH_SEED), khp, X, Y)

    # observation: full final velocity at T (built from M true segments)
    def seg(state):
        return time_integration(state, cfg, par._replace(t_end=h), rv)

    truth_state = P.build_state(truth_seed, khp, cfg, rv, X, Y)
    s = truth_state
    true_starts = [s]
    for _ in range(M_ - 1):
        s = seg(s); true_starts.append(s)
    obs_clean = P.velocity_of(seg(true_starts[-1]), rv)
    # base (unperturbed) final field: the seed's imprint on u_T is obs-base.
    # The base flow is KNOWN (fixed in build_state, not optimized), so the
    # recoverable signal is exactly this imprint -- scale the noise to IT (not the
    # base-flow-dominated full field) so NOISE is a true 1/SNR on what we recover.
    base = P.build_state(jnp.zeros_like(truth_seed), khp, cfg, rv, X, Y)
    for _ in range(M_):
        base = seg(base)
    base_final = P.velocity_of(base, rv)
    pert_rms = float(jnp.sqrt(jnp.mean((obs_clean - base_final) ** 2)))
    # additive Gaussian observation noise -> sets the mode-3 information floor
    nsd = NOISE * pert_rms
    obs = obs_clean + nsd * jax.random.normal(
        jax.random.PRNGKey(999 + TRUTH_SEED), obs_clean.shape)
    print(f"  noise model: pert_rms={pert_rms:.3e} nsd={nsd:.3e} (NOISE={NOISE})")

    def build_ic(seed):
        if KCTRL > 0:                       # project control onto the prior band
            seed = M.lowpass_x(seed, KCTRL)
        return P.build_state(seed, khp, cfg, rv, X, Y)

    # data-misfit noise floor: stop before fitting below it (overfitting noise)
    J_floor = 0.5 * nsd ** 2

    def defect_term(finals, starts):
        d = 0.0
        for j in range(M_ - 1):
            diff = finals[j] - starts[j + 1]
            if MSMODE == "soft":
                diff = M.lowpass_x(diff, KCUT)
            d = d + jnp.mean(diff ** 2)
        return d

    # Tikhonov regularization of the seed amplitude. The seed->u_T map is
    # ill-conditioned even within the low-k band: weakly-observed (small-sigma)
    # directions let the optimizer inflate the IC to chase the data residual
    # (lowk_err grows while J drops). An L2 penalty alpha||u0||^2 shrinks exactly
    # those null directions to 0 (minimum-norm solution), so the recovered IC =
    # truth projected on the observable subspace. alpha ~ noise level (L-curve).
    ALPHA = float(os.environ.get("KH_ALPHA", 0.0))

    def loss(theta):
        ic_seed = M.lowpass_x(theta["seed"], KCTRL) if KCTRL > 0 else theta["seed"]
        s0 = build_ic(theta["seed"])
        starts = [s0] + [theta["seg"][j] for j in range(M_ - 1)]
        finals = [seg(st) for st in starts]
        data = 0.5 * jnp.mean((P.velocity_of(finals[-1], rv) - obs) ** 2)
        total = data + 0.5 * ALPHA * jnp.mean(ic_seed ** 2)
        if M_ > 1:
            total = total + 0.5 * MU * defect_term(finals, starts)
        return total, data    # has_aux=True -> early-stop on the data misfit

    # init: cold seed (default) or a warm start near truth (KH_WARM>0) to probe
    # the basin/mode-2 directly; interior states warm-started by forward prop.
    WARM = float(os.environ.get("KH_WARM", 0.0))
    if WARM > 0:
        pert = P.random_broadband_seed(jax.random.PRNGKey(50 + TRUTH_SEED), khp, X, Y)
        seed0 = truth_seed + WARM * pert
    else:
        seed0 = 1e-3 * jax.random.normal(jax.random.PRNGKey(100 + TRUTH_SEED), truth_seed.shape)
    s = build_ic(seed0); interior = []
    for _ in range(M_ - 1):
        s = seg(s); interior.append(s)
    theta = {"seed": seed0,
             "seg": jnp.stack(interior) if interior else jnp.zeros((0, *s0_probe.shape))}

    # Adam (not L-BFGS): the optax zoom line-search compiles pathologically
    # slowly through the viscous KH segment loop; Adam's update is trivial to
    # compile and -- same optimizer for single/hard/soft -- keeps the comparison
    # fair.
    # gradient clipping tames the mode-1 magnitude explosion (~e^{lambda T}); the
    # residual difficulty single shooting then faces is mode-2 multimodality.
    LR = float(os.environ.get("KH_LR", 1e-3))
    CLIP = float(os.environ.get("KH_CLIP", 1.0))
    opt = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(LR))
    vgrad = jax.value_and_grad(loss, has_aux=True)

    @jax.jit
    def step(theta, st):
        (val, data), g = vgrad(theta)
        upd, st = opt.update(g, st)
        return optax.apply_updates(theta, upd), st, val, data

    def errs(seed):
        full = float(jnp.linalg.norm(seed - truth_seed) / jnp.linalg.norm(truth_seed))
        lowk = float(jnp.linalg.norm(M.lowpass_x(seed - truth_seed, KCUT))
                     / (jnp.linalg.norm(M.lowpass_x(truth_seed, KCUT)) + 1e-30))
        return full, lowk

    st = opt.init(theta); t0 = time.time(); hist = []
    best = (errs(theta["seed"])[1], theta["seed"])   # (lowk_err, seed) best so far
    stopped_at = STEPS
    for it in range(STEPS):
        theta, st, val, data = step(theta, st)
        lk_now = errs(theta["seed"])[1]
        if lk_now < best[0]:
            best = (lk_now, theta["seed"])
        if it % 20 == 0 or it == STEPS - 1:
            fe, lk = errs(theta["seed"])
            hist.append((it, float(val), fe, lk))
            print(f"  {MSMODE} M={M_} it={it:3d}: J={float(val):.3e} data={float(data):.3e} "
                  f"full_err={fe:.3f} lowk_err={lk:.3f}")
        # early stop once the data misfit reaches the noise floor (avoid
        # overfitting the noise, which drives the IC away from truth)
        if float(data) <= 1.1 * J_floor:
            stopped_at = it
            print(f"  early stop at it={it}: data={float(data):.3e} <= floor {J_floor:.3e}")
            break
    # report the seed at the (principled) early-stop point; best_lowk is an
    # oracle diagnostic (uses truth) for reference only.
    best_lowk = float(best[0])
    fe, lk = errs(theta["seed"])
    errk = np.asarray(M.per_kx_relative_error(theta["seed"], truth_seed, slab))
    runtime = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez(OUT, mode=MSMODE, M=M_, mu=MU, kcut=KCUT, kctrl=KCTRL, re=RE, t_g=TREC,
             n=N, seed=TRUTH_SEED, noise=NOISE, best_lowk=best_lowk, stopped_at=stopped_at,
             kx=np.asarray(M.streamwise_modes(khp.n)), errk=errk,
             full_err=fe, lowk_err=lk, final_loss=float(val), runtime=runtime,
             rec=np.asarray(theta["seed"]), truth=np.asarray(truth_seed))
    print(f"[done] {MSMODE} M={M_} mu={MU} kctrl={KCTRL} T={TREC:.0f}: lowk_err={lk:.3f} "
          f"best={best_lowk:.3f} stop@{stopped_at} {runtime:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
