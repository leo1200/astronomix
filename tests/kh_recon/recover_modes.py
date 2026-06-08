"""
KH reconstruction in the OBSERVABLE SUBSPACE -- the method that actually recovers.

The full-field control (~768 DOF after the x-band-limit) over-parametrizes the
truth seed by ~75x: the true seed is transverse only (svx=0), lives on a single
known shear-layer envelope env(y), and is band-limited in kx. The ~758 extra DOF
are an unobservable null space; an unconstrained optimizer fills them with junk
that fits the data while lowk_err stays ~1 (confirmed: cold + warm starts both
fail / are pushed out). Here we restrict the control to the PHYSICAL prior -- the
same functional form as the truth (env(y) * sum_k [c_k^cos cos + c_k^sin sin],
svy only) -- so the inverse is ~10-dimensional and fully observable (SVD cond~21
at short T). Single shooting then recovers at short T and FAILS only past the
SVD information frontier (mode 3) -- "a good method reaches the frontier, none
crosses it."

Env: KH_N, KH_TREC (t_g), KH_RE, KH_NOISE, KH_STEPS, KH_LR, KH_KMAX, KH_SEED, KH_OUT.
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
TREC = float(os.environ.get("KH_TREC", 10))
RE = float(os.environ.get("KH_RE", 2000))
NOISE = float(os.environ.get("KH_NOISE", 1e-2))
STEPS = int(os.environ.get("KH_STEPS", 300))
LR = float(os.environ.get("KH_LR", 3e-3))
KMAX = int(os.environ.get("KH_KMAX", 6))
KMIN = int(os.environ.get("KH_KMIN", 2))
TRUTH_SEED = int(os.environ.get("KH_SEED", 0))
OUT = os.environ.get("KH_OUT", f"data/modes_T{TREC:.0f}_Re{RE:.0f}.npz")


def main():
    khp = P.KHParams(n=N, reynolds=RE, k_min=KMIN, k_max=KMAX)
    T = TREC * khp.growth_time
    cfg, par = P.make_config_params(khp, T, snapshots=0, backward=True)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0_probe = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0_probe.shape)
    y = jnp.asarray(np.asarray(Y[0])); slab = M.shear_layer_slab(y, khp.yc, 0.12)

    # physical prior basis: env(y) * {cos,sin}(2 pi k x), k in [KMIN,KMAX], svy only
    ks = jnp.arange(KMIN, KMAX + 1)
    # Robustness knobs: the control envelope need NOT match the truth's. CTRL_ENVW
    # sets the control envelope width (default = truth's), NYB sets how many y-basis
    # Gaussians (centered on yc, geometric spread of widths) span the layer -- so we
    # can test recovery WITHOUT assuming the exact envelope (NYB>1 = flexible y-profile).
    CTRL_ENVW = float(os.environ.get("KH_CTRL_ENVW", khp.env_width))
    NYB = int(os.environ.get("KH_NYB", 1))
    widths = CTRL_ENVW * (np.geomspace(0.5, 2.0, NYB) if NYB > 1 else np.array([1.0]))
    envs = jnp.stack([jnp.exp(-((Y - khp.yc) / w) ** 2) for w in widths])  # (NYB,Nx,Ny)
    cosx = jnp.cos(2 * jnp.pi * ks[None, None, :] * X[..., None])          # (Nx,Ny,nk)
    sinx = jnp.sin(2 * jnp.pi * ks[None, None, :] * X[..., None])
    print(f"  control: env_w={CTRL_ENVW:.4f} (truth {khp.env_width:.4f}) NYB={NYB}")

    def seed_of(coeffs):                       # coeffs (NYB,nk,2) -> (2,Nx,Ny)
        svy = jnp.zeros(envs.shape[1:])
        for m in range(NYB):
            svy = svy + envs[m] * jnp.sum(coeffs[m, :, 0] * cosx + coeffs[m, :, 1] * sinx, axis=-1)
        return jnp.stack([jnp.zeros_like(svy), svy])

    truth_seed = P.random_broadband_seed(jax.random.PRNGKey(TRUTH_SEED), khp, X, Y)

    def seg(state):
        return time_integration(state, cfg, par._replace(t_end=T), rv)

    obs_clean = P.velocity_of(seg(P.build_state(truth_seed, khp, cfg, rv, X, Y)), rv)
    base = P.velocity_of(seg(P.build_state(jnp.zeros_like(truth_seed), khp, cfg, rv, X, Y)), rv)
    pert_rms = float(jnp.sqrt(jnp.mean((obs_clean - base) ** 2)))
    nsd = NOISE * pert_rms
    obs = obs_clean + nsd * jax.random.normal(jax.random.PRNGKey(999 + TRUTH_SEED), obs_clean.shape)
    J_floor = 0.5 * nsd ** 2
    print(f"  modes T={TREC} Re={RE} nk={len(ks)} pert_rms={pert_rms:.3e} nsd={nsd:.3e}")

    def loss(coeffs):
        s0 = P.build_state(seed_of(coeffs), khp, cfg, rv, X, Y)
        data = 0.5 * jnp.mean((P.velocity_of(seg(s0), rv) - obs) ** 2)
        return data

    def errs(coeffs):
        seed = seed_of(coeffs)
        full = float(jnp.linalg.norm(seed - truth_seed) / jnp.linalg.norm(truth_seed))
        lowk = float(jnp.linalg.norm(M.lowpass_x(seed - truth_seed, KMAX))
                     / (jnp.linalg.norm(M.lowpass_x(truth_seed, KMAX)) + 1e-30))
        return full, lowk

    coeffs = 1e-4 * jax.random.normal(jax.random.PRNGKey(100 + TRUTH_SEED), (NYB, len(ks), 2))
    opt = optax.adam(LR); st = opt.init(coeffs)
    vg = jax.value_and_grad(loss)

    @jax.jit
    def step(coeffs, st):
        v, g = vg(coeffs); upd, st = opt.update(g, st)
        return optax.apply_updates(coeffs, upd), st, v

    t0 = time.time(); stopped = STEPS; hist = []
    for it in range(STEPS):
        coeffs, st, v = step(coeffs, st)
        if it % 25 == 0 or it == STEPS - 1:
            fe, lk = errs(coeffs)
            hist.append((it, float(v), fe, lk))
            print(f"  it={it:3d}: data={float(v):.3e} full_err={fe:.3f} lowk_err={lk:.3f}")
        if float(v) <= 1.1 * J_floor:
            stopped = it; print(f"  early stop it={it}: data<=floor {J_floor:.2e}"); break
    fe, lk = errs(coeffs)
    errk = np.asarray(M.per_kx_relative_error(seed_of(coeffs), truth_seed, slab))
    rt = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez(OUT, t_g=TREC, re=RE, n=N, kmax=KMAX, noise=NOISE, full_err=fe, lowk_err=lk,
             stopped_at=stopped, runtime=rt, kx=np.asarray(M.streamwise_modes(khp.n)),
             errk=errk, rec=np.asarray(seed_of(coeffs)), truth=np.asarray(truth_seed),
             hist=np.array(hist))
    print(f"[done] modes T={TREC:.0f} Re={RE:.0f}: lowk_err={lk:.3f} full_err={fe:.3f} "
          f"stop@{stopped} {rt:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
