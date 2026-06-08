"""
KH reconstruction -- Stage 3: single-shooting (reduced-space) baseline.

Reconstruct the seed u0 from a single later snapshot u_T by backprop through the
solver. Diagnostics for the failure modes:
  * MODE 1 (gradient explosion): ||grad J||(T) at cold start grows ~ e^{lambda T}.
  * reconstruction quality per kx vs T (compare to the Stage-2 frontier).
  * MODE 2 (multimodality): multi-restart success spread + a 1-D loss slice.

Env: KH_MODE in {gradnorm, reconstruct, landscape} (default gradnorm),
     KH_N, KH_TREC (horizon in t_g for reconstruct/landscape), KH_STEPS.
"""

# ==== GPU selection ====
import os, sys
from autocvd import autocvd
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    autocvd(num_gpus=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ruff: noqa: E402
# =======================

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

import problem as P
import metrics as M

OUT = Path(__file__).parent / "figures"
DATA = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)

N = int(os.environ.get("KH_N", 192))
MODE = os.environ.get("KH_MODE", "gradnorm")
TREC = float(os.environ.get("KH_TREC", 80))
STEPS = int(os.environ.get("KH_STEPS", 120))


def setup(khp, Tg, backward=True):
    T = Tg * khp.growth_time
    cfg, par = P.make_config_params(khp, T, snapshots=0, backward=backward)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0 = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0.shape)
    truth = P.random_broadband_seed(jax.random.PRNGKey(0), khp, X, Y)
    obs = P.velocity_of(P.forward(truth, khp, cfg, par, rv, X, Y), rv)
    return cfg, par, rv, X, Y, truth, obs


def main():
    khp = P.KHParams(n=N)

    if MODE == "gradnorm":
        Tg_list = [10, 20, 40, 60, 80, 120, 160]
        gnorms = []
        for Tg in Tg_list:
            cfg, par, rv, X, Y, truth, obs = setup(khp, Tg)

            def J(s):
                v = P.velocity_of(P.forward(s, khp, cfg, par, rv, X, Y), rv)
                return 0.5 * jnp.mean((v - obs) ** 2)

            g = jax.grad(J)(jnp.zeros_like(truth))   # cold-start gradient
            gn = float(jnp.sqrt(jnp.sum(g ** 2)))
            gnorms.append(gn)
            print(f"  T={Tg:4.0f} t_g: ||grad J||_coldstart = {gn:.4e}")
        Ts = np.array(Tg_list) * khp.growth_time
        gnorms = np.array(gnorms)
        lam = np.polyfit(Ts, np.log(gnorms), 1)[0]
        np.savez(DATA / "ss_gradnorm.npz", Tg=np.array(Tg_list), gnorm=gnorms, lam=lam)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(Tg_list, gnorms, "o-", label="||grad J|| (cold start)")
        ax.semilogy(Tg_list, gnorms[0] * np.exp(lam * (Ts - Ts[0])), "k--",
                    label=rf"$e^{{{lam:.1f}\,T}}$")
        ax.set_xlabel("T / t_g"); ax.set_ylabel("gradient norm")
        ax.set_title("Single shooting: gradient explosion (mode 1)")
        ax.legend(); ax.grid(alpha=0.3, which="both")
        fig.tight_layout(); fig.savefig(OUT / "ss_gradnorm.png", dpi=160); plt.close(fig)
        print(f"fitted exponential rate lambda={lam:.2f}; figure -> {OUT}")

    elif MODE in ("reconstruct", "landscape"):
        cfg, par, rv, X, Y, truth, obs = setup(khp, TREC)
        y = jnp.asarray(np.asarray(Y[0]))
        slab = M.shear_layer_slab(y, khp.yc, 0.12)

        def J(s):
            v = P.velocity_of(P.forward(s, khp, cfg, par, rv, X, Y), rv)
            return 0.5 * jnp.mean((v - obs) ** 2)

        if MODE == "landscape":
            # 1-D slice along (truth - 0) and a random direction through cold start
            alphas = np.linspace(-0.5, 1.8, 41)
            rdir = jax.random.normal(jax.random.PRNGKey(3), truth.shape)
            rdir = rdir / jnp.sqrt(jnp.sum(rdir ** 2)) * jnp.sqrt(jnp.sum(truth ** 2))
            Ltruth = [float(J(a * truth)) for a in alphas]
            Lrand = [float(J(truth + (a - 1.0) * rdir)) for a in alphas]
            np.savez(DATA / f"ss_landscape_T{TREC:.0f}.npz",
                     alphas=alphas, Ltruth=Ltruth, Lrand=Lrand)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(alphas, Ltruth, "o-", label="along truth direction")
            ax.plot(alphas, Lrand, "s-", label="along random direction")
            ax.axvline(1.0, color="k", ls="--", label="truth")
            ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("loss J")
            ax.set_title(f"Loss landscape slice (T={TREC:.0f} t_g, mode 2)")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(OUT / f"ss_landscape_T{TREC:.0f}.png", dpi=160)
            plt.close(fig)
            print(f"landscape -> {OUT}")
            return

        # reconstruct: L-BFGS from several cold-ish starts -> success spread
        opt = optax.lbfgs()
        vg = optax.value_and_grad_from_state(J)

        @jax.jit
        def step(s, st):
            val, g = vg(s, state=st)
            upd, st = opt.update(g, st, s, value=val, grad=g, value_fn=J)
            return optax.apply_updates(s, upd), st, val

        truth_lk_err = lambda s: float(jnp.linalg.norm(M.lowpass_x(s - truth, 4))
                                       / (jnp.linalg.norm(M.lowpass_x(truth, 4)) + 1e-30))
        results = []
        for r in range(4):
            s = 1e-3 * jax.random.normal(jax.random.PRNGKey(100 + r), truth.shape)
            st = opt.init(s)
            for _ in range(STEPS):
                s, st, val = step(s, st)
            err = float(jnp.linalg.norm(s - truth) / jnp.linalg.norm(truth))
            errk = np.asarray(M.per_kx_relative_error(s, truth, slab))
            results.append((float(val), err, truth_lk_err(s), errk, s))
            print(f"  restart {r}: J={val:.3e} full_err={err:.3f} lowk_err={truth_lk_err(s):.3f}")
        # best restart per-kx error
        best = min(results, key=lambda z: z[0])
        kx = np.asarray(M.streamwise_modes(khp.n))
        np.savez(DATA / f"ss_reconstruct_T{TREC:.0f}.npz",
                 kx=kx, errk_best=best[3],
                 full_errs=np.array([r[1] for r in results]),
                 lowk_errs=np.array([r[2] for r in results]),
                 losses=np.array([r[0] for r in results]),
                 rec_best=np.asarray(best[4]), truth=np.asarray(truth))
        print(f"SS T={TREC:.0f}: full_err spread {[f'{r[1]:.2f}' for r in results]}, "
              f"lowk_err {[f'{r[2]:.2f}' for r in results]}")
        print(f"saved ss_reconstruct_T{TREC:.0f}.npz")


if __name__ == "__main__":
    main()
