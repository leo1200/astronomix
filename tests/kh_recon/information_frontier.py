"""
KH reconstruction -- Stage 2: the information frontier (independent ground truth).

For each horizon T we form the tangent map M_T = d u_T / d u0 around the
reference (seeded) trajectory and measure the per-streamwise-wavenumber GAIN

    sigma(kx, T) = rms_phase || M_T v_kx ||,   v_kx a unit input localized on the
                   shear layer with streamwise wavenumber kx,

via forward-mode AD (jax.jvp), vmapped over a log-sampled set of (kx, phase) so
the whole probe is one compiled batched call (no per-direction Python loop, no
reverse-mode -- far cheaper than a full randomized SVD). gain = "how much an
initial perturbation at scale kx still imprints on the observation u_T"; below
the noise floor eps it is unrecoverable (mode 3). The recoverable edge is
k_rec(T). We also track sigma_max(T) ~ e^{lambda T} and the mixing width h(T)
(prediction k_rec ~ 1/h).

Env: KH_N, KH_THORIZONS (comma list, t_g), KH_NK (kx samples), KH_EPS.
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
import matplotlib.pyplot as plt

import problem as P
import metrics as M

OUT = Path(__file__).parent / "figures"
DATA = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)

N = int(os.environ.get("KH_N", 192))
RE = float(os.environ.get("KH_RE", 2000))
T_G = [float(x) for x in os.environ.get("KH_THORIZONS", "20,60,100,160").split(",")]
NK = int(os.environ.get("KH_NK", 24))          # number of log-sampled kx
NOISE_FLOOR = float(os.environ.get("KH_EPS", 1e-2))
ENV_W = 0.06                                    # input localization width in y


def unit_mode_inputs(khp, X, Y, kx_list):
    """Build unit-norm single-kx inputs (cos & sin phase) localized on the layer.
    Returns array (n_inputs, 2, N, N) and the kx for each input."""
    env = jnp.exp(-((Y - khp.yc) / ENV_W) ** 2)
    inputs = []; kxs = []
    for kx in kx_list:
        for phase in (0.0, jnp.pi / 2):
            f = jnp.cos(2 * jnp.pi * kx * X + phase) * env
            v = jnp.stack([jnp.zeros_like(f), f])           # transverse-velocity input
            v = v / jnp.sqrt(jnp.sum(v ** 2))
            inputs.append(v); kxs.append(int(kx))
    return jnp.stack(inputs), np.array(kxs)


def run_for_T(khp, Tg, kx_list):
    T = Tg * khp.growth_time
    cfg, par = P.make_config_params(khp, T, snapshots=0, backward=False)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0 = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0.shape)
    seed = P.random_broadband_seed(jax.random.PRNGKey(0), khp, X, Y)

    def f(s):
        return P.velocity_of(P.forward(s, khp, cfg, par, rv, X, Y), rv)

    inputs, kxs = unit_mode_inputs(khp, X, Y, kx_list)
    # batched forward-mode tangent: one compiled call over all (kx, phase)
    batched = jax.jit(jax.vmap(lambda v: jax.jvp(f, (seed,), (v,))[1]))
    resp = batched(inputs)                                   # (n_inputs, 2, N, N)
    gains = np.asarray(jnp.sqrt(jnp.sum(resp ** 2, axis=(1, 2, 3))))  # ||M v||
    # rms over the two phases -> sigma(kx)
    sig_kx = np.array([np.sqrt(np.mean(gains[kxs == int(k)] ** 2)) for k in kx_list])

    res = P.forward(seed, khp, cfg, par, rv, X, Y)
    h = M.momentum_thickness(P.velocity_of(res, rv)[0], khp.box / khp.n, khp.dV,
                             jnp.asarray(np.asarray(Y[0])))
    return sig_kx, float(h), T


def main():
    khp = P.KHParams(n=N, reynolds=RE)
    kx_list = np.unique(np.round(np.geomspace(1, khp.n // 2, NK)).astype(int))
    print(f"Information frontier (per-kx gain): N={N}, kx={list(kx_list)}, T={T_G}")
    results = {}
    for Tg in T_G:
        sig_kx, h, T = run_for_T(khp, Tg, kx_list)
        smax = sig_kx.max()
        floor = NOISE_FLOOR * smax
        krec = int(kx_list[np.where(sig_kx > floor)[0]].max()) if np.any(sig_kx > floor) else 0
        results[Tg] = dict(sig_kx=sig_kx, h=h, krec=krec, smax=float(smax), T=T)
        print(f"  T={Tg:6.0f} t_g (t={T:.3f}): sigma_max={smax:.3e}, k_rec={krec:3d}, "
              f"1/h={1/h:6.1f}, h={h:.3f}")

    tag = "" if RE == 2000 else f"_Re{RE:.0f}"
    np.savez(DATA / f"frontier{tag}.npz", T_g=np.array(T_G), kx=kx_list,
             **{f"sigkx_{Tg}": results[Tg]["sig_kx"] for Tg in T_G},
             h=np.array([results[Tg]["h"] for Tg in T_G]),
             krec=np.array([results[Tg]["krec"] for Tg in T_G]),
             smax=np.array([results[Tg]["smax"] for Tg in T_G]))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ax = axs[0]
    for Tg in T_G:
        r = results[Tg]
        ax.loglog(kx_list, r["sig_kx"] / r["smax"], "o-", label=f"T={Tg:.0f} t_g")
    ax.axhline(NOISE_FLOOR, color="k", ls="--", label="noise floor")
    ax.set_xlabel("kx"); ax.set_ylabel(r"$\sigma(k_x)/\sigma_{max}$")
    ax.set_title("recoverability gain vs kx (information frontier)"); ax.legend(fontsize=8)

    ax = axs[1]
    krec = np.array([results[Tg]["krec"] for Tg in T_G])
    h = np.array([results[Tg]["h"] for Tg in T_G])
    ax.plot(T_G, krec, "o-", label="$k_{rec}(T)$")
    ax.plot(T_G, 1.0 / h, "s--", label="1/h(T)")
    ax.set_xlabel("T / t_g"); ax.set_ylabel("kx"); ax.set_title("frontier recedes; tracks 1/h")
    ax.legend()

    ax = axs[2]
    smax = np.array([results[Tg]["smax"] for Tg in T_G])
    Ts = np.array(T_G) * khp.growth_time
    ax.semilogy(Ts, smax, "o-", label=r"$\sigma_{max}(T)$")
    if len(Ts) > 1:
        lam = np.polyfit(Ts, np.log(smax), 1)[0]
        ax.semilogy(Ts, smax[0] * np.exp(lam * (Ts - Ts[0])), "k--", label=rf"$e^{{{lam:.1f}T}}$")
        print(f"sigma_max growth rate lambda = {lam:.2f} / time")
    ax.set_xlabel("T"); ax.set_ylabel(r"$\sigma_{max}$"); ax.set_title("tangent gain growth (mode 1)")
    ax.legend()
    fig.suptitle(f"KH Stage 2: information frontier (tangent per-kx gain, Re={RE:.0f})", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / f"stage2_frontier{tag}.png", dpi=160); plt.close(fig)
    print(f"figure -> {OUT / f'stage2_frontier{tag}.png'}")


if __name__ == "__main__":
    main()
