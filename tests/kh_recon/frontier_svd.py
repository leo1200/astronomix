"""
Information frontier via a structured SVD of the tangent map M_T = du_T/du0,
using ONLY forward-mode AD (no vjp / no backward integration -- which is what
crashed the naive vmapped-randomized-SVD: vmapping the reverse pass through the
OPEN_BOUNDARY ghost-cell padding of the checkpointed backward loop is not
vmap-safe).

Idea. The per-kx FORWARD gain ||M v_kx|| (information_frontier.py) is dominated
by the single leading Lyapunov direction: every input projects onto it and is
amplified ~e^{lambda T}, so it conflates "grows" with "recoverable" and gives no
receding edge. The SINGULAR VECTORS fix this. We restrict the input space to a
physically-relevant, shear-localized streamwise-mode basis V = {cos/sin(2 pi kx x)
* env(y)} (a few hundred vectors), apply the tangent map by a vmapped forward
jvp R = M V (this works -- forward only), then eigendecompose the small Gram
matrix G = R R^T. Its eigenvalues are sigma_a^2; the leading eigenvector is the
optimal precursor (absorbs the Lyapunov direction), and SUB-leading singular
values expose the next, orthogonal, recoverable directions. Projecting each
right singular vector U_a = sum_i c_a[i] V_i onto kx and assigning sigma_a to its
dominant kx gives a DISENTANGLED sigma(kx); the recoverable edge k_rec(T) is
where it crosses the noise floor eps.

Env: KH_N, KH_RE, KH_THORIZONS (t_g list), KH_NK (kx samples), KH_EPS.
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
import jax, jax.numpy as jnp, numpy as np
import matplotlib.pyplot as plt
import problem as P, metrics as M

OUT = Path(__file__).parent / "figures"; DATA = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
N = int(os.environ.get("KH_N", 96))
RE = float(os.environ.get("KH_RE", 2000))
T_G = [float(x) for x in os.environ.get("KH_THORIZONS", "20,60,100,160").split(",")]
NK = int(os.environ.get("KH_NK", 28))           # number of log-sampled kx
EPS = float(os.environ.get("KH_EPS", 1e-2))
ENV_W = 0.06                                     # input localization width in y


def mode_basis(khp, X, Y, kx_list):
    """Unit-norm shear-localized single-kx inputs (cos & sin). (n_in,2,N,N), kx."""
    env = jnp.exp(-((Y - khp.yc) / ENV_W) ** 2)
    V = []; kxs = []
    for kx in kx_list:
        for ph in (0.0, jnp.pi / 2):
            f = jnp.cos(2 * jnp.pi * int(kx) * X + ph) * env
            v = jnp.stack([jnp.zeros_like(f), f])        # transverse-velocity input
            V.append(v / jnp.sqrt(jnp.sum(v ** 2))); kxs.append(int(kx))
    return jnp.stack(V), np.array(kxs)


def run_T(khp, Tg, kx_list, slab):
    T = Tg * khp.growth_time
    cfg, par = P.make_config_params(khp, T, snapshots=0, backward=False)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0 = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0.shape)
    seed = P.random_broadband_seed(jax.random.PRNGKey(0), khp, X, Y)

    def f(s):
        return P.velocity_of(P.forward(s, khp, cfg, par, rv, X, Y), rv)

    V, kxs = mode_basis(khp, X, Y, kx_list)                 # (n_in,2,N,N)
    R = jax.jit(jax.vmap(lambda v: jax.jvp(f, (seed,), (v,))[1]))(V)  # M V, forward only
    Rf = R.reshape(R.shape[0], -1)                          # (n_in, 2*N*N)
    G = np.asarray(Rf @ Rf.T)                               # (n_in, n_in) Gram
    w, C = np.linalg.eigh(G)                                # ascending
    idx = np.argsort(w)[::-1]; w = w[idx]; C = C[:, idx]
    w = np.clip(w, 0, None); sig = np.sqrt(w)               # singular values of M|_V
    # right singular vectors in physical space: U_a = sum_i C[i,a] V_i
    Vn = np.asarray(V)                                      # (n_in,2,N,N)
    nsv = min(len(sig), 2 * len(kx_list))
    dom = np.empty(nsv, dtype=int)
    for a in range(nsv):
        Ua = np.tensordot(C[:, a], Vn, axes=(0, 0))         # (2,N,N)
        _, e = M.streamwise_energy_spectrum(jnp.asarray(Ua), slab)
        dom[a] = int(kx_list[int(np.argmax(np.asarray(e)[kx_list]))]) \
            if np.asarray(e)[kx_list].size else 0
    # disentangled per-kx sigma = largest singular value whose vector is that kx
    sig_kx = np.array([sig[dom == int(k)].max() if np.any(dom == int(k)) else 0.0
                       for k in kx_list])
    # diagnostic: the raw per-kx forward gain (Lyapunov-contaminated), for contrast
    gain_kx = np.array([np.sqrt(np.mean(
        np.sum(Rf[kxs == int(k)] ** 2, axis=1))) for k in kx_list])
    res = P.forward(seed, khp, cfg, par, rv, X, Y)
    h = M.momentum_thickness(P.velocity_of(res, rv)[0], khp.box / khp.n, khp.dV,
                             jnp.asarray(np.asarray(Y[0])))
    return sig, sig_kx, gain_kx, float(h), T


def main():
    khp = P.KHParams(n=N, reynolds=RE)
    kx_list = np.unique(np.round(np.geomspace(1, khp.n // 2, NK)).astype(int))
    y = jnp.asarray(np.asarray(P.coords(khp, P.make_config_params(khp, 1.0)[0])[1][0]))
    slab = M.shear_layer_slab(y, khp.yc, 0.12)
    tag = "" if RE == 2000 else f"_Re{RE:.0f}"
    print(f"Structured-SVD frontier N={N} Re={RE} kx={list(kx_list)} T={T_G}")
    R = {}
    for Tg in T_G:
        sig, sig_kx, gain_kx, h, T = run_T(khp, Tg, kx_list, slab)
        floor = EPS * sig[0]
        krec = int(kx_list[np.where(sig_kx > floor)[0]].max()) if np.any(sig_kx > floor) else 0
        R[Tg] = dict(sig=sig, sig_kx=sig_kx, gain_kx=gain_kx, h=h, krec=krec)
        print(f"  T={Tg:6.0f} t_g: sig_max={sig[0]:.3e} sig[1]={sig[1]:.3e} "
              f"sig_min={sig[-1]:.3e} k_rec={krec} 1/h={1/h:.1f}")
    np.savez(DATA / f"frontier_svd{tag}.npz", T_g=np.array(T_G), kx=kx_list,
             **{f"sigkx_{t}": R[t]["sig_kx"] for t in T_G},
             **{f"gainkx_{t}": R[t]["gain_kx"] for t in T_G},
             **{f"sig_{t}": R[t]["sig"] for t in T_G},
             h=np.array([R[t]["h"] for t in T_G]),
             krec=np.array([R[t]["krec"] for t in T_G]))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for t in T_G:
        r = R[t]
        ax[0].loglog(kx_list, r["sig_kx"] / r["sig"][0], "o-", label=f"T={t:.0f} t_g")
    ax[0].axhline(EPS, color="k", ls="--", label="noise floor")
    ax[0].set_xlabel("kx"); ax[0].set_ylabel("sigma(kx)/sigma_max (disentangled)")
    ax[0].set_title(f"SVD recoverability frontier (Re={RE:.0f})"); ax[0].legend(fontsize=8)
    # singular-value spectrum (rank of recoverable info)
    for t in T_G:
        s = R[t]["sig"]; ax[1].semilogy(np.arange(len(s)), s / s[0], "-", label=f"T={t:.0f}")
    ax[1].axhline(EPS, color="k", ls="--"); ax[1].set_xlabel("singular index a")
    ax[1].set_ylabel("sigma_a/sigma_max"); ax[1].set_title("tangent singular spectrum")
    ax[1].legend(fontsize=8)
    kr = np.array([R[t]["krec"] for t in T_G]); h = np.array([R[t]["h"] for t in T_G])
    ax[2].plot(T_G, kr, "o-", label="k_rec(T) [SVD]")
    ax[2].plot(T_G, 1 / h, "s--", label="1/h")
    ax[2].plot(T_G, 1 / (2 * np.pi * h), "^--", label="1/(2 pi h)")
    ax[2].set_xlabel("T/t_g"); ax[2].set_ylabel("kx"); ax[2].legend(); ax[2].set_title("frontier vs 1/h")
    fig.suptitle(f"KH structured-SVD information frontier (forward-jvp Gram, Re={RE:.0f})", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / f"frontier_svd{tag}.png", dpi=160); plt.close(fig)
    print(f"-> {OUT / f'frontier_svd{tag}.png'}")


if __name__ == "__main__":
    main()
