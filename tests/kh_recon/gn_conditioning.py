"""
Where (if anywhere) does multiple-shooting-GN beat single-shooting-GN? By the
condensing equivalence, MS-GN only wins when forming the single-shooting Jacobian
J = du_T/dp loses precision, i.e. when kappa(J) ~ e^{lambda T} exceeds float
precision -- MS avoids this by keeping e^{lambda T/M} blocks (never forming the
monodromy). BUT recovery is already dead once the information frontier bites:
sigma_min(J) * (seed scale) < noise (mode 3). So we measure both, vs horizon and
Re, by SVD of the GN Jacobian of the low-D (mode-space, kx2-6) IC:

  sigma_max/sigma_min(T)  -> kappa(T) ~ e^{lambda T}   (mode-1 / MS niche)
  sigma_min(T) vs noise   -> information frontier        (mode 3, method-independent)

If mode 3 precedes the precision wall, MS-GN cannot help recovery (its niche is
already unrecoverable). Higher Re (less viscous info loss) pushes mode 3 out and
may open the window.

Env: KH_N, KH_RE, KH_THORIZONS (t_g), KH_KMIN, KH_KMAX, KH_NOISE.
"""
# ==== GPU selection ====
import os, sys
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd
    autocvd(num_gpus=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ruff: noqa: E402
# =======================
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
import matplotlib.pyplot as plt
import problem as P
from astronomix import time_integration

OUT = Path(__file__).parent / "figures"; DATA = Path(__file__).parent / "data"
N = int(os.environ.get("KH_N", 64))
RE = float(os.environ.get("KH_RE", 2000))
KMIN = int(os.environ.get("KH_KMIN", 2)); KMAX = int(os.environ.get("KH_KMAX", 6))
T_G = [float(x) for x in os.environ.get("KH_THORIZONS", "10,20,30,40,60,80,120,160").split(",")]
NOISE = float(os.environ.get("KH_NOISE", 1e-2))


def main():
    khp = P.KHParams(n=N, reynolds=RE, k_min=KMIN, k_max=KMAX)
    cfg0, _ = P.make_config_params(khp, 1.0, snapshots=0, backward=False)
    rv = P.get_registered_variables(cfg0)
    X, Y = P.coords(khp, cfg0)
    s0p = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg0, rv, X, Y)
    cfg = P.finalize_config(cfg0, s0p.shape)
    ks = jnp.arange(KMIN, KMAX + 1)
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    cosx = jnp.cos(2 * jnp.pi * ks[None, None, :] * X[..., None])
    sinx = jnp.sin(2 * jnp.pi * ks[None, None, :] * X[..., None])

    def seed_of(c):
        svy = env * jnp.sum(c[:, 0] * cosx + c[:, 1] * sinx, axis=-1)
        return jnp.stack([jnp.zeros_like(svy), svy])

    # reference IC = the truth seed projected to these modes (use the broadband truth)
    truth = P.random_broadband_seed(jax.random.PRNGKey(0), khp, X, Y)
    # least-squares project truth svy onto the (cos,sin)*env basis to get c0
    A = jnp.stack([ (env*cosx[...,i]).ravel() for i in range(len(ks))]
                 +[ (env*sinx[...,i]).ravel() for i in range(len(ks))], axis=1)  # (Ncell, 2nk)
    coef, *_ = jnp.linalg.lstsq(A, truth[1].ravel(), rcond=None)
    c0 = jnp.stack([coef[:len(ks)], coef[len(ks):]], axis=1)  # (nk,2)

    rows = []
    for Tg in T_G:
        T = Tg * khp.growth_time
        par = P.make_config_params(khp, T, snapshots=0, backward=False)[1]

        def fwd(c):
            s = P.build_state(seed_of(c), khp, cfg, rv, X, Y)
            return P.velocity_of(time_integration(s, cfg, par._replace(t_end=T), rv), rv).ravel()

        J = np.asarray(jax.jacfwd(fwd)(c0)).reshape(-1, c0.size)   # (nobs, np)
        sv = np.linalg.svd(J, compute_uv=False)      # np singular values
        # noise floor on the obs (perturbation-relative); recover mode i if
        # sigma_i*||dp|| > nsd*sqrt(nobs). Use pert imprint rms for nsd.
        base = np.asarray(fwd(c0 * 0.0))
        ref = np.asarray(fwd(c0))
        pert_rms = float(np.sqrt(np.mean((ref - base) ** 2)))
        nsd = NOISE * pert_rms
        nfloor = nsd * np.sqrt(J.shape[0])            # ||noise|| in obs space
        rows.append((Tg, sv[0], sv[-1], sv[0] / sv[-1], nfloor, pert_rms))
        print(f"  T={Tg:6.1f}: smax={sv[0]:.3e} smin={sv[-1]:.3e} kappa={sv[0]/sv[-1]:.2e} "
              f"||noise||={nfloor:.3e}")

    R = np.array(rows)
    tag = "" if RE == 2000 else f"_Re{RE:.0f}"
    np.savez(DATA / f"gn_conditioning{tag}.npz", T_g=R[:, 0], smax=R[:, 1], smin=R[:, 2],
             kappa=R[:, 3], nfloor=R[:, 4], pert_rms=R[:, 5])

    Ts = R[:, 0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].semilogy(Ts, R[:, 3], "o-", label=r"$\kappa(J)=\sigma_{max}/\sigma_{min}$")
    ax[0].axhline(1 / np.finfo(np.float32).eps, color="C1", ls="--", label="float32 wall ($\\sim10^7$)")
    ax[0].axhline(1 / np.finfo(np.float64).eps, color="C3", ls="--", label="float64 wall ($\\sim10^{16}$)")
    ax[0].set_xlabel("horizon T / t_g"); ax[0].set_ylabel("condition number of GN Jacobian")
    ax[0].set_title(f"(a) single-shooting GN conditioning (Re={RE:.0f})\nMS-GN's niche: above the precision walls")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    ax[1].semilogy(Ts, R[:, 1], "o-", label=r"$\sigma_{max}$")
    ax[1].semilogy(Ts, R[:, 2], "s-", label=r"$\sigma_{min}$ (limits IC recovery)")
    ax[1].semilogy(Ts, R[:, 4], "k--", label="obs noise level")
    ax[1].set_xlabel("horizon T / t_g"); ax[1].set_ylabel("singular value of GN Jacobian")
    ax[1].set_title("(b) information frontier (mode 3):\n$\\sigma_{min}$ crossing noise kills recovery first")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle(f"Where MS-GN could help: mode-1 precision wall vs mode-3 info frontier (Re={RE:.0f})", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"fig_gn_conditioning{tag}.png", dpi=160); plt.close(fig)
    print(f"-> {OUT/f'fig_gn_conditioning{tag}.png'}")


if __name__ == "__main__":
    main()
