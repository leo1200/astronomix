"""
KH reconstruction -- Stage 1: regime selection (the go/no-go).

Runs the forward with a broadband seed (and a zero-seed reference) over a long
horizon and quantifies, per streamwise wavenumber kx and time T:

  * the perturbation energy spectrum  E(kx, T)  -- where small-scale structure
    is generated and where viscosity cuts it off;
  * the seed-memory correlation  C(kx, T) = corr(seed, final_perturbation)  --
    a cheap proxy for the recoverability frontier (high kx decorrelates =
    scrambled; low kx stays correlated = coherent);
  * the mixing-layer width h(T) (vorticity + momentum thickness);
  * vorticity snapshots.

GO criterion: there is a window of T where C(kx) is ~1 at low kx and ~0 at high
kx, with the crossover (a proxy k_rec) receding to lower kx as T grows -- i.e.
small scales scrambled, large scales coherent. If 2D stays globally coherent
(C high at all kx) or globally lost (C low everywhere), reconsider regime/3D.
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
from matplotlib.colors import LogNorm

import problem as P
import metrics as M

OUT = Path(__file__).parent / "figures"
DATA = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)


def main():
    khp = P.KHParams(n=256)
    T_max = 200 * khp.growth_time           # through roll-up into filamentation
    n_snap = 100
    config, params = P.make_config_params(khp, T_max, snapshots=n_snap)
    rv = P.get_registered_variables(config)
    X, Y = P.coords(khp, config)
    # finalize
    s0 = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, config, rv, X, Y)
    config = P.finalize_config(config, s0.shape)

    seed = P.random_broadband_seed(jax.random.PRNGKey(0), khp, X, Y)
    print(f"Forward (seed + zero-seed) to T={T_max:.3f} ({T_max/khp.growth_time:.0f} t_g) ...")
    res_seed = P.forward(seed, khp, config, params, rv, X, Y)
    res_zero = P.forward(jnp.zeros_like(seed), khp, config, params, rv, X, Y)
    times = np.asarray(res_seed.time_points)
    dx = khp.box / khp.n
    y = np.asarray(Y[0])
    slab = M.shear_layer_slab(jnp.asarray(y), khp.yc, 0.12)

    kx = np.asarray(M.streamwise_modes(khp.n))
    n_t = res_seed.states.shape[0]
    Espec = np.zeros((n_t, len(kx)))
    Corr = np.zeros((n_t, len(kx)))
    hvort = np.zeros(n_t); hmom = np.zeros(n_t)
    for i in range(n_t):
        v_s = P.velocity_of(res_seed.states[i], rv)
        v_z = P.velocity_of(res_zero.states[i], rv)
        pert = v_s - v_z
        _, Ek = M.streamwise_energy_spectrum(pert, y_slab=slab)
        Espec[i] = np.asarray(Ek)
        Corr[i] = np.asarray(M.per_kx_correlation(seed, pert, y_slab=slab))
        hvort[i] = float(M.vorticity_thickness(v_s[0], dx, khp.dV))
        hmom[i] = M.momentum_thickness(v_s[0], dx, khp.dV, jnp.asarray(y))

    # proxy frontier: largest kx with correlation still above 0.5
    krec = np.array([kx[np.where(Corr[i] > 0.5)[0]].max() if np.any(Corr[i] > 0.5) else 0
                     for i in range(n_t)])
    np.savez(DATA / "regime.npz", times=times, kx=kx, Espec=Espec, Corr=Corr,
             hvort=hvort, hmom=hmom, krec=krec)

    # ---- figures ----
    # (a) correlation heatmap C(kx, T) with proxy frontier
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ax = axs[0]
    im = ax.pcolormesh(times / khp.growth_time, kx, Corr.T, cmap="RdYlBu_r",
                       vmin=0, vmax=1, shading="auto")
    ax.plot(times / khp.growth_time, krec, "k-", lw=2, label="proxy $k_{rec}$ (C=0.5)")
    ax.set_yscale("log"); ax.set_ylim(1, khp.n // 2)
    ax.set_xlabel("T / t_g"); ax.set_ylabel("streamwise mode kx")
    ax.set_title("seed-memory correlation C(kx,T)"); ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="C")
    # (b) spectra at a few T
    ax = axs[1]
    idxs = [n_t // 6, n_t // 3, n_t // 2, 2 * n_t // 3, n_t - 1]
    for i in idxs:
        ax.loglog(kx[1:], Espec[i, 1:], label=f"T={times[i]/khp.growth_time:.0f} t_g")
    ax.set_xlabel("kx"); ax.set_ylabel("E_pert(kx)")
    ax.set_title("perturbation spectrum vs T"); ax.legend(fontsize=8)
    # (c) mixing width + frontier vs 1/h
    ax = axs[2]
    ax.plot(times / khp.growth_time, hvort, label="vorticity thickness h")
    ax.plot(times / khp.growth_time, hmom, label="momentum thickness")
    ax2 = ax.twinx()
    ax2.plot(times / khp.growth_time, krec, "k--", label="proxy $k_{rec}$")
    ax2.plot(times / khp.growth_time, 1.0 / (hvort + 1e-9), "r:", label="1/h")
    ax.set_xlabel("T / t_g"); ax.set_ylabel("layer width"); ax2.set_ylabel("kx")
    ax.set_title("mixing width h(T) and frontier"); ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("KH Stage 1: regime / recoverability frontier", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "stage1_regime.png", dpi=160); plt.close(fig)

    # vorticity panels
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for ax, i in zip(axs, idxs):
        w = M.vorticity(P.velocity_of(res_seed.states[i], rv), dx)
        vmax = float(np.percentile(np.abs(np.asarray(w)), 99))
        ax.imshow(np.asarray(w).T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax)
        ax.set_title(f"T={times[i]/khp.growth_time:.0f} t_g"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("KH vorticity over time"); fig.tight_layout()
    fig.savefig(OUT / "stage1_vorticity.png", dpi=150); plt.close(fig)

    # ---- verdict print ----
    print("kx where C drops below 0.5 (proxy k_rec) vs T/t_g:")
    for i in idxs:
        print(f"  T={times[i]/khp.growth_time:6.0f} t_g: k_rec~{krec[i]:4d}, "
              f"h_vort={hvort[i]:.3f}, C[k=2]={Corr[i, 2]:.2f}, "
              f"C[k=8]={Corr[i, 8]:.2f}, C[k=32]={Corr[i, min(32,len(kx)-1)]:.2f}")
    receding = krec[idxs[-1]] < krec[idxs[0]]
    print(f"GO/NO-GO: frontier receding (k_rec shrinks with T): {receding}")
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
