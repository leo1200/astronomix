"""
KH reconstruction -- Stage 0: forward sanity + differentiability check.

  1. Run the 2D viscous KH forward from base+broadband-seed; save vorticity
     snapshots + an animation; confirm billow roll-up.
  2. Differentiability: scalar loss on the final state, jax.grad w.r.t. the seed
     at a SHORT horizon; finite-difference check on a few random components.

Accept when: billows form; short-horizon gradient is finite and matches FD.
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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import problem as P

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def vorticity(v, dx):
    # omega_z = dvy/dx - dvx/dy  (periodic-ish central diff; ok for a diagnostic)
    dvydx = (jnp.roll(v[1], -1, 0) - jnp.roll(v[1], 1, 0)) / (2 * dx)
    dvxdy = (jnp.roll(v[0], -1, 1) - jnp.roll(v[0], 1, 1)) / (2 * dx)
    return dvydx - dvxdy


def main():
    khp = P.KHParams(n=256)
    print(f"KH: c_s={khp.dV/khp.mach:.2f}, P={khp.pressure:.2f}, mu={khp.viscosity:.2e}, "
          f"Re={khp.reynolds:.0f}, growth_time~{khp.growth_time:.3f}")

    # ---- forward run with snapshots ----
    t_end = 200 * khp.growth_time   # through roll-up into filamentation
    config, params = P.make_config_params(khp, t_end, snapshots=100)
    config = finalize(config, khp)
    rv = P.get_registered_variables(config)
    X, Y = P.coords(khp, config)
    seed = P.random_broadband_seed(jax.random.PRNGKey(0), khp, X, Y)

    print(f"Running forward to t_end={t_end:.3f} ({t_end/khp.growth_time:.0f} growth times) ...")
    result = P.forward(seed, khp, config, params, rv, X, Y)
    states = result.states
    times = result.time_points
    dx = khp.box / khp.n
    print(f"  {states.shape[0]} snapshots; final time {float(times[-1]):.3f}")

    # vorticity animation
    vmaps = []
    for i in range(states.shape[0]):
        v = P.velocity_of(states[i], rv)
        vmaps.append(vorticity(v, dx))
    vmaps = jnp.stack(vmaps)
    vmax = float(jnp.percentile(jnp.abs(vmaps[-1]), 99))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(vmaps[0].T, origin="lower", extent=(0, 1, 0, 1),
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("KH vorticity"); ax.set_xlabel("x (flow)"); ax.set_ylabel("y")
    ttl = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def upd(i):
        im.set_data(vmaps[i].T); ttl.set_text(f"t={float(times[i]):.2f}")
        return im, ttl
    FuncAnimation(fig, upd, frames=len(times), interval=100).save(
        OUT / "kh_forward.gif", writer=PillowWriter(fps=12), dpi=110)
    plt.close(fig)

    # initial vs final vorticity triptych
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, idx, lab in [(axs[0], 0, "t=0"),
                         (axs[1], len(times)//2, f"t={float(times[len(times)//2]):.2f}"),
                         (axs[2], -1, f"t={float(times[-1]):.2f}")]:
        ax.imshow(vmaps[idx].T, origin="lower", extent=(0, 1, 0, 1),
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(lab); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle("KH roll-up (vorticity)"); fig.tight_layout()
    fig.savefig(OUT / "kh_rollup.png", dpi=150); plt.close(fig)
    print(f"  figures -> {OUT}")

    # ---- differentiability + finite-difference check at SHORT horizon ----
    Tg = 8 * khp.growth_time
    cfg_b, par_b = P.make_config_params(khp, Tg, snapshots=0, backward=True)
    cfg_b = finalize(cfg_b, khp)

    def loss(s):
        final = P.forward(s, khp, cfg_b, par_b, rv, X, Y)
        v = P.velocity_of(final, rv)
        return 0.5 * jnp.mean(v[1] ** 2)   # mean transverse kinetic energy

    g = jax.grad(loss)(seed)
    gnorm = float(jnp.sqrt(jnp.sum(g**2)))
    L0 = float(loss(seed))
    print(f"short-horizon (t={Tg:.3f}) loss={L0:.3e} gradient norm={gnorm:.3e} "
          f"finite={bool(jnp.all(jnp.isfinite(g)))}")

    # DIRECTIONAL central-difference check (robust to float32 -- per-component
    # FD is dominated by roundoff because single-point sensitivities are ~1e-10
    # relative). Along the gradient direction the signal is maximal.
    ghat = g / (gnorm + 1e-30)
    rkey = jax.random.PRNGKey(7)
    rdir = jnp.zeros_like(seed).at[1].set(jax.random.normal(rkey, (khp.n, khp.n)))
    rdir = rdir / jnp.sqrt(jnp.sum(rdir**2))
    rel_errs = []
    for name, d in [("grad-dir", ghat), ("rand-dir", rdir)]:
        ad = float(jnp.sum(g * d))
        eps = 1e-2
        fd = (float(loss(seed + eps * d)) - float(loss(seed - eps * d))) / (2 * eps)
        rel = abs(fd - ad) / (abs(fd) + abs(ad) + 1e-30)
        rel_errs.append(rel)
        print(f"  {name}: AD={ad:.4e} FD(central,eps={eps:g})={fd:.4e} rel_err={rel:.2e}")
    ok = max(rel_errs) < 2e-2 and bool(jnp.all(jnp.isfinite(g)))
    print(f"STAGE 0 {'PASS' if ok else 'CHECK'}: max directional rel_err={max(rel_errs):.2e}")


def finalize(config, khp):
    # finalize_config needs the real state shape -> build a zero-seed state
    rv = P.get_registered_variables(config)
    X, Y = P.coords(khp, config)
    s = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, config, rv, X, Y)
    return P.finalize_config(config, s.shape)


if __name__ == "__main__":
    main()
