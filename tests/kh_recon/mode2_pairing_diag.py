"""
Diagnostic for the mode-2 (vortex-pairing) setting: roll up a 2-mode seed
(fundamental kx=2 + subharmonic kx=1) and save a vorticity time-montage so we can
pick the horizon T* that spans one PAIRING event (two billows -> one). Pairing is
the many-to-one folding that makes the terminal-state inverse multimodal (mode 2),
while the large scales stay recoverable (mode 3 generous).

Env: KH_N, KH_RE, KH_TEND (t_g), KH_A1 (subharmonic amp rel. to fundamental),
     KH_PHI1 (kx=1 phase), KH_NSNAP.
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
import problem as P, metrics as M

OUT = Path(__file__).parent / "figures"
N = int(os.environ.get("KH_N", 128))
RE = float(os.environ.get("KH_RE", 2000))
TEND = float(os.environ.get("KH_TEND", 90))
A1 = float(os.environ.get("KH_A1", 0.7))        # subharmonic amplitude (fund=1)
PHI1 = float(os.environ.get("KH_PHI1", 0.0))
NSNAP = int(os.environ.get("KH_NSNAP", 9))


def two_mode_seed(khp, X, Y, a2=1.0, a1=A1, phi2=0.0, phi1=PHI1):
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    svy = (a2 * jnp.cos(2 * jnp.pi * 2 * X + phi2)
           + a1 * jnp.cos(2 * jnp.pi * 1 * X + phi1)) * env
    svy = khp.seed_amp * khp.dV * svy / jnp.sqrt(jnp.mean(svy ** 2) + 1e-30)
    return jnp.stack([jnp.zeros_like(svy), svy])


SEEDAMP = float(os.environ.get("KH_SEEDAMP", 1e-2))


def main():
    khp = P.KHParams(n=N, reynolds=RE, seed_amp=SEEDAMP)
    T = TEND * khp.growth_time
    cfg, par = P.make_config_params(khp, T, snapshots=NSNAP, backward=False)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0 = P.build_state(two_mode_seed(khp, X, Y), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0.shape)
    out = P.time_integration(s0, cfg, par, rv)
    states = out.states if hasattr(out, "states") else out
    ts = np.asarray(out.time_points) if hasattr(out, "time_points") else np.linspace(0, T, NSNAP)
    dx = khp.box / khp.n
    fig, ax = plt.subplots(1, len(states), figsize=(2.1 * len(states), 2.4))
    for i, st in enumerate(states):
        w = np.asarray(M.vorticity(P.velocity_of(jnp.asarray(st), rv), dx))
        ax[i].imshow(w.T, origin="lower", cmap="RdBu_r")
        ax[i].set_title(f"{ts[i]/khp.growth_time:.0f} t_g", fontsize=8)
        ax[i].set_xticks([]); ax[i].set_yticks([])
    fig.suptitle(f"2-mode roll-up + pairing (kx=2 + {A1}*kx=1, Re={RE:.0f}, N={N})", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "mode2_pairing_diag.png", dpi=150); plt.close(fig)
    print(f"-> {OUT/'mode2_pairing_diag.png'}  times(t_g)={[f'{t/khp.growth_time:.0f}' for t in ts]}")


if __name__ == "__main__":
    main()
