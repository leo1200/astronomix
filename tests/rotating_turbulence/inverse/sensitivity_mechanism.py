"""
SS-vs-MS mechanism for the rotating-turbulence inverse problem (forward-only).

The single-shooting gradient through a window T tracks the forward sensitivity
of the barotropic-columnar observation P_large to a barotropic-columnar IC
perturbation,

    A(t) = || P_large(x(t; x0*+d)) - P_large(x(t; x0*)) || / ||d|| ,

    P_large = horizontal low-pass (|k_perp| <= k_c) of the vertical average of
              the horizontal velocity (the slow z-invariant columnar mode).

Single shooting over T is exposed to A(T); multiple shooting with m segments
caps it to A(T/m) << A(T). We measure A(t) from forward runs that share the
*identical* OU forcing realisation (fixed timestep + same seed) so the only
difference is the IC perturbation.

Procedure: spin up forced+rotating turbulence to a developed snapshot x0*, then
propagate x0* and x0*+eps*d under a common measurement forcing, averaging A(t)
over a few random columnar directions d. A log-linear fit gives a Lyapunov-like
rate and tau_L.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from astronomix import SimulationConfig, SimulationParams, get_registered_variables
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, ISOTHERMAL, NATIVE_JAX, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, SnapshotSettings, finalize_config,
)
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astronomix.time_stepping.time_integration import time_integration

N = 48
L = 1.0
CS = 1.0
NU = 1.0e-3
OMEGA = 4.0
KF_MODE = 3.0
TAU_F = 1.0
F0 = 1.0
K_CUT = 2.0
EPS = 1e-5
N_DIRS = 4
T_SPIN = 6.0
T_MEAS = 26.0              # ~6 tau_L (tau_L ~ 4.3) for a dramatic SS/MS gap
DT = 5.0e-3                 # fixed measurement timestep (CFL-stable at 48^3)
NSNAP = 130
MEAS_SEED = 12345
M_SHOW = [2, 4, 8, 16]

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def make_lowpass2d(nx, ny, kcut):
    kx = jnp.fft.fftfreq(nx) * nx
    ky = jnp.fft.fftfreq(ny) * ny
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    mask = (jnp.sqrt(KX ** 2 + KY ** 2) <= kcut).astype(jnp.float64)
    return lambda f: jnp.fft.ifftn(jnp.fft.fftn(f) * mask).real


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base = dict(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=ISOTHERMAL, mhd=False,
        dimensionality=3, num_cells=N, box_size=L, backend=NATIVE_JAX,
        enforce_positivity=False, progress_bar=False, diffusion=True, rotation=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True, ou_forcing=True),
    )
    fparams = TurbulentForcingParams(
        correlation_time=TAU_F, forcing_wavenumber=2.0 * np.pi * KF_MODE / L,
        forcing_amplitude=F0)
    rv = get_registered_variables(SimulationConfig(**base))
    lowpass = make_lowpass2d(N, N, K_CUT)
    vxi, vyi, vzi = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z
    tau_eddy = (L / 2) / 0.4

    # ---- spin-up to a developed snapshot x0* ----
    print(f"=== rotating-turbulence sensitivity mechanism {N}^3, "
          f"Omega={OMEGA}, T_spin={T_SPIN}, T_meas={T_MEAS} ===", flush=True)
    cfg_spin = SimulationConfig(random_seed=7, **base)
    p_spin = SimulationParams(isothermal_sound_speed=CS, t_end=T_SPIN, viscosity=NU,
                              rotation_rate=OMEGA, turbulent_forcing_params=fparams)
    d0 = jnp.ones((N, N, N)); z0 = jnp.zeros_like(d0)
    s0 = construct_primitive_state(config=cfg_spin, registered_variables=rv,
                                   density=d0, velocity_x=z0, velocity_y=z0, velocity_z=z0)
    cfg_spin = finalize_config(cfg_spin, s0.shape)
    x0 = time_integration(s0, cfg_spin, p_spin, rv)
    x0 = jax.block_until_ready(x0)
    urms = float(jnp.sqrt(jnp.mean(x0[vxi] ** 2 + x0[vyi] ** 2 + x0[vzi] ** 2)))
    print(f"spun up: u_rms = {urms:.3f}, Ma = {urms/CS:.3f}", flush=True)

    # ---- common measurement config: fixed dt + fixed seed -> identical forcing ----
    nsteps = int(round(T_MEAS / DT))
    cfg_meas = SimulationConfig(
        random_seed=MEAS_SEED, fixed_timestep=True, num_timesteps=nsteps,
        return_snapshots=True, num_snapshots=NSNAP,
        snapshot_settings=SnapshotSettings(return_states=True), **base)
    cfg_meas = finalize_config(cfg_meas, x0.shape)
    p_meas = SimulationParams(isothermal_sound_speed=CS, t_end=T_MEAS, viscosity=NU,
                              rotation_rate=OMEGA, turbulent_forcing_params=fparams)

    def p_large(states):  # states: (n_snap, nvars, N,N,N)
        vbx = states[:, vxi].mean(axis=3)        # vertical average -> (n_snap,N,N)
        vby = states[:, vyi].mean(axis=3)
        return np.stack([np.asarray(lowpass(jnp.asarray(vbx[i])))
                         for i in range(vbx.shape[0])]), \
               np.stack([np.asarray(lowpass(jnp.asarray(vby[i])))
                         for i in range(vby.shape[0])])

    truth = time_integration(x0, cfg_meas, p_meas, rv)
    tp = np.asarray(truth.time_points)
    base_states = np.asarray(truth.states)
    base_lx, base_ly = p_large(truth.states)

    rng = np.random.default_rng(0)
    full, large = [], []
    for d in range(N_DIRS):
        dx = np.asarray(lowpass(jnp.asarray(rng.standard_normal((N, N)))))
        dy = np.asarray(lowpass(jnp.asarray(rng.standard_normal((N, N)))))
        nrm = np.sqrt(np.mean(dx ** 2 + dy ** 2))
        dx, dy = dx / nrm, dy / nrm
        xp = x0.at[vxi].add(EPS * jnp.asarray(dx)[:, :, None])
        xp = xp.at[vyi].add(EPS * jnp.asarray(dy)[:, :, None])
        pert = time_integration(xp, cfg_meas, p_meas, rv)
        ps = np.asarray(pert.states)
        # full-state tangent (the proxy the single-shooting *adjoint* mirrors)
        A_full = np.sqrt(((ps - base_states) ** 2).sum(axis=(1, 2, 3, 4))) / EPS
        # barotropic-columnar observable (the slow target itself)
        plx, ply = p_large(pert.states)
        A_large = np.sqrt(((plx - base_lx) ** 2 + (ply - base_ly) ** 2).sum(axis=(1, 2))) / EPS
        full.append(A_full); large.append(A_large)
        print(f"  dir {d}: A_full {A_full[0]:.2e}->{A_full[-1]:.2e} "
              f"({A_full[-1]/(A_full[0]+1e-30):.1f}x);  "
              f"A_large {A_large[0]:.2e}->{A_large[-1]:.2e} "
              f"({A_large[-1]/(A_large[0]+1e-30):.1f}x)", flush=True)
    A_full = np.mean(full, axis=0)
    A_large = np.mean(large, axis=0)
    t_te = tp / tau_eddy

    grow = (tp > 0.5) & (A_full < 0.5 * A_full.max() + A_full[0])
    lam = np.polyfit(tp[grow], np.log(A_full[grow]), 1)[0] if grow.sum() >= 3 else \
        np.polyfit(tp[1:], np.log(A_full[1:] + 1e-30), 1)[0]
    tau_L = 1.0 / lam if lam > 0 else np.inf
    print(f"\nLyapunov rate lambda={lam:.3f}/t -> tau_L={tau_L:.2f} "
          f"= {tau_L/tau_eddy:.2f} eddy times (from A_full)", flush=True)

    def A_at(tt):
        return float(np.interp(tt, tp, A_full))
    T = tp[-1]
    print(f"single shooting gradient-amplification A_full(T={T/tau_eddy:.1f} t_e) "
          f"= {A_at(T):.3e}")
    for m in M_SHOW:
        print(f"  MS m={m}: A_full(T/{m}) = {A_at(T/m):.3e}  "
              f"(SS/MS gap = {A_at(T)/(A_at(T/m)+1e-30):.1f}x)")

    np.savez(DATA_DIR / f"rot_mechanism_N{N}.npz", t=tp, A_full=A_full,
             A_large=A_large, lam=lam, tau_L=tau_L, tau_eddy=tau_eddy)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.semilogy(t_te, A_full, "-o", ms=3, color="C0",
                label=r"$A_{full}(t)$: full-state tangent (SS adjoint blow-up)")
    ax.semilogy(t_te, A_large, "-s", ms=3, color="C2",
                label=r"$A_{large}(t)$: barotropic-columnar target (slow manifold)")
    if lam > 0 and np.isfinite(tau_L):
        ax.semilogy(t_te, A_full[0] * np.exp(lam * tp), "k--", alpha=0.5,
                    label=fr"$\sim e^{{\lambda t}},\ \tau_L={tau_L/tau_eddy:.1f}\,t_e$")
    T_te = t_te[-1]
    ax.scatter([T_te], [A_at(T)], color="C3", s=70, zorder=5,
               label=f"single shooting: $A_{{full}}(T)$ = {A_at(T):.1e}")
    for m, c in zip(M_SHOW, ("C1", "C4", "C5", "C6", "C7", "C8")):
        ax.scatter([T_te / m], [A_at(T / m)], color=c, s=55, zorder=5,
                   label=f"MS m={m}: $A_{{full}}(T/{m})$ = {A_at(T/m):.1e}")
    ax.set(xlabel=r"window $t$ / eddy time", ylabel="sensitivity amplification",
           title="rotating turbulence: columnar target is predictable, but the\n"
                 "single-shooting gradient (full tangent) blows up — MS caps it")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"rot_mechanism_N{N}.png", dpi=170)
    plt.close(fig)
    print(f"\nFigure -> rot_mechanism_N{N}.png")


if __name__ == "__main__":
    main()
