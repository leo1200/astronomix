"""
SS-vs-MS mechanism plot for the 2D RBC (Phase 6 / D2), the cheap way.

The single-shooting gradient through a window of length T is the adjoint
propagated backward across the whole window; its magnitude tracks the *forward*
sensitivity of the (large-scale) observation to a (large-scale) IC perturbation,

    A(t) = || P_lk(x(t; ctrl+dctrl)) - P_lk(x(t; ctrl)) ||  /  ||dctrl|| ,

which in a chaotic flow grows ~ e^{lambda t}.  Single shooting over a window T
is exposed to A(T); multiple shooting with m segments caps the back-prop to one
segment of length h = T/m, i.e. to A(T/m) <<  A(T).  That gap is the whole
mechanism -- and A(t) is measured from just two *forward* runs (Pallas-fast),
so it sidesteps the cost of differentiating through ~1500 low-Mach steps.

We perturb the truth large-scale IC by a few small random low-pass directions,
propagate truth and perturbed states with snapshots, and average the
amplification.  A log-linear fit of A(t) gives a Lyapunov-like rate lambda and
tau_L = 1/lambda (a Phase-2 by-product).  SS vs MS amplifications are read off
at T and T/m.
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

from astronomix import (
    SimulationConfig, SimulationParams,
    get_registered_variables, get_helper_data,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, IDEAL_GAS, PALLAS,
    REFLECTIVE_BOUNDARY, SIMPLE_SOURCE_TERM,
    BoundarySettings, BoundarySettings1D, SnapshotSettings,
    StaticFloatVector, StaticIntVector, finalize_config,
)
from astronomix.variable_registry.registered_variables import get_registered_variables as _grv
from astronomix.time_stepping.time_integration import time_integration

NY = 48
GAMMA_ASPECT = 2
LY = 1.0
LX = GAMMA_ASPECT * LY
GAMMA = 5.0 / 3.0
G = 0.25
T_BOT, T_TOP, P_BOT = 1.2, 0.8, 1.0
MU = 1.4e-3
KAPPA = 1.9e-3
K_CUT = 4.0
C_CFL = 0.4
EPS = 1e-4               # perturbation amplitude (tangent regime)
N_DIRS = 4
T_END_TAU = 8.0
NUM_SNAP = 120
M_SHOW = [2, 4, 8, 16]
SEED = 0

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def hydrostatic_background(y):
    dT = T_BOT - T_TOP
    T = T_BOT - (dT / LY) * y
    p = P_BOT * (T / T_BOT) ** (G * LY / dT)
    return T, p, p / T


def make_lowpass(nx, ny, kcut):
    kx = jnp.fft.fftfreq(nx) * nx
    ky = jnp.fft.fftfreq(ny) * ny
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    mask = (jnp.sqrt(KX ** 2 + KY ** 2) <= kcut).astype(jnp.float64)
    return lambda f: jnp.fft.ifftn(jnp.fft.fftn(f) * mask).real


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nx = GAMMA_ASPECT * NY
    v_ff = float(np.sqrt(G * (T_BOT - T_TOP) / (0.5 * (T_BOT + T_TOP)) * LY))
    tau_ff = LY / v_ff
    t_end = T_END_TAU * tau_ff

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS, dimensionality=2,
        backend=PALLAS, pallas_block_shape=(4, 4, 4), pallas_use_triton=True,
        pallas_interpret=False, progress_bar=False,
        num_cells=StaticIntVector(nx, NY, -1), box_size=StaticFloatVector(LX, LY, 1.0),
        external_potential=True, self_gravity_version=SIMPLE_SOURCE_TERM,
        diffusion=True, thermal_conduction=True,
        conduction_wall_axis=1, conduction_isothermal_walls=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
            BoundarySettings1D()),
        return_snapshots=True, num_snapshots=NUM_SNAP,
        snapshot_settings=SnapshotSettings(return_states=True),
    )
    rv = get_registered_variables(config)
    hd = get_helper_data(config)
    coords = hd.geometric_centers
    x, y = coords[..., 0], coords[..., 1]
    T_bg, p_bg, rho_bg = hydrostatic_background(y)
    probe = construct_primitive_state(
        config=config, registered_variables=rv, density=rho_bg,
        velocity_x=jnp.zeros_like(x), velocity_y=jnp.zeros_like(x), gas_pressure=p_bg)
    config = finalize_config(config, probe.shape)
    params = SimulationParams(
        C_cfl=C_CFL, gamma=GAMMA, t_end=t_end, viscosity=MU, thermal_conductivity=KAPPA,
        wall_temperature_low=T_BOT, wall_temperature_high=T_TOP, gravitational_potential=G * y)

    lowpass = make_lowpass(nx, NY, K_CUT)
    T_mean = 0.5 * (T_BOT + T_TOP)
    pidx, didx = rv.pressure_index, rv.density_index

    def build_ic(ctrl):
        return construct_primitive_state(
            config=config, registered_variables=rv, density=rho_bg,
            velocity_x=jnp.zeros_like(x), velocity_y=lowpass(ctrl), gas_pressure=p_bg)

    def run(ctrl):
        res = time_integration(build_ic(ctrl), config, params, rv)
        return np.asarray(res.states), np.asarray(res.time_points)

    rng = np.random.default_rng(SEED)
    truth_ctrl = jnp.asarray(lowpass(jnp.asarray(0.3 * v_ff * rng.standard_normal((nx, NY)))))
    base_states, tp = run(truth_ctrl)

    def obs_lk(states):  # large-scale temperature anomaly per snapshot
        T = states[:, pidx] / states[:, didx]
        return np.stack([np.asarray(lowpass(jnp.asarray(T[i] - T_mean))) for i in range(T.shape[0])])

    base_obs = obs_lk(base_states)

    print(f"=== adjoint/sensitivity mechanism, 2D RBC {nx}x{NY}, "
          f"tau_ff={tau_ff:.3f}, T={T_END_TAU} tau_ff ===", flush=True)
    amps = []
    for d in range(N_DIRS):
        dctrl = np.asarray(lowpass(jnp.asarray(rng.standard_normal((nx, NY)))))
        dctrl = dctrl / np.linalg.norm(dctrl)               # unit large-scale dir
        pert_states, _ = run(truth_ctrl + EPS * jnp.asarray(dctrl))
        pert_obs = obs_lk(pert_states)
        A = np.sqrt(((pert_obs - base_obs) ** 2).sum(axis=(1, 2))) / EPS   # A(t)
        amps.append(A)
        print(f"  dir {d}: A(0)={A[0]:.3e}, A(T)={A[-1]:.3e}, growth={A[-1]/(A[1]+1e-30):.1f}x",
              flush=True)
    A = np.mean(amps, axis=0)
    t_tau = tp / tau_ff

    # exponential fit over the growing (pre-saturation) range
    growing = (t_tau > 0.3) & (A > A[1]) & (A < 0.5 * A.max() + A[1])
    if growing.sum() >= 3:
        lam = np.polyfit(tp[growing], np.log(A[growing]), 1)[0]
    else:
        lam = np.polyfit(tp[t_tau > 0.2], np.log(A[t_tau > 0.2] + 1e-30), 1)[0]
    tau_L = 1.0 / lam if lam > 0 else np.inf
    print(f"\nLyapunov-like rate lambda = {lam:.3f} /t  ->  tau_L = {tau_L:.2f} "
          f"= {tau_L/tau_ff:.2f} tau_ff", flush=True)

    # SS vs MS amplification at T and T/m
    def A_at(t_target):
        return float(np.interp(t_target, tp, A))
    T = tp[-1]
    print(f"\nsingle shooting amplification A(T={T_END_TAU} tau_ff) = {A_at(T):.3e}")
    for m in M_SHOW:
        print(f"  multiple shooting m={m}: per-segment A(T/m={T_END_TAU/m:.2f} tau_ff) "
              f"= {A_at(T/m):.3e}   (SS/MS gap = {A_at(T)/ (A_at(T/m)+1e-30):.1f}x)")

    np.savez(DATA_DIR / f"rbc_mechanism_N{NY}.npz",
             t=tp, t_tau=t_tau, A=A, lam=lam, tau_L=tau_L, tau_ff=tau_ff)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(t_tau, A, "-o", ms=3, color="C0",
                label=r"$A(t)=\|\delta\,P_{lk}\,T(t)\|/\|\delta\mathrm{IC}\|$")
    if lam > 0 and np.isfinite(tau_L):
        ax.semilogy(t_tau, A[1] * np.exp(lam * (tp - tp[1])), "k--", alpha=0.6,
                    label=fr"$\sim e^{{\lambda t}},\ \tau_L={tau_L/tau_ff:.1f}\,\tau_{{ff}}$")
    T_tau = t_tau[-1]
    ax.scatter([T_tau], [A_at(T)], color="C3", zorder=5, s=70,
               label=f"single shooting: $A(T)$ = {A_at(T):.1e}")
    for m, c in zip(M_SHOW, ("C2", "C1", "C4")):
        ax.scatter([T_tau / m], [A_at(T / m)], color=c, zorder=5, s=55,
                   label=f"MS m={m}: $A(T/{m})$ = {A_at(T/m):.1e}")
    ax.set_xlabel(r"window length $t\ /\ \tau_{ff}$")
    ax.set_ylabel("large-scale sensitivity amplification $A$")
    ax.set_title("SS vs MS mechanism: gradient amplification grows with the window\n"
                 "(single shooting sees A(T); multiple shooting caps it at A(T/m))")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rbc_mechanism_N{NY}.png", dpi=180)
    plt.close(fig)
    print(f"\nFigure -> rbc_mechanism_N{NY}.png")


if __name__ == "__main__":
    main()
