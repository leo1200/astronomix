"""
Single shooting vs multiple shooting recovery of the barotropic-columnar
initial condition of rotating turbulence (the slow manifold), now enabled by the
checkpointed fixed-timestep loop.

Twin experiment.  Spin up forced+rotating turbulence to a developed snapshot
x0* (columns + 3D fluctuations).  The *control* is the barotropic-columnar
velocity (z-invariant, |k_perp| <= k_c); the 3D small scales are pinned to x0*.
build_ic(ctrl) replaces x0*'s columnar part with ctrl.  Recovery runs in
**decaying** rotating turbulence (forcing off) so the OU realisation needn't be
replayed across multiple-shooting segments -- the chaotic propagator and the
slow columnar manifold are both present, which is all the SS-vs-MS argument
needs.

Step-defect loss over m segments of length h = T/m, segment starts s_0..s_{m-1}
(s_0 = build_ic(ctrl)):

    L = || P_large(F_h(s_{m-1})) - target ||^2  +  (mu/2) mean_j ||F_h(s_j) - s_{j+1}||^2

P_large = horizontal low-pass of the vertical average of the horizontal
velocity.  m=1 is single shooting (full back-prop through T); m>1 caps the
back-prop to one segment.  We track the gradient norm (SS blow-up) and the
columnar recovery error.

Env knobs: RT_N, RT_TWIN (window in time units), RT_M ("1,2,4"), RT_STEPS,
RT_NSEG (fixed steps per segment).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import os
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from astronomix import SimulationConfig, SimulationParams, get_registered_variables
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, ISOTHERMAL, NATIVE_JAX, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, SnapshotSettings, finalize_config,
)
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astronomix.time_stepping.time_integration import time_integration, _time_integration
from astronomix.data_classes.simulation_helper_data import (
    _helper_data_requirements, get_helper_data as _get_helper_data,
)

N = int(os.environ.get("RT_N", 32))
L = 1.0
CS = 1.0
NU = 1.0e-3
OMEGA = 4.0
KF_MODE = 3.0
K_CUT = 2.0
DT = 8.0e-3                # fixed timestep (CFL-stable at 32^3)
NSEG = int(os.environ.get("RT_NSEG", 40))      # fixed steps per segment
T_TWIN_TAU = float(os.environ.get("RT_TWIN", 2.0))  # window in eddy times
M_LIST = [int(m) for m in os.environ.get("RT_M", "1,2,4").split(",")]
NUM_STEPS = int(os.environ.get("RT_STEPS", 40))
LEARNING_RATE = 1e-2
MU_DEFECT = 10.0
NUM_CHECKPOINTS = int(os.environ.get("RT_CKPT", 16))  # more ckpts -> less backward recompute
SEED = 0
T_SPIN = 6.0

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
    tau_eddy = (L / 2) / 0.4
    nseg_per_tau = int(round(tau_eddy / DT))
    T = T_TWIN_TAU * tau_eddy
    nsteps_T = int(round(T / DT))

    base = dict(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=ISOTHERMAL, mhd=False,
        dimensionality=3, num_cells=N, box_size=L, backend=NATIVE_JAX,
        enforce_positivity=False, progress_bar=False, diffusion=True, rotation=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)))
    rv = get_registered_variables(SimulationConfig(**base))
    lowpass = make_lowpass2d(N, N, K_CUT)
    vxi, vyi, vzi = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z

    # ---- spin up (forced) to a developed snapshot x0* ----
    print(f"=== rotating SS-vs-MS recovery {N}^3, T={T_TWIN_TAU} t_e ({nsteps_T} steps), "
          f"M={M_LIST} ===", flush=True)
    cfg_spin = SimulationConfig(
        random_seed=7,
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True, ou_forcing=True), **base)
    p_spin = SimulationParams(
        isothermal_sound_speed=CS, t_end=T_SPIN, viscosity=NU, rotation_rate=OMEGA,
        turbulent_forcing_params=TurbulentForcingParams(
            correlation_time=1.0, forcing_wavenumber=2 * np.pi * KF_MODE / L,
            forcing_amplitude=1.0))
    d0 = jnp.ones((N, N, N)); z0 = jnp.zeros_like(d0)
    s0 = construct_primitive_state(config=cfg_spin, registered_variables=rv,
                                   density=d0, velocity_x=z0, velocity_y=z0, velocity_z=z0)
    cfg_spin = finalize_config(cfg_spin, s0.shape)
    x0_star = jax.block_until_ready(time_integration(s0, cfg_spin, p_spin, rv))
    urms = float(jnp.sqrt(jnp.mean(x0_star[vxi]**2 + x0_star[vyi]**2 + x0_star[vzi]**2)))
    print(f"spun up x0*: u_rms={urms:.3f}, Ma={urms/CS:.3f}", flush=True)

    # decaying (no forcing) differentiable config, fixed timestep + checkpointing
    def make_cfg(nsteps):
        cfg = SimulationConfig(
            differentiation_mode=BACKWARDS, fixed_timestep=True, num_timesteps=nsteps,
            num_checkpoints=min(NUM_CHECKPOINTS, nsteps), return_snapshots=False,
            turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=False),
            **base)
        return finalize_config(cfg, x0_star.shape)
    requirements = _helper_data_requirements(make_cfg(NSEG))
    hd_pad = _get_helper_data(make_cfg(NSEG), None, padded=False, requirements=requirements)

    def params_for(nsteps):
        return SimulationParams(isothermal_sound_speed=CS, t_end=nsteps * DT,
                                viscosity=NU, rotation_rate=OMEGA)

    def columnar(v):  # barotropic columnar velocity (2, N, N)
        return jnp.stack([lowpass(v[vxi].mean(axis=2)), lowpass(v[vyi].mean(axis=2))])

    col_star = columnar(x0_star)
    # background = x0* with its columnar part removed
    bg = x0_star
    bg = bg.at[vxi].add(-jnp.broadcast_to(col_star[0][:, :, None], (N, N, N)))
    bg = bg.at[vyi].add(-jnp.broadcast_to(col_star[1][:, :, None], (N, N, N)))

    def build_ic(ctrl):
        c = jnp.stack([lowpass(ctrl[0]), lowpass(ctrl[1])])
        s = bg.at[vxi].add(jnp.broadcast_to(c[0][:, :, None], (N, N, N)))
        s = s.at[vyi].add(jnp.broadcast_to(c[1][:, :, None], (N, N, N)))
        return s

    def obs(state):
        return jnp.stack([lowpass(state[vxi].mean(axis=2)), lowpass(state[vyi].mean(axis=2))])

    # ---- target from the truth (control = columnar(x0*)) ----
    cfg_T = make_cfg(nsteps_T)
    truth_final = _time_integration(build_ic(col_star), cfg_T, params_for(nsteps_T), rv, hd_pad)
    target = obs(truth_final)
    tgt_energy = float(jnp.mean(target ** 2)) + 1e-30
    truth_lk = jnp.stack([lowpass(col_star[0]), lowpass(col_star[1])])
    truth_lk_norm = float(jnp.sqrt(jnp.sum(truth_lk ** 2)))

    def rec_err(ctrl):
        rec = np.stack([np.asarray(lowpass(jnp.asarray(ctrl[0]))),
                        np.asarray(lowpass(jnp.asarray(ctrl[1])))])
        return float(np.sqrt(np.sum((rec - np.asarray(truth_lk)) ** 2)) / truth_lk_norm)

    results = {}
    for m in M_LIST:
        nseg = max(nsteps_T // m, 1)
        cfg_seg = make_cfg(nseg)
        p_seg = params_for(nseg)
        print(f"\n--- m={m}: {m} segments of {nseg} steps ({nseg*DT/tau_eddy:.2f} t_e) ---",
              flush=True)

        def propagate(state):
            return _time_integration(state, cfg_seg, p_seg, rv, hd_pad)

        ctrl0 = jnp.zeros((2, N, N))
        s0c = build_ic(ctrl0)
        seg = []
        s = s0c
        for _ in range(m - 1):
            s = propagate(s); seg.append(s)
        seg0 = jnp.stack(seg) if seg else jnp.zeros((0, *s0c.shape))
        theta = {"ctrl": ctrl0, "seg": seg0}

        def loss(theta):
            s0c = build_ic(theta["ctrl"])
            starts = jnp.concatenate([s0c[None], theta["seg"]], axis=0) if m > 1 else s0c[None]
            finals = jnp.stack([propagate(starts[j]) for j in range(m)])
            data = jnp.mean((obs(finals[-1]) - target) ** 2) / tgt_energy
            defect = jnp.mean((finals[:-1] - starts[1:]) ** 2) if m > 1 else 0.0
            return data + 0.5 * MU_DEFECT * defect, (data, defect)

        vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
        opt = optax.adam(LEARNING_RATE)
        opt_state = opt.init(theta)
        hist = {"loss": [], "data": [], "defect": [], "err": [], "gnorm": []}
        t0 = time.time()
        for step in range(NUM_STEPS):
            (lv, (data, defect)), grads = vg(theta)
            gnorm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads))))
            updates, opt_state = opt.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            e = rec_err(theta["ctrl"])
            for k, v in zip(("loss", "data", "defect", "err", "gnorm"),
                            (float(lv), float(data), float(defect), e, gnorm)):
                hist[k].append(v)
            if step % 5 == 0 or step == NUM_STEPS - 1:
                print(f"  step {step:>3}: loss={float(lv):.3e} data={float(data):.3e} "
                      f"defect={float(defect):.3e} |grad|={gnorm:.3e} rec_err={e:.3f}",
                      flush=True)
        print(f"  m={m} done in {time.time()-t0:.1f}s, final rec_err={hist['err'][-1]:.3f}",
              flush=True)
        results[m] = {"hist": hist, "ctrl": np.asarray(theta["ctrl"])}

    np.savez(DATA_DIR / f"rot_ss_vs_ms_T{T_TWIN_TAU}.npz", tau_eddy=tau_eddy,
             T_tau=T_TWIN_TAU,
             **{f"err_m{m}": np.array(results[m]["hist"]["err"]) for m in results},
             **{f"gnorm_m{m}": np.array(results[m]["hist"]["gnorm"]) for m in results},
             **{f"loss_m{m}": np.array(results[m]["hist"]["loss"]) for m in results})

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for m in results:
        ax[0].semilogy(results[m]["hist"]["loss"], label=f"m={m}")
        ax[1].plot(results[m]["hist"]["err"], label=f"m={m}")
        ax[2].semilogy(results[m]["hist"]["gnorm"], label=f"m={m}")
    ax[0].set(xlabel="Adam step", ylabel="loss", title=f"Convergence (T={T_TWIN_TAU} t_e)")
    ax[1].set(xlabel="Adam step", ylabel="columnar recovery error", title="Recovery (lower=better)")
    ax[2].set(xlabel="Adam step", ylabel="|grad|", title="Gradient norm (SS blow-up vs MS)")
    for a in ax:
        a.legend(); a.grid(alpha=0.3, which="both")
    fig.suptitle("Rotating-turbulence columnar recovery: single (m=1) vs multiple shooting")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"rot_ss_vs_ms_T{T_TWIN_TAU}.png", dpi=170)
    plt.close(fig)
    print(f"\nFigure -> rot_ss_vs_ms_T{T_TWIN_TAU}.png")
    for m in results:
        print(f"  m={m}: final recovery error = {results[m]['hist']['err'][-1]:.3f}")


if __name__ == "__main__":
    main()
