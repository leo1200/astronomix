"""
Single shooting vs multiple shooting recovery of a large-scale 2D
Rayleigh-Benard initial condition from a filtered terminal observation.

This is the RBC analogue of ``tests/taylor_green/inverse/ss_vs_ms_recovery.py``
(worked example II of ``init_optim_theory.md``).  The differentiable forward
model is the full 2D RBC path built in Phase 0a/0b: FD WENO + reflective walls
+ external-potential gravity + viscosity + thermal conduction (isothermal
plates).

Control = the large-scale band (spectral low-pass, |k| <= K_CUT) of the initial
vertical-velocity perturbation; small scales pinned to the (zero-perturbation)
background, so the recoverable subspace is identical for every method and the
SS-vs-MS difference is the gradient/basin story, not identifiability.

Step-defect loss over m segments of length h = T_obs/m, segment-start states
s_0..s_{m-1} (s_0 = build_ic(ctrl)):

    L = || P_lk( F_h(s_{m-1}) ) - obs ||^2  +  (mu/2) mean_j || F_h(s_j) - s_{j+1} ||^2

m = 1 is exactly single shooting (full back-prop through T_obs); m > 1 caps the
back-prop to one segment.  We sweep the window T_obs (in free-fall times) and
compare large-scale IC recovery; SS is expected to degrade as the window grows
past the predictability horizon while MS holds.

Knobs via env: RBC_N, RBC_TOBS (tau_ff), RBC_M ("1,2,4"), RBC_STEPS.
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

from astronomix import (
    SimulationConfig, SimulationParams,
    get_registered_variables, get_helper_data,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, IDEAL_GAS, NATIVE_JAX,
    REFLECTIVE_BOUNDARY, SIMPLE_SOURCE_TERM,
    BoundarySettings, BoundarySettings1D, StaticFloatVector, StaticIntVector,
    finalize_config,
)
from astronomix.time_stepping.time_integration import _time_integration
from astronomix.data_classes.simulation_helper_data import (
    _helper_data_requirements, get_helper_data as _get_helper_data,
)

# ---- problem ----
NY = int(os.environ.get("RBC_N", 32))
GAMMA_ASPECT = 2
LY = 1.0
LX = GAMMA_ASPECT * LY
GAMMA = 5.0 / 3.0
G = 0.25
T_BOT, T_TOP, P_BOT = 1.2, 0.8, 1.0
MU = 1.4e-3                 # Ra ~ 5e4 effective, Pr ~ 1
KAPPA = 1.9e-3
K_CUT = 4.0
C_CFL = 0.4
NUM_CHECKPOINTS = 24

T_OBS_TAU = float(os.environ.get("RBC_TOBS", 2.0))   # window in free-fall times
M_LIST = [int(m) for m in os.environ.get("RBC_M", "1,2,4").split(",")]
NUM_STEPS = int(os.environ.get("RBC_STEPS", 50))
LEARNING_RATE = float(os.environ.get("RBC_LR", 1e-2))
MU_DEFECT = 10.0
TRUTH_AMP = 0.3            # truth IC amplitude as a fraction of v_ff
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

    def lowpass(field):
        return jnp.fft.ifftn(jnp.fft.fftn(field) * mask).real
    return lowpass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nx = GAMMA_ASPECT * NY

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS,
        dimensionality=2, backend=NATIVE_JAX,
        differentiation_mode=BACKWARDS, num_checkpoints=NUM_CHECKPOINTS,
        enforce_positivity=False, progress_bar=False,
        num_cells=StaticIntVector(nx, NY, -1),
        box_size=StaticFloatVector(LX, LY, 1.0),
        external_potential=True, self_gravity_version=SIMPLE_SOURCE_TERM,
        diffusion=True, thermal_conduction=True,
        conduction_wall_axis=1, conduction_isothermal_walls=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
            BoundarySettings1D(),
        ),
        return_snapshots=False,
    )
    rv = get_registered_variables(config)
    helper_data = get_helper_data(config)
    coords = helper_data.geometric_centers
    x, y = coords[..., 0], coords[..., 1]
    T_bg, p_bg, rho_bg = hydrostatic_background(y)
    probe = construct_primitive_state(
        config=config, registered_variables=rv, density=rho_bg,
        velocity_x=jnp.zeros_like(x), velocity_y=jnp.zeros_like(x),
        gas_pressure=p_bg,
    )
    config = finalize_config(config, probe.shape)
    requirements = _helper_data_requirements(config)
    hd_pad = _get_helper_data(config, None, padded=False, requirements=requirements)

    base_params = SimulationParams(
        C_cfl=C_CFL, gamma=GAMMA, t_end=1.0, viscosity=MU,
        thermal_conductivity=KAPPA, wall_temperature_low=T_BOT,
        wall_temperature_high=T_TOP, gravitational_potential=G * y,
    )

    lowpass = make_lowpass(nx, NY, K_CUT)
    T_mean = 0.5 * (T_BOT + T_TOP)
    v_ff = float(np.sqrt(G * (T_BOT - T_TOP) / T_mean * LY))
    tau_ff = LY / v_ff
    T_OBS = T_OBS_TAU * tau_ff
    vyi = rv.velocity_index.y
    pidx, didx = rv.pressure_index, rv.density_index

    def build_ic(ctrl):
        vy = lowpass(ctrl)
        return construct_primitive_state(
            config=config, registered_variables=rv, density=rho_bg,
            velocity_x=jnp.zeros_like(x), velocity_y=vy, gas_pressure=p_bg,
        )

    def propagate(state, h):
        return _time_integration(state, config, base_params._replace(t_end=h), rv, hd_pad)

    def obs_op(state):
        # large-scale terminal temperature anomaly = observation operator P_lk
        T = state[pidx] / state[didx]
        return lowpass(T - T_mean)

    # ---- truth: a large-scale vy seed ----
    rng = np.random.default_rng(SEED)
    truth_ctrl = jnp.asarray(lowpass(jnp.asarray(TRUTH_AMP * v_ff * rng.standard_normal((nx, NY)))))
    truth_ic = build_ic(truth_ctrl)
    truth_final = propagate(truth_ic, T_OBS)
    obs = obs_op(truth_final)
    obs_energy = float(jnp.mean(obs ** 2)) + 1e-30   # normalise the data term -> O(1) loss
    truth_lk = lowpass(truth_ctrl)
    truth_lk_norm = float(jnp.sqrt(jnp.sum(truth_lk ** 2)))

    def recovery_error(ctrl):
        rec = np.asarray(lowpass(ctrl))
        return float(np.sqrt(np.sum((rec - np.asarray(truth_lk)) ** 2)) / truth_lk_norm)

    print(f"=== SS vs MS, 2D RBC {nx}x{NY}, T_obs = {T_OBS_TAU} tau_ff "
          f"({T_OBS:.2f}), K_cut={K_CUT} ===", flush=True)

    results = {}
    for m in M_LIST:
        h = T_OBS / m
        print(f"\n--- m = {m} (segment length {h:.3f} = {h/tau_ff:.2f} tau_ff) ---",
              flush=True)
        ctrl0 = jnp.zeros((nx, NY))
        s0 = build_ic(ctrl0)
        seg = []
        s = s0
        for _ in range(m - 1):
            s = propagate(s, h)
            seg.append(s)
        seg0 = jnp.stack(seg) if seg else jnp.zeros((0, *s0.shape))
        theta = {"ctrl": ctrl0, "seg": seg0}

        def loss(theta):
            s0 = build_ic(theta["ctrl"])
            starts = jnp.concatenate([s0[None], theta["seg"]], axis=0) if m > 1 else s0[None]
            finals = jnp.stack([propagate(starts[j], h) for j in range(m)])
            data = jnp.mean((obs_op(finals[-1]) - obs) ** 2) / obs_energy
            if m > 1:
                defect = jnp.mean((finals[:-1] - starts[1:]) ** 2)
            else:
                defect = 0.0
            return data + 0.5 * MU_DEFECT * defect, (data, defect)

        vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
        opt = optax.adam(LEARNING_RATE)
        opt_state = opt.init(theta)
        hist = {"loss": [], "data": [], "defect": [], "err": [], "gnorm": []}
        t0 = time.time()
        for step in range(NUM_STEPS):
            (lval, (data, defect)), grads = vg(theta)
            gnorm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads))))
            updates, opt_state = opt.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            err = recovery_error(theta["ctrl"])
            for k, v in zip(("loss", "data", "defect", "err", "gnorm"),
                            (float(lval), float(data), float(defect), err, gnorm)):
                hist[k].append(v)
            if step % 5 == 0 or step == NUM_STEPS - 1:
                print(f"  step {step:>3}: loss={float(lval):.3e} data={float(data):.3e} "
                      f"defect={float(defect):.3e} |grad|={gnorm:.3e} rec_err={err:.3f}",
                      flush=True)
        print(f"  m={m} done in {time.time()-t0:.1f}s, final rec_err = {hist['err'][-1]:.3f}",
              flush=True)
        results[m] = {"hist": hist, "ctrl": np.asarray(theta["ctrl"])}

    np.savez(
        DATA_DIR / f"rbc_ss_vs_ms_Tobs{T_OBS_TAU}.npz",
        tau_ff=tau_ff, T_obs_tau=T_OBS_TAU, truth_lk=np.asarray(truth_lk),
        **{f"err_m{m}": np.array(results[m]["hist"]["err"]) for m in results},
        **{f"loss_m{m}": np.array(results[m]["hist"]["loss"]) for m in results},
        **{f"gnorm_m{m}": np.array(results[m]["hist"]["gnorm"]) for m in results},
        **{f"ctrl_m{m}": results[m]["ctrl"] for m in results},
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for m in results:
        axes[0].semilogy(results[m]["hist"]["loss"], label=f"m={m}")
        axes[1].plot(results[m]["hist"]["err"], label=f"m={m}")
        axes[2].semilogy(results[m]["hist"]["gnorm"], label=f"m={m}")
    axes[0].set(xlabel="Adam step", ylabel="loss",
                title=f"Convergence (T_obs={T_OBS_TAU} tau_ff)")
    axes[1].set(xlabel="Adam step", ylabel="large-scale IC recovery error",
                title="Recovery error (lower=better)")
    axes[2].set(xlabel="Adam step", ylabel="|grad|",
                title="Gradient norm (SS blow-up vs MS)")
    for a in axes:
        a.legend(); a.grid(alpha=0.3, which="both")
    fig.suptitle("Single shooting (m=1) vs multiple shooting (m>1) on 2D RBC")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rbc_ss_vs_ms_Tobs{T_OBS_TAU}.png", dpi=180)
    plt.close(fig)
    print(f"\nFigure -> rbc_ss_vs_ms_Tobs{T_OBS_TAU}.png")
    for m in results:
        print(f"  m={m}: final recovery error = {results[m]['hist']['err'][-1]:.3f}")


if __name__ == "__main__":
    main()
