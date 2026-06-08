"""
Single single-vs-multiple-shooting recovery optimization -- ONE job
(one truth seed, one horizon, one m), pinned to one GPU. Designed to be
launched many-at-once by run_money_sweep_parallel.py across the free GPUs
of a node.

Job parameters come from the environment (set by the dispatcher):
    JOB_TOBS        observation horizon in t_c            (float)
    JOB_M           shooting split (1 = single shooting)  (int)
    JOB_TRUTH_SEED  RNG seed for the random low-k truth IC (int)
    JOB_INIT_SEED   RNG seed for the optimization init    (int, optional)
    JOB_INIT_AMP    amplitude of the random init perturbation (float, default 0 = cold start)
    JOB_N           cells per dim                          (int, default 32)
    JOB_STEPS       max Adam steps                         (int, default 150)
    JOB_OUT         output .npz path                       (str)

GPU: if CUDA_VISIBLE_DEVICES is already pinned (by the dispatcher) we use
it directly; otherwise we fall back to autocvd.
"""

import os

# ==== GPU selection ====
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _cvd == "" or _cvd.lower() == "all":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# else: a single GPU was pinned by the dispatcher -- use it as-is
# ruff: noqa: E402
# =======================

import time

import jax
import jax.numpy as jnp
import optax

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_registered_variables,
    get_helper_data,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    finalize_config,
)
from astronomix.time_stepping.time_integration import _time_integration
from astronomix.data_classes.simulation_helper_data import (
    _helper_data_requirements,
    get_helper_data as _get_helper_data,
)

# ---- job parameters ----
T_OBS = float(os.environ["JOB_TOBS"])
M = int(os.environ["JOB_M"])
TRUTH_SEED = int(os.environ.get("JOB_TRUTH_SEED", 0))
INIT_SEED = int(os.environ.get("JOB_INIT_SEED", 1000))
INIT_AMP = float(os.environ.get("JOB_INIT_AMP", 0.0))
N = int(os.environ.get("JOB_N", 32))
MAX_STEPS = int(os.environ.get("JOB_STEPS", 150))
OPT = os.environ.get("JOB_OPT", "lbfgs").lower()   # "lbfgs" (default) or "adam"
JOB_MU = float(os.environ.get("JOB_MU", 10.0))     # consistency-defect penalty
OUT = os.environ["JOB_OUT"]

# ---- fixed physics / numerics ----
BOX = 2.0 * jnp.pi
V0 = 1.0
RHO0 = 1.0
MA0 = 0.3
GAMMA = 5.0 / 3.0
C_CFL = 0.4
K_CUT = float(os.environ.get("JOB_KCUT", 4.0))  # low-k control / obs / truth band
MU = JOB_MU               # consistency-defect penalty (tunable via JOB_MU)
LR = 3e-2                  # Adam learning rate
# store (almost) the whole trajectory so the backward pass barely recomputes
# -- 32^3 states are tiny, so this is the cheap way to speed up native backprop
NUM_CHECKPOINTS = 256
# stopping criteria (shared)
PATIENCE = 25              # plateau patience (iterations)
REL_TOL = 1e-3            # relative-loss-change plateau tolerance
GTOL = 1e-4               # L-BFGS gradient-norm stop


def random_lowk_velocity(seed):
    """Random divergence-free velocity with energy only in |k| <= K_CUT, rms V0."""
    key = jax.random.PRNGKey(seed)
    k1 = jnp.fft.fftfreq(N) * N
    KX, KY, KZ = jnp.meshgrid(k1, k1, k1, indexing="ij")
    kk = KX**2 + KY**2 + KZ**2
    band = ((kk > 0) & (kk <= K_CUT**2)).astype(jnp.float32)
    keys = jax.random.split(key, 6)

    def randc(ka, kb):
        return (jax.random.normal(keys[ka], (N, N, N))
                + 1j * jax.random.normal(keys[kb], (N, N, N))) * band

    fx, fy, fz = randc(0, 1), randc(2, 3), randc(4, 5)
    # project to divergence-free
    ks = jnp.where(kk == 0, 1.0, kk)
    kdotf = KX * fx + KY * fy + KZ * fz
    fx = fx - KX * kdotf / ks
    fy = fy - KY * kdotf / ks
    fz = fz - KZ * kdotf / ks
    vx = jnp.fft.ifftn(fx).real
    vy = jnp.fft.ifftn(fy).real
    vz = jnp.fft.ifftn(fz).real
    v = jnp.stack([vx, vy, vz])
    rms = jnp.sqrt(jnp.mean(vx**2 + vy**2 + vz**2))
    return V0 * v / rms


def main():
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        mhd=False,
        backend=NATIVE_JAX,
        differentiation_mode=BACKWARDS,
        num_checkpoints=NUM_CHECKPOINTS,
        enforce_positivity=False,
        progress_bar=False,
        dimensionality=3,
        num_cells=N,
        box_size=BOX,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        return_snapshots=False,
    )
    params = SimulationParams(C_cfl=C_CFL, gamma=GAMMA, t_end=1.0)
    rv = get_registered_variables(config)
    _ = get_helper_data(config)

    density = RHO0 * jnp.ones((N, N, N))
    p0 = RHO0 * V0**2 / (GAMMA * MA0**2)
    pressure = p0 * jnp.ones_like(density)
    vx_i, vy_i, vz_i = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z

    u_truth = random_lowk_velocity(TRUTH_SEED)
    probe = construct_primitive_state(
        config=config, registered_variables=rv, density=density,
        velocity_x=u_truth[0], velocity_y=u_truth[1], velocity_z=u_truth[2],
        gas_pressure=pressure,
    )
    config = finalize_config(config, probe.shape)
    requirements = _helper_data_requirements(config)
    hd_pad = _get_helper_data(config, None, padded=False, requirements=requirements)

    # low-pass mask (control band + observation operator)
    k1 = jnp.fft.fftfreq(N) * N
    KX, KY, KZ = jnp.meshgrid(k1, k1, k1, indexing="ij")
    lk_mask = (jnp.sqrt(KX**2 + KY**2 + KZ**2) <= K_CUT).astype(jnp.float32)

    def lowpass(field):
        return jnp.fft.ifftn(
            jnp.fft.fftn(field, axes=(-3, -2, -1)) * lk_mask, axes=(-3, -2, -1)
        ).real

    def build_ic(ctrl):
        v = lowpass(ctrl)
        return construct_primitive_state(
            config=config, registered_variables=rv, density=density,
            velocity_x=v[0], velocity_y=v[1], velocity_z=v[2], gas_pressure=pressure,
        )

    def propagate(state, h):
        return _time_integration(state, config, params._replace(t_end=h), rv, hd_pad)

    h = T_OBS / M
    truth_ic = build_ic(u_truth)
    truth_final = propagate(truth_ic, T_OBS)
    obs = lowpass(jnp.stack([truth_final[vx_i], truth_final[vy_i], truth_final[vz_i]]))
    truth_lk = lowpass(u_truth)
    truth_lk_norm = float(jnp.sqrt(jnp.sum(truth_lk**2)))

    def recovery_error(ctrl):
        rec = lowpass(ctrl)
        return float(jnp.sqrt(jnp.sum((rec - truth_lk) ** 2)) / truth_lk_norm)

    # init: cold start (optionally + random low-k perturbation for an init ensemble)
    ctrl0 = jnp.zeros_like(u_truth)
    if INIT_AMP > 0:
        ctrl0 = INIT_AMP * random_lowk_velocity(INIT_SEED)
    s0 = build_ic(ctrl0)
    seg_states = []
    s = s0
    for _ in range(M - 1):
        s = propagate(s, h)
        seg_states.append(s)
    seg0 = jnp.stack(seg_states) if seg_states else jnp.zeros((0, *s0.shape))
    theta = {"ctrl": ctrl0, "seg": seg0}

    def loss_parts(theta):
        s0 = build_ic(theta["ctrl"])
        if M > 1:
            starts = jnp.concatenate([s0[None], theta["seg"]], axis=0)
        else:
            starts = s0[None]
        finals = jnp.stack([propagate(starts[j], h) for j in range(M)])
        vf = jnp.stack([finals[-1][vx_i], finals[-1][vy_i], finals[-1][vz_i]])
        data = jnp.mean((lowpass(vf) - obs) ** 2)
        if M > 1:
            defect = jnp.mean((finals[:-1] - starts[1:]) ** 2)
        else:
            defect = 0.0
        return data + 0.5 * MU * defect, data

    def loss_scalar(theta):
        return loss_parts(theta)[0]

    loss_data = jax.jit(lambda th: loss_parts(th)[1])

    err_hist, loss_hist = [], []
    best_err = recovery_error(theta["ctrl"])
    t0 = time.time()
    stall = 0

    def track(theta, lval):
        nonlocal best_err, stall
        err = recovery_error(theta["ctrl"])
        err_hist.append(err); loss_hist.append(float(lval))
        best_err = min(best_err, err)
        if len(loss_hist) > 1:
            rel = abs(loss_hist[-2] - loss_hist[-1]) / (abs(loss_hist[-2]) + 1e-30)
            stall = stall + 1 if rel < REL_TOL else 0
        return stall >= PATIENCE

    if OPT == "lbfgs":
        # quasi-Newton with zoom line search: converge to a genuine local
        # minimum so a wrong recovered IC reflects multimodality, not
        # under-optimization. Same optimizer/settings for SS and MS.
        opt = optax.lbfgs()
        vg_state = optax.value_and_grad_from_state(loss_scalar)

        @jax.jit
        def lbfgs_step(theta, opt_state):
            value, grad = vg_state(theta, state=opt_state)
            updates, opt_state = opt.update(
                grad, opt_state, theta, value=value, grad=grad, value_fn=loss_scalar)
            theta = optax.apply_updates(theta, updates)
            gnorm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))
            return theta, opt_state, value, gnorm

        opt_state = opt.init(theta)
        for _ in range(MAX_STEPS):
            theta, opt_state, value, gnorm = lbfgs_step(theta, opt_state)
            plateaued = track(theta, value)
            if float(gnorm) < GTOL or plateaued:
                break
    else:  # adam
        vag = jax.jit(jax.value_and_grad(loss_scalar))
        opt = optax.adam(LR)
        opt_state = opt.init(theta)
        for _ in range(MAX_STEPS):
            lval, grads = vag(theta)
            updates, opt_state = opt.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            if track(theta, lval):
                break

    final_data = float(loss_data(theta))
    runtime = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    jnp.savez(
        OUT,
        t_obs=T_OBS, m=M, truth_seed=TRUTH_SEED, init_seed=INIT_SEED,
        init_amp=INIT_AMP, n_cells=N, opt=OPT, mu=MU, k_cut=K_CUT,
        err_hist=jnp.array(err_hist), loss_hist=jnp.array(loss_hist),
        final_err=err_hist[-1], best_err=best_err,
        final_data=final_data, n_steps=len(err_hist), runtime=runtime,
    )
    print(f"[done] T={T_OBS} m={M} seed={TRUTH_SEED} opt={OPT}: "
          f"final_err={err_hist[-1]:.3f} best={best_err:.3f} "
          f"iters={len(err_hist)} data={final_data:.2e} {runtime:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
