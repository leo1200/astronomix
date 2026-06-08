"""
Stage 2 of the inverse-modeling study: single shooting vs multiple
shooting recovery of a large-scale Taylor-Green initial condition from a
filtered late-time observation.

Implements worked example II of ``init_optim_theory.md`` (Section 8) on
the shared step-defect loss, with the reduced<->full-space limits exact:

    m = 1        single shooting   (control = IC; full back-prop)
    m = 2,4,8    multiple shooting (free interior segment states +
                                    consistency defects)
    m = N_t      full space        (limit, not run here)

**Control (axis-2 held fixed).** The control is the large-scale band:
the initial velocity is the low-pass (|k| <= K_CUT) projection of a
real field, with the small scales pinned to the prior. Identical
recoverable subspace for every method, so SS-vs-MS differences are the
gradient/basin story (axes 1 & 3), not identifiability.

**Step-defect loss.** Split [0, T_obs] into m segments of length
h = T_obs/m with segment-start states s_0..s_{m-1} (s_0 = build_ic(ctrl)):

    L(theta) = || P_lk( F_h(s_{m-1}) ) - obs ||^2
               + (mu/2) * mean_j || F_h(s_j) - s_{j+1} ||^2

with F_h = one segment of the differentiable hydro-FD integrator and
P_lk the low-pass observation operator. m=1 drops the defect term and is
exactly single shooting (full back-prop through T_obs).

**Outer loop.** F_h is ``_time_integration`` (the inner, side-effect-free
core that ``time_integration`` jits) run with t_end = h. The m segments
are propagated in a Python loop -- the outer loop around the integrator
-- and the whole loss is reverse-mode differentiated with jax.grad. (The
Pallas backend falls back to native JAX on the backward pass, so we use
NATIVE_JAX directly; 64^3 max for backprop, storage tuned via
num_checkpoints.)

**Prediction (Section 8).** Single shooting cliffs within ~1 turnover of
the predictability horizon (the adjoint piles up at small scales, see
adjoint_spectrum.py); multiple shooting caps the back-prop to one
segment and recovers the large scales across many turnovers.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

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
    PALLAS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    finalize_config,
)

# backend for the differentiable run: native everywhere, or Pallas forward +
# (native-fallback) backward -- the latter can be faster because the forward
# pass and the checkpoint recompute then use the fast Pallas kernels.
_BACKEND = PALLAS if os.environ.get("TGV_BACKEND", "native").lower() == "pallas" else NATIVE_JAX
from astronomix.time_stepping.time_integration import _time_integration
from astronomix.data_classes.simulation_helper_data import (
    _helper_data_requirements,
    get_helper_data as _get_helper_data,
)


# ---------------------------------------------------------------------------
#  Experiment parameters
# ---------------------------------------------------------------------------

# a few knobs are overridable via environment variables so the same script
# can do a cheap smoke test and drive the horizon sweep without edits
NUM_CELLS = int(os.environ.get("TGV_N", 64))
BOX_SIZE = 2.0 * jnp.pi
V0 = 1.0
RHO0 = 1.0
MA0 = 0.3
GAMMA = 5.0 / 3.0
C_CFL = 0.4
NUM_CHECKPOINTS = 16

K_CUT = 4.0            # low-k control / observation band (integer mode number)
T_OBS = float(os.environ.get("TGV_TOBS", 3.0))   # observation horizon (t_c = 1)
M_LIST = [int(m) for m in os.environ.get("TGV_M", "1,2,4").split(",")]
NUM_STEPS = int(os.environ.get("TGV_STEPS", 80))  # Adam steps
LEARNING_RATE = 3e-2
MU = 10.0              # consistency-defect penalty weight

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def build_config_params():
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        mhd=False,
        backend=_BACKEND,
        differentiation_mode=BACKWARDS,
        num_checkpoints=NUM_CHECKPOINTS,
        enforce_positivity=False,
        progress_bar=False,
        dimensionality=3,
        num_cells=NUM_CELLS,
        box_size=BOX_SIZE,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        return_snapshots=False,
    )
    params = SimulationParams(C_cfl=C_CFL, gamma=GAMMA, t_end=1.0)
    return config, params


def make_lowpass(n, kcut):
    k = jnp.fft.fftfreq(n) * n
    KX, KY, KZ = jnp.meshgrid(k, k, k, indexing="ij")
    mask = (jnp.sqrt(KX**2 + KY**2 + KZ**2) <= kcut).astype(jnp.float32)

    def lowpass(field):  # field (..., n, n, n)
        return jnp.fft.ifftn(
            jnp.fft.fftn(field, axes=(-3, -2, -1)) * mask, axes=(-3, -2, -1)
        ).real

    return lowpass


def tgv_velocity(helper_data):
    coords = helper_data.geometric_centers
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    vx = V0 * jnp.sin(x) * jnp.cos(y) * jnp.cos(z)
    vy = -V0 * jnp.cos(x) * jnp.sin(y) * jnp.cos(z)
    vz = jnp.zeros_like(x)
    return jnp.stack([vx, vy, vz])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    config, params = build_config_params()
    rv = get_registered_variables(config)
    helper_data = get_helper_data(config)

    density = RHO0 * jnp.ones((NUM_CELLS, NUM_CELLS, NUM_CELLS))
    p0 = RHO0 * V0**2 / (GAMMA * MA0**2)
    pressure = p0 * jnp.ones_like(density)

    u_truth = tgv_velocity(helper_data)
    probe = construct_primitive_state(
        config=config, registered_variables=rv, density=density,
        velocity_x=u_truth[0], velocity_y=u_truth[1], velocity_z=u_truth[2],
        gas_pressure=pressure,
    )
    config = finalize_config(config, probe.shape)

    # inner-core helper data (side-effect free integrator path)
    requirements = _helper_data_requirements(config)
    hd_pad = _get_helper_data(config, None, padded=False, requirements=requirements)

    lowpass = make_lowpass(NUM_CELLS, K_CUT)
    vx_i, vy_i, vz_i = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z

    def build_ic(ctrl):
        v = lowpass(ctrl)
        return construct_primitive_state(
            config=config, registered_variables=rv, density=density,
            velocity_x=v[0], velocity_y=v[1], velocity_z=v[2], gas_pressure=pressure,
        )

    def propagate(state, h):
        return _time_integration(
            state, config, params._replace(t_end=h), rv, hd_pad
        )

    # ---- synthetic observation: filtered late-time velocity of the truth ----
    truth_ic = build_ic(u_truth)           # truth is already low-k -> unchanged
    truth_final = propagate(truth_ic, T_OBS)
    obs = lowpass(jnp.stack([truth_final[vx_i], truth_final[vy_i], truth_final[vz_i]]))
    truth_lk = lowpass(u_truth)
    truth_lk_norm = float(jnp.sqrt(jnp.sum(truth_lk**2)))

    def recovery_error(ctrl):
        rec = lowpass(ctrl)
        return float(jnp.sqrt(jnp.sum((rec - truth_lk) ** 2)) / truth_lk_norm)

    results = {}
    for m in M_LIST:
        h = T_OBS / m
        print(f"\n=== m = {m}  (segment length h = {h:.3f} t_c) ===")

        # cold start: zero IC perturbation; warm interior states by forward
        # propagation of the guess so the initial defects are ~0
        ctrl0 = jnp.zeros_like(u_truth)
        s0 = build_ic(ctrl0)
        seg_states = []
        s = s0
        for _ in range(m - 1):
            s = propagate(s, h)
            seg_states.append(s)
        seg0 = jnp.stack(seg_states) if seg_states else jnp.zeros((0, *s0.shape))
        theta = {"ctrl": ctrl0, "seg": seg0}

        def loss(theta):
            s0 = build_ic(theta["ctrl"])
            if m > 1:
                starts = jnp.concatenate([s0[None], theta["seg"]], axis=0)
            else:
                starts = s0[None]
            finals = jnp.stack([propagate(starts[j], h) for j in range(m)])
            vf = jnp.stack([finals[-1][vx_i], finals[-1][vy_i], finals[-1][vz_i]])
            data = jnp.mean((lowpass(vf) - obs) ** 2)
            if m > 1:
                defects = finals[:-1] - starts[1:]
                defect = jnp.mean(defects ** 2)
            else:
                defect = 0.0
            return data + 0.5 * MU * defect, (data, defect)

        value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

        optimizer = optax.adam(LEARNING_RATE)
        opt_state = optimizer.init(theta)

        history = {"loss": [], "data": [], "defect": [], "err": []}
        t0 = time.time()
        for step in range(NUM_STEPS):
            (lval, (data, defect)), grads = value_and_grad(theta)
            updates, opt_state = optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            err = recovery_error(theta["ctrl"])
            history["loss"].append(float(lval))
            history["data"].append(float(data))
            history["defect"].append(float(defect))
            history["err"].append(err)
            if step % 10 == 0 or step == NUM_STEPS - 1:
                print(f"  step {step:>3}: loss={float(lval):.3e} "
                      f"data={float(data):.3e} defect={float(defect):.3e} "
                      f"rec_err={err:.3f}")
        dt = time.time() - t0
        print(f"  m={m} done in {dt:.1f}s, final recovery error = {history['err'][-1]:.3f}")
        results[m] = {"history": history, "ctrl": theta["ctrl"], "runtime": dt}

    # ---- save + plot ----
    jnp.savez(
        DATA_DIR / f"ss_vs_ms_Tobs{T_OBS}.npz",
        **{f"err_m{m}": jnp.array(results[m]["history"]["err"]) for m in results},
        **{f"loss_m{m}": jnp.array(results[m]["history"]["loss"]) for m in results},
        truth_lk=truth_lk,
        **{f"ctrl_m{m}": results[m]["ctrl"] for m in results},
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m in results:
        axes[0].semilogy(results[m]["history"]["loss"], label=f"m={m}")
        axes[1].plot(results[m]["history"]["err"], label=f"m={m}")
    axes[0].set(xlabel="Adam step", ylabel="loss", title=f"Convergence (T_obs={T_OBS} t_c)")
    axes[0].legend(); axes[0].grid(alpha=0.3, which="both")
    axes[1].set(xlabel="Adam step", ylabel="large-scale recovery error",
                title="IC recovery error (lower = better)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].axhline(0.0, color="k", lw=0.5)
    fig.suptitle("Single shooting (m=1) vs multiple shooting (m>1) on the TGV", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"ss_vs_ms_Tobs{T_OBS}.png", dpi=200)
    plt.close(fig)
    print(f"\nFigure -> {OUTPUT_DIR / f'ss_vs_ms_Tobs{T_OBS}.png'}")
    for m in results:
        print(f"  m={m}: final recovery error = {results[m]['history']['err'][-1]:.3f}")


if __name__ == "__main__":
    main()
