"""
The mismatch-penalty (mu) transition: multiple shooting -> single shooting.

The sawtooth in gradient_mechanism.py is the LOCAL per-segment sensitivity:
each full-space partial gradient dL/ds_j carries only a one-segment Jacobian
(capped at e^{lambda T/m}), for ANY finite mu -- mu only rescales the residual
magnitudes, it does not lengthen the back-prop horizon.

Single shooting is recovered in the *reduced* problem. As mu -> infinity the
consistency constraints F_h(s_j)=s_{j+1} are enforced exactly, the interior
states are slaved to the IC and can be eliminated, and the REDUCED gradient
dL_red/d(IC) chains the one-segment Jacobians across all m segments -> the full
e^{lambda T} single-shooting growth.

This script makes that explicit. For a sweep of mu we:
  1. hold the IC fixed (a slightly perturbed truth),
  2. inner-optimize the interior segment states to the weak-constraint optimum
     (L-BFGS, warm-started from the slaved trajectory),
  3. by the envelope theorem the reduced IC gradient is dL/d(IC) at that inner
     optimum; we record its total norm and its high-k band energy,
differentiating w.r.t. the FULL IC velocity field so the gradient carries real
small-scale content. Dashed references: the single-shooting gradient (the
mu->infinity target) and the small-mu (per-segment) value.

Hydro FD, 32^3, native backprop. cf. init_optim_theory.md Section 7 (local
per-segment sensitivity vs realized, mu-dependent conditioning).
"""

import os

# ==== GPU selection ====
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _cvd == "" or _cvd.lower() == "all":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from astronomix import (
    SimulationConfig, SimulationParams, get_registered_variables, get_helper_data,
)
from astronomix.analysis_helpers.energy_spectrum import LOG_BINNING, vector_field_energy_spectrum
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, IDEAL_GAS, NATIVE_JAX, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, finalize_config,
)
from astronomix.time_stepping.time_integration import _time_integration
from astronomix.data_classes.simulation_helper_data import _helper_data_requirements, get_helper_data as _ghd

N = int(os.environ.get("MU_N", 32))
T_WIN = float(os.environ.get("MU_TWIN", 3.0))
M = int(os.environ.get("MU_M", 4))
INNER_ITERS = int(os.environ.get("MU_INNER", 40))
MU_LIST = [float(x) for x in os.environ.get(
    "MU_LIST", "0.3,1,3,10,30,100,300,1000").split(",")]
EPS_PERT = 0.05            # IC perturbation amplitude (gives a nonzero misfit)
BOX = 2.0 * jnp.pi
V0, RHO0, MA0, GAMMA, C_CFL = 1.0, 1.0, 0.3, 5.0 / 3.0, 0.4
K_SPLIT = 4.0
NUM_CHECKPOINTS = 256
GTOL = 1e-5

OUT = Path(__file__).parent / "figures"
DATA = Path(__file__).parent / "data"


def main():
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS, mhd=False,
        backend=NATIVE_JAX, differentiation_mode=BACKWARDS, num_checkpoints=NUM_CHECKPOINTS,
        enforce_positivity=False, progress_bar=False, dimensionality=3,
        num_cells=N, box_size=BOX,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)),
        return_snapshots=False,
    )
    params = SimulationParams(C_cfl=C_CFL, gamma=GAMMA, t_end=1.0)
    rv = get_registered_variables(config)
    hd = get_helper_data(config)
    vx_i, vy_i, vz_i = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z

    density = RHO0 * jnp.ones((N, N, N))
    p0 = RHO0 * V0**2 / (GAMMA * MA0**2)
    pressure = p0 * jnp.ones_like(density)

    coords = hd.geometric_centers
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    u_truth = jnp.stack([V0 * jnp.sin(x) * jnp.cos(y) * jnp.cos(z),
                         -V0 * jnp.cos(x) * jnp.sin(y) * jnp.cos(z),
                         jnp.zeros_like(x)])

    probe = construct_primitive_state(config=config, registered_variables=rv,
        density=density, velocity_x=u_truth[0], velocity_y=u_truth[1],
        velocity_z=u_truth[2], gas_pressure=pressure)
    config = finalize_config(config, probe.shape)
    hd_pad = _ghd(config, None, padded=False, requirements=_helper_data_requirements(config))

    # low-pass observation operator
    k1 = jnp.fft.fftfreq(N) * N
    KX, KY, KZ = jnp.meshgrid(k1, k1, k1, indexing="ij")
    lk = (jnp.sqrt(KX**2 + KY**2 + KZ**2) <= K_SPLIT).astype(jnp.float32)
    lowpass = lambda f: jnp.fft.ifftn(jnp.fft.fftn(f, axes=(-3, -2, -1)) * lk, axes=(-3, -2, -1)).real

    def state_of(v0):
        return construct_primitive_state(config=config, registered_variables=rv,
            density=density, velocity_x=v0[0], velocity_y=v0[1], velocity_z=v0[2],
            gas_pressure=pressure)

    def propagate(s, hh):
        return _time_integration(s, config, params._replace(t_end=hh), rv, hd_pad)

    def velof(s):
        return jnp.stack([s[vx_i], s[vy_i], s[vz_i]])

    def highk(field3):
        kk, eg = vector_field_energy_spectrum(field3[0], field3[1], field3[2], binning=LOG_BINNING)
        return float(jnp.sum(eg[(kk / (2 * jnp.pi)) >= K_SPLIT]))

    h = T_WIN / M
    # truth observation and the (perturbed) IC we evaluate the gradient at
    obs = lowpass(velof(propagate(state_of(u_truth), T_WIN)))
    key = jax.random.PRNGKey(0)
    pert = jax.random.normal(key, u_truth.shape) * lk  # low-k random direction
    pert = jnp.fft.ifftn(jnp.fft.fftn(pert, axes=(-3, -2, -1)), axes=(-3, -2, -1)).real
    pert = pert / jnp.sqrt(jnp.mean(pert**2))
    v0_eval = u_truth + EPS_PERT * pert

    # --- single-shooting reference (the mu->infinity target) ---
    def loss_ss(v0):
        return 0.5 * jnp.mean((lowpass(velof(propagate(state_of(v0), T_WIN))) - obs) ** 2)
    g_ss = jax.grad(loss_ss)(v0_eval)
    ss_norm = float(jnp.sqrt(jnp.sum(g_ss**2))); ss_highk = highk(g_ss)
    print(f"single shooting: ||g||={ss_norm:.3e}  high-k={ss_highk:.3e}")

    # --- full-space loss L(v0, seg; mu) ---
    def full_loss(v0, seg, mu):
        s0 = state_of(v0)
        starts = jnp.concatenate([s0[None], seg], axis=0)
        finals = jnp.stack([propagate(starts[j], h) for j in range(M)])
        data = 0.5 * jnp.mean((lowpass(velof(finals[-1])) - obs) ** 2)
        defect = jnp.sum(jnp.stack([jnp.mean((finals[j] - starts[j + 1]) ** 2)
                                    for j in range(M - 1)]))
        return data + 0.5 * mu * defect

    # slaved interior init (the mu->infinity optimum of the penalty)
    s = state_of(v0_eval); slaved = []
    for _ in range(M - 1):
        s = propagate(s, h); slaved.append(s)
    seg_slaved = jnp.stack(slaved)

    def inner_optimize(seg0, mu):
        lf = lambda seg: full_loss(v0_eval, seg, mu)
        opt = optax.lbfgs()
        vg = optax.value_and_grad_from_state(lf)

        @jax.jit
        def step(seg, st):
            v, g = vg(seg, state=st)
            upd, st = opt.update(g, st, seg, value=v, grad=g, value_fn=lf)
            seg = optax.apply_updates(seg, upd)
            gn = jnp.sqrt(sum(jnp.sum(a**2) for a in jax.tree_util.tree_leaves(g)))
            return seg, st, v, gn

        st = opt.init(seg0); seg = seg0
        for _ in range(INNER_ITERS):
            seg, st, v, gn = step(seg, st)
            if float(gn) < GTOL:
                break
        return seg

    mus, red_norm, red_highk, defects = [], [], [], []
    seg = seg_slaved
    for mu in MU_LIST:
        seg = inner_optimize(seg, mu)               # warm-start across mu
        g_red = jax.grad(lambda v: full_loss(v, seg, mu))(v0_eval)
        rn = float(jnp.sqrt(jnp.sum(g_red**2))); rk = highk(g_red)
        # leftover consistency residual (how close to slaving)
        s0 = state_of(v0_eval)
        starts = jnp.concatenate([s0[None], seg], axis=0)
        fin = jnp.stack([propagate(starts[j], h) for j in range(M)])
        dres = float(jnp.sqrt(jnp.mean((fin[:-1] - starts[1:]) ** 2)))
        mus.append(mu); red_norm.append(rn); red_highk.append(rk); defects.append(dres)
        print(f"mu={mu:>7.1f}: ||g_red||={rn:.3e}  high-k={rk:.3e}  "
              f"defect={dres:.2e}  (-> SS norm {ss_norm:.2e})")

    jnp.savez(DATA / "mismatch_mu_sweep.npz",
              mu=jnp.array(mus), red_norm=jnp.array(red_norm),
              red_highk=jnp.array(red_highk), defect=jnp.array(defects),
              ss_norm=ss_norm, ss_highk=ss_highk, T_win=T_WIN, m=M)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.loglog(mus, red_norm, "o-", color="C0", label="reduced IC gradient $\\|g_{red}\\|$")
    a1.axhline(ss_norm, color="C3", ls="--", label="single shooting (m=1)")
    a1.axhline(red_norm[0], color="C0", ls=":", alpha=0.6, label="per-segment (small $\\mu$)")
    a1.set_xlabel(r"mismatch penalty  $\mu$"); a1.set_ylabel("gradient norm")
    a1.set_title(f"Effective IC gradient norm vs $\\mu$  (m={M}, T={T_WIN})\n"
                 r"$\mu\to\infty$ recovers single shooting")
    a1.legend(fontsize=9); a1.grid(True, which="both", alpha=0.3)

    a2.loglog(mus, red_highk, "o-", color="C0", label="reduced gradient high-k")
    a2.axhline(ss_highk, color="C3", ls="--", label="single shooting high-k")
    a2.set_xlabel(r"mismatch penalty  $\mu$"); a2.set_ylabel("high-k band energy")
    a2.set_title("Small-scale content of the effective IC gradient")
    a2.legend(fontsize=9); a2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "mismatch_mu_sweep.png", dpi=200)
    plt.close(fig)
    print(f"figure -> {OUT / 'mismatch_mu_sweep.png'}")


if __name__ == "__main__":
    main()
