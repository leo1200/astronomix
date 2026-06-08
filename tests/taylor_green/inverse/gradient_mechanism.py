"""
The gradient mechanism behind single- vs multiple-shooting on the TGV,
in two panels (cf. init_optim_theory.md Section 8 and the KS analogue).

(a) Adjoint sensitivity vs time-within-window (T = T_WIN).
    The adjoint at time t is how strongly a large-scale loss at the *end
    of its gradient path* responds to a perturbation of the state at t.
    We seed a large-scale (low-k) co-vector w at the path end and pull it
    back to t; the quantity plotted is the energy the adjoint has accrued
    in the high-k band (the small-scale pile-up), the tractable 3D
    stand-in for the propagator sigma_max used on KS.
      * single shooting  -> gradient path ends at the window end T, so the
        high-k adjoint energy grows monotonically toward t=0 (envelope
        ~ e^{2 lambda (T - t)});
      * multiple shooting (m segments) -> the path ends at the end of the
        current segment, so it is a SAWTOOTH that resets at every segment
        boundary, capped at the value reached over one segment T/m.

(b) Optimization basin / search-space expansion.
    Loss along the line ctrl = alpha * ctrl_truth (alpha=1 is the truth):
      * single shooting integrates the whole window, so the IC controls
        the late (turbulent) state through a strongly nonlinear map -> a
        rugged / broad, hard-to-locate basin;
      * multiple shooting frees the interior states (here fixed at the
        true segment starts), so the IC only has to match its own short
        first segment -> a clean, sharp, convex basin around alpha=1.
    This is the weak-constraint / "expand the search space" effect: the
    state-space lift smooths the landscape (axis 3, not axis 1).

Hydro FD, single GPU, 48^3 (backprop-friendly mechanism run). The adjoint
panel uses reverse-mode (native JAX) and the basin panel is forward-only
(Pallas) for speed.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_registered_variables,
    get_helper_data,
    time_integration,
)
from astronomix.analysis_helpers.energy_spectrum import (
    LOG_BINNING,
    vector_field_energy_spectrum,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FORWARDS,
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings,
    finalize_config,
)
from astronomix.time_stepping.time_integration import _time_integration
from astronomix.data_classes.simulation_helper_data import (
    _helper_data_requirements,
    get_helper_data as _get_helper_data,
)


NUM_CELLS = 48
BOX_SIZE = 2.0 * jnp.pi
V0 = 1.0
RHO0 = 1.0
MA0 = 0.3
GAMMA = 5.0 / 3.0
C_CFL = 0.4
NUM_CHECKPOINTS = 24

T_WIN = 4.0            # window length (turnover units) for both panels
T0_WARMUP = 2.0       # de-singularize: evolve the TGV this long before the
                      # adjoint window, so the base trajectory is no longer the
                      # pristine single-mode IC (which produces a deterministic
                      # transient-cascade spike in the first segment)
N_SNAP = 28           # reference snapshots sampled for panel (a)
M_SAWTOOTH = [4, 8]   # segment counts shown as sawtooth in panel (a)
M_BASIN = 4           # segment count for the basin panel
K_SPLIT = 4.0         # high-k band threshold (integer mode number)
EPS_TLM = 1e-3        # finite-difference amplitude for the tangent-linear growth
MU = 10.0
N_ALPHA = 41

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def boundaries():
    return BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    )


def base_config(backend, diff_mode, snapshots=False):
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        mhd=False,
        backend=backend,
        differentiation_mode=diff_mode,
        num_checkpoints=NUM_CHECKPOINTS,
        enforce_positivity=False,
        progress_bar=False,
        dimensionality=3,
        num_cells=NUM_CELLS,
        box_size=BOX_SIZE,
        boundary_settings=boundaries(),
        return_snapshots=snapshots,
        num_snapshots=N_SNAP if snapshots else 1,
        snapshot_settings=SnapshotSettings(return_states=True) if snapshots else SnapshotSettings(),
    )


def tgv_velocity(helper_data):
    coords = helper_data.geometric_centers
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    vx = V0 * jnp.sin(x) * jnp.cos(y) * jnp.cos(z)
    vy = -V0 * jnp.cos(x) * jnp.sin(y) * jnp.cos(z)
    vz = jnp.zeros_like(x)
    return jnp.stack([vx, vy, vz])


def make_lowpass(n, kcut):
    k = jnp.fft.fftfreq(n) * n
    KX, KY, KZ = jnp.meshgrid(k, k, k, indexing="ij")
    mask = (jnp.sqrt(KX**2 + KY**2 + KZ**2) <= kcut).astype(jnp.float32)

    def lowpass(field):
        return jnp.fft.ifftn(
            jnp.fft.fftn(field, axes=(-3, -2, -1)) * mask, axes=(-3, -2, -1)
        ).real

    return lowpass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rv = get_registered_variables(base_config(NATIVE_JAX, BACKWARDS))
    helper_data = get_helper_data(base_config(NATIVE_JAX, BACKWARDS))
    vidx = rv.velocity_index

    density = RHO0 * jnp.ones((NUM_CELLS, NUM_CELLS, NUM_CELLS))
    p0 = RHO0 * V0**2 / (GAMMA * MA0**2)
    pressure = p0 * jnp.ones_like(density)
    u_truth = tgv_velocity(helper_data)
    lowpass = make_lowpass(NUM_CELLS, K_SPLIT)

    # configs
    cfg_adj = base_config(NATIVE_JAX, BACKWARDS)
    cfg_fwd = base_config(PALLAS, FORWARDS)
    cfg_ref = base_config(PALLAS, FORWARDS, snapshots=True)

    probe = construct_primitive_state(
        config=cfg_adj, registered_variables=rv, density=density,
        velocity_x=u_truth[0], velocity_y=u_truth[1], velocity_z=u_truth[2],
        gas_pressure=pressure,
    )
    cfg_adj = finalize_config(cfg_adj, probe.shape)
    cfg_fwd = finalize_config(cfg_fwd, probe.shape)
    cfg_ref = finalize_config(cfg_ref, probe.shape)

    params = SimulationParams(C_cfl=C_CFL, gamma=GAMMA, t_end=1.0)
    requirements = _helper_data_requirements(cfg_adj)
    hd_pad = _get_helper_data(cfg_adj, None, padded=False, requirements=requirements)

    w = u_truth / jnp.sqrt(jnp.sum(u_truth**2))   # large-scale co-vector

    def build_state(velocity, rho, p, cfg):
        return construct_primitive_state(
            config=cfg, registered_variables=rv, density=rho,
            velocity_x=velocity[0], velocity_y=velocity[1], velocity_z=velocity[2],
            gas_pressure=p,
        )

    def highk_energy(vel3):
        """High-k band energy of a (3,N,N,N) velocity field (outside jit)."""
        k, eg = vector_field_energy_spectrum(vel3[0], vel3[1], vel3[2], binning=LOG_BINNING)
        k_mode = k / (2.0 * jnp.pi)
        return float(jnp.sum(eg[k_mode >= K_SPLIT]))

    # =====================================================================
    #  de-singularize: evolve the pristine TGV to t0 before the window so the
    #  base trajectory carries a developed spectrum (no pure-mode transient)
    # =====================================================================
    print(f"De-singularizing: evolving TGV to t0={T0_WARMUP} ...")
    warm = time_integration(
        build_state(u_truth, density, pressure, cfg_fwd),
        cfg_fwd, params._replace(t_end=T0_WARMUP), rv,
    )
    warm_vel = jnp.stack([warm[vidx.x], warm[vidx.y], warm[vidx.z]])
    warm_rho = warm[rv.density_index]
    warm_p = warm[rv.pressure_index]

    # =====================================================================
    #  (a) adjoint high-k sensitivity vs time-within-window (de-singularized)
    # =====================================================================
    print("Generating reference window from the de-singularized state ...")
    ref = time_integration(
        build_state(warm_vel, warm_rho, warm_p, cfg_ref),
        cfg_ref, params._replace(t_end=T_WIN), rv,
    )
    ref_states = ref.states          # (N_SNAP, vars, N,N,N)
    ref_times = jnp.asarray(ref.time_points)
    print(f"  {ref_states.shape[0]} snapshots over window [0,{T_WIN}] (traj t0+..)")

    # ---- tangent-linear forward growth of a large-scale perturbation --------
    # propagate w (the same large-scale direction) forward along the window and
    # track its high-k energy: the deterministic+chaotic forward cascade that
    # sets the upward envelope of the adjoint teeth.
    print("Tangent-linear forward growth (finite-difference) ...")
    pert = time_integration(
        build_state(warm_vel + EPS_TLM * w, warm_rho, warm_p, cfg_ref),
        cfg_ref, params._replace(t_end=T_WIN), rv,
    )
    pert_states = pert.states
    tlm_highk = []
    for i in range(ref_states.shape[0]):
        dv = (jnp.stack([pert_states[i][vidx.x], pert_states[i][vidx.y], pert_states[i][vidx.z]])
              - jnp.stack([ref_states[i][vidx.x], ref_states[i][vidx.y], ref_states[i][vidx.z]])) / EPS_TLM
        tlm_highk.append(highk_energy(dv))

    def adjoint_highk(base_vel, base_rho, base_p, horizon):
        def J(vel):
            s = build_state(vel, base_rho, base_p, cfg_adj)
            sf = _time_integration(s, cfg_adj, params._replace(t_end=horizon), rv, hd_pad)
            return jnp.sum(w[0] * sf[vidx.x] + w[1] * sf[vidx.y] + w[2] * sf[vidx.z])
        g = jax.grad(J)(base_vel)
        k, eg = vector_field_energy_spectrum(g[0], g[1], g[2], binning=LOG_BINNING)
        k_mode = k / (2.0 * jnp.pi)
        # boolean indexing is illegal under jit -> use a multiplicative mask
        return jnp.sum(jnp.where(k_mode >= K_SPLIT, eg, 0.0))

    adjoint_highk_jit = jax.jit(adjoint_highk)

    n_snap = ref_states.shape[0]
    t_grid = [float(ref_times[i]) for i in range(n_snap)]

    def highk_curve(path_end_fn):
        vals = []
        for i in range(n_snap):
            t_i = float(ref_times[i])
            t_e = path_end_fn(t_i)
            # clamp to >= ~1 step so the checkpointed loop always runs at
            # least one trip (zero-trip grad is an avoidable edge case); at a
            # segment boundary this is ~the seed w, i.e. the sawtooth reset
            horizon = max(t_e - t_i, 0.03)
            s_i = ref_states[i]
            base_vel = jnp.stack([s_i[vidx.x], s_i[vidx.y], s_i[vidx.z]])
            base_rho = s_i[rv.density_index]
            base_p = s_i[rv.pressure_index]
            val = float(adjoint_highk_jit(
                base_vel, base_rho, base_p, jnp.asarray(horizon, dtype=base_vel.dtype)
            ))
            vals.append(val)
        return vals

    print("Single-shooting adjoint (path ends at window end) ...")
    ss_curve = highk_curve(lambda t: T_WIN)

    ms_curves = {}
    for m in M_SAWTOOTH:
        h = T_WIN / m
        print(f"Multiple-shooting adjoint, m={m} (segment {h:.2f} t_c) ...")
        # path ends at the end of the segment containing t (next boundary up)
        def seg_end(t, h=h):
            import math
            b = (math.floor(t / h + 1e-9) + 1) * h
            return min(b, T_WIN)
        ms_curves[m] = highk_curve(seg_end)

    jnp.savez(
        DATA_DIR / "gradient_mechanism_adjoint.npz",
        t_grid=jnp.array(t_grid), ss=jnp.array(ss_curve),
        tlm_highk=jnp.array(tlm_highk),
        **{f"ms_m{m}": jnp.array(ms_curves[m]) for m in ms_curves},
    )

    # =====================================================================
    #  (b) optimization basin along ctrl = alpha * ctrl_truth
    # =====================================================================
    print("Building basin landscape ...")
    # observation: filtered late-time velocity of the truth
    truth_final = time_integration(
        build_state(u_truth, density, pressure, cfg_fwd),
        cfg_fwd, params._replace(t_end=T_WIN), rv,
    )
    obs = lowpass(jnp.stack([truth_final[vidx.x], truth_final[vidx.y], truth_final[vidx.z]]))

    # true segment-start states for the multiple-shooting (expanded) loss
    h = T_WIN / M_BASIN
    seg_starts_true = [build_state(u_truth, density, pressure, cfg_fwd)]
    s = seg_starts_true[0]
    for _ in range(M_BASIN - 1):
        s = time_integration(s, cfg_fwd, params._replace(t_end=h), rv)
        seg_starts_true.append(s)

    def ss_loss(alpha):
        s0 = build_state(alpha * u_truth, density, pressure, cfg_fwd)
        sf = time_integration(s0, cfg_fwd, params._replace(t_end=T_WIN), rv)
        vf = lowpass(jnp.stack([sf[vidx.x], sf[vidx.y], sf[vidx.z]]))
        return float(jnp.mean((vf - obs) ** 2))

    def ms_loss(alpha):
        # interior states fixed at truth; only s0 depends on alpha
        s0 = build_state(alpha * u_truth, density, pressure, cfg_fwd)
        starts = [s0] + seg_starts_true[1:]
        defect = 0.0
        for j in range(M_BASIN - 1):
            f = time_integration(starts[j], cfg_fwd, params._replace(t_end=h), rv)
            defect = defect + jnp.mean((f - starts[j + 1]) ** 2)
        # data term: last segment from the (fixed, true) last start -> constant
        f_last = time_integration(starts[-1], cfg_fwd, params._replace(t_end=h), rv)
        vf = lowpass(jnp.stack([f_last[vidx.x], f_last[vidx.y], f_last[vidx.z]]))
        data = jnp.mean((vf - obs) ** 2)
        return float(data + 0.5 * MU * defect)

    alphas = jnp.linspace(-0.5, 2.0, N_ALPHA)
    ss_land = [ss_loss(float(a)) for a in alphas]
    ms_land = [ms_loss(float(a)) for a in alphas]
    jnp.savez(
        DATA_DIR / "gradient_mechanism_basin.npz",
        alphas=alphas, ss=jnp.array(ss_land), ms=jnp.array(ms_land),
    )

    # =====================================================================
    #  figure
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.semilogy(t_grid, ss_curve, "o-", ms=3, color="C3", label="single shooting")
    for m, mk in zip(M_SAWTOOTH, ["s-", "^-"]):
        ax.semilogy(t_grid, ms_curves[m], mk, ms=3, label=f"multiple shooting m={m}")
        for b in [(i + 1) * (T_WIN / m) for i in range(m - 1)]:
            ax.axvline(b, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("time t within window  [t_c]")
    ax.set_ylabel("adjoint high-k band energy")
    ax.set_title(f"(a) Adjoint vs time, T={T_WIN} t_c (de-singularized):\n"
                 "SS blow-up (toward t=0) vs MS sawtooth")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (b) tangent-linear forward growth -> explains the upward envelope
    ax = axes[1]
    tlm_a = jnp.array(tlm_highk)
    ax.semilogy(t_grid, tlm_a, "o-", ms=3, color="C2",
                label="forward TLM high-k energy")
    # exponential fit over the window
    tt = jnp.asarray(t_grid)
    lam_f = float(jnp.polyfit(tt, jnp.log(tlm_a), 1)[0])
    c0 = float(jnp.log(tlm_a[0]))
    ax.semilogy(tt, jnp.exp(c0 + lam_f * tt), "k--", alpha=0.6,
                label=f"~ exp({lam_f:.2f} t)")
    ax.set_xlabel("time t within window  [t_c]")
    ax.set_ylabel("forward perturbation high-k energy")
    ax.set_title("(b) Tangent-linear forward growth of a\nlarge-scale perturbation "
                 "(sets the envelope)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ss_land_a = jnp.array(ss_land); ms_land_a = jnp.array(ms_land)
    ax.plot(alphas, ss_land_a / jnp.max(ss_land_a), "o-", ms=3, color="C3",
            label="single shooting (full window)")
    ax.plot(alphas, ms_land_a / jnp.max(ms_land_a), "s-", ms=3, color="C0",
            label=f"multiple shooting m={M_BASIN} (interior freed)")
    ax.axvline(1.0, color="k", ls="--", alpha=0.5, label="truth (alpha=1)")
    ax.set_xlabel(r"control amplitude $\alpha$  (ctrl = $\alpha\,$ctrl$_{truth}$)")
    ax.set_ylabel("loss (normalized)")
    ax.set_title("(c) Optimization basin:\nstate-space expansion sharpens/convexifies")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gradient_mechanism.png", dpi=200)
    plt.close(fig)
    print(f"Figure -> {OUTPUT_DIR / 'gradient_mechanism.png'}")


if __name__ == "__main__":
    main()
