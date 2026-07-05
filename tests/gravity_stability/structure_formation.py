"""Structure formation in driven ADIABATIC MHD turbulence with self-gravity,
stabilised by the positivity-preserving WENO flux limiter.

Driven (OU) magnetised turbulence with self-gravity on from t=0: the densest
turbulent fluctuations go gravitationally unstable and collapse into cores /
filaments while the rest of the box keeps a turbulent (roughly log-normal)
density distribution. We track the emergence of the high-density tail (the
star-formation signature), the dense-gas mass fraction, the rho-B amplification
relation, and keep mid-plane + peak-core slices.

Stability: config.positivity_preserving_flux (the recovered+extended Hu-Adams-Shu
FCT flux limiter) — no non-conservative floors needed.
"""

# ==== GPU selection ====
import os as _os
if not _os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse
import os
import time as walltime

import numpy as np
import jax.numpy as jnp

from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, IDEAL_GAS, PALLAS, PERIODIC_BOUNDARY,
    FOURTH_ORDER_CONSERVATIVE, GravityConfig, PositivityConfig,
    BoundarySettings, BoundarySettings1D, SimulationConfig, SnapshotSettings,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

# log-density PDF bins (log10 rho/rho0) and density thresholds for mass fractions
PDF_BINS = np.linspace(-2.0, 3.0, 101)
RHO_THRESHOLDS = np.array([1.0, 3.0, 10.0, 30.0, 100.0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--mturb", type=float, default=3.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--G", type=float, default=6.0)
    p.add_argument("--gamma", type=float, default=5.0 / 3.0)
    p.add_argument("--tcross", type=float, default=3.0)
    p.add_argument("--F0", type=float, default=3.5)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--kf", type=float, default=3.0 * np.pi)
    p.add_argument("--nsnap", type=int, default=60)
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--pp_flux", type=int, default=1)
    p.add_argument("--final_only", type=int, default=0,
                   help="memory-safe: no per-snapshot states; diagnostics from "
                        "the final state + cheap scalar time series (for N>=256/512)")
    p.add_argument("--rhomin", type=float, default=1e-4)
    p.add_argument("--pmin", type=float, default=1e-7)
    p.add_argument("--outdir", type=str, default="data_structure")
    p.add_argument("--tag", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rho0 = 1.0
    a = 1.0 / args.mturb
    P0 = rho0 * a ** 2 / args.gamma
    B0 = float(np.sqrt(2.0 * P0 / args.beta))

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS,
        backend=PALLAS, pallas_block_shape=(4, 4, 8), pallas_use_triton=True,
        pallas_interpret=False, progress_bar=False, dimensionality=3,
        num_cells=args.N, box_size=1.0, mhd=True,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=FOURTH_ORDER_CONSERVATIVE,
            poisson_manual_open_boundaries=False,
        ),
        positivity_config=PositivityConfig(
            default_positivity_protection=False,  # no floors; pp_flux is the guard
            preserving_flux=bool(args.pp_flux),
        ),
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True, ou_forcing=True, vacuum_protection=False,
        ),
        return_snapshots=True, num_snapshots=args.nsnap,
        snapshot_settings=SnapshotSettings(
            return_states=not bool(args.final_only),
            return_final_state=True,
        ),
    )

    t_cross = (config.box_size / 2.0) / 1.0
    params = SimulationParams(
        C_cfl=args.cfl, gamma=args.gamma, gravitational_constant=args.G,
        isothermal_sound_speed=a, t_end=args.tcross * t_cross,
        turbulent_forcing_params=TurbulentForcingParams(
            forcing_amplitude=args.F0, correlation_time=args.tau,
            forcing_wavenumber=args.kf,
        ),
        minimum_density=args.rhomin, minimum_pressure=args.pmin,
    )

    rv = get_registered_variables(config)
    density = jnp.ones((args.N, args.N, args.N), dtype=jnp.float32) * rho0
    zero = jnp.zeros_like(density)
    Bz = jnp.ones_like(density) * B0
    bxb, byb, bzb = initialize_interface_fields(zero, zero, Bz)
    initial_state = construct_primitive_state(
        config=config, registered_variables=rv, density=density,
        velocity_x=zero, velocity_y=zero, velocity_z=zero,
        gas_pressure=jnp.ones_like(density) * P0,
        magnetic_field_x=zero, magnetic_field_y=zero, magnetic_field_z=Bz,
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
    )
    config = finalize_config(config, initial_state.shape)

    lam_J = float(np.sqrt(np.pi * a ** 2 / (args.G * rho0)))
    print(f"[{args.tag}] N={args.N} M_turb~{args.mturb} beta={args.beta} G={args.G} "
          f"a={a:.3f} B0={B0:.4g} lam_J/L={lam_J:.3f} pp_flux={args.pp_flux}",
          flush=True)

    final_only = bool(args.final_only)
    t0 = walltime.time()
    result = time_integration(initial_state, config, params, rv)
    time_points = np.asarray(result.time_points)
    result.final_state.block_until_ready()
    print(f"[{args.tag}] wall {walltime.time()-t0:.1f}s final_only={final_only}",
          flush=True)

    di = rv.density_index
    vx_i, vy_i, vz_i = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z
    bx_i, by_i, bz_i = rv.magnetic_index.x, rv.magnetic_index.y, rv.magnetic_index.z

    def _pdf_massfrac(rho):
        rho_pos = np.maximum(rho, 1e-30)
        lr = np.log10(rho_pos)
        h, _ = np.histogram(lr, bins=PDF_BINS, density=True)
        hm, _ = np.histogram(lr, bins=PDF_BINS, weights=rho_pos / np.mean(rho_pos),
                             density=True)
        mf = np.array([float(np.sum(rho[rho >= thr]) / np.sum(rho))
                       for thr in RHO_THRESHOLDS])
        return h, hm, mf

    t_over_tc = time_points / t_cross

    if not final_only:
        states = result.states
        states.block_until_ready()
        ns = states.shape[0]
        Ms_t = np.zeros(ns); rhomax_t = np.zeros(ns); EB_t = np.zeros(ns)
        pdf_vol = np.zeros((ns, len(PDF_BINS) - 1))
        pdf_mass = np.zeros((ns, len(PDF_BINS) - 1))
        massfrac = np.zeros((ns, len(RHO_THRESHOLDS)))
        finite_t = np.zeros(ns, dtype=bool)
        first_bad = -1
        for s in range(ns):
            st = states[s]
            ok = bool(jnp.all(jnp.isfinite(st)))
            finite_t[s] = ok
            if first_bad < 0 and not ok:
                first_bad = s
            rho = np.asarray(jnp.nan_to_num(st[di]))
            v2 = np.asarray(jnp.nan_to_num(st[vx_i]**2 + st[vy_i]**2 + st[vz_i]**2))
            B2 = np.asarray(jnp.nan_to_num(st[bx_i]**2 + st[by_i]**2 + st[bz_i]**2))
            Ms_t[s] = np.sqrt(np.mean(v2)) / a
            rhomax_t[s] = float(np.max(rho))
            EB_t[s] = float(np.mean(0.5 * B2))
            pdf_vol[s], pdf_mass[s], massfrac[s] = _pdf_massfrac(rho)
        alive = np.where((rhomax_t > 0) & finite_t)[0]
        sidx = int(alive[-1]) if len(alive) else ns - 1
        fin = states[sidx]
    else:
        # memory-safe: only the final state is materialised
        fin = result.final_state
        ok = bool(jnp.all(jnp.isfinite(fin)))
        first_bad = -1 if ok else 0
        sidx = len(time_points) - 1
        rho = np.asarray(jnp.nan_to_num(fin[di]))
        v2 = np.asarray(jnp.nan_to_num(fin[vx_i]**2 + fin[vy_i]**2 + fin[vz_i]**2))
        B2 = np.asarray(jnp.nan_to_num(fin[bx_i]**2 + fin[by_i]**2 + fin[bz_i]**2))
        hv, hm, mf = _pdf_massfrac(rho)
        # single-entry "time series" = final state only
        Ms_t = np.array([np.sqrt(np.mean(v2)) / a])
        rhomax_t = np.array([float(np.max(rho))])
        EB_t = np.array([float(np.mean(0.5 * B2))])
        pdf_vol = hv[None, :]; pdf_mass = hm[None, :]
        massfrac = mf[None, :]

    rho_f = np.asarray(jnp.nan_to_num(fin[di]))
    B_f = np.asarray(jnp.sqrt(jnp.nan_to_num(
        fin[bx_i]**2 + fin[by_i]**2 + fin[bz_i]**2)))
    rb_bins = np.logspace(-1, np.log10(max(2.0, rho_f.max())), 30)
    rb_idx = np.digitize(rho_f.ravel(), rb_bins)
    rb_meanB = np.array([B_f.ravel()[rb_idx == k].mean() if np.any(rb_idx == k)
                         else np.nan for k in range(1, len(rb_bins))])
    rb_centers = 0.5 * (rb_bins[1:] + rb_bins[:-1])

    z = args.N // 2
    rho_slice = rho_f[:, :, z]
    B_slice = B_f[:, :, z]
    # column density (projection along z) — the observable
    coldens = np.asarray(jnp.nan_to_num(fin[di])).sum(axis=2) / args.N

    print(f"[{args.tag}] first_bad={first_bad} "
          + ("ALL FINITE" if first_bad < 0 else f"@t/tc={t_over_tc[first_bad]:.2f}"),
          flush=True)
    print(f"[{args.tag}] rho_max(t/tc): " +
          " ".join(f"{tc:.2f}:{rm:.0f}" for tc, rm in
                   zip(t_over_tc[::8], rhomax_t[::8])), flush=True)
    print(f"[{args.tag}] dense-mass-frac(rho>10) final={massfrac[-1,2]:.3f} "
          f"(rho>30)={massfrac[-1,3]:.3f} (rho>100)={massfrac[-1,4]:.3f}",
          flush=True)

    out = os.path.join(args.outdir, f"structure_{args.tag}.npz")
    np.savez(out, tag=args.tag, N=args.N, mturb=args.mturb, beta=args.beta,
             G=args.G, a=a, B0=B0, lam_J=lam_J, time_points=time_points,
             t_over_tc=t_over_tc, Ms_t=Ms_t, rhomax_t=rhomax_t, EB_t=EB_t,
             pdf_bins=PDF_BINS, pdf_vol=pdf_vol, pdf_mass=pdf_mass,
             rho_thresholds=RHO_THRESHOLDS, massfrac=massfrac,
             rb_centers=rb_centers, rb_meanB=rb_meanB,
             rho_slice=rho_slice, B_slice=B_slice, coldens=coldens,
             slice_t_over_tc=t_over_tc[sidx], first_bad=first_bad,
             EB0=float(0.5 * B0**2))
    print(f"[{args.tag}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
