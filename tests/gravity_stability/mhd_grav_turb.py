"""Driven ADIABATIC MHD turbulence with self-gravity -- stability test for the
energy-conserving FD self-gravity schemes.

Isothermal turbulence has no energy equation, so the gravity energy-source
instability cannot appear there; this test therefore uses an ideal-gas EOS so
the conservative gravity energy source is actually exercised. A uniform
magnetised box is stirred with OU forcing while self-gravity (periodic FFT /
Jeans swindle) pulls dense regions into collapse. We measure whether the run
stays finite and how strongly total energy drifts, for the baseline scheme vs
the stabilised (simple-blend) variant.

One case per invocation. Built on tests/turbulence/paper_turbulence.py.
"""

# ==== GPU selection ====
# Respect an externally-pinned GPU (CUDA_VISIBLE_DEVICES set by the caller via
# `autocvd -o`); otherwise self-select a free GPU.
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
    TurbulentForcingConfig,
    TurbulentForcingParams,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, IDEAL_GAS, PALLAS,
    PERIODIC_BOUNDARY,
    POSITIVITY_NONE, POSITIVITY_HARD_FLOOR, POSITIVITY_REDISTRIBUTE,
    POSITIVITY_CONSERVATIVE,
    SIMPLE_SOURCE, SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE,
    GravityConfig, PositivityConfig,
    BoundarySettings, BoundarySettings1D,
    SimulationConfig, SnapshotSettings, finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

_POS = {"none": POSITIVITY_NONE, "floor": POSITIVITY_HARD_FLOOR,
        "redist": POSITIVITY_REDISTRIBUTE, "cons": POSITIVITY_CONSERVATIVE}
_SCHEME = {"simple": SIMPLE_SOURCE, "second": SECOND_ORDER_CONSERVATIVE,
           "fourth": FOURTH_ORDER_CONSERVATIVE}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mturb", type=float, default=2.0, help="target sonic Mach")
    p.add_argument("--beta", type=float, default=1.0, help="initial plasma beta")
    p.add_argument("--G", type=float, default=1.0, help="gravitational constant")
    p.add_argument("--gamma", type=float, default=5.0 / 3.0)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--tcross", type=float, default=2.0, help="t_end in crossing times")
    p.add_argument("--F0", type=float, default=3.5)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--kf", type=float, default=3.0 * np.pi)
    p.add_argument("--nsnap", type=int, default=60)
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--scheme", choices=list(_SCHEME), default="fourth")
    p.add_argument("--stage_mode", choices=list(_POS), default="none")
    p.add_argument("--step_mode", choices=list(_POS), default="none")
    p.add_argument("--pp_flux", type=int, default=0,
                   help="positivity-preserving WENO flux limiter (reconstruction-level)")
    p.add_argument("--protect_pos", type=int, default=0,
                   help="default_positivity_protection (read-only clamps + floors)")
    p.add_argument("--vmaxcap", type=float, default=float("inf"),
                   help="per-stage REDISTRIBUTE velocity cap")
    p.add_argument("--vacuum_rest", type=int, default=0,
                   help="zero momentum in floored cells")
    p.add_argument("--protect", type=int, default=0, help="OU vacuum protection (prot)")
    p.add_argument("--rhomin", type=float, default=1e-4)
    p.add_argument("--pmin", type=float, default=1e-6)
    p.add_argument("--vmax", type=float, default=50.0)
    p.add_argument("--outdir", type=str, default="data_mhdgrav")
    p.add_argument("--tag", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rho0 = 1.0
    a = 1.0 / args.mturb                 # sound speed; v_rms ~ 1 normalisation
    vrms_target = 1.0
    P0 = rho0 * a ** 2 / args.gamma      # so sqrt(gamma P0/rho0) = a
    P_thermal = P0
    B0 = float(np.sqrt(2.0 * P_thermal / args.beta))

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        backend=PALLAS,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
        progress_bar=False,
        dimensionality=3,
        num_cells=args.N,
        box_size=1.0,
        mhd=True,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=_SCHEME[args.scheme],
            poisson_manual_open_boundaries=False,  # periodic gravity (Jeans swindle)
        ),
        positivity_config=PositivityConfig(
            default_positivity_protection=bool(args.protect_pos),
            preserving_flux=bool(args.pp_flux),
            per_stage_mode=_POS[args.stage_mode],
            per_step_mode=_POS[args.step_mode],
            vacuum_rest=bool(args.vacuum_rest),
        ),
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True, ou_forcing=True,
            vacuum_protection=bool(args.protect),
        ),
        return_snapshots=True,
        num_snapshots=args.nsnap,
        snapshot_settings=SnapshotSettings(
            return_states=True, return_total_energy=True,
        ),
    )

    L_inj = config.box_size / 2.0
    t_cross = L_inj / vrms_target

    params = SimulationParams(
        C_cfl=args.cfl,
        gamma=args.gamma,
        gravitational_constant=args.G,
        t_end=args.tcross * t_cross,
        turbulent_forcing_params=TurbulentForcingParams(
            forcing_amplitude=args.F0, correlation_time=args.tau,
            forcing_wavenumber=args.kf,
            protection_density_threshold=args.rhomin,
            protection_max_velocity=args.vmax,
        ),
        positivity_max_velocity=args.vmaxcap,
        minimum_density=args.rhomin,
        minimum_pressure=args.pmin,
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

    # Jeans length for reference: lambda_J = sqrt(pi c_s^2 / (G rho))
    lam_J = float(np.sqrt(np.pi * a ** 2 / (args.G * rho0))) if args.G > 0 else float("inf")
    print(f"[{args.tag}] scheme={args.scheme} pp_flux={args.pp_flux} "
          f"G={args.G} beta={args.beta} a={a:.3f} P0={P0:.4g} B0={B0:.4g} "
          f"lam_J/L={lam_J:.3f} cfl={args.cfl}", flush=True)

    t0 = walltime.time()
    result = time_integration(initial_state, config, params, rv)
    states = result.states
    time_points = np.asarray(result.time_points)
    total_energy = np.asarray(result.total_energy)
    states.block_until_ready()
    print(f"[{args.tag}] wall {walltime.time()-t0:.1f}s snaps={states.shape[0]}",
          flush=True)

    di = rv.density_index
    vx_i, vy_i, vz_i = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z
    pi = rv.pressure_index

    nsny = states.shape[0]
    Ms_t = np.zeros(nsny); rhomax_t = np.zeros(nsny); pmin_t = np.zeros(nsny)
    finite_t = np.zeros(nsny, dtype=bool)
    first_bad = -1
    for s in range(nsny):
        st = states[s]
        ok = bool(jnp.all(jnp.isfinite(st)))
        finite_t[s] = ok
        if first_bad < 0 and not ok:
            first_bad = s
        rho = st[di]
        v2 = st[vx_i] ** 2 + st[vy_i] ** 2 + st[vz_i] ** 2
        P = st[pi]
        vrms = float(jnp.sqrt(jnp.mean(jnp.nan_to_num(v2))))
        Ms_t[s] = vrms / a
        rhomax_t[s] = float(jnp.nanmax(rho))
        pmin_t[s] = float(jnp.nanmin(P))

    t_over_tc = time_points / t_cross
    bad_str = (f"first_bad_snap={first_bad} @t/tc={t_over_tc[first_bad]:.3f}"
               if first_bad >= 0 else "ALL FINITE")
    print(f"[{args.tag}] {bad_str}", flush=True)
    print(f"[{args.tag}] rho_max final={rhomax_t[max(0, (first_bad-1) if first_bad>0 else nsny-1)]:.2f} "
          f"p_min(min over run)={np.nanmin(pmin_t):.3e}", flush=True)
    print(f"[{args.tag}] M_s(t/tc): " +
          " ".join(f"{tc:.2f}:{m:.2f}" for tc, m in zip(t_over_tc[::6], Ms_t[::6])),
          flush=True)

    out = os.path.join(args.outdir, f"mhdgrav_{args.tag}.npz")
    np.savez(out, tag=args.tag, scheme=args.scheme, pp_flux=args.pp_flux,
             G=args.G,
             beta=args.beta, N=args.N, a=a, B0=B0, lam_J=lam_J,
             time_points=time_points, t_over_tc=t_over_tc,
             total_energy=total_energy, Ms_t=Ms_t, rhomax_t=rhomax_t,
             pmin_t=pmin_t, finite_t=finite_t, first_bad=first_bad)
    print(f"[{args.tag}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
