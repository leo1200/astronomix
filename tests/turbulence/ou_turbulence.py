"""
Driven isothermal MHD turbulence with Ornstein-Uhlenbeck (temporally
correlated) forcing, reproducing the turbulent-flow test of the HOW-MHD
paper (https://arxiv.org/pdf/2304.04360) but with OU forcing instead of the
white-in-time, constant-energy-injection driving used there.

We characterise the runs by two dimensionless numbers
  - sonic Mach number     M_s = v_rms / c_s
  - Alfven Mach number    M_A = v_rms / c_A,  c_A = ||B|| / sqrt(rho)  (mu0 = 1)

With rho_0 = 1 and the OU forcing tuned so that the stationary v_rms ~ 1, the
targets are set by
  - c_s = 1 / M_s      (isothermal sound speed)
  - B_0 = 1 / M_A      (uniform guide field along z)

The actual M_s, M_A measured in the stationary state will differ somewhat from
the targets; the goal is to cover the interesting (M_s, M_A) regimes.

One (M_s, M_A) case per invocation so cases can be run independently / in
parallel and calibrated. Diagnostics + slices + spectra are written to an npz;
make_fig_ou.py builds the comparison figures.

Uses the PALLAS backend throughout (isothermal-MHD WENO Pallas kernel) and
autocvd to grab a single genuinely-free GPU.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse
import os
import time as walltime

import numpy as np
import jax.numpy as jnp

from astronomix._fluid_equations._equations import get_absolute_velocity
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    ISOTHERMAL,
    PALLAS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
    SimulationConfig,
    SnapshotSettings,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.analysis_helpers.energy_spectrum import (
    get_kinetic_energy_spectrum,
    get_magnetic_energy_spectrum,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=float, required=True, help="target sonic Mach number")
    p.add_argument("--ma", type=float, required=True, help="target Alfven Mach number")
    p.add_argument("--N", type=int, default=64, help="cells per dimension")
    p.add_argument("--tcross", type=float, default=5.0, help="t_end in crossing times")
    p.add_argument("--F0", type=float, default=1.0, help="OU forcing amplitude")
    p.add_argument("--tau", type=float, default=0.5, help="OU correlation time")
    p.add_argument("--kf", type=float, default=2.0 * np.pi * 2.0,
                   help="OU forcing peak wavenumber (physical, 2*pi*n/L)")
    p.add_argument("--nsnap", type=int, default=60, help="number of snapshots")
    p.add_argument("--cfl", type=float, default=1.5)
    p.add_argument("--protect", action="store_true",
                   help="enable positivity + vacuum protection (supersonic)")
    p.add_argument("--outdir", type=str, default="data")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    M_s_aim = args.ms
    M_A_aim = args.ma
    sound_speed = 1.0 / M_s_aim
    B_0 = 1.0 / M_A_aim

    supersonic = M_s_aim >= 1.0
    protect = args.protect or supersonic

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=ISOTHERMAL,
        backend=PALLAS,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
        memory_analysis=False,
        progress_bar=True,
        dimensionality=3,
        num_cells=args.N,
        positivity_config=PositivityConfig(default_positivity_protection=protect),
        box_size=1.0,
        mhd=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
        ),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True,
            ou_forcing=True,
            vacuum_protection=protect,
        ),
        return_snapshots=True,
        num_snapshots=args.nsnap,
        snapshot_settings=SnapshotSettings(return_states=True),
    )

    t_cross = (config.box_size / 2) / 1.0  # assumes v_rms ~ 1

    params = SimulationParams(
        C_cfl=args.cfl,
        isothermal_sound_speed=sound_speed,
        t_end=args.tcross * t_cross,
        turbulent_forcing_params=TurbulentForcingParams(
            forcing_amplitude=args.F0,
            correlation_time=args.tau,
            forcing_wavenumber=args.kf,
            protection_density_threshold=0.02,
            protection_max_velocity=50.0,
        ),
        minimum_density=0.02,
    )

    registered_variables = get_registered_variables(config)

    density = jnp.ones((args.N, args.N, args.N), dtype=jnp.float32)
    zero = jnp.zeros_like(density)
    magnetic_field_z = jnp.ones_like(density) * B_0
    bxb, byb, bzb = initialize_interface_fields(zero, zero, magnetic_field_z)
    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
        magnetic_field_x=zero,
        magnetic_field_y=zero,
        magnetic_field_z=magnetic_field_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
    )

    config = finalize_config(config, initial_state.shape)

    tag = args.tag or f"Ms{M_s_aim}_MA{M_A_aim}_N{args.N}"
    print(f"[{tag}] starting: M_s_aim={M_s_aim}, M_A_aim={M_A_aim}, c_s={sound_speed:.3f}, "
          f"B_0={B_0:.4f}, F0={args.F0}, tau={args.tau}, kf={args.kf:.2f}, protect={protect}")

    t0 = walltime.time()
    result = time_integration(initial_state, config, params, registered_variables)
    states = result.states
    time_points = np.asarray(result.time_points)
    states.block_until_ready()
    print(f"[{tag}] integration wall time: {walltime.time() - t0:.1f} s, "
          f"snapshots={states.shape[0]}")

    di = registered_variables.density_index
    vx_i = registered_variables.velocity_index.x
    vy_i = registered_variables.velocity_index.y
    vz_i = registered_variables.velocity_index.z
    bx_i = registered_variables.magnetic_index.x
    by_i = registered_variables.magnetic_index.y
    bz_i = registered_variables.magnetic_index.z

    # v_rms(t) and Mach numbers over time (stationarity diagnostic)
    rho_t = states[:, di]
    vmag2_t = states[:, vx_i] ** 2 + states[:, vy_i] ** 2 + states[:, vz_i] ** 2
    v_rms_t = np.asarray(jnp.sqrt(jnp.mean(vmag2_t, axis=(1, 2, 3))))
    bmag2_t = states[:, bx_i] ** 2 + states[:, by_i] ** 2 + states[:, bz_i] ** 2
    cA_t = np.asarray(jnp.sqrt(jnp.mean(bmag2_t / rho_t, axis=(1, 2, 3))))
    Ms_t = v_rms_t / sound_speed
    MA_t = v_rms_t / np.maximum(cA_t, 1e-12)

    # stationary values = mean over the last third
    n3 = max(1, len(v_rms_t) // 3)
    v_rms_stat = float(np.mean(v_rms_t[-n3:]))
    Ms_stat = float(np.mean(Ms_t[-n3:]))
    MA_stat = float(np.mean(MA_t[-n3:]))
    rho_min = float(jnp.min(rho_t[-1]))
    rho_max = float(jnp.max(rho_t[-1]))
    finite = bool(np.all(np.isfinite(v_rms_t)))

    print(f"[{tag}] stationary v_rms={v_rms_stat:.3f}  M_s={Ms_stat:.3f}  M_A={MA_stat:.3f}  "
          f"rho[min,max]=[{rho_min:.3f},{rho_max:.3f}]  finite={finite}")

    # final-state spectra
    final = states[-1]
    k, kin_spec = get_kinetic_energy_spectrum(
        final[vx_i], final[vy_i], final[vz_i], final[di]
    )
    _, mag_spec = get_magnetic_energy_spectrum(final[bx_i], final[by_i], final[bz_i])

    # mid-plane slices (final): density, |v|, |B|
    z = args.N // 2
    dens_slice = np.asarray(final[di][:, :, z])
    vmag_slice = np.asarray(jnp.sqrt(final[vx_i] ** 2 + final[vy_i] ** 2 + final[vz_i] ** 2)[:, :, z])
    bmag_slice = np.asarray(jnp.sqrt(final[bx_i] ** 2 + final[by_i] ** 2 + final[bz_i] ** 2)[:, :, z])

    out = os.path.join(args.outdir, f"ou_{tag}.npz")
    np.savez(
        out,
        ms_aim=M_s_aim, ma_aim=M_A_aim, N=args.N, sound_speed=sound_speed, B_0=B_0,
        F0=args.F0, tau=args.tau, kf=args.kf, tcross=args.tcross, cfl=args.cfl, protect=protect,
        time_points=time_points, v_rms_t=v_rms_t, Ms_t=Ms_t, MA_t=MA_t,
        v_rms_stat=v_rms_stat, Ms_stat=Ms_stat, MA_stat=MA_stat,
        rho_min=rho_min, rho_max=rho_max, finite=finite,
        k=np.asarray(k), kinetic_spectrum=np.asarray(kin_spec), magnetic_spectrum=np.asarray(mag_spec),
        dens_slice=dens_slice, vmag_slice=vmag_slice, bmag_slice=bmag_slice,
    )
    print(f"[{tag}] wrote {out}")


if __name__ == "__main__":
    main()
