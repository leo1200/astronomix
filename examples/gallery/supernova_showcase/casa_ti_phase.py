"""Astronomix twin of the AthenaK thermal-instability box (Guo-Kim-Stone).

Runs the ti_box_128.athinput setup with the high-order FD/WENO methodology:
uniform n_p = 1 cm^-3 medium at T = 471 K in the GKS unit system
(pc / [mu m_u per cm^-3] / Myr), mainline-AthenaK-exact ISM cooling +
heating (KI2002 + Schure SPEX + CGOLS, hrate = 5e-26 erg/s; rate function
cross-validated to 0.05%), and solenoidal white-in-time driving matched to
the reference run's measured v_rms. The box heats to the warm equilibrium
branch (~6600 K), the driving seeds perturbations, and thermal instability
condenses the cold (~180 K) phase over tens of Myr.

Compare against /export/data/lstorcks/athenak_ref/ti128_prod: v_rms(t),
mean temperature, phase fractions and the density PDF at matched times.
"""

# ==== GPU selection (before jax import; pq presets CUDA_VISIBLE_DEVICES) ====
import os
import sys
_NUM_GPUS = 1
if "--gpus" in sys.argv:
    _NUM_GPUS = int(sys.argv[sys.argv.index("--gpus") + 1])
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=_NUM_GPUS)
# ruff: noqa: E402
# =======================

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from astropy import units as u
import astropy.constants as const

from astronomix import (
    SimulationParams,
    SnapshotSettings,
    get_registered_variables,
    finalize_config,
    construct_primitive_state,
    time_integration,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)
from astronomix.units.unit_helpers import CodeUnits

from _common import GAMMA, make_fd_config, ism_ti_cooling_setup
from cassiopeia_realistic import FIGURES_DIR

# ↓───────────────────────────────────────────────────────────────────────↓
# GKS unit system (ti.athinput <units>): length pc, time Myr, and the mass
# unit chosen so rho_code = 1 is one particle per cm^3 at mu = 0.618.
# code velocity = pc/Myr = 0.978 km/s; temp_unit = 71.08 K (T_code = p/rho).
# ↑───────────────────────────────────────────────────────────────────────↑
MU_ATH = 0.618
BOX_SIZE = 64.0                  # pc
T_INIT_K = 471.233476            # GKS temp (Kelvin)
HRATE_CGS = 5.0e-26              # erg/s per particle
DFLOOR, PFLOOR = 1e-4, 1e-2      # athinput floors (code units)
SEED = 7


def gks_code_units():
    code_length = 1.0 * u.pc
    code_velocity = (1.0 * u.pc / (1.0 * u.Myr)).to(u.km / u.s)
    code_mass = (MU_ATH * const.u * (1.0 / u.cm ** 3) * (1.0 * u.pc) ** 3).to(u.g)
    return CodeUnits(code_length, code_mass, code_velocity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--t-end", type=float, default=50.0, help="Myr")
    ap.add_argument("--vrms", type=float, default=8.4,
                    help="target driven rms velocity (km/s); einj = M v^3 / L_f")
    ap.add_argument("--einj", type=float, default=0.0, help="override (code, box total)")
    ap.add_argument("--ou", action="store_true",
                    help="Ornstein-Uhlenbeck forcing (AthenaK's driver type: "
                         "tcorr-correlated, band-peaked) instead of white")
    ap.add_argument("--f0", type=float, default=17.0,
                    help="OU forcing amplitude (code acceleration; calibrate "
                         "to the target v_rms)")
    ap.add_argument("--tcorr", type=float, default=0.5, help="OU tau (Myr)")
    ap.add_argument("--kf", type=float, default=0.589,
                    help="OU peak wavenumber (2 pi n/L; n=6 for the k=4-8 band)")
    ap.add_argument("--limiter-alpha", type=float, default=0.0)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--weno-z", action="store_true",
                    help="WENO-Z nonlinear weights (scale-invariant, keeps the "
                         "optimal linear weights at smooth extrema)")
    ap.add_argument("--weno-eps-rel", type=float, default=0.0,
                    help="relative WENO epsilon: add eps_rel*(amx*|q|)^2 so the "
                         "smoothness indicators are compared to the local data "
                         "scale instead of an absolute 1e-7")
    ap.add_argument("--native", action="store_true",
                    help="force the NATIVE_JAX backend (required by "
                         "--weno-eps-rel; also use as the backend control). "
                         "--weno-z now runs on Pallas too.")
    ap.add_argument("--band", type=int, nargs=2, default=None,
                    metavar=("NLOW", "NHIGH"),
                    help="AthenaK-style discrete driving band in mode number "
                         "(ti.athinput uses 16 32; the gate deck uses 4 8)")
    ap.add_argument("--expo", type=float, default=5.0/3.0,
                    help="driving power-law exponent (AthenaK <turb_driving> expo)")
    ap.add_argument("--ou-exact", action="store_true",
                    help="normalise the OU field to inject exactly einj*dt "
                         "(AthenaK dedt) instead of constant --f0")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="PRNG seed (TI onset is stochastic: vary for an ensemble)")
    ap.add_argument("--restart", type=str, default=None,
                    help="npz (rho/press/vx/vy/vz) to start from instead of the "
                         "uniform 471 K box -- e.g. a CONDENSED AthenaK state, "
                         "to test whether astronomix SUSTAINS a two-phase "
                         "medium it cannot itself nucleate")
    ap.add_argument("--conduction-order", type=int, default=2, choices=(2, 4),
                    help="formal order of the conduction stencils (4 = "
                         "consistent with the 5th-order hydro)")
    ap.add_argument("--explicit-cooling", action="store_true",
                    help="AthenaK-style single forward-Euler cooling source per "
                         "stage (dt limited to the local thermal time) instead "
                         "of the implicit fixed point")
    ap.add_argument("--no-cooling", action="store_true",
                    help="disable radiative cooling+heating (profiling control)")
    ap.add_argument("--conduction", type=float, default=0.0,
                    help="isotropic thermal DIFFUSIVITY alpha (AthenaK "
                         "<hydro> alpha_iso; GKS ti.athinput uses 0.24 = "
                         "1e7 cgs at n=1). Sets a resolved Field length so "
                         "TI clump survival is physics, not grid dissipation")
    ap.add_argument("--no-positivity", action="store_true",
                    help="run WITHOUT the showcase positivity stack (no "
                         "floors, no preserving_flux LLF blending — the "
                         "blending diffuses forming cold clumps; transonic "
                         "TI does not need blast armour). NOTE: NaN'd at "
                         "x32/64^3 — floors ARE needed; see --no-ppflux")
    ap.add_argument("--n0", type=float, default=10.0,
                    help="mean number density (cm^-3). GKS use n0 = 10, which "
                         "is the UNSTABLE thermal equilibrium at T = 471 K "
                         "(the unstable branch spans n = 2.68-19.3); n0 = 1 "
                         "instead sits on the STABLE warm branch and never "
                         "condenses")
    ap.add_argument("--vpert", type=float, default=0.0,
                    help="one-time divergence-free Kolmogorov velocity seed, "
                         "rms in km/s (GKS use sigma_v ~ 1 km/s over "
                         "3.2 pc < lambda < 16 pc, applied ONCE then left to "
                         "decay -- this is what triggers TI at the unstable "
                         "equilibrium, no driving needed)")
    ap.add_argument("--rho-floor", type=float, default=0.0,
                    help="override minimum_density (AthenaK ti.athinput uses "
                         "1e-4; paper_turbulence uses 0.02). 0 = use the default")
    ap.add_argument("--tfloor-K", type=float, default=0.0,
                    help="hard Athena-style temperature floor in K (p >= rho*T~), "
                         "applied per step AND per stage like AthenaK's EOS "
                         "tfloor; ti.athinput uses 5 K. 0 = off")
    ap.add_argument("--pos-mode", choices=("floor","redist","conservative"),
                    default="floor",
                    help="per-stage/per-step positivity mode (HARD_FLOOR / "
                         "REDISTRIBUTE / CONSERVATIVE). The pp flux limiter "
                         "stays ON in all of them.")
    ap.add_argument("--deepvoid", action="store_true",
                    help="also enable the deep-void LLF blend (high-Mach "
                         "void overshoot guard)")
    ap.add_argument("--turb-positivity", action="store_true",
                    help="use the paper_turbulence.py high-Mach recipe instead "
                         "of the blast recipe: prot vacuum_protection + "
                         "hard floor per stage/step + vacuum_rest, NO "
                         "preserving_flux, density floor 0.02 (validated to "
                         "M_turb ~ 10)")
    ap.add_argument("--hllc-blend", action="store_true",
                    help="blend the positivity limiter toward HLLC instead of "
                         "first-order LLF (preserves contacts, so cold "
                         "condensations survive the blend)")
    ap.add_argument("--no-ppflux", action="store_true",
                    help="keep the floors/nan_safe/vacuum_rest but disable "
                         "the positivity-preserving flux limiter (its LLF "
                         "blending at extrema diffuses forming cold clumps)")
    ap.add_argument("--nsnap", type=int, default=26)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--save-state", type=str,
                    default="/export/data/lstorcks/supernova_showcase/casa_ti_n128.npz")
    args = ap.parse_args()

    print(f"[casa-ti] devices: {jax.devices()}  x64={jax.config.jax_enable_x64}")

    cu = gks_code_units()
    cooling_config, cooling_params = (None, None)
    if not args.no_cooling:
      cooling_config, cooling_params = ism_ti_cooling_setup(
        cu, hrate_cgs=HRATE_CGS, mu_athena=MU_ATH, floor_temperature_K=10.0,
        hydrogen_mass_fraction=0.7, metal_mass_fraction=0.02,
        resolution_limiter_alpha=args.limiter_alpha,
        explicit=args.explicit_cooling,
      )
    snaps = SnapshotSettings(
        return_states=False, return_final_state=True,
        return_total_mass=True, return_total_energy=True,
        return_internal_energy=True, return_kinetic_energy=True,
    )
    extra = {}
    if args.turb_positivity:
        from astronomix.option_classes.simulation_config import (
            PositivityConfig, POSITIVITY_HARD_FLOOR, POSITIVITY_REDISTRIBUTE,
            POSITIVITY_CONSERVATIVE)
        _M = {"floor": POSITIVITY_HARD_FLOOR, "redist": POSITIVITY_REDISTRIBUTE,
              "conservative": POSITIVITY_CONSERVATIVE}
        extra["positivity_config"] = PositivityConfig(
            default_positivity_protection=True,
            per_stage_mode=_M[args.pos_mode],
            per_step_mode=_M[args.pos_mode],
            vacuum_rest=True,
            preserving_flux=True,          # pp limiter ON (user directive)
            deepvoid_blend=args.deepvoid,
            per_step_specific_floor=args.tfloor_K > 0.0,
            per_stage_specific_floor=args.tfloor_K > 0.0,
            nan_safe=True,
        )
    if args.native or args.weno_eps_rel > 0.0:
        from astronomix.option_classes.simulation_config import (
            BackendConfig, NATIVE_JAX)
        extra["backend_config"] = BackendConfig(backend=NATIVE_JAX)
    if args.weno_z:
        extra["weno_z"] = True
    if args.weno_eps_rel > 0.0:
        extra["weno_epsilon_relative"] = args.weno_eps_rel
    if args.no_positivity:
        from astronomix.option_classes.simulation_config import PositivityConfig
        extra["positivity_config"] = PositivityConfig()
    elif args.hllc_blend:
        from _common import fd_positivity as _fdp
        extra["positivity_config"] = _fdp()._replace(blend_fallback_hllc=True)
    elif args.no_ppflux:
        from astronomix.option_classes.simulation_config import (
            PositivityConfig, POSITIVITY_HARD_FLOOR)
        extra["positivity_config"] = PositivityConfig(
            per_stage_mode=POSITIVITY_HARD_FLOOR,
            per_step_mode=POSITIVITY_HARD_FLOOR,
            preserving_flux=False,
            nan_safe=True,
            vacuum_rest=True,
        )
    config = make_fd_config(
        BOX_SIZE, args.n, mhd=False,
        cooling_config=cooling_config,
        snapshot_settings=snaps, num_snapshots=args.nsnap,
        random_seed=args.seed,
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True, ou_forcing=args.ou,
            ou_exact_injection=args.ou_exact,
            vacuum_protection=args.turb_positivity,
            banded_spectrum=args.band is not None),
        thermal_conduction=args.conduction > 0.0,
        conduction_density_weighted=True,
        conduction_order=args.conduction_order,
        **extra,
    )
    rv = get_registered_variables(config)

    N = args.n
    shape = (N, N, N)
    rho = jnp.full(shape, args.n0)
    restart = np.load(args.restart) if args.restart else None
    if restart is not None:
        assert int(restart["num_cells"]) == N, "restart grid must match --n"
        rho = jnp.asarray(restart["rho"])
    # p = rho * T_code with T_code = T_K / temp_unit; temp_unit = mu m_u v_u^2/k_B
    temp_unit_K = float((MU_ATH * const.u * (1.0 * cu.code_velocity) ** 2
                         / const.k_B).to(u.K).value)
    p0 = args.n0 * T_INIT_K / temp_unit_K      # p = n * T~ (T~ = T_K/temp_unit)
    p = jnp.full(shape, p0)
    zeros = jnp.zeros_like(rho)
    vx = vy = vz = zeros
    if args.vpert > 0.0 and restart is None:
        # one-time divergence-free Kolmogorov-like seed (GKS: sigma_v ~ 1 km/s
        # over 3.2 pc < lambda < 16 pc, applied ONCE and left to decay). At the
        # unstable equilibrium this is all TI needs -- no sustained driving.
        kx = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=BOX_SIZE / N)
        KX = kx.reshape(N, 1, 1); KY = kx.reshape(1, N, 1); KZ = kx.reshape(1, 1, N)
        k2 = KX ** 2 + KY ** 2 + KZ ** 2
        kk = jnp.sqrt(k2)
        k_lo = 2.0 * jnp.pi / 16.0          # lambda = 16 pc
        k_hi = 2.0 * jnp.pi / 3.2           # lambda = 3.2 pc
        band = (kk >= k_lo) & (kk <= k_hi)
        amp = jnp.where(band, jnp.where(kk > 0, kk ** (-11.0 / 6.0), 0.0), 0.0)
        keys = jax.random.split(jax.random.PRNGKey(args.seed), 6)
        comps = []
        for c in range(3):
            noise = (jax.random.normal(keys[2 * c], (N, N, N))
                     + 1j * jax.random.normal(keys[2 * c + 1], (N, N, N)))
            comps.append(amp * noise)
        cwx, cwy, cwz = comps
        k2s = jnp.where(k2 == 0.0, 1.0, k2)
        div = (KX * cwx + KY * cwy + KZ * cwz) / k2s      # solenoidal projection
        cwx, cwy, cwz = cwx - KX * div, cwy - KY * div, cwz - KZ * div
        wx = jnp.real(jnp.fft.ifftn(cwx))
        wy = jnp.real(jnp.fft.ifftn(cwy))
        wz = jnp.real(jnp.fft.ifftn(cwz))
        norm = jnp.sqrt(jnp.mean(wx ** 2 + wy ** 2 + wz ** 2) + 1e-30)
        sig = float((args.vpert * u.km / u.s).to(cu.code_velocity).value)
        vx, vy, vz = sig * wx / norm, sig * wy / norm, sig * wz / norm
        print(f"[casa-ti] velocity seed: sigma_v={args.vpert} km/s ({sig:.4f} code) "
              f"over lambda 3.2-16 pc, solenoidal, one-time (decaying)")
    if restart is not None:
        p = jnp.asarray(restart["press"])
        vx = jnp.asarray(restart["vx"]); vy = jnp.asarray(restart["vy"])
        vz = jnp.asarray(restart["vz"])
        T_r = np.asarray(p / rho) * temp_unit_K
        print(f"[casa-ti] restart {args.restart}: T[{T_r.min():.0f},{T_r.max():.3g}] K  "
              f"f_cold={float(np.asarray(rho)[T_r < 184].sum() / np.asarray(rho).sum()):.4f}")
    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=rho, velocity_x=vx, velocity_y=vy, velocity_z=vz,
        gas_pressure=p,
    )
    config = finalize_config(config, state.shape)

    v_target = float((args.vrms * u.km / u.s).to(cu.code_velocity).value)
    L_f = BOX_SIZE / 2.0
    M_box = float(jnp.sum(rho)) * (BOX_SIZE / N) ** 3
    einj = args.einj if args.einj > 0 else M_box * v_target ** 3 / L_f
    params = SimulationParams(
        gamma=GAMMA, C_cfl=args.cfl, t_end=args.t_end,
        minimum_density=(args.rho_floor if args.rho_floor > 0.0
                         else (0.02 if args.turb_positivity else DFLOOR)),
        minimum_pressure=PFLOOR,
        # Athena tfloor: p >= rho * (k T_floor / (mu m_u)) in code units
        minimum_specific_pressure=(args.tfloor_K / temp_unit_K
                                   if args.tfloor_K > 0.0 else 0.0),
        thermal_conductivity=args.conduction,
        cooling_params=cooling_params,
        turbulent_forcing_params=TurbulentForcingParams(
            energy_injection_rate=einj,
            correlation_time=args.tcorr,
            forcing_wavenumber=args.kf,
            forcing_amplitude=args.f0,
            forcing_nlow=(args.band[0] if args.band else 0),
            forcing_nhigh=(args.band[1] if args.band else 0),
            forcing_expo=args.expo,
            protection_density_threshold=0.02,
            protection_max_velocity=50.0,
        ),
    )
    print(f"[casa-ti] N={N} box={BOX_SIZE} pc  T0={T_INIT_K:.0f} K (p0={p0:.4f}) "
          f"temp_unit={temp_unit_K:.2f} K")
    print(f"[casa-ti] target v_rms={args.vrms} km/s ({v_target:.3f} code) "
          f"einj={einj:.3e}  t_end={args.t_end} Myr  limiter={args.limiter_alpha} "
          f"alpha_cond={args.conduction} (chi={(GAMMA-1)*args.conduction:.3f} pc^2/Myr)")

    result = time_integration(state, config, params, rv, sharding=None)
    jax.block_until_ready(result)

    ke = np.asarray(result.kinetic_energy)
    tm = np.asarray(result.total_mass)
    tp = np.asarray(result.time_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        vrms = np.sqrt(np.maximum(2.0 * ke / np.maximum(tm, 1e-30), 0.0))
    vrms_kms = vrms * float((1.0 * cu.code_velocity).to(u.km / u.s).value)
    print(f"[casa-ti] time_points:  {np.array2string(tp, precision=1, max_line_width=220)}")
    print(f"[casa-ti] total_energy: {np.array2string(np.asarray(result.total_energy), precision=4, max_line_width=220)}")
    print(f"[casa-ti] v_rms [km/s]: {np.array2string(vrms_kms, precision=2, max_line_width=220)}")

    fs = np.asarray(result.final_state)
    rho_f = fs[rv.density_index]
    p_f = fs[rv.pressure_index]
    T_K = (p_f / rho_f) * temp_unit_K

    # GKS phase cuts (ti.athinput): cold < 184 K < unstable < 5050 K < warm
    cold = (T_K < 184.0); warm = (T_K > 184.0) & (T_K < 5050.0)
    hot = T_K >= 5050.0
    mcold = rho_f[cold].sum(); mtot = rho_f.sum()
    print(f"[casa-ti] final T_K[{T_K.min():.0f},{T_K.max():.3g}]  "
          f"rho[{rho_f.min():.3e},{rho_f.max():.3e}]")
    print(f"[casa-ti] mass fractions: cold(T<184) {mcold/mtot:.3f}  "
          f"unstable {rho_f[warm].sum()/mtot:.3f}  warm/hot {rho_f[hot].sum()/mtot:.3f}")
    print(f"[casa-ti] volume fractions: cold {cold.mean():.3f}  "
          f"unstable {warm.mean():.3f}  warm/hot {hot.mean():.3f}")

    np.savez_compressed(
        args.save_state, rho=rho_f, press=p_f,
        vx=fs[rv.velocity_index.x], vy=fs[rv.velocity_index.y],
        vz=fs[rv.velocity_index.z],
        box=float(BOX_SIZE), age=args.t_end, num_cells=N)
    print(f"[casa-ti] saved {args.save_state}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    mid = N // 2
    im0 = axes[0].imshow(np.log10(rho_f[:, :, mid]).T, origin="lower", cmap="magma")
    axes[0].set_title("log10 rho (midplane)"); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.log10(T_K[:, :, mid]).T, origin="lower", cmap="coolwarm")
    axes[1].set_title("log10 T [K]"); plt.colorbar(im1, ax=axes[1])
    axes[2].hist2d(np.log10(rho_f).ravel(), np.log10(T_K).ravel(), bins=120,
                   norm=matplotlib.colors.LogNorm())
    axes[2].set_xlabel("log10 n"); axes[2].set_ylabel("log10 T [K]")
    axes[2].set_title("phase diagram")
    out = FIGURES_DIR / "casa_ti_phase.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[casa-ti] saved {out}")


if __name__ == "__main__":
    main()
