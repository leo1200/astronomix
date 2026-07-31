"""Phase A of the two-phase Cas A pipeline: a turbulently driven CSM box.

Drives solenoidal turbulence (white-in-time, exact energy-injection rate,
k ~ 2 -> ~3.5 pc driving scale) in a UNIFORM periodic medium with REAL
radiative cooling, so density structure develops self-consistently: every
clump is created by the solver with correlated velocity and pressure,
instead of being imposed as a cold knife-edge the dynamics would never
produce (the failure mode of every 512^3 substructure gate). Phase B
(``cassiopeia_realistic --ambient-from``) modulates this saturated
turbulence onto the smooth wind + shell profile (density and pressure
scaled together, preserving the temperature field and the local
correlations) and injects the ejecta on top.

Why uniform and not the wind+shell profile directly: any meaningful
driving time bulk-advects gas by ~ the forcing scale, which smears the
ordered pc-scale CSM (an N=64 probe fully dispersed the shell in two
turnovers). Only the SMALL-scale structure ever destabilized the blast --
the smooth profile never did -- so we let the solver make the small
scales and keep the ordered background analytic.

The driving phase is transonic (target v_rms ~ 15 km/s vs c_s(1e4 K) ~ 12
km/s), so it needs none of the blast-phase armour: no dual energy, no
temperature floor, full cooling (the limiter is OFF by default here -- at
Mach ~ 1 the isothermal compressions it would suppress are exactly the
structure we want, and they are bounded at the modest isothermal jump).
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
    get_helper_data,
    finalize_config,
    construct_primitive_state,
    time_integration,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)

from _common import (
    GAMMA,
    snr_code_units,
    make_fd_config,
    temperature_K,
    schure_cooling_setup,
)

# medium parameters shared with the blast script (import, don't duplicate)
from cassiopeia_realistic import (
    BOX_SIZE,
    N_C,
    WIND_TEMPERATURE,
    MASS_PER_NUCLEUS,
    SEED,
    FIGURES_DIR,
)


def build(num_cells, t_end, einj, uniform_nc=5.0, num_snapshots=21,
          limiter_alpha=0.0):
    code_units = snr_code_units()
    cooling_config, cooling_params = schure_cooling_setup(
        code_units, floor_temperature_K=1e4,
        hydrogen_mass_fraction=1.0 - 0.28 - 0.02, metal_mass_fraction=0.02,
        resolution_limiter_alpha=limiter_alpha,
    )
    snaps = SnapshotSettings(
        return_states=False, return_final_state=True,
        return_total_mass=True, return_total_energy=True,
        return_internal_energy=True, return_kinetic_energy=True,
    )
    config = make_fd_config(
        BOX_SIZE, num_cells, mhd=False,
        cooling_config=cooling_config,
        snapshot_settings=snaps, num_snapshots=num_snapshots,
        random_seed=SEED,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=True),
    )
    registered_variables = get_registered_variables(config)

    # ---- UNIFORM periodic medium at 1e4 K (structure comes from driving;
    # phase B modulates it onto the analytic wind + shell profile) ----
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(code_units.code_density).value)
    p_per_n = float((const.k_B * WIND_TEMPERATURE / u.cm ** 3).to(code_units.code_pressure).value)

    shape = (num_cells, num_cells, num_cells)
    rho = jnp.full(shape, uniform_nc * rho_per_n)
    p = jnp.full(shape, uniform_nc * p_per_n)   # T = 1e4 K everywhere

    zeros = jnp.zeros_like(rho)
    initial_state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho, velocity_x=zeros, velocity_y=zeros, velocity_z=zeros,
        gas_pressure=p,
    )
    config = finalize_config(config, initial_state.shape)

    dfloor = float(N_C * rho_per_n) * 1e-3
    pfloor = float(N_C * p_per_n) * 1e-2
    params = SimulationParams(
        gamma=GAMMA, C_cfl=0.3, t_end=t_end,
        minimum_density=dfloor, minimum_pressure=pfloor,
        cooling_params=cooling_params,
        turbulent_forcing_params=TurbulentForcingParams(
            energy_injection_rate=einj,
        ),
    )

    dx = BOX_SIZE / num_cells
    M_box = float(jnp.sum(rho)) * dx ** 3
    return initial_state, config, params, registered_variables, code_units, M_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="cells per axis")
    ap.add_argument("--vrms", type=float, default=15.0,
                    help="target saturated rms velocity (km/s); sets the "
                         "energy-injection rate eps = M v^3 / L_f unless "
                         "--einj overrides")
    ap.add_argument("--einj", type=float, default=0.0,
                    help="energy injection rate (code energy / code time, "
                         "box total); 0 = derive from --vrms")
    ap.add_argument("--t-end", type=float, default=0.0,
                    help="driving time (code, ~978 yr each); 0 = two "
                         "turnovers of the k=2 driving scale at --vrms")
    ap.add_argument("--turnovers", type=float, default=2.0,
                    help="driving duration in eddy turnovers (with --t-end 0)")
    ap.add_argument("--limiter-alpha", type=float, default=0.0,
                    help="cooling-resolution limiter (0 = full cooling; the "
                         "transonic driving phase is safe without it)")
    ap.add_argument("--uniform-nc", type=float, default=5.0,
                    help="uniform medium number density (cm^-3); the absolute "
                         "value only matters for the cooling response, phase B "
                         "renormalises to the profile mean")
    ap.add_argument("--nsnap", type=int, default=21)
    ap.add_argument("--gpus", type=int, default=1,
                    help="consumed by the autocvd preamble (single-GPU run)")
    ap.add_argument("--save-state", type=str,
                    default="/export/data/lstorcks/supernova_showcase/casa_turb_medium.npz",
                    help="npz path for the driven medium (phase B input)")
    args = ap.parse_args()

    print(f"[casa-turb] devices: {jax.devices()}  x64={jax.config.jax_enable_x64}")

    cu = snr_code_units()
    v_target = float((args.vrms * u.km / u.s).to(cu.code_velocity).value)
    L_f = BOX_SIZE / 2.0                       # forcing spectrum peaks at k ~ 2

    # provisional mass for the eps estimate (cheap rebuild below reuses it)
    state, config, params, rv, cu, M_box = build(
        args.n, 1.0, 1.0, uniform_nc=args.uniform_nc,
        num_snapshots=args.nsnap, limiter_alpha=args.limiter_alpha,
    )
    einj = args.einj if args.einj > 0 else M_box * v_target ** 3 / L_f
    t_turn = L_f / v_target
    t_end = args.t_end if args.t_end > 0 else args.turnovers * t_turn
    params = params._replace(
        t_end=t_end,
        turbulent_forcing_params=params.turbulent_forcing_params._replace(
            energy_injection_rate=einj,
        ),
    )

    print(f"[casa-turb] N={args.n} target v_rms={args.vrms} km/s "
          f"({v_target:.4f} code)  einj={einj:.3e}  M_box={M_box:.1f}")
    print(f"[casa-turb] t_turn={t_turn:.0f}  t_end={t_end:.0f} code "
          f"(~{t_end * 978:.0f} yr)  limiter_alpha={args.limiter_alpha}")

    snaps = time_integration(state, config, params, rv, sharding=None)
    jax.block_until_ready(snaps)

    # v_rms(t) from the kinetic-energy series: v_rms = sqrt(2 KE / M)
    ke = np.asarray(snaps.kinetic_energy)
    tm = np.asarray(snaps.total_mass)
    tp = np.asarray(snaps.time_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        vrms_kms = np.sqrt(np.maximum(2.0 * ke / np.maximum(tm, 1e-30), 0.0))
    vrms_kms = vrms_kms * float((1.0 * cu.code_velocity).to(u.km / u.s).value)
    print(f"[casa-turb] time_points:  {np.array2string(tp, precision=0, max_line_width=200)}")
    print(f"[casa-turb] total_energy: {np.array2string(np.asarray(snaps.total_energy), precision=4, max_line_width=200)}")
    print(f"[casa-turb] v_rms [km/s]: {np.array2string(vrms_kms, precision=1, max_line_width=200)}")

    fs = np.asarray(snaps.final_state)
    rho = fs[rv.density_index]
    p = fs[rv.pressure_index]
    T = temperature_K(rho, p, cu)

    # structure diagnostics inside the region the remnant will occupy
    N = args.n
    dxc = BOX_SIZE / N
    xs = (np.arange(N) + 0.5) * dxc - BOX_SIZE / 2
    r3 = np.sqrt(xs[:, None, None] ** 2 + xs[None, :, None] ** 2 + xs[None, None, :] ** 2)
    inner = r3 < 2.5
    clump = float((rho[inner] ** 2).mean() / rho[inner].mean() ** 2)
    print(f"[casa-turb] final rho[{rho.min():.3e},{rho.max():.3e}] "
          f"T[{np.nanmin(T):.1e},{np.nanmax(T):.1e}]K")
    print(f"[casa-turb] clumping <rho^2>/<rho>^2 (r<2.5pc) = {clump:.2f}  "
          f"rho contrast p5/p95 = {np.percentile(rho[inner],5):.3f}/"
          f"{np.percentile(rho[inner],95):.3f}")

    np.savez_compressed(
        args.save_state, rho=rho, press=p,
        vx=fs[rv.velocity_index.x], vy=fs[rv.velocity_index.y],
        vz=fs[rv.velocity_index.z],
        box=float(BOX_SIZE), age=0.0, num_cells=args.n)
    print(f"[casa-turb] saved driven medium {args.save_state}")

    # quick-look figure: midplane density + temperature + density PDF
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    mid = N // 2
    im0 = axes[0].imshow(np.log10(rho[:, :, mid]).T, origin="lower",
                         extent=[-BOX_SIZE/2, BOX_SIZE/2] * 2, cmap="magma")
    axes[0].set_title("log10 rho (z midplane)"); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.log10(np.maximum(T[:, :, mid], 1.0)).T, origin="lower",
                         extent=[-BOX_SIZE/2, BOX_SIZE/2] * 2, cmap="viridis")
    axes[1].set_title("log10 T [K] (z midplane)"); plt.colorbar(im1, ax=axes[1])
    axes[2].hist(np.log10(rho[inner]).ravel(), bins=100, histtype="step")
    axes[2].set_xlabel("log10 rho (r<2.5 pc)"); axes[2].set_yscale("log")
    axes[2].set_title(f"density PDF, clumping={clump:.2f}")
    out = FIGURES_DIR / "casa_turb_phase.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[casa-turb] saved {out}")


if __name__ == "__main__":
    main()
