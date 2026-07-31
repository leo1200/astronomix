"""Phase 2 of the TI cross-validation: supernova into the multiphase medium.

Loads the driven-TI medium from ``casa_ti_phase.py`` (npz) and injects the
Guo-Kim-Stone supernova exactly as the AthenaK ``ti_box`` restart path does:
mass_snr = 197.85 and etot_snr = 3.469e9 (~1e51 erg) added UNIFORMLY inside
a sharp sphere of radius ``--r-snr`` (analytic volume normalisation — kept
bit-comparable to the reference, not the tapered/renormalised snr_sedov
variant), then evolves with the same mainline-matched ISM cooling + heating.

Blast-phase guards (the TI medium is self-consistent, but the SN is a
Mach ~100 event into 5 K clumps): cooling resolution limiter (default
alpha = 4) and the per-step density-scaled temperature floor at the
AthenaK tfloor (5 K) — the direct analogue of snr.athinput's
dfloor/pfloor/tfloor + fofc stack.
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

from _common import (GAMMA, make_fd_config, fd_positivity,
                     ism_ti_cooling_setup, tapered_sphere_weight)
from cassiopeia_realistic import FIGURES_DIR
from casa_ti_phase import gks_code_units, MU_ATH, BOX_SIZE, HRATE_CGS, DFLOOR, PFLOOR

MASS_SNR = 197.85148
ETOT_SNR = 3.4691068e9
T_FLOOR_ATHENA_K = 5.0        # snr.athinput tfloor = 7e-2 code = 5 K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("medium", help="npz from casa_ti_phase.py")
    ap.add_argument("--t-end", type=float, default=1.0, help="Myr (snr.athinput tlim)")
    ap.add_argument("--r-snr", type=float, default=1.0,
                    help="injection radius (pc); snr.athinput uses radius = 1.00")
    ap.add_argument("--limiter-alpha", type=float, default=4.0)
    ap.add_argument("--implicit-cooling", action="store_true",
                    help="use the implicit cooling solver; the default is the "
                         "AthenaK-matched EXPLICIT update (~50x faster and "
                         "what the reference uses)")
    ap.add_argument("--taper-cells", type=float, default=0.0,
                    help="tanh-taper the injection over this many cells with "
                         "EXACT mass/energy renormalisation (the showcase "
                         "recipe: a sharp injection edge NaNs regardless). "
                         "0 = sharp sphere (AthenaK-matched)")
    ap.add_argument("--deepvoid", action="store_true",
                    help="deep-void LLF blend: kills the high-Mach WENO "
                         "overshoot in blast-evacuated near-vacuum cells")
    ap.add_argument("--rho-floor", type=float, default=0.0,
                    help="override minimum_density (blast voids spike as "
                         "mom/rho_floor; paper_turbulence uses 0.02)")
    ap.add_argument("--vmax-cap", type=float, default=0.0,
                    help="positivity_max_velocity cap (code units); 0 = off")
    ap.add_argument("--dual-energy", action="store_true",
                    help="Bryan+95 dual energy. The SN is KE-dominated and the "
                         "ambient contains 70 K clumps, so E - KE cancels "
                         "catastrophically in float32 and the recovered "
                         "pressure goes negative — the same failure the Cas A "
                         "showcase hit above N~240")
    ap.add_argument("--no-tfloor", action="store_true",
                    help="disable the per-step density-scaled 5 K floor")
    ap.add_argument("--nsnap", type=int, default=11)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--save-state", type=str,
                    default="/export/data/lstorcks/supernova_showcase/casa_ti_snr.npz")
    args = ap.parse_args()

    print(f"[ti-snr] devices: {jax.devices()}  x64={jax.config.jax_enable_x64}")

    d = np.load(args.medium)
    N = int(d["num_cells"])
    rho = np.array(d["rho"]); press = np.array(d["press"])
    vx = np.array(d["vx"]); vy = np.array(d["vy"]); vz = np.array(d["vz"])

    # SN injection, bit-matched to the AthenaK ti_box restart path
    dx = BOX_SIZE / N
    xs = (np.arange(N) + 0.5) * dx - BOX_SIZE / 2
    r3 = np.sqrt(xs[:, None, None]**2 + xs[None, :, None]**2 + xs[None, None, :]**2)
    if args.taper_cells > 0.0:
        w = np.asarray(tapered_sphere_weight(
            jnp.asarray(r3), args.r_snr, dx, args.taper_cells))
        wsum = w.sum() * dx ** 3
        inside = w / wsum          # exact renormalisation: sum(inside)*dV = 1
        vol = 1.0                  # amplitudes below are then M and E exactly
    else:
        inside = (r3 < args.r_snr).astype(float)
        vol = 4.0 / 3.0 * np.pi * args.r_snr ** 3
    # AthenaK adds to CONSERVED (rho, E): momentum unchanged -> velocity of
    # loaded gas drops mass-weightedly; energy addition is thermal.
    gm1 = GAMMA - 1.0
    e_int = press / gm1
    mom = [rho * v for v in (vx, vy, vz)]
    rho_new = rho + inside * (MASS_SNR / vol)
    e_int_new = e_int + inside * (ETOT_SNR / vol)
    vx, vy, vz = [m / rho_new for m in mom]
    press_new = e_int_new * gm1
    n_in = int(inside.sum())
    print(f"[ti-snr] N={N} injected M={MASS_SNR:.2f} E={ETOT_SNR:.3e} in "
          f"r<{args.r_snr} pc ({n_in} cells; discrete/analytic vol "
          f"{n_in*dx**3/vol:.3f})")

    cu = gks_code_units()
    cooling_config, cooling_params = ism_ti_cooling_setup(
        cu, hrate_cgs=HRATE_CGS, mu_athena=MU_ATH, floor_temperature_K=10.0,
        hydrogen_mass_fraction=0.7, metal_mass_fraction=0.02,
        resolution_limiter_alpha=args.limiter_alpha,
        explicit=not args.implicit_cooling,
    )
    snaps = SnapshotSettings(
        return_states=False, return_final_state=True,
        return_total_mass=True, return_total_energy=True,
        return_internal_energy=True, return_kinetic_energy=True,
    )
    extra = {}
    if args.deepvoid:
        pc = fd_positivity(tfloor=not args.no_tfloor)
        extra["positivity_config"] = pc._replace(deepvoid_blend=True)
    if args.dual_energy:
        extra["dual_energy"] = True
    if not args.no_tfloor:
        extra["positivity_config"] = fd_positivity(tfloor=True)
    config = make_fd_config(
        BOX_SIZE, N, mhd=False,
        cooling_config=cooling_config,
        snapshot_settings=snaps, num_snapshots=args.nsnap,
        **extra,
    )
    rv = get_registered_variables(config)
    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=jnp.asarray(rho_new), velocity_x=jnp.asarray(vx),
        velocity_y=jnp.asarray(vy), velocity_z=jnp.asarray(vz),
        gas_pressure=jnp.asarray(press_new),
    )
    config = finalize_config(config, state.shape)

    # tfloor scale: p >= rho * T~(5 K) matching snr.athinput's tfloor
    temp_unit_K = float((MU_ATH * const.u * (1.0 * cu.code_velocity) ** 2
                         / const.k_B).to(u.K).value)
    spfloor = T_FLOOR_ATHENA_K / temp_unit_K   # p/rho at 5 K in code units
    params = SimulationParams(
        gamma=GAMMA, C_cfl=0.3, t_end=args.t_end,
        minimum_density=(args.rho_floor if args.rho_floor > 0 else DFLOOR),
        minimum_pressure=PFLOOR,
        minimum_specific_pressure=spfloor,
        positivity_max_velocity=(args.vmax_cap if args.vmax_cap > 0
                                 else float("inf")),
        cooling_params=cooling_params,
    )
    print(f"[ti-snr] t_end={args.t_end} Myr  limiter={args.limiter_alpha} "
          f"tfloor={'off' if args.no_tfloor else f'{T_FLOOR_ATHENA_K} K'} "
          f"(spfloor={spfloor:.4f})")

    result = time_integration(state, config, params, rv, sharding=None)
    jax.block_until_ready(result)

    tp = np.asarray(result.time_points)
    te = np.asarray(result.total_energy)
    print(f"[ti-snr] time_points:  {np.array2string(tp, precision=2, max_line_width=220)}")
    print(f"[ti-snr] total_energy: {np.array2string(te, precision=4, max_line_width=220)}")

    fs = np.asarray(result.final_state)
    rho_f = fs[rv.density_index]; p_f = fs[rv.pressure_index]
    T_K = (p_f / rho_f) * temp_unit_K
    mtot = rho_f.sum()
    print(f"[ti-snr] final T_K[{T_K.min():.1f},{T_K.max():.3g}] "
          f"rho[{rho_f.min():.3e},{rho_f.max():.3e}]")
    print(f"[ti-snr] hot(>1e5K) volume fraction {float((T_K>1e5).mean()):.3f}  "
          f"mass fraction {float(rho_f[T_K>1e5].sum()/mtot):.3f}")

    np.savez_compressed(
        args.save_state, rho=rho_f, press=p_f,
        vx=fs[rv.velocity_index.x], vy=fs[rv.velocity_index.y],
        vz=fs[rv.velocity_index.z],
        box=float(BOX_SIZE), age=args.t_end, num_cells=N)
    print(f"[ti-snr] saved {args.save_state}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    mid = N // 2
    im0 = axes[0].imshow(np.log10(rho_f[:, :, mid]).T, origin="lower", cmap="magma")
    axes[0].set_title("log10 rho (midplane)"); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.log10(np.maximum(T_K[:, :, mid], 1.0)).T,
                         origin="lower", cmap="coolwarm")
    axes[1].set_title("log10 T [K]"); plt.colorbar(im1, ax=axes[1])
    axes[2].hist2d(np.log10(rho_f).ravel(), np.log10(np.maximum(T_K, 1.0)).ravel(),
                   bins=120, norm=matplotlib.colors.LogNorm())
    axes[2].set_xlabel("log10 n"); axes[2].set_ylabel("log10 T [K]")
    axes[2].set_title("phase diagram (post-SN)")
    out = FIGURES_DIR / "casa_ti_snr.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[ti-snr] saved {out}")


if __name__ == "__main__":
    main()
