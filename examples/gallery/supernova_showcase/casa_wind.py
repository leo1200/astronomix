"""
The progenitor's wind, blown into a turbulent ISM: a self-consistent CSM.

WHY THIS EXISTS
---------------
``casa_orlando.py`` imposes its circumstellar medium analytically -- an ``r^-2``
wind with a fitted normalisation ``n_w``, times a log-normal clumping field,
plus one Gaussian shell at a hand-placed radius with a hand-chosen density
(Orlando et al. 2022 Eq. 1). Both are stand-ins for the same physical object:
the wind-blown bubble the progenitor spent its life inflating.

That matters more than it sounds, and the reason is morphological. The
synthetic image reads as **Tycho** rather than Cas A -- a near-circular filled
disc with a closed, smooth, limb-brightened rim -- and there are exactly two
symmetries in the model that could produce that:

1. the ejecta seed is statistically isotropic (addressed by
   ``_common.explosion_plume_field``), and
2. **the ambient medium is spherically symmetric**, so the forward shock is a
   sphere and the swept-up wind limb-brightens into a clean closed ring
   whatever the ejecta do.

The plume field settled which of the two matters for the outline, and the
answer was unambiguous: across four 256^3 runs -- control, plumes, plumes with
composition mixing, and plumes with a 6.5 % kinetic-energy velocity coupling --
the forward-shock position-angle spread was **identical to three decimals**
(min 2.290, max 2.581, spread 0.292 pc). Ejecta structure does not reach the
forward shock by 350 yr. **The outline is set by what the blast runs INTO**,
which is this script.

A wind bubble blown into a turbulent medium is aspherical for free: the shell
is corrugated where the swept ISM was denser, and the bubble breaks out where
it was thinner. Nothing is fitted to make that happen, which is the whole point
-- ``OVERVIEW.md`` §5 forbids tuning structure against the morphology metric,
and a mechanism that produces asphericity as a by-product of the physics is
admissible where an imposed asymmetry would not be.

WHAT IT DOES
------------
Two phases sharing one turbulent driving field, following
``examples/gallery/stellar_wind/turbulent_stellar_wind.py``::

    phase 1   drive turbulence in a uniform ISM, no wind   -> a real turbulent
              density/velocity field rather than a smooth box
    phase 2   switch the wind on, keep driving             -> a wind bubble
              with a swept-up shell, corrugated by the turbulence it ate

and writes the final cube plus the diagnostics that decide whether it is
usable: the angle-averaged radial density profile against the fitted ``r^-2``,
the radial column, and where the swept-up shell sits and how much it carries.

WHAT IT DOES NOT DO YET
-----------------------
**It is not wired into ``casa_orlando.py``.** The mapping stage takes a 1D
spherical profile, and consuming a 3D CSM cube means the ejecta can no longer
be laid down by a spherically symmetric mass coordinate -- which is what
``ejecta_mass_coordinate`` and therefore the whole composition model rest on.
That is a real piece of design work, not a plumbing change, and it should be
done only after the diagnostics below show this CSM is physically right.

So: run this, look at the profile, and compare it with the imposed model.
Judge the CSM before adopting it.

CALIBRATION TARGET
------------------
The imposed model this has to reproduce (or improve on) is, at 350 yr:
``n_w = 0.928 cm^-3`` at ``r_fs_ref = 2.5 pc`` on an ``r^-2`` law, and the
Orlando shell at ~1.5-1.9 pc carrying a column of ``20 x 0.02 cm^-3 pc``
(``CALIBRATION.md`` Result 5 -- the showcase shell carried 27x too much).

An RSG wind reaching ``n = 0.93 cm^-3`` at 2.5 pc needs roughly
``Mdot ~ 2.6e-5 Msun/yr`` at ``v_w = 10 km/s``; those are the defaults, and
``--report`` prints what was actually achieved so the two can be compared
rather than assumed.

USAGE
-----
    ./run.sh casa_wind.py --n 128 --spinup-only        # look at the ISM first
    ./run.sh casa_wind.py --n 128 --save-csm csm.npz
"""

# ==== GPU selection ====
import os
import sys
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
from pathlib import Path

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# units and constants
from astropy import units as u
import astropy.constants as const

# astronomix
from astronomix import (
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
    WindParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes import WindConfig
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)

# shared showcase helpers
from _common import FIGURES_DIR, GAMMA, MASS_PER_NUCLEUS, centered_radius, snr_code_units


# =============================================================================
# ============ ↓ Setup parameters ↓ ===========================================
# =============================================================================
BOX_SIZE = 7.0              # pc -- the SAME box casa_orlando.py uses, so the
                            # CSM can be handed over cell for cell
#: RSG wind. The mass-loss rate is set to land n = 0.93 cm^-3 at 2.5 pc on an
#: r^-2 law, which is the fitted n_w the imposed model uses.
WIND_MDOT_MSUN_PER_YR = 2.6e-5
WIND_VELOCITY_KMS = 10.0
#: How long the wind blows. An RSG phase is ~1e5 yr, and at 10 km/s the wind
#: takes ~3.4e5 yr to cross 3.5 pc -- so the bubble is NOT in steady state
#: across the box on an RSG timescale, and this duration is a knob whose effect
#: on the profile is exactly what --report is for.
WIND_DURATION_YR = 3.0e5
#: Ambient ISM into which the bubble is blown.
ISM_NUMBER_DENSITY = 2.0    # cm^-3
ISM_TEMPERATURE_K = 3.0e4
#: Turbulent driving. The energy injection rate sets the rms Mach number; the
#: default is the order of magnitude used by the stellar-wind gallery example.
TURB_ENERGY_INJECTION = 4.3e34   # erg/s, converted below
SPINUP_YR = 5.0e4


def build_config(args, code_units, wind=False):
    """The solver configuration for either phase."""
    return SimulationConfig(
        progress_bar=True,
        donate_state=True,
        dimensionality=3,
        box_size=args.box,
        num_cells=args.n,
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=not args.no_turbulence),
        wind_config=WindConfig(stellar_wind=wind,
                               num_injection_cells=args.injection_cells),
        # PERIODIC on every face. The bubble must not reach the boundary within
        # the run -- if it does, the box is too small and the profile is
        # wrapping, which --report checks for explicitly.
        boundary_settings=BoundarySettings(
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY,
                               right_boundary=PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY,
                               right_boundary=PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY,
                               right_boundary=PERIODIC_BOUNDARY),
        ),
    )


def initial_ism(config, registered_variables, args, code_units):
    """A uniform ISM at rest, for the turbulence to work on."""
    shape = (args.n,) * 3
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3)
                      .to(code_units.code_density).value)
    p_per_n = float((const.k_B * ISM_TEMPERATURE_K * u.K / u.cm ** 3)
                    .to(code_units.code_pressure).value)

    rho = jnp.full(shape, args.ism_density * rho_per_n)
    p = jnp.full(shape, args.ism_density * p_per_n)
    zero = jnp.zeros(shape)
    return construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho, velocity_x=zero, velocity_y=zero, velocity_z=zero,
        gas_pressure=p,
    )


def radial_profile(field, r, *, nbins=80, r_max=None):
    """Angle-averaged radial profile of a 3D field, on ``nbins`` shells."""
    r = np.asarray(r).ravel()
    f = np.asarray(field).ravel()
    r_max = r_max if r_max is not None else r.max()
    edges = np.linspace(0.0, r_max, nbins + 1)
    idx = np.digitize(r, edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    counts = np.bincount(idx[ok], minlength=nbins)
    sums = np.bincount(idx[ok], weights=f[ok], minlength=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return centers, mean, counts


def report(state, registered_variables, helper_data, args, code_units):
    """Everything needed to judge whether this CSM can replace the imposed one."""
    rho = np.asarray(state[registered_variables.density_index])
    r, X, Y, Z = centered_radius(helper_data, args.box, args.n)
    r = np.asarray(r)
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3)
                      .to(code_units.code_density).value)
    n = rho / rho_per_n

    centers, n_of_r, _ = radial_profile(n, r, r_max=0.5 * args.box)

    # the imposed model this has to beat
    n_imposed = 0.928 * (2.5 / np.maximum(centers, 1e-3)) ** 2

    print(f"\n[wind] angle-averaged radial profile (n in cm^-3):")
    print(f"[wind] {'r [pc]':>8s} {'n_sim':>10s} {'n_imposed':>10s} {'ratio':>8s} "
          f"{'sigma/mean':>10s}")
    for i in range(0, len(centers), max(1, len(centers) // 14)):
        # the ANGULAR scatter at fixed radius is the number that matters for
        # the morphology: it is zero by construction in the imposed model
        shell = np.abs(r - centers[i]) < (0.5 * args.box / 80)
        scatter = (np.std(n[shell]) / np.mean(n[shell])
                   if shell.sum() > 10 else np.nan)
        print(f"[wind] {centers[i]:8.3f} {n_of_r[i]:10.4f} {n_imposed[i]:10.4f} "
              f"{n_of_r[i] / n_imposed[i]:8.3f} {scatter:10.3f}")

    # is the bubble still inside the box?
    edge = n_of_r[-1]
    print(f"\n[wind] ISM at the box edge: {edge:.3f} cm^-3 against the "
          f"{args.ism_density:.1f} it started at "
          f"({'UNDISTURBED -- the bubble is contained' if abs(edge / args.ism_density - 1) < 0.15 else 'DISTURBED -- the bubble has reached the boundary and the profile is wrapping; use a larger --box'})")

    # the swept-up shell: the outermost density maximum
    finite = np.isfinite(n_of_r)
    if finite.sum() > 3:
        i_shell = int(np.nanargmax(np.where(centers > 0.2, n_of_r, -np.inf)))
        print(f"[wind] swept-up shell peak: n = {n_of_r[i_shell]:.3f} cm^-3 at "
              f"r = {centers[i_shell]:.3f} pc "
              f"(the imposed Orlando shell sits at 1.5-1.9 pc)")

    # the column, which is what actually decelerates the blast
    dr = centers[1] - centers[0]
    col = np.nansum(n_of_r * dr)
    print(f"[wind] radial column to the box edge: {col:.3f} cm^-3 pc "
          f"(the imposed shell alone carries 20 x 0.02 = 0.4)")
    return centers, n_of_r, n


def figure(centers, n_of_r, n3d, args, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].loglog(centers, n_of_r, label="simulated wind bubble")
    ax[0].loglog(centers, 0.928 * (2.5 / np.maximum(centers, 1e-3)) ** 2,
                 "--", label=r"imposed $n_w (2.5/r)^2$")
    ax[0].set_xlabel("r [pc]"); ax[0].set_ylabel(r"$n$ [cm$^{-3}$]")
    ax[0].legend(); ax[0].set_title("angle-averaged profile")

    mid = args.n // 2
    im = ax[1].imshow(n3d[:, mid, :], origin="lower", norm=LogNorm(),
                      extent=[-args.box / 2, args.box / 2] * 2, cmap="magma")
    ax[1].set_title("density slice: the corrugated shell is the point")
    ax[1].set_xlabel("x [pc]"); ax[1].set_ylabel("z [pc]")
    fig.colorbar(im, ax=ax[1], label=r"$n$ [cm$^{-3}$]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"[wind] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=128, help="cells per axis")
    ap.add_argument("--box", type=float, default=BOX_SIZE, help="box size (pc)")
    ap.add_argument("--ism-density", type=float, default=ISM_NUMBER_DENSITY,
                    help="ambient ISM number density (cm^-3)")
    ap.add_argument("--mdot", type=float, default=WIND_MDOT_MSUN_PER_YR,
                    help="wind mass-loss rate (Msun/yr)")
    ap.add_argument("--v-wind", type=float, default=WIND_VELOCITY_KMS,
                    help="terminal wind velocity (km/s)")
    ap.add_argument("--wind-duration", type=float, default=WIND_DURATION_YR,
                    help="how long the wind blows (yr)")
    ap.add_argument("--spinup", type=float, default=SPINUP_YR,
                    help="turbulence spin-up time (yr)")
    ap.add_argument("--injection-cells", type=int, default=10,
                    help="cells the wind is injected into")
    ap.add_argument("--no-turbulence", action="store_true",
                    help="no turbulent driving -- the control, which must give "
                         "a SPHERICAL bubble and zero angular scatter")
    ap.add_argument("--spinup-only", action="store_true",
                    help="stop after the turbulence phase")
    ap.add_argument("--save-csm", default=None, help="write the final cube here")
    args = ap.parse_args()

    code_units = snr_code_units()
    yr = float((1.0 * u.yr).to(code_units.code_time).value)

    config = build_config(args, code_units, wind=False)
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    state = initial_ism(config, rv, args, code_units)
    config = finalize_config(config, state.shape)

    mdot = (args.mdot * u.Msun / u.yr).to(
        code_units.code_mass / code_units.code_time).value
    v_wind = (args.v_wind * u.km / u.s).to(code_units.code_velocity).value
    e_inj = float((TURB_ENERGY_INJECTION * u.erg / u.s).to(
        code_units.code_energy / code_units.code_time).value)

    params = SimulationParams(
        C_cfl=0.4, gamma=GAMMA,
        minimum_density=1e-6, minimum_pressure=1e-12,
        wind_params=WindParams(wind_mass_loss_rate=mdot,
                               wind_final_velocity=v_wind),
        turbulent_forcing_params=TurbulentForcingParams(
            energy_injection_rate=e_inj),
    )

    print(f"[wind] {args.n}^3 in a {args.box} pc box "
          f"({args.box / args.n:.4f} pc per cell)")
    print(f"[wind] wind: Mdot = {args.mdot:.2e} Msun/yr at {args.v_wind:.0f} km/s "
          f"for {args.wind_duration:.2e} yr")
    print(f"[wind] a free wind would give n = "
          f"{_free_wind_n(args.mdot, args.v_wind, 2.5):.3f} cm^-3 at 2.5 pc "
          f"(the imposed model uses 0.928)")

    # ---- phase 1: turbulence -------------------------------------------
    if not args.no_turbulence:
        print(f"[wind] phase 1: turbulence spin-up, {args.spinup:.1e} yr")
        params = params._replace(t_end=args.spinup * yr)
        state = time_integration(state, config, params, rv)

    if not args.spinup_only:
        # ---- phase 2: the wind ------------------------------------------
        print(f"[wind] phase 2: wind on, driving continues, "
              f"{args.wind_duration:.1e} yr")
        config = build_config(args, code_units, wind=True)
        config = finalize_config(config, state.shape)
        params = params._replace(t_end=args.wind_duration * yr)
        state = time_integration(state, config, params, rv)

    centers, n_of_r, n3d = report(state, rv, helper_data, args, code_units)
    figure(centers, n_of_r, n3d, args,
           FIGURES_DIR / f"casa_wind_n{args.n}"
                         f"{'_noturb' if args.no_turbulence else ''}.png")

    if args.save_csm:
        np.savez_compressed(
            args.save_csm,
            rho=np.asarray(state[rv.density_index]),
            press=np.asarray(state[rv.pressure_index]),
            vx=np.asarray(state[rv.velocity_index.x]),
            vy=np.asarray(state[rv.velocity_index.y]),
            vz=np.asarray(state[rv.velocity_index.z]),
            box=float(args.box), num_cells=args.n,
            argv=np.array(" ".join(sys.argv)),
        )
        print(f"[wind] saved {args.save_csm}")


def _free_wind_n(mdot_msun_yr, v_kms, r_pc):
    """Number density of an unimpeded r^-2 wind, for the calibration check."""
    mdot = (mdot_msun_yr * u.Msun / u.yr).to(u.g / u.s).value
    v = (v_kms * u.km / u.s).to(u.cm / u.s).value
    r = (r_pc * u.pc).to(u.cm).value
    rho = mdot / (4.0 * np.pi * r ** 2 * v)
    return rho / (MASS_PER_NUCLEUS * const.m_p.cgs.value)


if __name__ == "__main__":
    main()
