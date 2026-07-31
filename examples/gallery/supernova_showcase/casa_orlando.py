"""
Cassiopeia A following Orlando et al. -- the calibrated Route B setup.

``cassiopeia_realistic.py`` builds its remnant from scratch on the 3D grid: a
cold homologous ball already 1.5 pc across, tagged with an age ``R / v_max``.
That skips the phase that sets the answer (the progenitor wind holds ~3 M_sun --
an entire ejecta mass -- inside 1.5 pc), so its shock radii and its age are both
unreliable, and its explosion parameters were never checked against anything.

This script does it the way Orlando et al. (2016) do:

  **Stage 1 (1D, ``casa_calibrate_1d.py``).** The same ejecta profile expanding
  into the same r^-2 wind, started at 0.05 pc where free expansion still holds,
  evolved spherically and *calibrated* against r_FS, r_RS, v_FS, v_RS and the
  post-shock density. Costs seconds.

  **Stage 2 (this script).** The calibrated 1D profile is mapped onto the 3D
  grid at a chosen age (``--map-age``, default 150 yr, where the blast is at
  ~1.27 pc and there are still >100 cells per remnant radius), and the
  multi-dimensional structure is imposed *there*:

    * small-scale ejecta clumping interior to the reverse shock (Orlando et al.
      2012: power-law perturbations, clump size ~2% of the remnant radius,
      maximum contrast ~5) -- seeds the Rayleigh-Taylor fingering at the
      contact discontinuity;
    * the five large-scale anisotropies of Orlando et al. (2016) Table 4
      (``--pistons``) -- the Fe-rich knots that produce the layer inversion and
      the Si-rich NE jet / SW counter-jet. Without these essentially no Fe is
      shocked by 350 yr;
    * the asymmetric circumstellar shell of Orlando et al. (2022) Eq. 1
      (``--shell``), placed *ahead* of the blast wave so the interaction is
      computed rather than imposed -- this is what drives a reflected shock
      back into the ejecta and makes the reverse shock move inward in the west;
    * circumstellar clumping in the wind.

  The remnant is then evolved to 350 yr, so the reported age is a real age and
  the shock radii are predictions.

Solver recipe is the one that survived the 512^3 stability campaign: WENO
finite difference, SSP-RK4, Bryan+95 dual energy (the ejecta core is cold and
kinetic-energy dominated -- float32 pressure recovery there is cancellation
noise without it), the positivity-preserving flux limiter, and the cold-crush
LLF blend. With ``--cooling`` add the resolution limiter and the temperature
floor, without which a resolved radiative shell ram-crushes without bound.

Examples::

    # calibrate (seconds, CPU)
    ./run.sh casa_calibrate_1d.py --n 4000 --energy-51 2.09 --ejecta-mass 3.0 \\
        --n-w 0.928 --age 150 --save-profile casa_1d_map150.npz

    # map and evolve (queue this)
    pq sub -t a100 -n 4 -- ./run.sh casa_orlando.py casa_1d_map150.npz \\
        --n 512 --gpus 4 --shell --cooling --save-state casa_orlando_n512.npz
"""

# ==== GPU selection ====
# --gpus must be known BEFORE jax initialises. Under pq, CUDA_VISIBLE_DEVICES
# is already pinned to the assigned GPUs; only fall back to autocvd otherwise.
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

# general
import argparse
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# units and constants
from astropy import units as u
import astropy.constants as const

# astronomix containers
from astronomix import (
    SimulationParams,
    SnapshotSettings,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

# shared showcase helpers
from _common import (
    GAMMA,
    FIGURES_DIR,
    IIB_LAYERS,
    MASS_PER_NUCLEUS,
    centered_radius,
    chandra_deep_figure,
    ejecta_mass_coordinate,
    fd_positivity,
    POSITIVITY_HARD_FLOOR,
    POSITIVITY_REDISTRIBUTE,
    POSITIVITY_CONSERVATIVE,
    layered_composition,
    make_fd_config,
    map_1d_profile,
    prior_shock_history,
    orlando_csm_shell,
    orlando_piston_fields,
    realistic_figure,
    schure_cooling_setup,
    snr_code_units,
    temperature_K,
    turbulent_field,
)

#: The species carried as passive scalars. Hydrogen is NOT tracked: mass
#: fractions sum to one, so it is recovered as ``1 - sum(the rest)`` and a whole
#: 3D field is saved. The ejecta/circumstellar discriminator is separate because
#: composition alone cannot provide it — the outer ejecta layers are H/He too.
TRACKED_SPECIES = ("Fe", "Si", "O", "He")
SCALAR_NAMES = ("C_ej",) + tuple(f"C_{s}" for s in TRACKED_SPECIES)


# =============================================================================
# ============ ↓ Setup parameters ↓ ===========================================
# =============================================================================
BOX_SIZE = 7.0                  # pc; half-width 3.5 pc > the final r_FS = 2.52
TARGET_AGE_YR = 350.0           # Fesen et al. 2006
WIND_TEMPERATURE_K = 1e4
SEED = 7

# --- small-scale ejecta clumping (Orlando et al. 2012) -----------------------
CLUMP_SIZE_FRACTION = 0.02      # clump size as a fraction of the remnant radius
CLUMP_MAX_CONTRAST = 5.0        # nu_max
CLUMP_SLOPE = -1.0              # xi

# --- circumstellar clumping --------------------------------------------------
CSM_SIGMA = 0.4
CSM_K_MIN, CSM_K_MAX = 4, 20

# --- the Orlando et al. (2022) shell, model W15-IIb-sh-HD-1eta-az ------------
SHELL_RADIUS = 1.5              # pc
SHELL_THICKNESS = 0.02          # pc (Gaussian sigma) -- sub-cell, see below
SHELL_DENSITY = 20.0            # cm^-3
SHELL_THETA_DEG = 30.0
SHELL_PHI_DEG = 50.0
SHELL_SCALE_LENGTH = 0.7        # pc
# =============================================================================
# ============ ↑ Setup parameters ↑ ===========================================
# =============================================================================


def _smoothstep(x):
    """Smooth 0->1 ramp on a tanh, for blending regions without a seam."""
    return 0.5 * (1.0 + jnp.tanh(x))


def build(args, sharding=None):
    """Map the calibrated 1D profile into 3D and impose the multi-D structure."""
    code_units = snr_code_units()
    num_cells = args.n
    dx = args.box / num_cells

    cooling_config = cooling_params = None
    if args.cooling:
        # the resolution limiter cutoff is held FIXED IN PHYSICAL LENGTH
        # (alpha = 4 at 256^3 in this box, so alpha ~ 0.109 pc / dx): scaling it
        # in cells instead only delays the radiative-shell crush.
        alpha = args.limiter_alpha
        if alpha is None:
            alpha = 0.109 / dx
        cooling_config, cooling_params = schure_cooling_setup(
            code_units, floor_temperature_K=1e4,
            hydrogen_mass_fraction=1.0 - 0.28 - 0.02, metal_mass_fraction=0.02,
            resolution_limiter_alpha=float(alpha),
            explicit=args.explicit_cooling,
            max_cooling_fraction=args.max_cool_fraction,
            clamp_to_floor=args.clamp_floor,
        )

    snaps = SnapshotSettings(
        return_states=False, return_final_state=True,
        return_total_mass=True, return_total_energy=True,
        return_internal_energy=True, return_kinetic_energy=True,
    )
    extra = dict(random_seed=SEED, dual_energy=True)
    if args.conduction:
        # Conduction supplies the physical length the pure-hydro model lacks.
        # An ideal optically-thin radiative shock collapses without bound, so
        # SOMETHING has to set the layer thickness; here every candidate has
        # been numerical (the floor revert, the LLF blend, the resolution
        # limiter), which is why refining the grid only made the crush worse.
        # The Field length is the physical answer and it is resolvable.
        extra["thermal_conduction"] = True
    if args.composition:
        # An ejecta/CSM discriminator plus the chemical stratification, and the
        # shock bookkeeping that gives the ionization age and the time since
        # shocking. Without these the X-ray synthesis has to assume one
        # metallicity everywhere and collisional ionization equilibrium, which
        # is where the ~3.5x luminosity deficit and the wrong radial
        # distribution of the brightness came from.
        extra["num_passive_scalars"] = len(SCALAR_NAMES)
        extra["track_shock_history"] = True
        # every user scalar here is a mass fraction, so declare it: the ratio
        # recovery can run away in the near-vacuum interior where the fast
        # pistons and radiative cooling compress the same cells
        extra["passive_scalar_bounds"] = tuple((0.0, 1.0) for _ in SCALAR_NAMES)
    # Positivity protection is applied ALWAYS, not only with --cooling. It used
    # to be gated on cooling, which meant every adiabatic run silently used the
    # default PositivityConfig() -- no preserving_flux, no cold-crush blend, no
    # floor, no vacuum_rest -- despite this module's docstring listing them as
    # part of the recipe. Those runs are marginal: two completed and a third,
    # identical but for the node, blew up with rho ~ 1e18.
    # ``tfloor`` stays cooling-only: with real cooling the shocked shell reaches
    # the isothermal jump and a CONSTANT pressure floor leaves it pressureless
    # against ram crushing, but for an adiabatic run the density-scaled floor
    # has nothing to do.
    extra["positivity_config"] = fd_positivity(
        tfloor=bool(args.cooling), coldcrush_factor=args.coldcrush_factor,
        mode={"floor": POSITIVITY_HARD_FLOOR,
              "redistribute": POSITIVITY_REDISTRIBUTE,
              "conservative": POSITIVITY_CONSERVATIVE}[args.positivity])
    config = make_fd_config(args.box, num_cells, mhd=False,
                            cooling_config=cooling_config,
                            snapshot_settings=snaps, num_snapshots=args.nsnap,
                            **extra)
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config, sharding)

    r, X, Y, Z = centered_radius(helper_data, args.box, num_cells)
    r_safe = jnp.maximum(r, 0.5 * dx)

    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(code_units.code_density).value)
    p_per_n = float((const.k_B * WIND_TEMPERATURE_K * u.K / u.cm ** 3).to(code_units.code_pressure).value)

    # -------------------------------------------------------------
    # ====== ↓ Stage 2a: map the calibrated 1D profile ↓ ==========
    # -------------------------------------------------------------
    rho, v_r, p, meta = map_1d_profile(r, args.profile)
    r_fs = meta["r_fs"]
    r_rs = meta["r_rs"]
    map_age = meta["measured_age_yr"]
    n_w = meta.get("cfg_n_w", 0.928)
    r_fs_ref = meta.get("cfg_r_fs_ref", 2.5)
    n_c = meta.get("cfg_n_c", 0.1)
    print(f"[orlando] mapped {args.profile}: age {map_age:.1f} yr, "
          f"r_FS = {r_fs:.3f} pc, r_RS = {r_rs:.3f} pc, "
          f"E = {meta.get('cfg_energy_erg', np.nan):.3e} erg, "
          f"M_ej = {meta.get('cfg_ejecta_mass_msun', np.nan):.2f} Msun, "
          f"n_w = {n_w:.3f} cm^-3")
    print(f"[orlando] resolution: {r_fs / dx:.0f} cells per remnant radius at the "
          f"mapping time (Orlando's invariant is > 100), "
          f"{2.52 / dx:.0f} at 350 yr")

    keys = jax.random.split(jax.random.PRNGKey(SEED), 4)

    # -------------------------------------------------------------
    # ====== ↓ Stage 2b: circumstellar structure (r > r_FS) ↓ =====
    # -------------------------------------------------------------
    # The 1D profile already carries the correct pre-shock wind (same n_w, same
    # r^-2 law), so the CSM structure enters as a MULTIPLICATIVE modulation of
    # it, switched on only ahead of the blast wave -- inside the shock the
    # solver's own 1D solution is left alone.
    outside = _smoothstep((r - 1.02 * r_fs) / (2.0 * dx))
    csm_clump = 1.0
    if args.csm_sigma > 0:
        g = turbulent_field(num_cells, keys[0], kmin=CSM_K_MIN, kmax=CSM_K_MAX, slope=-1.0)
        csm_clump = jnp.exp(args.csm_sigma * g - 0.5 * args.csm_sigma ** 2)
    modulation = 1.0 + outside * (csm_clump - 1.0)
    rho = rho * modulation
    p = p * modulation                      # preserves T ahead of the shock

    shell_info = None
    if args.shell:
        # broaden the sub-cell shell to a resolvable width at FIXED surface
        # density, so the column the blast runs into -- what sets both the
        # deceleration and the reflected shock -- is preserved
        rho_shell, shell_info = orlando_csm_shell(
            r, X, Y, Z,
            shell_radius=args.shell_radius, thickness=SHELL_THICKNESS,
            peak_number_density=args.shell_density,
            theta_deg=SHELL_THETA_DEG, phi_deg=SHELL_PHI_DEG,
            scale_length=SHELL_SCALE_LENGTH, rho_per_n=rho_per_n,
            min_thickness=2.5 * dx,
        )
        if args.shell_radius < 1.05 * r_fs:
            raise SystemExit(
                f"the shell at {args.shell_radius} pc is inside the mapped blast "
                f"wave at {r_fs:.2f} pc: it would be imposed on already-shocked "
                f"gas instead of being run into. Map earlier (--map-age) or move "
                f"the shell out.")
        rho = rho + rho_shell
        # the shell is cool and dense: give it the ambient temperature rather
        # than the local (hot) 1D pressure
        p = p + rho_shell / rho_per_n * p_per_n
        print(f"[orlando] shell: sigma {SHELL_THICKNESS} -> {shell_info['thickness']:.4f} pc "
              f"({shell_info['thickness'] / dx:.1f} cells), n_sh {args.shell_density} -> "
              f"{shell_info['peak_number_density']:.2f} cm^-3 at fixed column; "
              f"peak densities {shell_info['n_peak_near']:.0f} / "
              f"{shell_info['n_peak_far']:.1f} cm^-3 across the gradient")

    # -------------------------------------------------------------
    # ====== ↓ Stage 2c: ejecta structure (r < r_RS) ↓ ============
    # -------------------------------------------------------------
    # Everything here multiplies the DENSITY only: the pressure stays on the
    # smooth mapped profile, so the structure is isobaric (dense knots cold,
    # rarefied regions hot) rather than carrying a spurious pressure imprint.
    inside = 1.0 - _smoothstep((r - r_rs) / (2.0 * dx))
    ejecta_mult = jnp.ones_like(r)
    velocity_mult = jnp.ones_like(r)

    if args.clump_sigma > 0:
        # clump size ~2% of the remnant radius -> a wavenumber band; at 512^3 in
        # a 7 pc box that is right at the grid scale, so it is clamped and the
        # achieved clump size reported.
        k_clump = args.box / (CLUMP_SIZE_FRACTION * r_fs)
        k_hi = int(min(k_clump, num_cells // 6))
        k_lo = max(4, int(k_hi / 3))
        g = turbulent_field(num_cells, keys[1], kmin=k_lo, kmax=k_hi, slope=CLUMP_SLOPE)
        # log-normal so the perturbation is strictly positive (a linear
        # 1 + sigma*g clips to vacuum in the tail and speckles the ejecta), with
        # sigma set so the ~3-sigma peak reaches the Orlando maximum contrast
        sigma = np.log(CLUMP_MAX_CONTRAST) / 3.0 * args.clump_sigma
        ejecta_mult = ejecta_mult * jnp.exp(sigma * g - 0.5 * sigma ** 2)
        print(f"[orlando] ejecta clumps: k = {k_lo}-{k_hi} "
              f"(size {args.box / k_hi:.4f} pc = {100 * args.box / k_hi / r_fs:.1f}% of "
              f"r_FS, target {100 * CLUMP_SIZE_FRACTION:.0f}%; "
              f"{args.box / k_hi / dx:.1f} cells), sigma_ln = {sigma:.3f}")

    piston_info = piston_species = None
    if args.pistons:
        d_mult, v_mult, piston_species, piston_info = orlando_piston_fields(
            X, Y, Z, r_fs, rho, dx ** 3, smoothing_cells=2.0, dx=dx)
        ejecta_mult = ejecta_mult * d_mult
        velocity_mult = velocity_mult * v_mult
        for k in piston_info:
            print(f"[orlando]   knot {k['name']:12s} {k['m_knot']:.4f} Msun of "
                  f"{k['species']:2s} at r = "
                  f"{np.sqrt(sum(c*c for c in k['center'])):.3f} pc, radius "
                  f"{k['radius']:.3f} pc = {k['r_frac']:.3f} R (Orlando lists "
                  f"0.02-0.10; solved to hold chi_n = {k['chi_n']:.0f}), "
                  f"{k['cells_across']:.1f} cells across")

    # apply, conserving the ejecta mass under the CLUMPING (which only
    # redistributes) while letting the PISTONS add mass, as Orlando's do
    cell_vol = dx ** 3
    if args.clump_sigma > 0 or args.pistons:
        w = inside
        # write the perturbation as rho * [(1 - w) + w * mult * c]; the
        # renormalisation c then touches ONLY the perturbed interior, so
        # rescaling the clumping cannot quietly rescale the circumstellar
        # medium as well.
        m_smooth = float(jnp.sum(rho * w) * cell_vol)
        m_clumped = float(jnp.sum(rho * w * ejecta_mult) * cell_vol)
        c = 1.0 if args.pistons else m_smooth / m_clumped
        rho = rho * ((1.0 - w) + w * ejecta_mult * c)
        added = c * m_clumped - m_smooth
        if args.pistons:
            print(f"[orlando] pistons + clumping add {added:.4f} Msun "
                  f"({100 * added / m_smooth:.1f}% of the mapped interior mass; "
                  f"Orlando's five knots total 0.25 Msun = 5% of M_ej)")
        else:
            print(f"[orlando] clumping is mass-neutral (renormalised by "
                  f"{c:.5f}; it only redistributes ejecta mass)")

    # velocity: radial, from the mapped profile, boosted inside the pistons
    v_r = v_r * (1.0 + inside * (velocity_mult - 1.0))
    vx = v_r * X / r_safe
    vy = v_r * Y / r_safe
    vz = v_r * Z / r_safe

    # -------------------------------------------------------------
    # ====== ↓ Stage 2d: composition and the ejecta tracer ↓ ======
    # -------------------------------------------------------------
    scalars = None
    if args.composition:
        # The enclosed-mass coordinate of the 1D profile IS the Lagrangian label
        # (1D spherical flow preserves the ordering exactly), so the chemical
        # stratification can be laid down at the mapping time without the 1D
        # stage having carried any composition itself.
        m_ej = float((meta.get("cfg_ejecta_mass_msun", 3.0) * u.Msun).to(code_units.code_mass).value)
        m_coord, C_ej, r_cd = ejecta_mass_coordinate(
            r, args.profile, ejecta_mass=m_ej,
            interior_wind_mass=meta.get("M_wind_inside_r0", 0.0))
        comp = layered_composition(m_coord, C_ej)
        if piston_species is not None:
            # The knots carry their OWN composition -- this is what produces the
            # observed layer inversion (Fe outside Si). Blend the knot material
            # in by mass: a cell that is half knot material ends up half that
            # species. Without this the knots merely compress whatever the
            # smooth stratification already had there, and no Fe reaches the
            # reverse shock early -- exactly Orlando's point.
            m_tot = sum(piston_species.values())
            frac = m_tot / jnp.maximum(rho, 1e-30)
            frac = jnp.clip(frac, 0.0, 1.0)
            for sp in TRACKED_SPECIES:
                add = piston_species.get(sp, 0.0)
                f_sp = jnp.clip(jnp.asarray(add) / jnp.maximum(rho, 1e-30), 0.0, 1.0) \
                    if not isinstance(add, float) else 0.0
                comp[sp] = comp[sp] * (1.0 - frac) + f_sp
        scalars = jnp.stack([C_ej] + [comp[s] for s in TRACKED_SPECIES])
        shocked_ej = float(jnp.sum(jnp.where(r > r_rs, C_ej * rho, 0.0)) * dx ** 3)
        print(f"[orlando] composition: contact discontinuity at {r_cd:.3f} pc "
              f"(reverse shock {r_rs:.3f}, forward shock {r_fs:.3f}); "
              f"layers {'/'.join(f'{s}:{mass:.2f}' for s, mass in IIB_LAYERS)} Msun")
        print(f"[orlando] scalars: {', '.join(SCALAR_NAMES)} + "
              f"4 shock-history; shocked ejecta at the mapping time = "
              f"{shocked_ej:.3f} Msun (observed 2.8-3.7 at 350 yr)")

    fields = dict(density=rho, velocity_x=vx, velocity_y=vy, velocity_z=vz,
                  gas_pressure=p)
    initial_state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        sharding=sharding, passive_scalars=scalars, gamma=GAMMA, **fields)

    if args.composition:
        # Material shocked BEFORE the mapping time must not restart its clock at
        # zero, or the ionization age is under-counted by the whole pre-mapping
        # history. Seed it from the 1D run's own shock trajectories.
        f_sh, t_since, rho_t = prior_shock_history(
            r, args.profile, rho, map_age=map_age, code_units=code_units,
            mass_coordinate=m_coord, ejecta_fraction=C_ej)
        if t_since is not None:
            i_hist = (registered_variables.passive_scalar_index
                      + registered_variables.num_passive_scalars - 4)
            initial_state = initial_state.at[i_hist + 1].set(f_sh)
            initial_state = initial_state.at[i_hist + 2].set(t_since)
            initial_state = initial_state.at[i_hist + 3].set(rho_t)
            seeded = float(jnp.mean(t_since > 0))
            t_max_yr = float((float(jnp.max(t_since)) * code_units.code_time).to(u.yr).value)
            print(f"[orlando] shock history seeded: {100 * seeded:.1f}% of the box "
                  f"already shocked at the mapping time, longest for "
                  f"{t_max_yr:.0f} yr")
        else:
            print("[orlando] WARNING: the 1D profile carries no shock-radius "
                  "history, so the ionization age restarts at the mapping time "
                  "and is a LOWER BOUND -- regenerate the profile")
    config = finalize_config(config, initial_state.shape)

    t_end = float(((args.age - map_age) * u.yr).to(code_units.code_time).value)
    dfloor = float(n_c * rho_per_n) * 1e-3
    pfloor = float(n_c * p_per_n) * 1e-2
    params = SimulationParams(
        gamma=GAMMA, C_cfl=0.3, t_end=t_end,
        minimum_density=dfloor, minimum_pressure=pfloor,
        minimum_specific_pressure=float(p_per_n / rho_per_n),
        cooling_params=cooling_params,
        thermal_conductivity=float(args.kappa) if args.conduction else 0.0,
    )

    ke = float(jnp.sum(0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2)) * cell_vol)
    print(f"[orlando] evolving {map_age:.0f} -> {args.age:.0f} yr "
          f"(t_end = {t_end:.4f} code), KE = "
          f"{float((ke * code_units.code_energy).to(u.erg).value):.3e} erg, "
          f"v_max = {float((float(jnp.max(jnp.abs(v_r))) * code_units.code_velocity).to(u.km / u.s).value):.0f} km/s")
    return (initial_state, config, params, registered_variables, code_units,
            helper_data, dict(map_age=map_age, r_fs=r_fs, r_rs=r_rs,
                              n_w=n_w, r_fs_ref=r_fs_ref, n_c=n_c,
                              shell=shell_info, pistons=piston_info))


# =============================================================================
# ============ ↓ Shock diagnostics on the 3D state ↓ ==========================
# =============================================================================
def measure_shocks_3d(rho, v_r, r, *, age_yr, code_units, n_w, r_fs_ref, n_c,
                      rho_per_n, nbins=120, homology_tol=0.08, fs_contrast=2.0,
                      shocked_fraction=None, ejecta_fraction=None):
    """Angle-averaged r_FS and r_RS from a 3D state, using the 1D definitions.

    The showcase's old estimator read the reverse shock off a median-temperature
    profile, which breaks as soon as the interior cools or contains
    pressure-floored cavities (it has reported anything from 0.01 to 1.09 pc on
    states whose real r_RS is ~1.5 pc). These are the same two criteria the 1D
    calibration uses -- a density contrast against the KNOWN analytic wind for
    the forward shock, and the departure from homologous expansion for the
    reverse shock -- so 1D and 3D are measured identically and can be compared.

    When the run carried the shock-history scalars, the reverse shock is taken
    straight from them instead: it is the inner edge of the SHOCKED EJECTA, and
    ``shocked_fraction`` says exactly which material that is. That is both more
    direct and more robust than the kinematic proxy — a chi_v = 4.2 piston makes
    a large volume of the interior non-homologous, and the homology criterion
    then finds no unshocked shell at all and returns NaN, which is what it did
    on the first piston run.

    Returns a dict with the two radii (pc) and the radial profiles used.
    """
    r = np.asarray(r); rho = np.asarray(rho); v_r = np.asarray(v_r)
    t_code = float((age_yr * u.yr).to(code_units.code_time).value)

    edges = np.linspace(0.0, float(r.max()), nbins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nbins - 1)
    cnt = np.bincount(idx, minlength=nbins).astype(float)
    rc = 0.5 * (edges[:-1] + edges[1:])

    rho_mean = np.bincount(idx, weights=rho.ravel(), minlength=nbins) / np.maximum(cnt, 1)
    # fraction of each shell that is still expanding homologously
    v_hom = np.maximum(r.ravel(), 1e-12) / t_code
    hom = (np.abs(v_r.ravel() - v_hom) < homology_tol * v_hom).astype(float)
    hom_frac = np.bincount(idx, weights=hom, minlength=nbins) / np.maximum(cnt, 1)

    n_amb = n_w * (r_fs_ref / np.maximum(rc, 1e-6)) ** 2 + n_c
    shocked = (rho_mean > fs_contrast * n_amb * rho_per_n) & (cnt > 0)
    r_fs = float(rc[shocked].max()) if np.any(shocked) else np.nan
    # reverse shock
    if shocked_fraction is not None and ejecta_fraction is not None:
        # the inner edge of the shocked ejecta, straight from the tracer
        w_ej = np.asarray(ejecta_fraction).ravel()
        f_sh = np.asarray(shocked_fraction).ravel()
        num = np.bincount(idx, weights=(f_sh * w_ej), minlength=nbins)
        den = np.bincount(idx, weights=w_ej, minlength=nbins)
        shocked_frac_ej = np.where(den > 1e-12, num / np.maximum(den, 1e-30), np.nan)
        inner = (rc < r_fs) & (rc > 0.05 * r_fs) & np.isfinite(shocked_frac_ej)
        unshocked = inner & (shocked_frac_ej < 0.5)
        r_rs = float(rc[unshocked].max()) if np.any(unshocked) else np.nan
    else:
        # fallback: outermost radius where most of the shell is still homologous
        band = (rc < r_fs) & (rc > 0.05 * r_fs) & (hom_frac > 0.5)
        r_rs = float(rc[band].max()) if np.any(band) else np.nan
        shocked_frac_ej = None
    return dict(r_fs=r_fs, r_rs=r_rs, rc=rc, rho_mean=rho_mean, hom_frac=hom_frac,
                shocked_frac_ej=shocked_frac_ej)


def shock_speed_vs_position_angle(rho, r, X, Y, Z, *, n_w, r_fs_ref, n_c,
                                  rho_per_n, n_angles=36, fs_contrast=2.0,
                                  nbins=120):
    """Forward-shock radius as a function of position angle in the plane of the sky.

    Orlando et al. (2022) identify the shock radius/velocity versus position
    angle as the single most discriminating diagnostic for the circumstellar
    shell parameters, averaged over ~10 degree cones. The Earth vantage point is
    on the -y axis, so the plane of the sky is (x, z) and the position angle is
    measured in it.

    The radius is taken from the CONE-AVERAGED radial density profile, not from
    the outermost individual cell above the contrast. With log-normal
    circumstellar clumping at sigma = 0.4 the +3 sigma tail reaches ~3x the mean,
    so a per-cell test finds unshocked clumps out at the box corner and reports
    the box diagonal (4.95 pc in a 7 pc box) as the shock radius. Averaging
    within the cone first removes the clumping the same way the angle-averaged
    estimator does.

    Returns ``(angles_deg, r_fs_per_angle)``.
    """
    r = np.asarray(r).ravel(); rho = np.asarray(rho).ravel()
    X = np.asarray(X).ravel(); Y = np.asarray(Y).ravel(); Z = np.asarray(Z).ravel()

    # cone half-angle ~10 deg around each direction in the plane of the sky
    pa = np.rad2deg(np.arctan2(Z, X)) % 360.0
    in_plane = np.abs(Y) / np.maximum(r, 1e-12) < np.sin(np.deg2rad(10.0))

    edges = np.linspace(0.0, float(r.max()), nbins + 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    n_amb = n_w * (r_fs_ref / np.maximum(rc, 1e-6)) ** 2 + n_c
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)

    angles = np.linspace(0.0, 360.0, n_angles, endpoint=False)
    width = 360.0 / n_angles
    out = np.full(n_angles, np.nan)
    for i, a in enumerate(angles):
        sel = in_plane & (np.abs(((pa - a + 180.0) % 360.0) - 180.0) < width)
        if not np.any(sel):
            continue
        cnt = np.bincount(idx[sel], minlength=nbins).astype(float)
        tot = np.bincount(idx[sel], weights=rho[sel], minlength=nbins)
        mean = np.divide(tot, cnt, out=np.zeros(nbins), where=cnt > 0)
        hit = (cnt > 0) & (mean > fs_contrast * n_amb * rho_per_n)
        if np.any(hit):
            out[i] = rc[hit].max()
    return angles, out
# =============================================================================
# ============ ↑ Shock diagnostics on the 3D state ↑ ==========================
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", help="casa_calibrate_1d.py --save-profile npz")
    ap.add_argument("--n", type=int, default=256, help="cells per axis")
    ap.add_argument("--box", type=float, default=BOX_SIZE,
                    help="box side (pc). The default 7 pc is sized for the "
                         "350 yr remnant; a SMALLER box mapped at an EARLIER "
                         "age is the first stage of the Orlando expanding-grid "
                         "approach, and buys cells per remnant radius exactly "
                         "when the pistons are imposed")
    ap.add_argument("--age", type=float, default=TARGET_AGE_YR, help="target age (yr)")
    ap.add_argument("--nsnap", type=int, default=21, help="number of snapshots")
    ap.add_argument("--gpus", type=int, default=1, help="number of GPUs (x-axis sharding)")
    ap.add_argument("--cooling", action="store_true", help="radiative cooling (+ limiter + tfloor)")
    ap.add_argument("--positivity", choices=("floor", "redistribute", "conservative"),
                    default="floor",
                    help="positivity enforcement. The default hard floor is "
                         "NON-CONSERVATIVE and is how the failing runs create "
                         "mass (17 Msun -> 1e12 Msun)")
    ap.add_argument("--conduction", action="store_true",
                    help="isotropic thermal conduction; sets a PHYSICAL layer "
                         "thickness (Field length) for the radiative shell, "
                         "which no numerical guard can do. Slow: explicit, with "
                         "a parabolic dt constraint")
    ap.add_argument("--kappa", type=float, default=0.05,
                    help="constant thermal conductivity (code units)")
    ap.add_argument("--clamp-floor", action="store_true",
                    help="clamp a cooling cell ONTO the temperature floor instead "
                         "of discarding its update; more correct, but the revert is "
                         "load-bearing crush protection -- see CoolingParams")
    ap.add_argument("--max-cool-fraction", type=float, default=0.0,
                    help="cap the fractional temperature drop per hydro step "
                         "(0 = off); the local alternative to a wide LLF blend")
    ap.add_argument("--explicit-cooling", action="store_true",
                    help="forward cooling update (cooling time enters the CFL): "
                         "far slower, but rules out a one-step backward-Euler "
                         "jump as the cause of the piston crush")
    ap.add_argument("--coldcrush-factor", type=float, default=8.0,
                    help="width of the first-order LLF blend at crushed cells, "
                         "in multiples of the floor temperature (PositivityConfig)")
    ap.add_argument("--limiter-alpha", type=float, default=None,
                    help="cooling resolution-limiter alpha (default: fixed 0.109 pc cutoff)")
    ap.add_argument("--clump-sigma", type=float, default=1.0,
                    help="scale factor on the Orlando clumping amplitude (0 = smooth)")
    ap.add_argument("--csm-sigma", type=float, default=CSM_SIGMA, help="wind clumping sigma")
    ap.add_argument("--shell", action="store_true", help="add the Orlando et al. (2022) CSM shell")
    ap.add_argument("--shell-radius", type=float, default=SHELL_RADIUS, help="r_sh (pc)")
    ap.add_argument("--shell-density", type=float, default=SHELL_DENSITY, help="n_sh (cm^-3)")
    ap.add_argument("--composition", action="store_true",
                    help="carry passive scalars: an ejecta/CSM discriminator, the "
                         "chemical stratification (Fe, Si, O, He; H is the "
                         "remainder), and the shock history that yields the "
                         "ionization age and the electron/ion relaxation. Costs "
                         "~9 extra state variables plus advection working arrays, "
                         "so it roughly doubles the memory at a given resolution")
    ap.add_argument("--pistons", action="store_true",
                    help="add the five Orlando et al. (2016) Table 4 anisotropies. NOTE: "
                         "cold dense ejecta substructure has repeatedly crossed the "
                         "512^3 stability threshold -- gate it at 256^3 first")
    ap.add_argument("--save-state", type=str, default=None, help="write the final state npz")
    ap.add_argument("--ic-only", action="store_true", help="build and report the IC, do not run")
    args = ap.parse_args()

    sharding = None
    if args.gpus > 1:
        from jax.sharding import PartitionSpec as P
        mesh = jax.make_mesh((args.gpus,), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, P(None, "x"))
        jax.config.update("jax_use_shardy_partitioner", False)

    state, config, params, rv, cu, helper_data, meta = build(args, sharding=sharding)

    if args.ic_only:
        rho = np.asarray(state[rv.density_index])
        p = np.asarray(state[rv.pressure_index])
        T = temperature_K(rho, p, cu)
        print(f"[orlando] IC: rho [{rho.min():.3e}, {rho.max():.3e}], "
              f"T [{np.nanmin(T):.2e}, {np.nanmax(T):.2e}] K")
        return

    snaps = time_integration(state, config, params, rv, sharding=sharding)
    jax.block_until_ready(snaps)
    te = np.asarray(snaps.total_energy)
    print(f"[orlando] time_points:  {np.array2string(np.asarray(snaps.time_points), precision=4, max_line_width=200)}")
    print(f"[orlando] total_energy: {np.array2string(te, precision=4, max_line_width=200)}")

    # MASS is the diagnostic that actually catches this setup failing, and it
    # was requested but never printed. Every blown-up run here CREATES mass --
    # 17.4 Msun becomes 1e12 Msun -- so the failure is a conservation violation,
    # NOT the radiative collapse it looks like. (A cooled shock is isothermal,
    # so its compression ratio is bounded by ~Mach^2 ~ 1e5; rho ~ 1e18 is eleven
    # orders past anything physical.) Watch this, not energy: energy stayed
    # conserved to 0.02% in runs whose mass had already run away.
    tm = np.asarray(snaps.total_mass)
    print(f"[orlando] total_mass:   {np.array2string(tm, precision=4, max_line_width=200)}")
    good = tm[np.isfinite(tm) & (tm > 0)]
    if good.size and good.max() > 1.01 * good[0]:
        print(f"[orlando] MASS NOT CONSERVED: {good[0]:.4e} -> {good.max():.4e} "
              f"({good.max() / good[0]:.3e}x). The run is manufacturing mass; "
              f"nothing downstream of this point is physical.")

    fs = np.asarray(snaps.final_state)
    rho = fs[rv.density_index]
    p = fs[rv.pressure_index]
    vx, vy, vz = (fs[rv.velocity_index.x], fs[rv.velocity_index.y], fs[rv.velocity_index.z])

    if args.save_state:
        extra_fields = {}
        if rv.passive_scalars_active:
            i0 = rv.passive_scalar_index
            for k, name in enumerate(SCALAR_NAMES):
                extra_fields[name] = fs[i0 + k]
            # the library-managed block, in its fixed order
            i_hist = i0 + rv.num_passive_scalars - 4
            for j, name in enumerate(("entropy_initial", "shocked_fraction",
                                      "time_since_shock", "density_time")):
                extra_fields[name] = fs[i_hist + j]
        np.savez_compressed(args.save_state, rho=rho, press=p, vx=vx, vy=vy, vz=vz,
                            box=float(args.box), age=float(args.age),
                            num_cells=args.n, map_age=meta["map_age"],
                            **extra_fields)
        print(f"[orlando] saved state {args.save_state}"
              + (f" (+ {len(extra_fields)} scalar fields)" if extra_fields else ""))

    # ==== ↓ Refuse to report numbers from a state that has left the physical
    # regime ↓ ================================================================
    # The dt-collapse guard inside the solver only fires when dt drops below the
    # float resolution of the clock. A run can instead carry a blown-up state
    # all the way to t_end -- exit code 0, no ABORT, and a perfectly plausible
    # "measured at 350 yr" line -- while holding rho ~ 1e18 and NaN scalars.
    # That happened with the per-step cooling cap, and the printed radii were
    # taken at face value. Check the state, not the exit status.
    rho_max, p_max = float(np.max(rho)), float(np.max(p))
    n_bad = int(np.sum(~np.isfinite(rho)) + np.sum(~np.isfinite(p)))
    if n_bad or rho_max > 1e6 or p_max > 1e9:
        print(f"[orlando] REFUSING to report shock radii: the final state is "
              f"unphysical (rho_max = {rho_max:.3e}, p_max = {p_max:.3e}, "
              f"{n_bad} non-finite cells). Check the log for an ABORT above; if "
              f"there is none the run reached t_end carrying this state, which "
              f"is the WORSE case. Treat earlier snapshots with suspicion too.")
        return
    # ==== ↑ ===================================================================

    # measured shock radii, with the same criteria the 1D calibration used
    r, X, Y, Z = centered_radius(helper_data, args.box, args.n)
    r_np = np.asarray(r)
    r_safe = np.maximum(r_np, 0.5 * args.box / args.n)
    v_r = (vx * np.asarray(X) + vy * np.asarray(Y) + vz * np.asarray(Z)) / r_safe
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(cu.code_density).value)
    # the wind the shock detector compares against must be the CALIBRATED one
    # that produced this profile, not a hard-coded copy
    wind = dict(n_w=meta["n_w"], r_fs_ref=meta["r_fs_ref"], n_c=meta["n_c"])
    f_sh = f_ej = None
    if rv.passive_scalars_active:
        i0 = rv.passive_scalar_index
        f_ej = fs[i0]                                   # C_ej
        f_sh = fs[i0 + rv.num_passive_scalars - 3]      # shocked_fraction
    m = measure_shocks_3d(rho, v_r, r_np, age_yr=args.age, code_units=cu,
                          rho_per_n=rho_per_n, shocked_fraction=f_sh,
                          ejecta_fraction=f_ej, **wind)
    # A radius beyond the inscribed sphere is not a shock: the detector has
    # flagged the domain corners, where the 1/r^2 ambient it compares against
    # has fallen below the local density. Seen at N = 192 and 224, which report
    # r_FS = 6.005 pc -- the box DIAGONAL half-length -- from entirely healthy
    # states. Say so rather than printing it next to the observed value.
    if not np.isfinite(m["r_fs"]) or m["r_fs"] > 0.5 * args.box:
        print(f"[orlando] r_FS = {m['r_fs']:.3f} pc is outside the inscribed "
              f"sphere (half-box {0.5 * args.box:.3f} pc): the shock detector "
              f"has locked onto the domain corners, NOT a measurement. The "
              f"state itself may be fine -- check rho_max before concluding "
              f"anything about the run.")
    print(f"[orlando] measured at {args.age:.0f} yr: r_FS = {m['r_fs']:.3f} pc "
          f"(observed 2.52 +- 0.20), r_RS = {m['r_rs']:.3f} pc (observed 1.58 +- 0.16)")

    angles, r_pa = shock_speed_vs_position_angle(
        rho, r_np, np.asarray(X), np.asarray(Y), np.asarray(Z),
        rho_per_n=rho_per_n, **wind)
    print(f"[orlando] r_FS vs position angle: min {np.nanmin(r_pa):.3f}, "
          f"max {np.nanmax(r_pa):.3f}, spread {np.nanmax(r_pa) - np.nanmin(r_pa):.3f} pc")

    T = temperature_K(rho, p, cu)
    dx_cm = (args.box / args.n) * float((1.0 * cu.code_length).to(u.cm).value)
    # name the figures after the state, or parallel legs of a ladder silently
    # overwrite each other's output
    stem = (Path(args.save_state).stem if args.save_state
            else f"casa_orlando_n{args.n}")
    realistic_figure(rho, T, r_np, args.box,
                     title=f"Cas A, Orlando Route B ({args.age:.0f} yr, "
                           f"mapped at {meta['map_age']:.0f} yr)",
                     out_path=FIGURES_DIR / f"{stem}_realistic.png")
    chandra_deep_figure(rho, p, cu, args.box, dx_cm,
                        FIGURES_DIR / f"{stem}_chandra.png", observe=True)
    print(f"[orlando] wrote figures/{stem}_{{realistic,chandra}}.png")


if __name__ == "__main__":
    main()
