"""
Cassiopeia A, the realistic version: clumpy ejecta hitting a turbulent,
asymmetric circumstellar medium, with radiative cooling (and optional conduction).

This builds on ``cassiopeia.py`` (cold freely-expanding ejecta, 1.5e51 erg,
3.3 M_sun, into the r^-2 progenitor wind) and adds the ingredients that make a
young core-collapse remnant look the way it does in observations and in the
Orlando et al. (2021, 2025) Cas A models:

  * a **dense, asymmetric circumstellar shell** at ~1.7 pc -- the pre-supernova
    eruptive-mass-loss shell that Orlando et al. (2025) invoke for Cas A's
    "Green Monster" (n ~ 180 cm^-3, denser than the shocked main shell), here a
    Gaussian shell, lopsided toward +z;
  * a **turbulent / clumpy medium**: band-limited log-normal density
    fluctuations in the wind and shell, so the circumstellar gas is structured
    rather than perfectly smooth;
  * **clumpy ejecta**: fractional density perturbations in the ejecta that, on
    contact with the reverse shock and the dense shell, grow into
    Rayleigh-Taylor fingers and fragment into knots and filaments;
  * **radiative cooling** (Schure et al. 2009 ISM curve, implicit) -- radiative
    losses in the shocked shell enhance the fragmentation into thin dense
    filaments;
  * **optional thermal conduction** (``--conduction``): isotropic constant-kappa;
    off by default because the explicit parabolic timestep in the near-vacuum
    hot bubble is expensive.

Same high-order finite-difference (WENO) solver and single-precision
positivity-preserving flux limiter as the rest of the showcase.

Resolution matters here: the instabilities and filaments are resolution-limited,
so the structure sharpens with ``--n``. The default 128^3 shows the clumpy shell
and the onset of fingering in a few minutes on one GPU; 256^3 resolves the
filaments far better. Writes ``figures/cassiopeia_realistic.png``.
"""

# ==== GPU selection ====
# --gpus must be known BEFORE jax initialises, so pre-parse it here.
# Under a scheduler (pq) CUDA_VISIBLE_DEVICES is already pinned to the
# assigned GPUs -- only fall back to autocvd for interactive runs.
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

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# units and constants
from astropy import units as u
import astropy.constants as const

# astronomix containers / functions
from astronomix import (
    SimulationParams,
    SnapshotSettings,
    get_registered_variables,
    get_helper_data,
    finalize_config,
    construct_primitive_state,
    time_integration,
)

# shared showcase helpers
from _common import (
    GAMMA,
    FIGURES_DIR,
    snr_code_units,
    make_fd_config,
    fd_positivity,
    centered_radius,
    freely_expanding_ejecta,
    temperature_K,
    radial_velocity_field,
    realistic_figure,
    chandra_deep_figure,
    schure_cooling_setup,
    turbulent_field,
    nickel_bubble_field,
    dense_csm_shell,
    xray_figure,
    multiwavelength_figure,
)


# ---------------------------------------------------------------------------
# problem parameters (Orlando et al. 2021, 2025)
# ---------------------------------------------------------------------------
BOX_SIZE = 7.0                 # pc, box [-3.5, 3.5]^3
# NOTE: an E = 2e51 + N_C = 0.05 retune (to reach the observed 2.5 pc forward
# shock faster) was tried and REVERTED: the stronger shock into the thinner
# ambient crosses the 512^3 radiative-crush threshold all by itself (isolation
# probe blew at t=0.028 with no bubbles and no knots), while everything is
# stable at 1.5e51. The forward-shock radius is instead matched by running
# longer (t_end ~ 0.35 -> FS ~ 2.3-2.4 pc at the measured ~5,000 km/s).
EXPLOSION_ENERGY = 1.5e51       # erg
EJECTA_MASS = 3.3               # Msun
EJECTA_RADIUS = 1.5             # pc
# r^-2 wind
N_W, R_FS, N_C = 0.8, 2.5, 0.1  # n(r) = N_W (R_FS/r)^2 + N_C  [cm^-3]
WIND_TEMPERATURE = 1e4 * u.K
MASS_PER_NUCLEUS = 1.4
# dense circumstellar shell ("Green Monster", Orlando et al. 2025)
SHELL_RADIUS = 1.7              # pc
SHELL_THICKNESS = 0.18          # pc (Gaussian sigma)
SHELL_PEAK_DENSITY = 60.0       # cm^-3 (Orlando quote ~180; softened for the showcase grid)
SHELL_ASYMMETRY = 0.5           # 0 = spherical, 1 = fully one-sided (denser toward +z)
# turbulence / clumping
CSM_SIGMA = 0.4                 # log-normal density fluctuation amplitude in the CSM
CSM_K_MIN, CSM_K_MAX = 4, 20    # CSM wavenumber band (small-scale wind clumps)
# Ejecta: large-scale-dominated perturbations (a few big Ni/Fe-like plumes plus
# smaller clumps), matching the Orlando et al. picture where the remnant
# morphology is set by large-scale explosion asymmetries interacting with the
# reverse shock -- not fine speckle. Low k + steep red slope -> big plumes.
EJECTA_CLUMP_SIGMA = 0.5        # fractional density clumping of the ejecta
EJECTA_K_MIN, EJECTA_K_MAX = 2, 10
EJECTA_SLOPE = -2.0             # steeper (redder) => more power on large scales
# Small-scale (knot) clump component: the real X-ray-bright ejecta is a lace
# of 0.02-0.05 pc line-emitting knots; the large-scale spectrum above sets the
# plumes but seeds nothing at knot scales, so the shell fragments too little.
EJECTA_SMALL_SIGMA = 0.25
EJECTA_SMALL_K_MIN, EJECTA_SMALL_K_MAX = 10, 40
EJECTA_SMALL_SLOPE = -1.0
SEED = 7
# Bipolar ejecta jet (Cas A's NE jet / weaker SW counter-jet: fast Si-rich
# ejecta "pistons" from the explosion; Orlando et al. 2016, 2022). Modeled as
# a cone in which the ejecta edge is stretched outward (the homologous v ∝ r
# then makes the jet material the fastest) and the steep r^-9 envelope is
# flattened to an effective r^-JET_SLOPE piston, so the cone carries real mass.
JET_AXIS = (-1.0, 1.0, 0.25)    # NE-ish in the x-y projection (upper left)
JET_OPENING_DEG = 14.0          # Gaussian half-opening angle of the cone
JET_ELONGATION = 0.5            # fractional edge-radius stretch of the NE lobe
JET_COUNTER_FRACTION = 0.6      # SW counter-jet strength relative to NE
JET_SLOPE = 3.0                 # effective envelope slope inside the cone


def build(num_cells, t_end, cooling=True, conduction=False, kappa=0.05,
          clump_sigma=EJECTA_CLUMP_SIGMA, csm_sigma=CSM_SIGMA,
          shell_density=SHELL_PEAK_DENSITY, num_snapshots=5,
          dual_energy=False, low_mem=False, jet=False, ni_bubbles=True,
          knot_sigma=EJECTA_SMALL_SIGMA, energy_erg=EXPLOSION_ENERGY,
          ambient_nc=N_C, limiter_alpha=4.0, tfloor=False, tfloor_stage=False,
          ambient_from=None, sharding=None):
    code_units = snr_code_units()
    cooling_config, cooling_params = (None, None)
    if cooling:
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
    extra = dict(random_seed=SEED)
    if conduction:
        extra["thermal_conduction"] = True
    if dual_energy:
        # Bryan+95 dual energy: keeps the cold, kinetic-energy-dominated ejecta
        # core's pressure recovery out of the float32 catastrophic-cancellation
        # regime (routes WENO through the native backend).
        extra["dual_energy"] = True
    if low_mem:
        # 2N-storage LSRK4 + donated state buffers: one fewer full-state
        # register at peak — for hero-resolution runs at the memory edge.
        from astronomix.option_classes.simulation_config import RK4_LSRK
        extra["time_integrator"] = RK4_LSRK
        extra["donate_state"] = True
    if tfloor or tfloor_stage:
        # density-scaled pressure floor (Athena tfloor at the cooling floor
        # temperature) — the isothermal support that stops radiatively cooled
        # shock layers from ram-crushing without bound. Only for runs with
        # real cooling; the adiabatic hero recipe stays floor-free. The
        # per-stage variant closes the intra-step crush window.
        extra["positivity_config"] = fd_positivity(
            tfloor=tfloor, tfloor_stage=tfloor_stage)
    config = make_fd_config(BOX_SIZE, num_cells, mhd=False,
                            cooling_config=cooling_config,
                            snapshot_settings=snaps, num_snapshots=num_snapshots,
                            **extra)
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config, sharding)

    # -------------------------------------------------------------
    # ====== ↓ Initial condition (clumpy ejecta + CSM) ↓ ==========
    # -------------------------------------------------------------
    r, X, Y, Z = centered_radius(helper_data, BOX_SIZE, num_cells)
    dx = BOX_SIZE / num_cells
    r_safe = jnp.maximum(r, 0.5 * dx)

    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(code_units.code_density).value)
    p_per_n = float((const.k_B * WIND_TEMPERATURE / u.cm ** 3).to(code_units.code_pressure).value)

    # smooth r^-2 wind + dense asymmetric shell. The wind cusp is capped inside
    # the ejecta: uncapped, the innermost cells (r ~ dx/2) get an immobile
    # ~10^4 cm^-3 wind knot that the ejecta is merely ADDED onto -- it survives
    # the whole run as a spurious bright dot / projection artifact at the box
    # centre (and grows ∝ N^2). The wind profile only matters outside the
    # region the explosion has swept anyway.
    r_wind = jnp.maximum(r_safe, 0.5 * EJECTA_RADIUS)
    n_wind = N_W * (R_FS / r_wind) ** 2 + ambient_nc
    rho_wind = n_wind * rho_per_n
    rho_shell = dense_csm_shell(
        r, X, Y, Z, shell_radius=SHELL_RADIUS, shell_thickness=SHELL_THICKNESS,
        peak_number_density=shell_density, rho_per_n=rho_per_n,
        asymmetry=SHELL_ASYMMETRY,
    )
    rho_csm_smooth = rho_wind + rho_shell

    # CSM small-scale structure: either the legacy imposed log-normal clumping,
    # or (--ambient-from) a SELF-CONSISTENT driven-turbulence box from
    # casa_turb_phase.py modulated onto the smooth profile -- density and
    # pressure are scaled by the same profile factor, so the temperature field
    # and the local density/velocity/pressure correlations the solver built
    # survive; the ambient turbulent velocity is added mass-weighted after the
    # ejecta are laid down.
    keys = jax.random.split(jax.random.PRNGKey(SEED), 4)
    v_ambient = None
    if ambient_from is not None:
        turb = np.load(ambient_from)
        if int(turb["num_cells"]) != num_cells:
            raise ValueError(
                f"--ambient-from grid ({int(turb['num_cells'])}^3) does not "
                f"match --n {num_cells} (resampling not implemented)")
        rho_t = jnp.asarray(turb["rho"])
        T_t = jnp.asarray(turb["press"]) / rho_t          # code T~ field
        mod = rho_t / jnp.mean(rho_t)
        rho_csm = rho_csm_smooth * mod
        p_csm = rho_csm * T_t                              # preserves T(x)
        v_ambient = (jnp.asarray(turb["vx"]), jnp.asarray(turb["vy"]),
                     jnp.asarray(turb["vz"]))
    else:
        csm_clump = 1.0
        if csm_sigma > 0:
            d_csm = turbulent_field(num_cells, keys[0], kmin=CSM_K_MIN, kmax=CSM_K_MAX, slope=-1.0)
            csm_clump = jnp.exp(csm_sigma * d_csm - 0.5 * csm_sigma ** 2)
        rho_csm = rho_csm_smooth * csm_clump
        # keep the CSM in rough pressure equilibrium at the wind temperature
        p_csm = rho_csm / rho_per_n * p_per_n

    # clumpy ejecta: large-scale plumes + clumps that grow into Rayleigh-Taylor
    # fingers at the reverse shock (the dominant Cas A morphology driver).
    # Ejecta clumping. The large-scale plume component keeps the ORIGINAL
    # linear ``1 + sigma*g`` statistics: this exact distribution is what every
    # stable 512^3 run used, and a log-normal refactor of it (heavier +3sigma
    # tail -> clump peaks ~2x denser) pushed the reverse-shock/clump crush
    # over the stability threshold (hero-7 forensics, r~1.2 pc, off-seam).
    # The OPTIONAL small-scale knot component multiplies in as a log-normal
    # factor (strictly positive, no vacuum clipping), and the optional
    # Ni-bubble factor multiplies on top.
    clump_multiplier = None
    if clump_sigma > 0:
        gbig = turbulent_field(num_cells, keys[1], kmin=EJECTA_K_MIN,
                               kmax=EJECTA_K_MAX, slope=EJECTA_SLOPE)
        clump_multiplier = jnp.clip(1.0 + clump_sigma * gbig, 0.0, None)
        if knot_sigma > 0:
            gsmall = turbulent_field(num_cells, keys[2],
                                     kmin=EJECTA_SMALL_K_MIN,
                                     kmax=min(EJECTA_SMALL_K_MAX, num_cells // 6),
                                     slope=EJECTA_SMALL_SLOPE)
            clump_multiplier = clump_multiplier * jnp.exp(
                knot_sigma * gsmall - 0.5 * knot_sigma ** 2)

    # radioactive Ni-bubble structure: evacuated bubbles with compressed walls
    # -> the ring-shaped interior ejecta emission of the real remnant
    if ni_bubbles:
        bubbles = nickel_bubble_field(X, Y, Z, keys[3], EJECTA_RADIUS)
        clump_multiplier = bubbles if clump_multiplier is None else clump_multiplier * bubbles

    ejecta_clump = None if clump_multiplier is None else clump_multiplier - 1.0

    # bipolar jet: elongate the ejecta edge along the cone and flatten the
    # r^-9 envelope there to an effective r^-JET_SLOPE, so the cone is a fast
    # massive piston (NE lobe full strength, SW counter-jet weaker)
    edge_radius_field = None
    if jet:
        axn = jnp.asarray(JET_AXIS, dtype=r.dtype)
        axn = axn / jnp.sqrt(jnp.sum(axn ** 2))
        mu = (X * axn[0] + Y * axn[1] + Z * axn[2]) / r_safe
        w = 1.0 - jnp.cos(jnp.deg2rad(JET_OPENING_DEG))
        cone = (jnp.exp(-((1.0 - mu) / w) ** 2)
                + JET_COUNTER_FRACTION * jnp.exp(-((1.0 + mu) / w) ** 2))
        edge_radius_field = EJECTA_RADIUS * (1.0 + JET_ELONGATION * cone)
        r_core = 0.5 * EJECTA_RADIUS          # core_fraction default of the ejecta
        envelope_boost = jnp.where(
            r > r_core, cone * (r_safe / r_core) ** (9.0 - JET_SLOPE), 0.0)
        ejecta_clump = envelope_boost if ejecta_clump is None else ejecta_clump + envelope_boost

    fields, info = freely_expanding_ejecta(
        helper_data, code_units, BOX_SIZE, num_cells,
        explosion_energy_erg=energy_erg, ejecta_mass_msun=EJECTA_MASS,
        ejecta_radius=EJECTA_RADIUS, rho_ambient=rho_csm, p_ambient=p_csm,
        mass_per_nucleus=MASS_PER_NUCLEUS, clump_field=ejecta_clump,
        edge_radius_field=edge_radius_field,
    )

    # driven-turbulence ambient: add its velocity field mass-weighted (the
    # ejecta velocity is already the ejecta-mass-weighted contribution)
    if v_ambient is not None:
        w_amb = rho_csm / fields["density"]
        fields["velocity_x"] = fields["velocity_x"] + w_amb * v_ambient[0]
        fields["velocity_y"] = fields["velocity_y"] + w_amb * v_ambient[1]
        fields["velocity_z"] = fields["velocity_z"] + w_amb * v_ambient[2]

    initial_state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        sharding=sharding, **fields
    )
    config = finalize_config(config, initial_state.shape)

    # Floors: keep density above a small fraction of the un-shocked wind floor, and
    # pressure at a small fraction of the un-shocked wind pressure. The pressure
    # floor is NOT set arbitrarily tiny -- a near-zero floor lets cooled/clumped
    # cells lose their sound speed and the high-Mach WENO reconstruction across
    # them cascades to vacuum. This modest floor keeps the coldest gas subsonic.
    dfloor = float(ambient_nc * rho_per_n) * 1e-3
    pfloor = float(ambient_nc * p_per_n) * 1e-2
    # Effective temperature floor (p >= rho * p_per_n/rho_per_n corresponds to
    # T = WIND_TEMPERATURE = 1e4 K, matching the cooling floor): a constant
    # pressure floor cannot resist ram-pressure crushing of radiatively cooled
    # dense gas -- at 512^3 the jet piston crushed floored cone cells to
    # rho ~ 1e16 and collapsed dt. This floor stiffens with compression and
    # caps the density at the physical isothermal shock jump.
    spfloor = float(p_per_n / rho_per_n)
    params = SimulationParams(
        gamma=GAMMA, C_cfl=0.3, t_end=t_end,
        minimum_density=dfloor, minimum_pressure=pfloor,
        minimum_specific_pressure=spfloor,
        cooling_params=cooling_params,
        thermal_conductivity=float(kappa) if conduction else 0.0,
    )
    # -------------------------------------------------------------
    # ====== ↑ Initial condition (clumpy ejecta + CSM) ↑ ==========
    # -------------------------------------------------------------

    t0_yr = float((EJECTA_RADIUS * code_units.code_length / (info["v_max_kms"] * u.km / u.s)).to(u.yr).value)
    age_yr = t0_yr + float((t_end * code_units.code_time).to(u.yr).value)
    print(f"[casa-real] N={num_cells} dx={info['dx']:.3f}pc v_max={info['v_max_kms']:.0f} km/s "
          f"cooling={cooling} conduction={conduction} jet={jet} bubbles={ni_bubbles}")
    print(f"[casa-real] shell: n_peak={shell_density} cm^-3 @ {SHELL_RADIUS} pc "
          f"(sigma {SHELL_THICKNESS} pc, asym {SHELL_ASYMMETRY}); "
          f"CSM sigma={csm_sigma}, ejecta clump sigma={clump_sigma}")
    print(f"[casa-real] E={info['KE_achieved']:.3e}/{info['E_target']:.3e} "
          f"M_ej={info['M_ej_achieved']:.3f}/{info['M_ej_target']:.3f} evolved to ~{age_yr:.0f} yr")
    return initial_state, config, params, registered_variables, code_units, age_yr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="cells per axis")
    ap.add_argument("--t-end", type=float, default=0.25,
                    help="evolution time (code units ~978 yr); default lands the clumpy "
                         "remnant near Cas A's ~350 yr age / ~2.5 pc shock radius")
    ap.add_argument("--no-cooling", action="store_true", help="disable radiative cooling")
    ap.add_argument("--conduction", action="store_true", help="enable thermal conduction (slow)")
    ap.add_argument("--kappa", type=float, default=0.05, help="thermal conductivity (if --conduction)")
    ap.add_argument("--clump-sigma", type=float, default=EJECTA_CLUMP_SIGMA,
                    help="ejecta clumping amplitude (0 = smooth)")
    ap.add_argument("--csm-sigma", type=float, default=CSM_SIGMA,
                    help="CSM log-normal density fluctuation amplitude (0 = smooth)")
    ap.add_argument("--shell-density", type=float, default=SHELL_PEAK_DENSITY,
                    help="dense CSM shell peak number density (cm^-3; 0 = no shell)")
    ap.add_argument("--save-state", type=str, default=None,
                    help="npz path to save the final density/pressure (for offline re-imaging)")
    ap.add_argument("--nsnap", type=int, default=5, help="number of diagnostic snapshots")
    ap.add_argument("--jet", action="store_true",
                    help="add Cas A's bipolar ejecta jet (NE jet + weaker SW "
                         "counter-jet): fast massive pistons along a cone")
    ap.add_argument("--knot-sigma", type=float, default=EJECTA_SMALL_SIGMA,
                    help="small-scale (knot) ejecta clump amplitude (0 = off)")
    ap.add_argument("--energy-51", type=float, default=EXPLOSION_ENERGY / 1e51,
                    help="explosion energy in 1e51 erg (2.0 matches the "
                         "observed 2.5 pc forward shock faster; needs the "
                         "cooling limiter to be stable at 512^3)")
    ap.add_argument("--ambient-nc", type=float, default=N_C,
                    help="uniform ambient floor density N_C (cm^-3)")
    ap.add_argument("--limiter-alpha", type=float, default=4.0,
                    help="cooling-resolution limiter threshold in cells "
                         "(suppress cooling where l_cool < alpha*dx; 0 = off)")
    ap.add_argument("--tfloor", action="store_true",
                    help="per-step density-scaled pressure floor at the "
                         "cooling floor temperature (isothermal support for "
                         "radiatively cooled dense layers; use with cooling)")
    ap.add_argument("--tfloor-stage", action="store_true",
                    help="apply the density-scaled floor inside every RK "
                         "stage as well (closes the intra-step crush window; "
                         "only with real cooling)")
    ap.add_argument("--ambient-from", type=str, default=None,
                    help="npz from casa_turb_phase.py: modulate the driven-"
                         "turbulence box onto the smooth wind+shell profile "
                         "instead of imposed log-normal CSM clumping")
    ap.add_argument("--no-bubbles", action="store_true",
                    help="disable the radioactive Ni-bubble ejecta structure "
                         "(evacuated bubbles + compressed walls -> the "
                         "ring-shaped interior emission of the real remnant)")
    ap.add_argument("--dual-energy", action="store_true",
                    help="Bryan+95 dual-energy formalism (fixes the float32 "
                         "cold-core pressure cancellation at high N)")
    ap.add_argument("--low-mem", action="store_true",
                    help="LSRK4 low-storage integrator + donated state buffers. "
                         "CAUTION: LSRK4 is not SSP, so the positivity-"
                         "preserving flux limiter's guarantee does not hold on "
                         "this cold-blast problem -- observed to inflate energy "
                         "at N=64 and collapse dt at 512^3. Prefer the default "
                         "SSP integrator on larger GPUs.")
    ap.add_argument("--gpus", type=int, default=1,
                    help="domain-decompose along x across this many GPUs "
                         "(must divide 8 for the ghost padding; consumed "
                         "before jax initialises)")
    args = ap.parse_args()

    print(f"[casa-real] devices: {jax.devices()}  x64={jax.config.jax_enable_x64}")

    # Multi-GPU: domain-decompose along the x axis (periodic BCs are
    # sharding-safe; the ghost padding needs the GPU count to divide 8).
    sharding = None
    if args.gpus > 1:
        # JAX >= 0.10 defaults to the Shardy partitioner, which rejects this
        # codebase's integer mesh-axis names; fall back to GSPMD.
        jax.config.update("jax_use_shardy_partitioner", False)
        from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
        from astronomix.option_classes.simulation_config import (
            VARAXIS, XAXIS, YAXIS, ZAXIS,
        )
        # Auto (not the JAX>=0.10 default Explicit) axes so the library's
        # with_sharding_constraint calls work.
        mesh = jax.make_mesh(
            (1, args.gpus, 1, 1), (VARAXIS, XAXIS, YAXIS, ZAXIS),
            axis_types=(AxisType.Auto,) * 4,
        )
        sharding = NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))

    state, config, params, rv, cu, age_yr = build(
        args.n, args.t_end, cooling=not args.no_cooling,
        conduction=args.conduction, kappa=args.kappa,
        clump_sigma=args.clump_sigma, csm_sigma=args.csm_sigma,
        shell_density=args.shell_density, num_snapshots=args.nsnap,
        dual_energy=args.dual_energy, low_mem=args.low_mem, jet=args.jet,
        ni_bubbles=not args.no_bubbles, knot_sigma=args.knot_sigma,
        energy_erg=args.energy_51 * 1e51, ambient_nc=args.ambient_nc,
        limiter_alpha=args.limiter_alpha, tfloor=args.tfloor,
        tfloor_stage=args.tfloor_stage, ambient_from=args.ambient_from,
        sharding=sharding,
    )
    snaps = time_integration(state, config, params, rv, sharding=sharding)
    jax.block_until_ready(snaps)
    # energy/time series -- shows exactly when (if) the run diverges
    print(f"[casa-real] time_points:  {np.array2string(np.asarray(snaps.time_points), precision=4, max_line_width=200)}")
    print(f"[casa-real] total_energy: {np.array2string(np.asarray(snaps.total_energy), precision=4, max_line_width=200)}")

    fs = np.asarray(snaps.final_state)
    rho = fs[rv.density_index]
    p = fs[rv.pressure_index]
    T = temperature_K(rho, p, cu)
    helper_data = get_helper_data(config, sharding)
    r, _ = radial_velocity_field(fs, rv, helper_data, BOX_SIZE, args.n, cu)

    te = np.asarray(snaps.total_energy)
    print(f"[casa-real] final rho[{rho.min():.3e},{rho.max():.3e}] "
          f"T[{np.nanmin(T):.1e},{np.nanmax(T):.1e}]K  total_energy {te[0]:.3e} -> {te[-1]:.3e}")

    # Save the state BEFORE imaging: a plotting hiccup must never cost the
    # simulation itself (hero runs take hours). Re-image offline from the npz.
    # Velocities enable kinematic imaging offline (ejecta-vs-CSM
    # discrimination, Doppler views like the Green Monster's -2300 km/s
    # blueshift).
    if args.save_state:
        np.savez_compressed(
            args.save_state, rho=rho, press=p,
            vx=fs[rv.velocity_index.x], vy=fs[rv.velocity_index.y],
            vz=fs[rv.velocity_index.z],
            box=float(BOX_SIZE), age=age_yr, num_cells=args.n)
        print(f"[casa-real] saved state {args.save_state}")

    cond_tag = "+ conduction" if args.conduction else ""
    out = realistic_figure(
        rho, T, r, BOX_SIZE,
        title=f"Cassiopeia A (realistic: clumpy ejecta, dense CSM shell, cooling {cond_tag}, "
              f"~{age_yr:.0f} yr)",
        out_path=FIGURES_DIR / "cassiopeia_realistic.png",
    )
    print(f"[casa-real] saved {out}")

    # Chandra-like synthetic X-ray view for side-by-side comparison with the
    # real telescope image.
    dx_cm = (BOX_SIZE / args.n) * float((1.0 * cu.code_length).to(u.cm).value)
    xout = xray_figure(
        rho, p, cu, BOX_SIZE, dx_cm,
        title=f"Cassiopeia A -- synthetic Chandra X-ray (~{age_yr:.0f} yr)",
        out_path=FIGURES_DIR / "cassiopeia_xray.png",
    )
    print(f"[casa-real] saved {xout}")

    # Deep single-band Chandra-style press image (blue on black).
    cout = chandra_deep_figure(
        rho, p, cu, BOX_SIZE, dx_cm,
        out_path=FIGURES_DIR / "cassiopeia_chandra.png",
    )
    print(f"[casa-real] saved {cout}")

    # X-ray + infrared multiwavelength composite (Chandra + JWST style).
    mout = multiwavelength_figure(
        rho, p, cu, BOX_SIZE, dx_cm,
        title=f"Cassiopeia A -- synthetic X-ray + infrared composite (~{age_yr:.0f} yr)",
        out_path=FIGURES_DIR / "cassiopeia_composite.png",
    )
    print(f"[casa-real] saved {mout}")


if __name__ == "__main__":
    main()
