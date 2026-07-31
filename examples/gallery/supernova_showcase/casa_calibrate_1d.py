"""
Cassiopeia A, step 1 of the Orlando ladder: calibrate the explosion in 1D.

The 3D showcase runs (``cassiopeia_realistic.py``) start a cold, homologous
ejecta ball already at 1.5 pc and tag it with an age ``R / v_max``. That skips
the phase that actually sets the answer: an r^-2 wind with n_w = 0.8 cm^-3 at
2.5 pc already contains ~3.3 M_sun -- an entire ejecta mass -- inside 1.5 pc, so
by that radius the remnant should long since have swept up its own mass, driven
a reverse shock and given a large part of its energy to the shocked wind.

This script does that phase properly and cheaply, following Orlando et al.
(2016)'s Route B: a **1D spherically symmetric** run of the same cold,
freely-expanding ejecta into the same r^-2 progenitor wind, started small
(``--r0``, default 0.05 pc, where the swept wind mass is still only ~3% of
M_ej so free expansion is an excellent approximation) and evolved to Cas A's
age. Every snapshot is measured against the observational targets:

    r_FS  = 2.52 +- 0.20 pc      (Gotthelf et al. 2001)
    r_RS  = 1.58 +- 0.16 pc      (Gotthelf et al. 2001)
    v_FS  = 5000-5500 km/s       (Vink et al. 1998, 2022)
    v_RS  = 2000-4000 km/s       (observer frame, E/N; Vink et al. 2022)
    n_post= 3-5 cm^-3            (post-shock CSM; Lee et al. 2014)

so a parameter set can be accepted or rejected in seconds instead of after an
eight-hour 512^3 run. ``--scan`` sweeps the degenerate (E_SN, M_ej, n_w) block
and prints a ranked table; the winning profile is written to an npz that
``cassiopeia_realistic.py --profile-from`` maps into 3D at the chosen time,
which is where the multi-D structure (clumping, pistons, CSM shell) is imposed.

Shock definitions used here (all measured on the actual profile, no fitting):

  * the **reverse shock** is the outer edge of the still-freely-expanding
    ejecta: the largest radius at which the velocity is still homologous,
    ``|v - r/t| < homology_tol * r/t``. Unshocked ejecta satisfies v = r/t
    exactly, so this is sharp and needs no thresholds on density or entropy.
  * the **forward shock** is the outermost radius at which the density exceeds
    twice the (known, analytic) ambient wind profile -- a strong shock
    compresses by 4, so the factor 2 sits safely inside the jump.
  * ``v_FS``/``v_RS`` are finite differences of those radii between snapshots,
    i.e. shock speeds in the observer frame, exactly as quoted from proper
    motions.

Runs in float64 on the CPU in well under a minute; no GPU needed.
"""

# ==== precision / device ====
# 1D is tiny: run in float64 (the cold ejecta core has a ~10^-6 pressure
# contrast, which is exactly the float32 cancellation regime the 3D runs need
# the dual-energy formalism for) and default to the CPU so a calibration sweep
# never has to queue for a GPU.
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
if "--gpu" not in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import itertools
from pathlib import Path

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# units and constants
from astropy import units as u
import astropy.constants as const

# astronomix constants
from astronomix import SPHERICAL
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    POSITIVITY_HARD_FLOOR,
    PositivityConfig,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
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
    MASS_PER_NUCLEUS,
    ejecta_radial_shape,
    schure_cooling_setup,
    snr_code_units,
)


# =============================================================================
# ============ ↓ Observational targets (Cas A at 350 yr) ↓ ====================
# =============================================================================
AGE_YR = 350.0                      # Fesen et al. 2006
TARGETS = {
    # name:        (value, tolerance, unit)
    "r_FS":        (2.52, 0.20, "pc"),      # Gotthelf et al. 2001
    "r_RS":        (1.58, 0.16, "pc"),      # Gotthelf et al. 2001
    "v_FS":        (5250.0, 250.0, "km/s"),  # Vink et al. 1998, 2022
    "v_RS":        (3000.0, 1000.0, "km/s"),  # observer frame E/N, Vink et al. 2022
    "n_post":      (4.0, 1.0, "cm^-3"),     # Lee et al. 2014
    # The constraint the original calibration MISSED. It fixed the shock radii
    # and speeds but said nothing about how far the reverse shock has eaten into
    # the ejecta IN MASS -- and the 3D runs then came out with 96% of the ejecta
    # shocked against the ~87-90% observed, which shows up as too much shocked
    # Si and a reverse shock 0.35 pc too far in.
    "m_unshocked": (0.35, 0.10, "Msun"),    # DeLaney et al. 2014; Hwang & Laming 2012
}
# =============================================================================
# ============ ↑ Observational targets (Cas A at 350 yr) ↑ ====================
# =============================================================================


# =============================================================================
# ============ ↓ 1D initial condition ↓ =======================================
# =============================================================================
def wind_number_density(r, *, n_w, r_fs_ref, n_c, r_cap):
    """The progenitor's r^-2 RSG wind plus a uniform floor, in cm^-3.

    ``n(r) = n_w (r_fs_ref / r)^2 + n_c``, with the cusp capped inside
    ``r_cap`` (the region the explosion has swept clean anyway; uncapped it
    puts an immobile ~10^4 cm^-3 knot at the origin).
    """
    return n_w * (r_fs_ref / jnp.maximum(r, r_cap)) ** 2 + n_c


def build_1d(cfg):
    """Assemble the 1D spherical initial state and the matching config/params.

    Returns ``(state, config, params, registered_variables, helper_data,
    code_units, info)`` where ``info`` carries the exact achieved mass and
    energy plus ``t0_yr``, the age the homologous profile already represents
    (``t0 = 1/s`` for ``v = s r``, exactly).
    """
    code_units = snr_code_units()

    snaps = SnapshotSettings(
        return_states=True, return_final_state=True,
        return_total_mass=True, return_total_energy=True,
    )

    cooling_config = cooling_params = None
    if cfg["cooling"]:
        cooling_config, cooling_params = schure_cooling_setup(
            code_units, floor_temperature_K=1e4,
            hydrogen_mass_fraction=1.0 - 0.28 - 0.02, metal_mass_fraction=0.02,
        )

    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        geometry=SPHERICAL,
        dimensionality=1,
        box_size=cfg["r_max"],
        num_cells=cfg["num_cells"],
        # REQUIRED here: the plain 2nd-order MUSCL/minmod reconstruction NaNs on
        # the very first step on the r^-9 ejecta envelope (verified: the failure
        # threshold is v_max ~ 100-500 km/s, is insensitive to the ejecta
        # temperature, the CFL number and the density/pressure floors, and
        # happens in Cartesian geometry too -- it is the reconstruction
        # overshooting on the steep envelope, not the geometric source terms).
        # The first-order fallback limits the offending cells only; 1D is cheap
        # enough to buy the lost accuracy back with resolution (--converge).
        first_order_fallback=True,
        # ALSO REQUIRED: the origin cell evacuates (rho falls ~10 orders of
        # magnitude by 80 yr as the cold, essentially pressureless homologous
        # core expands off r = 0 against the 2p/r geometric source) and the
        # finite-volume evolve has no positivity protection of its own. The
        # per-step hard floor lives in ``_iteration_level_updates`` and is
        # solver-agnostic, so it applies here. The floor sits ~3000x below the
        # ambient n_c and only ever bites deep inside the reverse shock, where
        # it cannot touch the calibration targets -- ``--verbose`` prints the
        # mass it injects so that stays checkable.
        positivity_config=PositivityConfig(
            per_step_mode=POSITIVITY_HARD_FLOOR, nan_safe=True, vacuum_rest=True),
        return_snapshots=True,
        snapshot_settings=snaps,
        num_snapshots=cfg["num_snapshots"],
        progress_bar=cfg["progress"],
        **({"cooling_config": cooling_config} if cooling_config is not None else {}),
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    r = helper_data.geometric_centers
    dx = cfg["r_max"] / cfg["num_cells"]
    cell_vol = helper_data.cell_volumes

    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(code_units.code_density).value)
    p_per_n = float((const.k_B * cfg["wind_temperature_K"] * u.K / u.cm ** 3).to(code_units.code_pressure).value)

    # ambient: the progenitor wind
    n_amb = wind_number_density(r, n_w=cfg["n_w"], r_fs_ref=cfg["r_fs_ref"],
                                n_c=cfg["n_c"], r_cap=0.5 * cfg["r0"])
    rho_amb = n_amb * rho_per_n
    p_amb = n_amb * p_per_n

    # ejecta: the same flat-core + steep-envelope profile the 3D IC uses
    E = float((cfg["energy_erg"] * u.erg).to(code_units.code_energy).value)
    M_ej = float((cfg["ejecta_mass_msun"] * u.Msun).to(code_units.code_mass).value)
    r_core = cfg["core_fraction"] * cfg["r0"]
    shape = ejecta_radial_shape(r, r_core, cfg["r0"], dx,
                                envelope_slope=cfg["envelope_slope"],
                                inner_slope=cfg["inner_slope"],
                                taper_cells=cfg["taper_cells"])

    # renormalise to the target ejecta mass (exactly, independent of resolution)
    d_rho = M_ej / jnp.sum(shape * cell_vol)
    m_ej = d_rho * shape
    rho = rho_amb + m_ej

    # homologous v = s r on the ejecta mass, s renormalised to KE == E exactly
    r_safe = jnp.maximum(r, 0.5 * dx)
    integrand = jnp.sum(m_ej ** 2 * r ** 2 / rho * cell_vol)
    s = jnp.sqrt(E / (0.5 * integrand))
    v = m_ej * s * r / rho

    # cold ejecta blended into the ambient pressure
    p_cold = (rho / rho_per_n) * float(
        (const.k_B * cfg["ejecta_temperature_K"] * u.K / u.cm ** 3).to(code_units.code_pressure).value)
    p = p_amb * (1.0 - shape) + p_cold * shape

    state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho, velocity_x=v, gas_pressure=p,
    )
    config = finalize_config(config, state.shape)

    # the age the homologous profile already represents: v = s r  =>  t0 = 1/s
    t0_yr = float(((1.0 / s) * code_units.code_time).to(u.yr).value)
    t_end = float(((cfg["age_yr"] - t0_yr) * u.yr).to(code_units.code_time).value)

    params = SimulationParams(
        C_cfl=cfg["cfl"], gamma=cfg["gamma"], t_end=t_end,
        minimum_density=cfg["minimum_density"],
        minimum_pressure=cfg["minimum_pressure"],
    )
    if cooling_params is not None:
        params = params._replace(cooling_params=cooling_params)

    info = dict(
        t0_yr=t0_yr,
        t_end_code=t_end,
        M_ej_achieved=float(jnp.sum(m_ej * cell_vol)),
        M_ej_target=M_ej,
        KE_achieved=float(jnp.sum(0.5 * rho * v ** 2 * cell_vol)),
        E_target=E,
        v_max_kms=float((float(jnp.max(v)) * code_units.code_velocity).to(u.km / u.s).value),
        M_wind_inside_r0=float(jnp.sum(jnp.where(r < cfg["r0"], rho_amb, 0.0) * cell_vol)),
        cells_across_ejecta=cfg["r0"] / dx,
        rho_per_n=rho_per_n,
    )
    return state, config, params, registered_variables, helper_data, code_units, info
# =============================================================================
# ============ ↑ 1D initial condition ↑ =======================================
# =============================================================================


# =============================================================================
# ============ ↓ Shock diagnostics ↓ ==========================================
# =============================================================================
def measure_snapshot(r, rho, v, p, *, age_yr, cfg, rho_per_n, code_units,
                     homology_tol=0.05, fs_contrast=2.0):
    """Measure r_FS, r_RS and the post-shock density on one radial profile.

    See the module docstring for the definitions. Radii come back in pc, the
    post-shock number density in cm^-3; ``None`` if a shock is not found (e.g.
    before the reverse shock has formed).
    """
    r = np.asarray(r); rho = np.asarray(rho); v = np.asarray(v)
    t_code = float((age_yr * u.yr).to(code_units.code_time).value)

    n_amb = np.asarray(wind_number_density(
        jnp.asarray(r), n_w=cfg["n_w"], r_fs_ref=cfg["r_fs_ref"],
        n_c=cfg["n_c"], r_cap=0.5 * cfg["r0"]))
    rho_amb = n_amb * rho_per_n

    # forward shock: outermost radius compressed above fs_contrast x ambient
    shocked = rho > fs_contrast * rho_amb
    r_fs = float(r[shocked].max()) if np.any(shocked) else None

    # reverse shock: outer edge of the still-homologous (unshocked) ejecta
    r_rs = None
    if r_fs is not None:
        v_hom = r / t_code
        # only look inside the forward shock and outside the innermost cells
        # (where v_hom -> 0 makes the relative test meaningless)
        band = (r < r_fs) & (r > 5.0 * cfg["r_max"] / cfg["num_cells"])
        homologous = band & (np.abs(v - v_hom) < homology_tol * v_hom)
        if np.any(homologous):
            r_rs = float(r[homologous].max())

    # post-shock density: mean over the outer 5% of the shocked region
    n_post = None
    if r_fs is not None:
        shell = (r > 0.95 * r_fs) & (r <= r_fs)
        if np.any(shell):
            n_post = float(np.mean(rho[shell]) / rho_per_n)

    m_unshocked = None
    if r_rs is not None:
        shell = 4.0 * np.pi * r ** 2 * np.asarray(rho) * np.gradient(r)
        m_unshocked = float(np.sum(np.where(r <= r_rs, shell, 0.0))
                            - cfg.get("_m_wind_interior", 0.0))
    return dict(r_fs=r_fs, r_rs=r_rs, n_post=n_post, m_unshocked=m_unshocked)


def measure_run(snaps, helper_data, registered_variables, code_units, cfg, info):
    """Turn a snapshot series into the full diagnostic table (radii + speeds)."""
    r = np.asarray(helper_data.geometric_centers)
    states = np.asarray(snaps.states)
    t_code = np.asarray(snaps.time_points)
    age = info["t0_yr"] + (t_code * code_units.code_time).to(u.yr).value

    rows = []
    for k in range(states.shape[0]):
        st = states[k]
        m = measure_snapshot(
            r, st[registered_variables.density_index],
            st[registered_variables.velocity_index],
            st[registered_variables.pressure_index],
            age_yr=age[k], cfg=cfg, rho_per_n=info["rho_per_n"],
            code_units=code_units)
        m["age_yr"] = float(age[k])
        rows.append(m)

    # shock speeds in the observer frame: centred differences of the radii
    pc_per_yr_to_kms = float((1.0 * u.pc / u.yr).to(u.km / u.s).value)
    for key, vkey in (("r_fs", "v_fs"), ("r_rs", "v_rs")):
        vals = [row[key] for row in rows]
        ages = [row["age_yr"] for row in rows]
        for k in range(len(rows)):
            lo, hi = max(k - 1, 0), min(k + 1, len(rows) - 1)
            if vals[lo] is None or vals[hi] is None or hi == lo:
                rows[k][vkey] = None
            else:
                rows[k][vkey] = (vals[hi] - vals[lo]) / (ages[hi] - ages[lo]) * pc_per_yr_to_kms
    return rows


def score(row):
    """Sum of |deviation| / tolerance over the observational targets (lower is better)."""
    keys = (("r_fs", "r_FS"), ("r_rs", "r_RS"), ("v_fs", "v_FS"),
            ("v_rs", "v_RS"), ("n_post", "n_post"), ("m_unshocked", "m_unshocked"))
    total, missing = 0.0, 0
    for k, tk in keys:
        val = row.get(k)
        target, tol, _ = TARGETS[tk]
        if val is None:
            missing += 1
            continue
        total += abs(val - target) / tol
    return total + 10.0 * missing
# =============================================================================
# ============ ↑ Shock diagnostics ↑ ==========================================
# =============================================================================


def _enclosed_mass_at(snaps, helper_data, registered_variables, radii):
    """``M(<r)`` at a given radius, snapshot by snapshot (code mass).

    Used to record the reverse shock's Lagrangian position: the enclosed mass is
    the material label a 1D spherical flow preserves, so a parcel's shocking time
    is found by inverting ``m_RS(t)`` rather than the shock RADIUS, which the
    parcel has long since left behind.
    """
    states = np.asarray(snaps.states)
    cell_vol = np.asarray(helper_data.cell_volumes)
    r = np.asarray(helper_data.geometric_centers)
    out = np.full(states.shape[0], np.nan)
    for k in range(states.shape[0]):
        if radii[k] is None or not np.isfinite(radii[k]):
            continue
        shell = states[k][registered_variables.density_index] * cell_vol
        out[k] = float(np.sum(np.where(r <= radii[k], shell, 0.0)))
    return out


def default_cfg(**over):
    """The baseline calibration configuration (the current showcase parameters)."""
    cfg = dict(
        # explosion
        energy_erg=1.5e51,
        ejecta_mass_msun=3.3,
        envelope_slope=9.0,
        inner_slope=1.0,   # standard core-collapse inner index; see CALIBRATION.md
        core_fraction=0.5,
        r0=0.05,                    # pc, initial ejecta radius (~3% of M_ej swept)
        ejecta_temperature_K=100.0,
        taper_cells=3.0,
        # circumstellar medium
        n_w=0.8, r_fs_ref=2.5, n_c=0.1,
        wind_temperature_K=1e4,
        # grid / integration
        r_max=4.0, num_cells=2000, cfl=0.4, gamma=GAMMA,
        num_snapshots=36, age_yr=AGE_YR,
        minimum_density=1e-6, minimum_pressure=1e-12,
        cooling=False, progress=False,
    )
    cfg.update(over)
    return cfg


def run_one(cfg):
    """Build, run and measure one 1D calibration model."""
    state, config, params, rv, helper_data, cu, info = build_1d(cfg)
    cfg["_m_wind_interior"] = info["M_wind_inside_r0"]
    snaps = time_integration(state, config, params, rv)
    rows = measure_run(snaps, helper_data, rv, cu, cfg, info)
    # how much mass/energy the density floor injected: the floor only fires in
    # the evacuated core, but it is not conservative, so keep it visible.
    mass = np.asarray(snaps.total_mass)
    energy = np.asarray(snaps.total_energy)
    info["mass_drift"] = float(mass[-1] / mass[0] - 1.0)
    info["energy_drift"] = float(energy[-1] / energy[0] - 1.0)
    return rows, info, snaps, helper_data, rv, cu


def _fmt(v, prec=2):
    return "  --  " if v is None else f"{v:6.{prec}f}"


def print_table(rows, info, title=""):
    print(f"\n=== {title} ===" if title else "")
    print(f"    t0 = {info['t0_yr']:.1f} yr (age the homologous IC already represents), "
          f"v_max = {info['v_max_kms']:.0f} km/s, "
          f"M_wind(<r0) = {info['M_wind_inside_r0']:.4f} Msun "
          f"({100 * info['M_wind_inside_r0'] / info['M_ej_target']:.1f}% of M_ej)")
    if "mass_drift" in info:
        print(f"    conservation: mass {100 * info['mass_drift']:+.4f}%, "
              f"energy {100 * info['energy_drift']:+.4f}% (floor injection)")
    print(f"    {'age[yr]':>8} {'r_FS[pc]':>9} {'r_RS[pc]':>9} "
          f"{'v_FS[km/s]':>11} {'v_RS[km/s]':>11} {'n_post':>8} {'M_unsh':>8}")
    for row in rows:
        m = (row["v_fs"] / 977792.2 * row["age_yr"] / row["r_fs"]
             if row["v_fs"] and row["r_fs"] else None)
        print(f"    {row['age_yr']:8.1f} {_fmt(row['r_fs'], 3):>9} {_fmt(row['r_rs'], 3):>9} "
              f"{_fmt(row['v_fs'], 0):>11} {_fmt(row['v_rs'], 0):>11} "
              f"{_fmt(row['n_post'], 2):>8} {_fmt(row.get('m_unshocked'), 3):>8}")


def final_row(rows, age_yr):
    """The snapshot closest to the target age."""
    return min(rows, key=lambda r: abs(r["age_yr"] - age_yr))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="run on a GPU instead of the CPU")
    ap.add_argument("--n", type=int, default=2000, help="number of radial cells")
    ap.add_argument("--r-max", type=float, default=4.0, help="outer radius (pc)")
    ap.add_argument("--r0", type=float, default=0.05, help="initial ejecta radius (pc)")
    ap.add_argument("--energy-51", type=float, default=1.5, help="explosion energy / 1e51 erg")
    ap.add_argument("--ejecta-mass", type=float, default=3.3, help="ejecta mass (Msun)")
    ap.add_argument("--n-w", type=float, default=0.8, help="wind density at r_fs_ref (cm^-3)")
    ap.add_argument("--envelope-slope", type=float, default=9.0, help="ejecta envelope index")
    ap.add_argument("--inner-slope", type=float, default=1.0,
                    help="inner ejecta density index delta (rho ~ r^-delta inside "
                         "the core radius); 0 = flat core. This is what controls "
                         "the unshocked ejecta mass")
    ap.add_argument("--gamma", type=float, default=GAMMA,
                    help="adiabatic index (4/3..5/3 probes CR back-reaction crudely)")
    ap.add_argument("--age", type=float, default=AGE_YR, help="target age (yr)")
    ap.add_argument("--cooling", action="store_true", help="include radiative cooling")
    ap.add_argument("--progress", action="store_true", help="live progress bar")
    ap.add_argument("--scan", nargs="?", const="coarse", choices=["coarse", "fine"],
                    default=None,
                    help="sweep the degenerate explosion parameters and rank against "
                         "the observational targets ('coarse' = E_SN/M_ej/n_w, "
                         "'fine' = refined + envelope slope)")
    ap.add_argument("--converge", action="store_true", help="resolution convergence check")
    ap.add_argument("--save-profile", type=str, default=None,
                    help="write the final radial profile to this npz (for the 3D mapping)")
    args = ap.parse_args()

    base = dict(num_cells=args.n, r_max=args.r_max, r0=args.r0,
                energy_erg=args.energy_51 * 1e51, ejecta_mass_msun=args.ejecta_mass,
                n_w=args.n_w, envelope_slope=args.envelope_slope,
                inner_slope=args.inner_slope, gamma=args.gamma,
                age_yr=args.age, cooling=args.cooling, progress=args.progress)

    if args.converge:
        for n in (1000, 2000, 4000, 8000):
            cfg = default_cfg(**{**base, "num_cells": n})
            rows, info, *_ = run_one(cfg)
            row = final_row(rows, cfg["age_yr"])
            print(f"N = {n:5d}  r_FS = {_fmt(row['r_fs'], 3)} pc  r_RS = {_fmt(row['r_rs'], 3)} pc  "
                  f"v_FS = {_fmt(row['v_fs'], 0)} km/s  n_post = {_fmt(row['n_post'], 2)}  "
                  f"score = {score(row):.2f}")
        return

    if args.scan:
        if args.scan == "coarse":
            # the degenerate (E_SN, M_ej, n_w) block, spanning the Orlando
            # Route A (1.5e51 / 3.3 Msun) and Route B (2.3e51 / 4 Msun) values
            # and the CR-back-reaction-thinned winds of Orlando et al. (2022)
            grid = dict(
                energy_erg=[1.5e51, 2.0e51, 2.3e51, 2.6e51],
                ejecta_mass_msun=[3.0, 3.3, 4.0, 5.0],
                n_w=[0.42, 0.62, 0.8, 0.9],
            )
        else:
            # refinement around the coarse optimum, now including the ejecta
            # envelope slope -- the parameter the reverse-shock radius is most
            # sensitive to (a steeper envelope decelerates earlier)
            grid = dict(
                energy_erg=[1.8e51, 2.0e51, 2.2e51, 2.4e51],
                ejecta_mass_msun=[3.0, 3.3, 3.6],
                n_w=[0.8, 0.9, 1.0],
                envelope_slope=[7.0, 9.0, 12.0],
            )
        results = []
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            cfg = default_cfg(**{**base, **dict(zip(keys, combo))})
            rows, info, *_ = run_one(cfg)
            row = final_row(rows, cfg["age_yr"])
            label = " ".join(f"{k.replace('energy_erg', 'E').replace('ejecta_mass_msun', 'M')}"
                             f"={(v / 1e51 if k == 'energy_erg' else v):g}"
                             for k, v in zip(keys, combo))
            results.append((score(row), label, row))
            print(f"{label:44s} -> r_FS {_fmt(row['r_fs'], 3)} r_RS {_fmt(row['r_rs'], 3)} "
                  f"v_FS {_fmt(row['v_fs'], 0)} v_RS {_fmt(row['v_rs'], 0)} "
                  f"n_post {_fmt(row['n_post'], 2)}  score {score(row):7.2f}", flush=True)
        results.sort(key=lambda x: x[0])
        print("\n=== ranked (lower score = closer to Cas A) ===")
        for s, label, row in results[:12]:
            print(f"  score {s:7.2f}  {label:44s} "
                  f"r_FS {_fmt(row['r_fs'], 3)} r_RS {_fmt(row['r_rs'], 3)} "
                  f"v_FS {_fmt(row['v_fs'], 0)} v_RS {_fmt(row['v_rs'], 0)} "
                  f"n_post {_fmt(row['n_post'], 2)}")
        return

    cfg = default_cfg(**base)
    rows, info, snaps, helper_data, rv, cu = run_one(cfg)
    print_table(rows, info,
                title=f"E = {cfg['energy_erg'] / 1e51:.2f}e51 erg, M_ej = {cfg['ejecta_mass_msun']} Msun, "
                      f"n_w = {cfg['n_w']} cm^-3, n = {cfg['envelope_slope']}")
    row = final_row(rows, cfg["age_yr"])
    print(f"\n    at {row['age_yr']:.0f} yr: score = {score(row):.2f} "
          f"(targets r_FS {TARGETS['r_FS'][0]}+-{TARGETS['r_FS'][1]}, "
          f"r_RS {TARGETS['r_RS'][0]}+-{TARGETS['r_RS'][1]} pc, "
          f"v_FS {TARGETS['v_FS'][0]:.0f}+-{TARGETS['v_FS'][1]:.0f}, "
          f"v_RS {TARGETS['v_RS'][0]:.0f}+-{TARGETS['v_RS'][1]:.0f} km/s, "
          f"n_post {TARGETS['n_post'][0]}+-{TARGETS['n_post'][1]} cm^-3)")

    if args.save_profile:
        fs = np.asarray(snaps.final_state)
        np.savez_compressed(
            args.save_profile,
            r=np.asarray(helper_data.geometric_centers),
            rho=fs[rv.density_index], v=fs[rv.velocity_index],
            press=fs[rv.pressure_index],
            measured_age_yr=row["age_yr"], t0_yr=info["t0_yr"],
            # needed by the 3D mapping to place the contact discontinuity: this
            # circumstellar mass started INSIDE the ejecta radius, so in the
            # enclosed-mass ordering it is mixed through the ejecta rather than
            # sitting below it
            M_wind_inside_r0=info["M_wind_inside_r0"],
            M_ej=info["M_ej_achieved"],
            # the shock-radius history, so the 3D mapping can work out how long
            # each parcel has ALREADY been shocked at the mapping time -- without
            # it the ionization age would restart from zero there and be
            # under-counted by up to the whole pre-mapping history
            history_age_yr=np.array([row_["age_yr"] for row_ in rows]),
            history_r_fs=np.array([np.nan if row_["r_fs"] is None else row_["r_fs"]
                                   for row_ in rows]),
            history_r_rs=np.array([np.nan if row_["r_rs"] is None else row_["r_rs"]
                                   for row_ in rows]),
            # The ENCLOSED MASS at the reverse shock, snapshot by snapshot. The
            # shocked ejecta has moved a long way since it was shocked, so
            # inverting r_RS(t) at a parcel's present radius is meaningless for
            # it; its Lagrangian label is the enclosed mass, which 1D spherical
            # flow preserves exactly, so m_RS(t) inverts correctly.
            history_m_rs=_enclosed_mass_at(
                snaps, helper_data, rv, [row_["r_rs"] for row_ in rows]),
            r_fs=row["r_fs"] if row["r_fs"] else np.nan,
            r_rs=row["r_rs"] if row["r_rs"] else np.nan,
            v_fs=row["v_fs"] if row["v_fs"] else np.nan,
            v_rs=row["v_rs"] if row["v_rs"] else np.nan,
            n_post=row["n_post"] if row["n_post"] else np.nan,
            # the full configuration, so the 3D mapping inherits the exact
            # calibrated wind and explosion parameters rather than a copy
            **{f"cfg_{k}": v for k, v in cfg.items() if isinstance(v, (int, float))},
        )
        print(f"    saved profile -> {args.save_profile}")

    # a quick look at the profile evolution
    out = Path(FIGURES_DIR) / "casa_calibrate_1d.png"
    states = np.asarray(snaps.states)
    r = np.asarray(helper_data.geometric_centers)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    ages = [row_["age_yr"] for row_ in rows]
    picks = np.linspace(1, len(ages) - 1, 6).astype(int)
    for k in picks:
        lbl = f"{ages[k]:.0f} yr"
        axes[0].loglog(r, states[k][rv.density_index] / info["rho_per_n"], lw=1.2, label=lbl)
        axes[1].plot(r, states[k][rv.velocity_index] * 1000.0, lw=1.2, label=lbl)
    axes[0].set(xlabel="r [pc]", ylabel=r"$n$ [cm$^{-3}$]", xlim=(1e-2, cfg["r_max"]))
    axes[1].set(xlabel="r [pc]", ylabel="v [km/s]", xlim=(0, cfg["r_max"]))
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    ax = axes[2]
    ax.plot(ages, [row_["r_fs"] for row_ in rows], "-o", ms=3, label="$r_{FS}$")
    ax.plot(ages, [row_["r_rs"] for row_ in rows], "-o", ms=3, label="$r_{RS}$")
    for key, tk in (("r_fs", "r_FS"), ("r_rs", "r_RS")):
        val, tol, _ = TARGETS[tk]
        ax.errorbar([AGE_YR], [val], yerr=[tol], fmt="k*", ms=10, capsize=4)
    ax.set(xlabel="age [yr]", ylabel="radius [pc]")
    ax.legend()
    fig.savefig(out, dpi=140)
    print(f"    saved {out}")


if __name__ == "__main__":
    main()
