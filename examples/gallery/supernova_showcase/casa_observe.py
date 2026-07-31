"""
Synthetic *Chandra* observations of the simulated remnant -- a real forward
model, not a rendering.

``_common.chandra_deep_figure`` produces a press-style image: a bremsstrahlung
proxy, cells sorted into three bands by a hard temperature cut, an ad-hoc
density weighting for "knots", and a two-scale unsharp mask. It looks like Cas A
because of the unsharp mask. Nothing in it can be compared to data.

This script instead runs the standard simulation-to-observation pipeline:

    npz state -> yt uniform grid -> pyXSIM (AtomDB/APEC emissivity, per cell,
    Doppler-shifted by the local velocity field) -> photon list -> SIMPUT
    -> SOXS ``instrument_simulator`` (real ACIS ARF + RMF + PSF image +
    particle/Galactic backgrounds, Poisson) -> a Chandra event file

so the output is an event list in detector coordinates with an energy column,
which can be binned exactly like the real ``evt2`` files in
``/export/data/lstorcks/chandra_casa`` and compared count-for-count. ``--compare``
does that: it re-bins the synthetic events onto the same tangent-plane grid as
``make_epoch_images.py`` (1024 pixels of 0.492", centred on Cas A) and writes a
side-by-side figure with the real epoch.

What is faithful here:
  * AtomDB emissivities (continuum + lines) folded through the real ACIS-S
    response, so band ratios and the line-dominated morphology are physical
    rather than assumed;
  * TBabs photoelectric absorption at Cas A's N_H;
  * the real Chandra PSF image, the ACIS particle background and the Galactic
    foreground, and Poisson statistics at the actual exposure;
  * Doppler shifts from the simulated velocity field (this is what makes the
    line emission asymmetric across the remnant).

  * **the simulated composition**, per cell and per element, when the state was
    produced by ``casa_orlando.py --composition``: the ejecta layers emit with
    the abundances the simulation actually carried through the reverse shock,
    not with an assumed uniform metallicity. Cas A's X-ray emission is
    line-dominated ejecta emission, so this is the difference between a plausible
    picture and a comparable one.

What is still approximate, and must be stated with any figure:
  * **collisional ionization equilibrium.** Cas A's bulk plasma sits at
    n_e t ~ 1e11 cm^-3 s, far from equilibrium, so the He- and H-like line ratios
    are wrong. The ingredients for the fix now exist — the shock history gives
    the ionization age per cell (``casa_plasma.py`` turns it into the observable
    EM(kT, n_e t) distribution) — but pyXSIM's ``NEISourceModel`` wants per-ion
    fields rather than an ionization age, so the swap is not a one-liner.
  * **hydrogen-free ejecta.** APEC normalises emission to hydrogen (``n_e n_H``
    and el/H abundance ratios), which formally diverges in the Fe layer. A
    hydrogen floor and an abundance cap keep it finite; both are reported per
    run. See :func:`abundance_fields`.
  * no non-thermal (synchrotron) component: the blast-wave rim will be fainter
    relative to the ejecta than in the real image.
  * states written before the passive scalars existed have no composition; for
    those, ``--ejecta-zmet`` applies a crude density/temperature-selected
    enhancement instead, and says so.

Runs in the separate CPU-only ``xrayobs`` venv (yt/pyxsim/soxs pull in their own
numpy), NOT the astx env:

    /export/home/lstorcks/xrayobs/bin/python casa_observe.py \\
        /export/data/lstorcks/supernova_showcase/casa_n512_radiative.npz \\
        --exposure 143.5 --compare 2004
"""

# general
import argparse
import os
from pathlib import Path

# numerics
import numpy as np

# units and constants
from astropy import constants as const
from astropy import units as u

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# =============================================================================
# ============ ↓ Cas A on the sky ↓ ===========================================
# =============================================================================
RA0, DEC0 = 350.8583, 58.8149        # Cas A centre (same as make_epoch_images.py)
DISTANCE_KPC = 3.4                   # Reed et al. 1995
NH_CASA = 1.2                        # 1e22 cm^-2, Galactic column toward Cas A
PIXEL_ARCSEC = 0.492                 # native ACIS pixel
NPIX_COMPARE = 1024                  # same grid as make_epoch_images.py
REAL_EPOCH_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")

# the showcase code units: pc / Msun / 1000 km s^-1
CODE_LENGTH = (1.0 * u.pc).to(u.cm).value
CODE_MASS = (1.0 * u.Msun).to(u.g).value
CODE_VELOCITY = (1000.0 * u.km / u.s).to(u.cm / u.s).value
CODE_DENSITY = CODE_MASS / CODE_LENGTH ** 3
CODE_PRESSURE = CODE_DENSITY * CODE_VELOCITY ** 2
MU_PARTICLE = 0.61                   # matches _common.temperature_K
# =============================================================================
# ============ ↑ Cas A on the sky ↑ ===========================================
# =============================================================================


# Solar (Anders & Grevesse, the "angr" table SOXS/pyXSIM default) abundances
# expressed as MASS fractions relative to hydrogen, which is what converts a
# simulated mass fraction into the "in solar units" abundance pyXSIM wants.
SOLAR_MASS_RATIO_TO_H = {"He": 0.3911, "O": 0.01362, "Si": 9.97e-4, "Fe": 2.613e-3}

#: species carried by ``casa_orlando.py --composition``
TRACKED_SPECIES = ("Fe", "Si", "O", "He")


def load_state(path):
    """Read a showcase ``--save-state`` npz into cgs physical fields."""
    d = np.load(path)
    rho = np.asarray(d["rho"], dtype=np.float64) * CODE_DENSITY          # g/cm^3
    press = np.asarray(d["press"], dtype=np.float64) * CODE_PRESSURE     # erg/cm^3
    temp = (MU_PARTICLE * const.m_p.cgs.value / const.k_B.cgs.value) * press / rho
    vel = None
    if "vx" in d:
        vel = tuple(np.asarray(d[k], dtype=np.float64) * CODE_VELOCITY
                    for k in ("vx", "vy", "vz"))
    box_pc = float(d["box"]) if "box" in d else 7.0
    age_yr = float(d["age"]) if "age" in d else np.nan

    # passive scalars, when the run carried them
    composition = None
    if "C_ej" in d:
        composition = {s: np.asarray(d[f"C_{s}"], dtype=np.float64)
                       for s in TRACKED_SPECIES if f"C_{s}" in d}
        composition["ejecta"] = np.asarray(d["C_ej"], dtype=np.float64)
    shock = None
    if "density_time" in d:
        shock = dict(
            density_time=np.asarray(d["density_time"], dtype=np.float64),
            time_since_shock=np.asarray(d["time_since_shock"], dtype=np.float64),
        )
    return dict(density=rho, temperature=temp, velocity=vel,
                box_pc=box_pc, age_yr=age_yr, num_cells=rho.shape[0],
                composition=composition, shock=shock)


def abundance_fields(composition, *, max_abundance, h_floor):
    """Turn simulated mass fractions into pyXSIM abundances (solar units).

    pyXSIM, like every APEC-based model, normalises emission to hydrogen: the
    emission measure is ``n_e n_H`` and abundances are ratios to solar
    ``el/H``. Supernova ejecta are the pathological case for that convention —
    the Fe layer is essentially hydrogen-free, so the abundance ratio diverges
    while ``n_H`` goes to zero, and only their product is well behaved.

    Two explicit, reported approximations keep this numerically sane:
    a floor on the hydrogen mass fraction and a cap on the abundance
    enhancement. The result is not a pure-metal plasma calculation, but it is a
    great deal closer than assuming one metallicity everywhere, which is what
    left the synthetic image ~3.5x under-luminous and peaking at the wrong
    radius.

    Returns ``(h_fraction, {element: abundance in solar units}, info)``.
    """
    metals = {s: composition[s] for s in TRACKED_SPECIES if s in composition}
    x_h = np.clip(1.0 - sum(metals.values()), h_floor, 1.0)

    abund, info = {}, {}
    for el, x_el in metals.items():
        a = (x_el / x_h) / SOLAR_MASS_RATIO_TO_H[el]
        capped = np.minimum(a, max_abundance)
        info[el] = dict(median=float(np.median(a)), p99=float(np.percentile(a, 99)),
                        capped_fraction=float(np.mean(a > max_abundance)))
        abund[el] = capped
    return x_h, abund, info


def make_yt_dataset(state, *, zmet, ejecta_zmet, ejecta_temperature_K,
                    max_abundance=50.0, h_floor=1e-3):
    """Wrap the state in a yt uniform grid with the fields pyXSIM needs."""
    import yt

    n = state["num_cells"]
    half = 0.5 * state["box_pc"] * CODE_LENGTH
    bbox = np.array([[-half, half]] * 3)

    # use explicit ("gas", ...) field tuples: with bare names yt registers them
    # under ("stream", ...) and does not alias the velocities, which pyXSIM
    # then cannot find
    data = {
        ("gas", "density"): (state["density"], "g/cm**3"),
        ("gas", "temperature"): (state["temperature"], "K"),
    }
    # Always register velocities, even for the older states that were saved
    # before ``--save-state`` kept them: pyXSIM's default is to look for
    # ("gas", "velocity_*") whether or not we ask for Doppler shifts, so a
    # missing field is a hard error rather than "no shifting".
    vel = state["velocity"] or (np.zeros_like(state["density"]),) * 3
    for name, arr in zip(("velocity_x", "velocity_y", "velocity_z"), vel):
        data[("gas", name)] = (arr, "cm/s")

    # Composition. With passive scalars the abundances are the SIMULATED ones,
    # per cell and per element; the two fallbacks below exist only for states
    # written before the scalars existed.
    if state["composition"] is not None:
        x_h, abund, info = abundance_fields(
            state["composition"], max_abundance=max_abundance, h_floor=h_floor)
        data[("gas", "H_fraction")] = (x_h, "dimensionless")
        for el, a in abund.items():
            # units MUST be "Zsun" (they are solar-unit abundances, so this is
            # also the honest label). pyXSIM masks the hydrogen fraction to the
            # emitting cells before using it, then for any var_elem field NOT in
            # Zsun divides the conversion factor by that masked array and
            # multiplies it against the FULL-length element field -- which
            # raises "operands could not be broadcast together" the moment
            # h_fraction is supplied as a field. Declaring Zsun takes that
            # branch out.
            data[("gas", f"{el}_abundance")] = (a, "Zsun")
        for el, i in info.items():
            print(f"[casa-obs] {el}: median {i['median']:.2f} solar, "
                  f"p99 {i['p99']:.1f}, {100 * i['capped_fraction']:.2f}% of cells "
                  f"capped at {max_abundance}x")
    elif ejecta_zmet is not None:
        # legacy stand-in: no ejecta tracer, so select by the only thing
        # available -- dense, hot material interior to the blast wave
        rho_med = np.median(state["density"])
        is_ejecta = ((state["density"] > 3.0 * rho_med)
                     & (state["temperature"] > ejecta_temperature_K))
        data[("gas", "metallicity")] = (np.where(is_ejecta, ejecta_zmet, zmet), "Zsun")

    ds = yt.load_uniform_grid(
        data, [n, n, n], length_unit="cm", bbox=bbox,
        nprocs=1, default_species_fields="ionized",
    )
    return ds


def make_events(state, args):
    """Photon list -> SIMPUT -> Chandra event file. Returns the event-file path."""
    import pyxsim
    import soxs

    ds = make_yt_dataset(state, zmet=args.zmet, ejecta_zmet=args.ejecta_zmet,
                         ejecta_temperature_K=args.ejecta_temperature,
                         max_abundance=args.max_abundance, h_floor=args.h_floor)
    sp = ds.all_data()

    # With the simulated composition available, every element pyXSIM knows about
    # varies per cell and hydrogen is a field too; otherwise fall back to a
    # single metallicity.
    var_elem = h_fraction = None
    if state["composition"] is not None:
        var_elem = {el: ("gas", f"{el}_abundance") for el in TRACKED_SPECIES
                    if el in state["composition"]}
        h_fraction = ("gas", "H_fraction")
        Zmet = args.zmet          # the un-tracked trace metals
    else:
        Zmet = ("gas", "metallicity") if args.ejecta_zmet is not None else args.zmet
    source = pyxsim.CIESourceModel(
        "apec", args.emin, args.emax, args.nbins, Zmet,
        var_elem=var_elem, h_fraction=h_fraction,
        # do not let the cold, unshocked ejecta (which is at the pressure floor
        # and whose float32 temperature is meaningless) contribute
        kT_min=args.kt_min,
        binscale="log",
        thermal_broad=True,
        abund_table="angr",
    )

    # Generate more photons than we will need, then sub-sample at projection: a
    # collecting area above Chandra's lets soxs draw the real number. Beware the
    # scaling -- the photon list holds area x exposure photons, and Cas A is
    # bright enough that the careless combination (3000 cm^2 x 50 ks, emitting
    # down to 1e5 K) produced 4.7e9 photons, a 38 GB file and 64 GB resident.
    # The intermediates therefore go to scratch on /export/data, not to $HOME.
    prefix = os.path.join(args.scratch, os.path.basename(args.out))
    os.makedirs(args.scratch, exist_ok=True)
    n_ph, n_cell = pyxsim.make_photons(
        f"{prefix}_photons", sp, 0.0, args.area, args.exposure * 1e3, source,
        dist=(DISTANCE_KPC, "kpc"),
        # Doppler shifts from the simulated velocity field: this is what makes
        # the line emission asymmetric across the remnant (zero, and therefore
        # a no-op, for states saved without velocities)
        velocity_fields=[("gas", "velocity_x"), ("gas", "velocity_y"),
                         ("gas", "velocity_z")],
    )
    if state["velocity"] is None:
        print("[casa-obs] NOTE: this state carries no velocities -- the line "
              "emission is unshifted (no Doppler structure)")
    print(f"[casa-obs] {n_ph:.3e} photons from {n_cell:.3e} cells")

    n_ev = pyxsim.project_photons(
        f"{prefix}_photons", f"{prefix}_events", args.los, (RA0, DEC0),
        absorb_model="tbabs", nH=args.nh, abund_table="angr",
        # smear each cell's photons over the cell so the projection is not
        # a lattice of delta functions at the sub-arcsecond ACIS pixel scale
        kernel="gaussian",
    )
    print(f"[casa-obs] {n_ev:.3e} photons survive absorption + projection")

    simput = f"{prefix}_simput"
    el = pyxsim.EventList(f"{prefix}_events.h5")
    el.write_to_simput(simput, overwrite=True)

    evtfile = f"{prefix}_evt.fits"
    soxs.instrument_simulator(
        f"{simput}_simput.fits", evtfile, (args.exposure * 1e3, "s"),
        args.instrument, (RA0, DEC0), overwrite=True,
        instr_bkgnd=not args.no_background,
        foreground=not args.no_background,
        ptsrc_bkgnd=False,
    )
    print(f"[casa-obs] wrote {evtfile}")
    return evtfile


# =============================================================================
# ============ ↓ Binning onto the real-data sky grid ↓ ========================
# =============================================================================
def bin_events_to_grid(evtfile, *, emin=0.5, emax=7.0, npix=NPIX_COMPARE,
                       scale_arcsec=PIXEL_ARCSEC):
    """Bin a SOXS event file onto the same tangent-plane grid the real data uses.

    SOXS writes the same structure as a Chandra ``evt2``: sky ``X``/``Y``
    columns with the tangent-plane WCS in the ``TCRVL``/``TCRPX``/``TCDLT``
    keywords and an ``ENERGY`` column in eV. So this is deliberately the same
    inverse-gnomonic conversion followed by the same forward projection that
    ``/export/data/lstorcks/chandra_casa/make_epoch_images.py`` applies to the
    real data -- synthetic and real go through identical code onto an identical
    grid, which is the point of the exercise.
    """
    from astropy.io import fits

    with fits.open(evtfile) as f:
        hdu = f["EVENTS"]
        names = [c.name for c in hdu.columns]
        ix = names.index("X") + 1
        iy = names.index("Y") + 1
        h = hdu.header
        x = np.asarray(hdu.data["X"], dtype=np.float64)
        y = np.asarray(hdu.data["Y"], dtype=np.float64)
        energy = np.asarray(hdu.data["ENERGY"], dtype=np.float64)   # eV
        crvx, crpx, cdlx = h[f"TCRVL{ix}"], h[f"TCRPX{ix}"], h[f"TCDLT{ix}"]
        crvy, crpy, cdly = h[f"TCRVL{iy}"], h[f"TCRPX{iy}"], h[f"TCDLT{iy}"]

    sel = (energy > emin * 1e3) & (energy < emax * 1e3)
    x, y = x[sel], y[sel]

    # inverse gnomonic (TAN) projection: sky pixels -> RA, Dec
    xi = np.deg2rad((x - crpx) * cdlx)
    eta = np.deg2rad((y - crpy) * cdly)
    ra0_t, dec0_t = np.deg2rad(crvx), np.deg2rad(crvy)
    rho_t = np.hypot(xi, eta)
    c_t = np.arctan(rho_t)
    denom = np.where(rho_t == 0, 1.0, rho_t)
    with np.errstate(invalid="ignore"):
        dec = np.arcsin(np.cos(c_t) * np.sin(dec0_t)
                        + eta * np.sin(c_t) * np.cos(dec0_t) / denom)
        ra = ra0_t + np.arctan2(
            xi * np.sin(c_t),
            rho_t * np.cos(dec0_t) * np.cos(c_t) - eta * np.sin(dec0_t) * np.sin(c_t))
    ra, dec = np.rad2deg(ra), np.rad2deg(dec)

    scale = scale_arcsec / 3600.0
    ra_r, dec_r = np.deg2rad(ra), np.deg2rad(dec)
    ra0, dec0 = np.deg2rad(RA0), np.deg2rad(DEC0)
    cosc = (np.sin(dec0) * np.sin(dec_r)
            + np.cos(dec0) * np.cos(dec_r) * np.cos(ra_r - ra0))
    xi = np.cos(dec_r) * np.sin(ra_r - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec_r)
           - np.sin(dec0) * np.cos(dec_r) * np.cos(ra_r - ra0)) / cosc
    px = npix / 2 - np.rad2deg(xi) / scale        # RA increases to the left
    py = npix / 2 + np.rad2deg(eta) / scale
    img, _, _ = np.histogram2d(py, px, bins=npix, range=[[0, npix], [0, npix]])
    return img


def load_real_epoch(label):
    """Load one binned real Chandra epoch (counts and exposure)."""
    path = REAL_EPOCH_DIR / f"epoch_{label}.npz"
    if not path.exists():
        raise SystemExit(f"no real epoch {label}; have "
                         f"{sorted(p.stem[6:] for p in REAL_EPOCH_DIR.glob('epoch_*.npz'))}")
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["counts"], dtype=np.float64), float(d["exposure"])


def comparison_figure(syn, real, *, out_path, syn_exposure_ks, real_exposure_ks,
                      label, age_yr, crop_arcsec=200.0):
    """Side-by-side synthetic / real Chandra image on identical scales."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    cmap = LinearSegmentedColormap.from_list("chandra_blue", [
        (0.00, "#000005"), (0.15, "#04102e"), (0.35, "#0a3f8f"),
        (0.60, "#1f7fd4"), (0.82, "#7fc4ef"), (1.00, "#f2fbff")])

    half = int(crop_arcsec / PIXEL_ARCSEC)
    c = NPIX_COMPARE // 2
    sl = slice(c - half, c + half)

    def prep(img, smooth=1.0):
        a = gaussian_filter(img[sl, sl], smooth)
        hi = np.percentile(a[a > 0], 99.6) if np.any(a > 0) else 1.0
        x = np.clip(a / max(hi, 1e-30), 0.0, 1.0)
        return np.arcsinh(x / 0.02) / np.arcsinh(1.0 / 0.02)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), facecolor="black")
    ext = [crop_arcsec, -crop_arcsec, -crop_arcsec, crop_arcsec]
    for ax, img, title in (
            (axes[0], prep(syn), f"astronomix, synthetic ACIS-S\n"
                                 f"{syn_exposure_ks:.0f} ks, {syn.sum():.3g} counts"
                                 f"{'' if np.isnan(age_yr) else f', t = {age_yr:.0f} yr'}"),
            (axes[1], prep(real), f"Chandra, real (epoch {label})\n"
                                  f"{real_exposure_ks:.0f} ks, {real.sum():.3g} counts")):
        ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=1,
                  interpolation="bilinear")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_facecolor("black")
        ax.tick_params(colors="0.6", labelsize=8)
        ax.set_xlabel("offset [arcsec]", color="0.6", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="black")
    print(f"[casa-obs] saved {out_path}")


def radial_profile_figure(syn, real, *, out_path, label):
    """Azimuthally averaged surface-brightness profiles -- the quantitative test.

    The forward shock shows up as the outer break; comparing the two profiles
    is how the calibrated shock radii are checked against the data in counts
    space rather than by eye.

    Note that the real profile keeps a shallow tail well beyond the forward
    shock which the synthetic one does not reproduce: that is the Chandra
    dust-scattering halo plus the far PSF wings, neither of which SOXS models
    (it convolves with the core PSF image only). Compare the position of the
    outer break, not the flux beyond it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = NPIX_COMPARE // 2
    yy, xx = np.mgrid[:NPIX_COMPARE, :NPIX_COMPARE]
    rr = np.hypot(xx - c, yy - c) * PIXEL_ARCSEC
    bins = np.arange(0, 260, 4.0)
    idx = np.digitize(rr.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)

    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    for img, lbl, style in ((syn, "astronomix (synthetic)", "-"),
                            (real, f"Chandra {label}", "--")):
        tot = np.bincount(idx[ok], weights=img.ravel()[ok], minlength=len(bins) - 1)
        area = np.bincount(idx[ok], minlength=len(bins) - 1)
        prof = np.where(area > 0, tot / np.maximum(area, 1), 0.0)
        prof = prof / max(prof.max(), 1e-30)
        ax.semilogy(0.5 * (bins[:-1] + bins[1:]), np.maximum(prof, 1e-4), style, label=lbl)
    # observed shock radii at 3.4 kpc
    for r_pc, name, col in ((2.52, "$r_{FS}$", "tab:red"), (1.58, "$r_{RS}$", "tab:orange")):
        arcsec = np.rad2deg(r_pc / (DISTANCE_KPC * 1e3)) * 3600.0
        ax.axvline(arcsec, color=col, ls=":", lw=1.2)
        ax.text(arcsec, 1.1, name, color=col, ha="center", fontsize=9)
    ax.set(xlabel="radius [arcsec]", ylabel="normalised surface brightness (0.5-7 keV)",
           ylim=(1e-4, 2.0))
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=150)
    print(f"[casa-obs] saved {out_path}")
# =============================================================================
# ============ ↑ Binning onto the real-data sky grid ↑ ========================
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", help="showcase --save-state npz")
    ap.add_argument("--out", default=None, help="output prefix (default: from the state name)")
    ap.add_argument("--los", default="y", choices=["x", "y", "z"],
                    help="line of sight. Default y: Orlando et al. put the Earth "
                         "vantage point on the -y axis, so the plane of the sky is "
                         "(x, z) -- this is the convention casa_orlando.py's CSM "
                         "shell orientation and position-angle diagnostic assume. "
                         "The older cassiopeia_realistic.py states have no such "
                         "convention (their asymmetry is simply lopsided toward +z), "
                         "so for those the choice is free.")
    ap.add_argument("--exposure", type=float, default=20.0,
                    help="exposure (ks). Cas A delivers ~300 counts/s to ACIS-S, so "
                         "20 ks already gives 6e6 counts -- photon statistics are "
                         "never the limitation, and the photon list stays small. The "
                         "comparison is done in counts/s, not raw counts.")
    ap.add_argument("--instrument", default="chandra_aciss_cy0",
                    help="soxs instrument (the real Cas A observations are ACIS-7 = S3)")
    ap.add_argument("--nh", type=float, default=NH_CASA, help="N_H / 1e22 cm^-2")
    ap.add_argument("--zmet", type=float, default=1.0, help="ambient metallicity (Zsun)")
    ap.add_argument("--ejecta-zmet", type=float, default=None,
                    help="crude metallicity for the dense hot (ejecta-like) gas, "
                         "in Zsun -- a stand-in until composition tracers exist")
    ap.add_argument("--ejecta-temperature", type=float, default=1e6,
                    help="temperature above which dense gas counts as shocked ejecta (K)")
    ap.add_argument("--emin", type=float, default=0.3, help="source model E_min (keV)")
    ap.add_argument("--emax", type=float, default=12.0, help="source model E_max (keV)")
    ap.add_argument("--nbins", type=int, default=3000, help="source model spectral bins")
    ap.add_argument("--kt-min", type=float, default=0.09,
                    help="minimum kT that emits (keV; 0.09 keV = 1e6 K). Cooler gas "
                         "radiates almost entirely below 0.3 keV, where N_H = 1.2e22 "
                         "transmits nothing -- generating those photons only to "
                         "absorb them costs memory and buys no counts.")
    ap.add_argument("--area", type=float, default=800.0,
                    help="photon-generation collecting area (cm^2); must exceed "
                         "Chandra's peak effective area (~600 cm^2 for ACIS-S at 1 keV)")
    ap.add_argument("--max-abundance", type=float, default=50.0,
                    help="cap on the per-element abundance enhancement (solar units); "
                         "the Fe layer is hydrogen-free, so the el/H ratio APEC "
                         "works in formally diverges there")
    ap.add_argument("--h-floor", type=float, default=1e-3,
                    help="floor on the hydrogen mass fraction, for the same reason")
    ap.add_argument("--no-background", action="store_true", help="no instrumental/sky background")
    ap.add_argument("--scratch", default="/export/data/lstorcks/supernova_showcase/xray_scratch",
                    help="where the (large) photon/event/SIMPUT intermediates go")
    ap.add_argument("--compare", default=None,
                    help="also bin onto the real-data grid and compare with this epoch "
                         "(e.g. 2004)")
    ap.add_argument("--events", default=None,
                    help="skip the simulation and re-bin/compare an existing event file")
    args = ap.parse_args()

    if args.out is None:
        args.out = str(FIGURES_DIR.parent / Path(args.state).stem)

    state = load_state(args.state)
    print(f"[casa-obs] {args.state}: {state['num_cells']}^3, box {state['box_pc']} pc, "
          f"age {state['age_yr']:.0f} yr, "
          f"T [{state['temperature'].min():.2e}, {state['temperature'].max():.2e}] K")

    evtfile = args.events or make_events(state, args)

    syn = bin_events_to_grid(evtfile)
    syn_exp = args.exposure * 1e3
    np.savez_compressed(f"{args.out}_synimg.npz", counts=syn, exposure=syn_exp)
    print(f"[casa-obs] synthetic image: {syn.sum():.4g} counts in {args.exposure:.0f} ks "
          f"= {syn.sum() / syn_exp:.1f} counts/s (0.5-7 keV)")

    if args.compare:
        real, real_exp = load_real_epoch(args.compare)
        # The count RATE is the quantitative test of the emission model: it is
        # set by the emission measure (density squared times volume) folded
        # through the real response, with nothing free to tune. Cas A delivers
        # ~300 counts/s to ACIS-S in this band. (The real observations are
        # piled up at that rate and SOXS does not model pileup, so the real
        # number is a slight under-estimate of the true incident rate.)
        print(f"[casa-obs] count rate: synthetic {syn.sum() / syn_exp:.1f} vs real "
              f"{real.sum() / real_exp:.1f} counts/s "
              f"(ratio {syn.sum() / syn_exp / (real.sum() / real_exp):.2f})")
        comparison_figure(syn, real,
                          out_path=FIGURES_DIR / f"{Path(args.out).name}_vs_chandra_{args.compare}.png",
                          syn_exposure_ks=args.exposure, real_exposure_ks=real_exp / 1e3,
                          label=args.compare, age_yr=state["age_yr"])
        radial_profile_figure(syn, real,
                              out_path=FIGURES_DIR / f"{Path(args.out).name}_radial_{args.compare}.png",
                              label=args.compare)


if __name__ == "__main__":
    main()
