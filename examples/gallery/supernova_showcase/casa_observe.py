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
  * **the electron temperature**, not the single-fluid one. Behind a collisionless
    shock the electrons are heated to ~0.3 keV while the ions take the rest, and
    Coulomb equilibration takes thousands of years at Cas A's density; the
    spectrum is set by T_e. See :mod:`_plasma`.
  * **the electron density**, from the same composition: fully ionized ejecta
    carry ~1.7x fewer electrons per gram than cosmic gas, and both the emission
    measure and the ionization age scale with it.

  * **non-equilibrium ionization** with ``--nei``: the ion populations come from
    the simulated (kT_e, n_e t) of each parcel rather than from the assumption
    that it has reached collisional equilibrium, which at Cas A's n_e t ~ 1e11
    it has not. See :mod:`_nei`.

What is still approximate, and must be stated with any figure:
  * **collisional ionization equilibrium**, unless ``--nei`` is given. Cas A's
    bulk plasma sits an order of magnitude short of equilibrium, so CIE gets the
    line-to-continuum and He/H-like ratios wrong, in a direction the spectral
    comparison measures directly.
  * **hydrogen-free ejecta.** APEC normalises to hydrogen, which formally
    diverges when there is none. :func:`emission_fields` sets a per-cell
    reference hydrogen density so that every METAL density is exact and only the
    hydrogen continuum is affected; the run reports what that costs (0.05 % of
    the free-free emission at the default ``--max-abundance``).
  * **four tracers, nine elements.** The carried species stand for whole
    nucleosynthetic layers, and are divided into elements by the fixed mass
    ratios in ``_plasma.TRACER_SPLIT``. Relative abundances WITHIN a layer are
    therefore assumed, not simulated.
  * **full ionization** in the mean molecular weights (see :mod:`_plasma`);
    ~10-20 % in ``mu_e`` for the Fe-rich cells only.
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

# the shared plasma physics (also used by casa_plasma.py)
from _plasma import (
    ATOMIC,
    CODE_DENSITY,
    CODE_LENGTH,
    CODE_VELOCITY,
    KEV_IN_K,
    M_P,
    SOLAR_NUMBER_RATIO_TO_H,
    plasma_state,
)

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

# =============================================================================
# ============ ↑ Cas A on the sky ↑ ===========================================
# =============================================================================


#: species carried by ``casa_orlando.py --composition``
TRACKED_SPECIES = ("Fe", "Si", "O", "He")


def load_state(path):
    """Read a showcase ``--save-state`` npz, keeping the fields in code units.

    The conversion to physical quantities is deliberately NOT done here: the
    temperature, the electron density and the ionization age all depend on the
    composition the run carried, and that physics lives in :mod:`_plasma` so
    that this script and ``casa_plasma.py`` cannot disagree about the same cell.
    """
    d = np.load(path)
    fields = {k: np.asarray(d[k], dtype=np.float64) for k in d.files
              if np.asarray(d[k]).ndim == 3}
    return dict(fields=fields,
                box_pc=float(d["box"]) if "box" in d else 7.0,
                age_yr=float(d["age"]) if "age" in d else np.nan,
                num_cells=fields["rho"].shape[0],
                has_velocity="vx" in fields)


def emission_fields(state, *, max_abundance, two_temperature=True):
    """The self-consistent set pyXSIM needs: ``n_e``, ``n_H``, abundances, ``T_e``.

    APEC -- like every X-ray plasma code -- normalises to hydrogen: the emission
    measure is ``n_e n_H V`` and an element enters as ``A_el``, its abundance
    relative to solar, so the modelled emission from element ``el`` is
    proportional to ``n_e * (A_el r_sun,el n_H)``. Supernova ejecta are the
    pathological case, because ``n_H`` there is essentially zero and the ratio
    diverges.

    The resolution is to notice that only the PRODUCT is physical. Writing
    ``n_el = A_el r_sun,el n_H``, any positive ``n_H`` reproduces the true
    ``n_el`` provided ``A_el`` is set to match, and the only quantity that
    depends on the choice is the emission of hydrogen itself. So instead of
    flooring the hydrogen mass fraction (which silently CHANGES the plasma) this
    picks, per cell, the smallest reference hydrogen density that keeps every
    abundance inside ``max_abundance``:

        ``n_H,ref = max(n_H,true, max_el n_el / (A_max r_sun,el))``

    and then sets ``A_el = n_el / (r_sun,el n_H,ref)`` exactly. Every metal
    density is then correct by construction, at the cost of a spurious hydrogen
    continuum in the hydrogen-free cells -- which is bounded by
    ``1 / (A_max r_sun,el Z_el^2)`` relative to that element's own free-free
    emission, i.e. 0.2 % for oxygen and 1.4 % for silicon at ``A_max = 1e4``.
    That is a far smaller error than a factor-of-ten deficit in the Fe and Si
    line emission, which is what the old cap of 50 was producing: it left the
    iron knots with a tenth of their iron and the silicon layer with a
    twenty-sixth of its silicon.

    Returns a dict of cgs fields plus a report of what was done.
    """
    ps = plasma_state(state["fields"], two_temperature=two_temperature)
    X, n_e = ps["X"], ps["n_e"]                 # X is per ELEMENT (TRACER_SPLIT)
    rho = state["fields"]["rho"] * CODE_DENSITY

    n_el = {el: rho * x / (ATOMIC[el][0] * M_P)
            for el, x in X.items() if el != "H"}
    n_H_true = rho * X["H"] / (ATOMIC["H"][0] * M_P)
    n_H_ref = n_H_true.copy() if np.ndim(n_H_true) else np.full_like(n_e, n_H_true)
    for el, n in n_el.items():
        n_H_ref = np.maximum(n_H_ref, n / (max_abundance * SOLAR_NUMBER_RATIO_TO_H[el]))

    abund = {el: n / (SOLAR_NUMBER_RATIO_TO_H[el] * np.maximum(n_H_ref, 1e-30))
             for el, n in n_el.items()}

    # What the invented hydrogen costs, as a fraction of the whole remnant's
    # free-free emission: the spurious part is n_e (n_H,ref - n_H,true) against
    # the real n_e (n_H,true + sum_el n_el Z_el^2). Weighting by n_e makes this
    # the number that actually matters -- a large relative error in a cell that
    # emits nothing is not an error in the observation.
    z2 = sum(n * ATOMIC[el][1] ** 2 for el, n in n_el.items())
    excess = np.maximum(n_H_ref - n_H_true, 0.0)
    ff_spurious = float(np.sum(n_e * excess))
    ff_real = float(np.sum(n_e * (n_H_true + z2)))
    report = dict(
        invented_fraction=float(np.mean(n_H_ref > 1.0000001 * n_H_true)),
        spurious_ff=ff_spurious / max(ff_real, 1e-30),
        abundance_max=float(max(np.max(a) for a in abund.values())),
    )
    return dict(n_e=n_e, n_H=n_H_ref, abundances=abund, T_e=ps["T_e"], T=ps["T"],
                net=ps["net"], info=ps["info"], report=report,
                moments=ps["moments"])


def ion_abundance_fields(em, net, *, threshold=0.02, kt_emitting=0.5):
    """Per-ION abundances (solar units) from the ionization age, for NEI.

    The ionization state of a parcel shocked once is a function of
    ``(kT_e, n_e t)``, both of which the simulation carries, so this is a table
    lookup rather than a network integration -- see :mod:`_nei`. Each ion of
    element ``el`` gets ``A_el * f_ion``, which is what an APEC NEI model wants.

    Every ion kept costs a full 3D field, so ions are dropped below
    ``threshold`` of their element's X-ray-emitting mass; ``kt_emitting``
    excludes the cells too cool to contribute counts, which would otherwise keep
    near-neutral ions alive on the strength of gas that emits nothing in band.
    The retained fraction per element is reported, since anything dropped is
    emission thrown away.

    Returns ``({"O^7": field, ...}, report)``.
    """
    import _nei

    kt_grid, net_grid, table = _nei.load_table()
    kT_e = em["T_e"] / KEV_IN_K
    emitting = kT_e > kt_emitting

    fields, report = {}, {}
    for el, a_el in em["abundances"].items():
        if el not in table:
            continue
        f = _nei.interpolate_fractions(table[el], kt_grid, net_grid, kT_e, net)
        w = em["n_e"] * a_el * em["n_H"] * emitting * np.sqrt(np.maximum(kT_e, 0.0))
        w_tot = float(w.sum())
        share = np.array([float((fi * w).sum()) / max(w_tot, 1e-300) for fi in f])
        keep = np.where(share >= threshold)[0]
        for ion in keep:
            fields[f"{el}^{ion}"] = a_el * f[ion]
        report[el] = dict(kept=len(keep), covered=float(share[keep].sum()),
                          mean_charge=float((share * np.arange(len(share))).sum()))
        del f
    return fields, report


def describe_emission(em):
    """Print what the plasma model did, so no figure is produced silently."""
    i, r = em["info"], em["report"]
    if not i["composition_tracked"]:
        print("[casa-obs] NOTE: no composition scalars -- cosmic abundances "
              "everywhere, and the ejecta temperature is understated ~3x")
        return
    if i["two_temperature"]:
        w = (em["n_e"] ** 2) * (em["T_e"] > 1e6)
        ratio = float(np.average(em["T_e"] / em["T"], weights=w))
        kt = float(np.average(em["T_e"], weights=w)) / KEV_IN_K
        print(f"[casa-obs] electron temperature: EM-weighted T_e/T = {ratio:.3f} "
              f"(1 = full equilibration), kT_e = {kt:.2f} keV; the spectrum is "
              f"computed from T_e")
    else:
        print("[casa-obs] NOTE: single-temperature plasma (T_e = T_i = T), "
              "which over-predicts the hard emission of recently shocked gas")
    print(f"[casa-obs] hydrogen reference: invented in "
          f"{100 * r['invented_fraction']:.1f}% of cells, costing "
          f"{100 * r['spurious_ff']:.2f}% of the total free-free emission; "
          f"peak abundance {r['abundance_max']:.3g} solar")


def make_yt_dataset(state, em, *, zmet, ejecta_zmet, ejecta_temperature_K):
    """Wrap the state in a yt uniform grid with the fields pyXSIM needs.

    The emission measure is supplied EXPLICITLY rather than left to yt. yt's
    ``("gas", "emission_measure")`` is ``n_e n_H dV`` with ``n_e`` and ``n_H``
    derived from the density under ``default_species_fields="ionized"``, i.e.
    from a cosmic composition -- which is wrong by 1.8x in ``n_e`` and by orders
    of magnitude in ``n_H`` in the ejecta, and was inconsistent with the
    per-element abundances handed to the same source model.
    """
    import yt

    n = state["num_cells"]
    f = state["fields"]
    half = 0.5 * state["box_pc"] * CODE_LENGTH
    bbox = np.array([[-half, half]] * 3)
    dv = (state["box_pc"] * CODE_LENGTH / n) ** 3

    # use explicit ("gas", ...) field tuples: with bare names yt registers them
    # under ("stream", ...) and does not alias the velocities, which pyXSIM
    # then cannot find
    data = {
        ("gas", "density"): (f["rho"] * CODE_DENSITY, "g/cm**3"),
        # THE ELECTRON temperature: it is the electrons that excite the lines
        # and radiate the continuum, and behind a fast collisionless shock they
        # are far colder than the ions (see _plasma.electron_ion_temperatures).
        # Using the single-fluid temperature here over-predicted the hard
        # emission of the youngest-shocked gas.
        ("gas", "temperature"): (em["T_e"], "K"),
        ("gas", "emission_measure_neneh"): (em["n_e"] * em["n_H"] * dv, "cm**-3"),
    }
    # Always register velocities, even for the older states that were saved
    # before ``--save-state`` kept them: pyXSIM's default is to look for
    # ("gas", "velocity_*") whether or not we ask for Doppler shifts, so a
    # missing field is a hard error rather than "no shifting".
    zero = np.zeros_like(f["rho"])
    for name, key in (("velocity_x", "vx"), ("velocity_y", "vy"), ("velocity_z", "vz")):
        data[("gas", name)] = ((f[key] * CODE_VELOCITY) if key in f else zero, "cm/s")

    if em.get("ions"):
        # NEI: one field per ION, each already scaled by its element's abundance
        for name, a in em["ions"].items():
            data[("gas", f"{name.replace('^', '_')}_abundance")] = (a, "Zsun")
    elif em["abundances"]:
        for el, a in em["abundances"].items():
            # units MUST be "Zsun" (they are solar-unit abundances, so this is
            # also the honest label). pyXSIM masks the hydrogen fraction to the
            # emitting cells before using it, then for any var_elem field NOT in
            # Zsun divides the conversion factor by that masked array and
            # multiplies it against the FULL-length element field -- which
            # raises "operands could not be broadcast together" the moment
            # h_fraction is supplied as a field. Declaring Zsun takes that
            # branch out.
            data[("gas", f"{el}_abundance")] = (a, "Zsun")
    elif ejecta_zmet is not None:
        # legacy stand-in: no ejecta tracer, so select by the only thing
        # available -- dense, hot material interior to the blast wave
        rho_med = np.median(f["rho"])
        is_ejecta = (f["rho"] > 3.0 * rho_med) & (em["T"] > ejecta_temperature_K)
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

    em = emission_fields(state, max_abundance=args.max_abundance,
                         two_temperature=not args.single_temperature)
    if args.nei:
        if em["net"] is None:
            raise SystemExit("--nei needs the ionization age: rerun "
                             "casa_orlando.py with --composition")
        em["ions"], ion_report = ion_abundance_fields(
            em, em["net"], threshold=args.ion_threshold)
        for el, r in ion_report.items():
            print(f"[casa-obs] {el:2s}: <Z> = {r['mean_charge']:5.2f}, "
                  f"{r['kept']} ions carrying {100 * r['covered']:.1f}% of the "
                  f"emitting mass")
        print(f"[casa-obs] NEI: {len(em['ions'])} ion fields")
    describe_emission(em)
    ds = make_yt_dataset(state, em, zmet=args.zmet, ejecta_zmet=args.ejecta_zmet,
                         ejecta_temperature_K=args.ejecta_temperature)
    sp = ds.all_data()

    common = dict(
        # our own n_e n_H dV, from the simulated composition
        emission_measure_field=("gas", "emission_measure_neneh"),
        # do not let the cold, unshocked ejecta (which is at the pressure floor
        # and whose float32 temperature is meaningless) contribute
        kT_min=args.kt_min,
        binscale="log",
        # thermal broadening uses the single temperature it is given, i.e. T_e;
        # the ions are hotter, but even at kT_i = 30 keV the Fe-K line broadens
        # by ~5 eV against ACIS's ~120 eV resolution, so it does not matter here
        thermal_broad=True,
        abund_table="angr",
    )
    if em.get("ions"):
        # Every emitting element must be listed ion by ion: in NEI mode the
        # model has no "metallicity" to fall back on, which is the honest
        # behaviour -- an unlisted element simply does not emit.
        source = pyxsim.NEISourceModel(
            args.emin, args.emax, args.nbins,
            {name: ("gas", f"{name.replace('^', '_')}_abundance")
             for name in em["ions"]},
            **common)
    else:
        # With the simulated composition available every element varies per cell
        # and the emission measure carries the true electron and metal
        # densities, so ``Zmet`` covers only the elements no tracer stands for.
        var_elem = None
        if em["abundances"]:
            var_elem = {el: ("gas", f"{el}_abundance") for el in em["abundances"]}
            Zmet = args.zmet
        else:
            Zmet = ("gas", "metallicity") if args.ejecta_zmet is not None else args.zmet
        source = pyxsim.CIESourceModel(
            "apec", args.emin, args.emax, args.nbins, Zmet,
            var_elem=var_elem, **common)

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
    if not state["has_velocity"]:
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
    px, py, energy, _ = read_events(evtfile, npix=npix, scale_arcsec=scale_arcsec)
    sel = (energy > emin) & (energy < emax)
    img, _, _ = np.histogram2d(py[sel], px[sel], bins=npix, range=[[0, npix], [0, npix]])
    return img


def read_events(evtfile, *, npix=NPIX_COMPARE, scale_arcsec=PIXEL_ARCSEC):
    """Event sky coordinates -> pixels on the common grid, plus energies in keV.

    Works on both a SOXS event file and a real Chandra ``evt2``: they have the
    same structure (sky ``X``/``Y`` with the tangent-plane WCS in the
    ``TCRVL``/``TCRPX``/``TCDLT`` keywords, and an energy column in eV), only
    the column-name case differs. Deliberately the same inverse-gnomonic
    conversion followed by the same forward projection that
    ``/export/data/lstorcks/chandra_casa/make_epoch_images.py`` applies to the
    real data, so synthetic and real land on an identical grid through identical
    code -- which is the point of the exercise.

    Returns ``(px, py, energy_keV, exposure_s)``.
    """
    from astropy.io import fits

    with fits.open(evtfile) as f:
        hdu = f["EVENTS"]
        names = {c.name.upper(): c.name for c in hdu.columns}
        order = [c.name.upper() for c in hdu.columns]
        ix, iy = order.index("X") + 1, order.index("Y") + 1
        h = hdu.header
        x = np.asarray(hdu.data[names["X"]], dtype=np.float64)
        y = np.asarray(hdu.data[names["Y"]], dtype=np.float64)
        energy = np.asarray(hdu.data[names["ENERGY"]], dtype=np.float64) * 1e-3  # keV
        exposure = float(h.get("EXPOSURE", h.get("ONTIME", np.nan)))
        crvx, crpx, cdlx = h[f"TCRVL{ix}"], h[f"TCRPX{ix}"], h[f"TCDLT{ix}"]
        crvy, crpy, cdly = h[f"TCRVL{iy}"], h[f"TCRPX{iy}"], h[f"TCDLT{iy}"]

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
    return px, py, energy, exposure


#: ACIS-S responses SOXS ships, by Chandra cycle. Cycle n was observed in
#: 1999 + n, and what changes between them is chiefly the molecular
#: contamination on the optical blocking filter, which by cycle 20 absorbs most
#: of the flux below ~1 keV. Matching the cycle to the epoch is therefore not a
#: detail: it is the difference between comparing plasma models and comparing
#: filter thicknesses.
ACIS_S_CYCLES = (0, 10, 22, 28)


def instrument_for_epoch(label):
    """The SOXS ACIS-S response closest in Chandra cycle to a data epoch."""
    if label is None:
        return "chandra_aciss_cy0"
    try:
        year = int(str(label)[:4])
    except ValueError:
        return "chandra_aciss_cy0"
    cycle = min(ACIS_S_CYCLES, key=lambda c: abs(1999 + c - year))
    name = f"chandra_aciss_cy{cycle}"
    off = abs(1999 + cycle - year)
    print(f"[casa-obs] instrument {name} for epoch {label}"
          + (f" (nearest available cycle; {off} yr of contamination buildup "
             f"unaccounted for)" if off else " (exact match)"))
    return name


def load_real_epoch(label):
    """Load one binned real Chandra epoch (counts and exposure)."""
    path = REAL_EPOCH_DIR / f"epoch_{label}.npz"
    if not path.exists():
        raise SystemExit(f"no real epoch {label}; have "
                         f"{sorted(p.stem[6:] for p in REAL_EPOCH_DIR.glob('epoch_*.npz'))}")
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["counts"], dtype=np.float64), float(d["exposure"])


# =============================================================================
# ============ ↓ The spectral comparison ↓ ====================================
# =============================================================================
#: Energy bins for the spectral comparison (keV). 50 eV is about half the ACIS
#: resolution, so the He-alpha complexes are resolved without over-binning.
SPECTRUM_EBINS = np.arange(0.4, 8.001, 0.05)

#: Bands worth quoting separately, each dominated by one thing.
SPECTRAL_BANDS = (
    ("0.5-1.5 (O, Ne, Fe-L)", 0.5, 1.5),
    ("1.5-2.1 (Si He-a)", 1.5, 2.1),
    ("2.1-2.8 (S He-a)", 2.1, 2.8),
    ("2.8-4.2 (Ar, Ca)", 2.8, 4.2),
    ("4.2-6.0 (continuum)", 4.2, 6.0),
    ("6.0-7.0 (Fe-K)", 6.0, 7.0),
)


def event_spectrum(px, py, energy, *, radius_arcsec, ebins=SPECTRUM_EBINS):
    """Counts per energy bin inside a circular aperture on the common grid."""
    c = NPIX_COMPARE / 2
    rr = np.hypot(px - c, py - c) * PIXEL_ARCSEC
    return np.histogram(energy[rr < radius_arcsec], bins=ebins)[0].astype(float)


def real_epoch_spectrum(label, *, radius_arcsec, ebins=SPECTRUM_EBINS):
    """Spectrum of the real epoch, from the raw ``evt2`` files, on the same grid.

    Cached next to the epoch images, because it means re-reading tens of
    millions of events. The obsids belonging to an epoch are identified by the
    year in ``DATE-OBS``, and the summed exposure is checked against the one
    ``make_epoch_images.py`` recorded -- if they disagree the epoch definitions
    have drifted apart and the comparison would be silently wrong.
    """
    from astropy.io import fits

    cache = REAL_EPOCH_DIR / f"epoch_{label}_spectrum.npz"
    if cache.exists():
        d = np.load(cache)
        if d["ebins"].shape == ebins.shape and np.allclose(d["ebins"], ebins) \
                and float(d["radius"]) == radius_arcsec:
            return np.asarray(d["counts"], dtype=np.float64), float(d["exposure"])

    evt_dir = REAL_EPOCH_DIR.parent / "evt2"
    counts, exposure, used = np.zeros(len(ebins) - 1), 0.0, []
    for path in sorted(evt_dir.glob("acisf*_evt2.fits.gz")):
        with fits.open(path) as f:
            date = f["EVENTS"].header.get("DATE-OBS", "")
        if not date.startswith(str(label)):
            continue
        px, py, energy, exp = read_events(path)
        counts += event_spectrum(px, py, energy, radius_arcsec=radius_arcsec,
                                 ebins=ebins)
        exposure += exp
        used.append(path.name)
    if not used:
        raise SystemExit(f"no evt2 file with DATE-OBS in {label} under {evt_dir}")

    _, exp_ref = load_real_epoch(label)
    if abs(exposure - exp_ref) > 0.02 * exp_ref:
        print(f"[casa-obs] WARNING: epoch {label} exposure from evt2 "
              f"{exposure / 1e3:.1f} ks but the binned image says "
              f"{exp_ref / 1e3:.1f} ks -- different obsid sets, so the spectrum "
              f"and the image are not of the same data")
    print(f"[casa-obs] real spectrum from {', '.join(used)} "
          f"({exposure / 1e3:.1f} ks)")
    np.savez_compressed(cache, counts=counts, exposure=exposure, ebins=ebins,
                        radius=radius_arcsec)
    return counts, exposure


def spectrum_figure(syn, syn_exp, real, real_exp, *, out_path, label,
                    ebins=SPECTRUM_EBINS):
    """Synthetic and real Chandra spectra of the same sky region, in counts/s/keV.

    This is the test the images cannot do. The morphology is set by the
    hydrodynamics; the SPECTRUM is set by the plasma model -- the electron
    temperature, the composition and (still missing) the non-equilibrium
    ionization -- so this is where those show up. Both sides are folded through
    the ACIS response and are absorbed by the same column, so no unfolding is
    involved and nothing here is fitted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = 0.5 * (ebins[:-1] + ebins[1:])
    de = np.diff(ebins)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1.2]),
                             constrained_layout=True)
    ax = axes[0]
    ax.step(e, real / real_exp / de, where="mid", color="k", lw=1.1,
            label=f"Chandra {label}")
    ax.step(e, syn / syn_exp / de, where="mid", color="tab:red", lw=1.1,
            label="astronomix (synthetic)")
    for name, lo, hi in (("O/Ne/Fe-L", 0.5, 1.5), ("Si", 1.78, 1.94),
                         ("S", 2.38, 2.52), ("Ar", 3.06, 3.20),
                         ("Ca", 3.83, 3.97), ("Fe-K", 6.4, 6.75)):
        ax.axvspan(lo, hi, color="0.85", zorder=0)
        ax.text(0.5 * (lo + hi), 1.4, name, ha="center", fontsize=7, color="0.4")
    ax.set(yscale="log", ylabel="counts s$^{-1}$ keV$^{-1}$", xlim=(0.4, 8.0))
    ax.legend(fontsize=9)
    ax.set_title("spectrum inside the same aperture, through the same response")

    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (syn / syn_exp) / (real / real_exp)
    ax.step(e, ratio, where="mid", color="tab:red", lw=1.0)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set(xlabel="energy [keV]", ylabel="synthetic / real", yscale="log",
           ylim=(0.02, 50.0))
    fig.savefig(out_path, dpi=150)
    print(f"[casa-obs] saved {out_path}")


def report_bands(syn, syn_exp, real, real_exp, ebins=SPECTRUM_EBINS):
    """Print band count rates for both, which is the comparison in numbers."""
    print(f"    {'band [keV]':<24}{'synthetic':>12}{'real':>10}{'ratio':>8}")
    for name, lo, hi in SPECTRAL_BANDS:
        sel = (0.5 * (ebins[:-1] + ebins[1:]) > lo) & (0.5 * (ebins[:-1] + ebins[1:]) < hi)
        s, r = syn[sel].sum() / syn_exp, real[sel].sum() / real_exp
        print(f"    {name:<24}{s:>12.2f}{r:>10.2f}{s / max(r, 1e-30):>8.2f}")
# =============================================================================
# ============ ↑ The spectral comparison ↑ ====================================
# =============================================================================


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
    ap.add_argument("--instrument", default=None,
                    help="soxs instrument (the real Cas A observations are "
                         "ACIS-7 = S3). Default: the ACIS-S response of the "
                         "cycle closest to --compare, because the contamination "
                         "layer on the optical blocking filter thickens by the "
                         "year and mostly absorbs BELOW 1.5 keV -- comparing a "
                         "cycle-0 synthetic spectrum with a cycle-20 "
                         "observation is a soft-band error, not a model error")
    ap.add_argument("--aperture", type=float, default=200.0,
                    help="radius (arcsec) of the aperture the spectra are "
                         "extracted in, for both synthetic and real")
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
    ap.add_argument("--max-abundance", type=float, default=1.0e4,
                    help="largest per-element abundance (solar units) the model "
                         "will express. This is NOT a cap on the plasma: it sets "
                         "the reference hydrogen density that keeps the el/H "
                         "ratios finite in hydrogen-free ejecta, and the metal "
                         "densities are exact for any value (see "
                         ":func:`emission_fields`). Lowering it re-introduces the "
                         "old error -- at 50 the iron knots emit as if they held "
                         "a tenth of their iron")
    ap.add_argument("--nei", action="store_true",
                    help="non-equilibrium ionization: take the ion populations "
                         "from the simulated (kT_e, n_e t) instead of assuming "
                         "collisional equilibrium. Cas A's bulk plasma sits an "
                         "order of magnitude short of equilibrium, so CIE gets "
                         "the line-to-continuum ratio and the He/H-like ratios "
                         "wrong -- measurably: the CIE spectrum is 0.30x the "
                         "observed 0.5-1.5 keV rate and 2.9x at Fe-K")
    ap.add_argument("--ion-threshold", type=float, default=0.02,
                    help="drop ions holding less than this fraction of their "
                         "element's X-ray-emitting mass (each costs a 3D field)")
    ap.add_argument("--single-temperature", action="store_true",
                    help="use the single-fluid temperature instead of T_e. Only "
                         "for showing what the two-temperature model changes: "
                         "Coulomb equilibration is far from complete at 350 yr, "
                         "so T_e = T is not a defensible approximation here")
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
    if args.instrument is None:
        args.instrument = instrument_for_epoch(args.compare)

    state = load_state(args.state)
    print(f"[casa-obs] {args.state}: {state['num_cells']}^3, box {state['box_pc']} pc, "
          f"age {state['age_yr']:.0f} yr, "
          f"scalars {sorted(k for k in state['fields'] if k.startswith('C_') or k in ('shocked_fraction', 'time_since_shock', 'density_time'))}")

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

        # ---- the spectral comparison ----------------------------------------
        px, py, energy, _ = read_events(evtfile)
        syn_spec = event_spectrum(px, py, energy, radius_arcsec=args.aperture)
        real_spec, real_spec_exp = real_epoch_spectrum(
            args.compare, radius_arcsec=args.aperture)
        print(f"[casa-obs] band count rates inside r < {args.aperture:.0f}\":")
        report_bands(syn_spec, syn_exp, real_spec, real_spec_exp)
        spectrum_figure(syn_spec, syn_exp, real_spec, real_spec_exp,
                        out_path=FIGURES_DIR / f"{Path(args.out).name}_spectrum_{args.compare}.png",
                        label=args.compare)


if __name__ == "__main__":
    main()
