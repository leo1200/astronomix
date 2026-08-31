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

  * **the dust-scattering halo** with ``--halo``: the same N_H that absorbs also
    scatters, and at 1.2e22 the 1 keV scattering depth is 0.71. Every photon is
    given ``Poisson(tau_sca(E))`` deflections drawn from a Mie calculation on an
    MRN grain population, at random positions along the 3.4 kpc. This is a model
    of the SIGHTLINE, not of the remnant: it adds no flux, it moves it, and it
    is the only candidate with the right radial shape for the 8.8 % of Chandra's
    counts that lie outside the forward shock. See :mod:`_dusthalo`.

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
  * **no scattering halo unless ``--halo`` is given**, so by default the model
    puts nothing at all outside the forward shock, where Chandra puts 8.8 % of
    its counts.
  * the halo's one assumption is where the dust sits along the sightline; the
    default spreads it uniformly, ``--halo-screen`` puts it in a single wall.
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

# ---- MPI ---------------------------------------------------------------
# ~98% of the wall time of this script is one call, pyxsim.make_photons, and it
# already contains yt's parallel_objects -- it was running on 1 of the node's
# 192 cores only because mpi4py was absent and the dataset was a single grid.
# Launch under mpirun to use it:
#
#     mpirun -n 16 /export/home/lstorcks/xrayobs/bin/python casa_observe.py ...
#
# Serial behaviour is unchanged when the module is missing or -n 1 is used, so
# nothing below is conditional on having MPI. Every rank builds the full plasma
# state (the numpy prep is not distributed), so the memory cost is per-rank:
# ~8 GB at 256^3 and ~110 GB at 512^3, which is what bounds the rank count.
try:
    from mpi4py import MPI as _MPI
    _MPI_SIZE = _MPI.COMM_WORLD.Get_size()
    _MPI_RANK = _MPI.COMM_WORLD.Get_rank()
except Exception:
    _MPI_SIZE, _MPI_RANK = 1, 0

if _MPI_SIZE > 1:
    # pyXSIM writes the photon list COLLECTIVELY, which needs an h5py built
    # against parallel HDF5. The wheel is serial, so every rank instead tries to
    # create the same file and the run dies inside h5py with "unable to lock
    # file" -- an error that says nothing about the actual cause. Refuse up
    # front instead.
    #
    # This is not worth fixing by building parallel HDF5: the work in this
    # study is many INDEPENDENT observations (a T_e scan, a plume ladder, halo
    # variants), so task-level concurrency -- N serial runs at once on a
    # 192-core node -- gives the same throughput with no new dependency and no
    # risk to a validated pipeline. Use that.
    import h5py as _h5py
    if not _h5py.get_config().mpi:
        raise SystemExit(
            f"casa_observe.py was launched under MPI (-n {_MPI_SIZE}) but h5py "
            f"{_h5py.__version__} has no MPI support, so pyXSIM cannot write "
            f"the photon list collectively. Run it SERIALLY and get throughput "
            f"by running independent observations concurrently instead.")

#: Grids per rank in the yt decomposition. More than one because the emitting
#: cells sit in a thin shell, so an equal-volume split is not an equal-work one.
_GRIDS_PER_RANK = 4


def _is_root():
    """True on the one rank that should print, plot and write final products."""
    return _MPI_RANK == 0


def _say(*a, **k):
    """print() on the root rank only -- otherwise every message appears N times."""
    if _is_root():
        print(*a, **k)

# the shared plasma physics (also used by casa_plasma.py)
from _plasma import (
    ATOMIC,
    CODE_DENSITY,
    CODE_LENGTH,
    CODE_VELOCITY,
    KEV_IN_K,
    M_P,
    SOLAR_NUMBER_RATIO_TO_H,
    TE_MODELS,
    TRACER_SPLIT_PRESETS,
    plasma_state,
    set_tracer_split,
    tracer_split_report,
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


def emission_fields(state, *, max_abundance, two_temperature=True,
                    te_model="ghavamian", kT_e_shock_keV=0.3,
                    beta_shock=0.05):
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
    ps = plasma_state(state["fields"], two_temperature=two_temperature,
                      te_model=te_model, kT_e_shock_keV=kT_e_shock_keV,
                      beta_shock=beta_shock)
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
        _say("[casa-obs] NOTE: no composition scalars -- cosmic abundances "
              "everywhere, and the ejecta temperature is understated ~3x")
        return
    if i["two_temperature"]:
        w = (em["n_e"] ** 2) * (em["T_e"] > 1e6)
        ratio = float(np.average(em["T_e"] / em["T"], weights=w))
        kt = float(np.average(em["T_e"], weights=w)) / KEV_IN_K
        setting = (f"{i['kT_e_shock_keV']:.2f} keV" if i["te_model"] == "ghavamian"
                   else f"beta = {i['beta_shock']:.3f}" if i["te_model"] == "beta"
                   else "no free parameter")
        _say(f"[casa-obs] electron temperature: EM-weighted T_e/T = {ratio:.3f} "
              f"(1 = full equilibration), kT_e = {kt:.2f} keV; the spectrum is "
              f"computed from T_e. Shock heating model '{i['te_model']}' "
              f"({setting})")
    else:
        _say("[casa-obs] NOTE: single-temperature plasma (T_e = T_i = T), "
              "which over-predicts the hard emission of recently shocked gas")
    _say(f"[casa-obs] hydrogen reference: invented in "
          f"{100 * r['invented_fraction']:.1f}% of cells, costing "
          f"{100 * r['spurious_ff']:.2f}% of the total free-free emission; "
          f"peak abundance {r['abundance_max']:.3g} solar")


def make_yt_dataset(state, em, *, zmet, ejecta_zmet, ejecta_temperature_K,
                    em_scale=1.0):
    """Wrap the state in a yt uniform grid with the fields pyXSIM needs.

    The emission measure is supplied EXPLICITLY rather than left to yt. yt's
    ``("gas", "emission_measure")`` is ``n_e n_H dV`` with ``n_e`` and ``n_H``
    derived from the density under ``default_species_fields="ionized"``, i.e.
    from a cosmic composition -- which is wrong by 1.8x in ``n_e`` and by orders
    of magnitude in ``n_H`` in the ejecta, and was inconsistent with the
    per-element abundances handed to the same source model.
    """
    import yt

    if _MPI_SIZE > 1:
        # must precede the load: yt decides then whether the grids are
        # distributed over the communicator
        yt.enable_parallelism()

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
        ("gas", "emission_measure_neneh"): (em["n_e"] * em["n_H"] * dv * em_scale,
                                            "cm**-3"),
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

    # nprocs is what makes MPI worth anything here. pyXSIM's make_photons loops
    # over the dataset's GRIDS through yt's parallel_objects, so a dataset built
    # as a single grid runs on one rank however many are launched -- the cost is
    # ~98% of the pipeline and it was being paid serially on a 192-core node.
    # Splitting into a few grids per rank also balances the load, which matters
    # because the emitting cells are concentrated in a thin shell and an
    # equal-VOLUME decomposition is not an equal-WORK one.
    nprocs = 1 if _MPI_SIZE == 1 else _GRIDS_PER_RANK * _MPI_SIZE
    ds = yt.load_uniform_grid(
        data, [n, n, n], length_unit="cm", bbox=bbox,
        nprocs=nprocs, default_species_fields="ionized",
    )
    return ds


def scratch_prefix(args):
    """Where this run's (large) intermediates live. Keyed on ``--out``, so two
    variants of the same state -- halo on and halo off -- do not overwrite each
    other's event files."""
    os.makedirs(args.scratch, exist_ok=True)
    return os.path.join(args.scratch, os.path.basename(args.out))


def make_photon_events(state, args, *, suffix="", em_scale=1.0):
    """State -> pyXSIM photon list -> projected, absorbed event list (.h5).

    Stops before the telescope, because the interstellar dust comes first: see
    :func:`apply_dust_halo`.

    ``suffix`` names the scratch files, so the sub-grid split (:mod:`_subgrid`)
    can call this once per phase without the two collidng. ``em_scale`` is the
    phase's VOLUME FRACTION: the emission measure is ``n_e n_H dV``, so a phase
    occupying a fraction of each cell emits that fraction -- which is the only
    change the volume split needs, because pyXSIM is handed the emission measure
    explicitly rather than deriving it from a volume.
    """
    import pyxsim

    em = emission_fields(state, max_abundance=args.max_abundance,
                         two_temperature=not args.single_temperature,
                         te_model=args.te_model,
                         kT_e_shock_keV=args.kt_e_shock,
                         beta_shock=args.beta_shock)
    if args.nei:
        if em["net"] is None:
            raise SystemExit("--nei needs the ionization age: rerun "
                             "casa_orlando.py with --composition")
        em["ions"], ion_report = ion_abundance_fields(
            em, em["net"], threshold=args.ion_threshold)
        for el, r in ion_report.items():
            _say(f"[casa-obs] {el:2s}: <Z> = {r['mean_charge']:5.2f}, "
                  f"{r['kept']} ions carrying {100 * r['covered']:.1f}% of the "
                  f"emitting mass")
        _say(f"[casa-obs] NEI: {len(em['ions'])} ion fields")
    describe_emission(em)
    ds = make_yt_dataset(state, em, zmet=args.zmet, ejecta_zmet=args.ejecta_zmet,
                         ejecta_temperature_K=args.ejecta_temperature,
                         em_scale=em_scale)
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
    prefix = scratch_prefix(args) + suffix
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
        _say("[casa-obs] NOTE: this state carries no velocities -- the line "
              "emission is unshifted (no Doppler structure)")
    _say(f"[casa-obs] {n_ph:.3e} photons from {n_cell:.3e} cells")

    n_ev = pyxsim.project_photons(
        f"{prefix}_photons", f"{prefix}_events", args.los, (RA0, DEC0),
        absorb_model="tbabs", nH=args.nh, abund_table="angr",
        # smear each cell's photons over the cell so the projection is not
        # a lattice of delta functions at the sub-arcsecond ACIS pixel scale
        kernel="gaussian",
    )
    _say(f"[casa-obs] {n_ev:.3e} photons survive absorption + projection")
    return f"{prefix}_events.h5"


def apply_dust_halo(h5_in, args):
    """Scatter the projected photons off the interstellar dust in the sightline.

    This belongs exactly here -- after ``project_photons`` has applied
    photoelectric absorption, before the telescope sees anything -- because that
    is where it happens physically. TBabs and dust scattering are separate
    processes on the same column and both are driven by the same ``--nh``.

    Writes a copy of the pyXSIM event list with the sky coordinates moved. The
    photon count is unchanged: this redistributes flux, and the photons pushed
    off the detector are lost from the aperture exactly as they are lost from
    the real observation. See :mod:`_dusthalo`.
    """
    import shutil

    import h5py
    import pyxsim

    from _dusthalo import scatter_sky_positions

    h5_out = f"{scratch_prefix(args)}_events_halo.h5"
    if os.path.abspath(h5_out) == os.path.abspath(h5_in):
        raise SystemExit("--halo would overwrite its own input; give a different --out")
    shutil.copyfile(h5_in, h5_out)
    with h5py.File(h5_out, "r+") as f:
        ra, dec = f["data/xsky"][:], f["data/ysky"][:]
        energy = f["data/eobs"][:]
        r_before = _offset_arcsec(ra, dec)
        ra, dec, _ = scatter_sky_positions(
            ra, dec, energy, nh=args.nh, profile=args.halo_profile,
            screen_x=args.halo_screen, seed=args.halo_seed)
        f["data/xsky"][:] = ra
        f["data/ysky"][:] = dec
        r_after = _offset_arcsec(ra, dec)
        # pyXSIM's EventList IGNORES the path it is handed and re-reads the
        # filenames recorded INSIDE the file, so a copied-and-edited event list
        # silently serves up the original photons. This must be updated, and
        # the round trip below is checked rather than assumed -- the failure is
        # completely silent otherwise: the run completes and the count rate
        # comes out identical to the no-halo case.
        f["info"].attrs["filenames"] = [h5_out]

    check = pyxsim.EventList(h5_out)
    if list(check.filenames) != [h5_out]:
        raise SystemExit(f"the halo event list still points at {check.filenames}; "
                         "SOXS would read the un-scattered photons")

    # What the scattering does to the aperture, in the units the count-rate
    # comparison is quoted in. This is the number the halo was built to settle:
    # a halo cannot add photons, so any change here is photons LEAVING.
    print("[casa-obs] photon fraction inside an aperture, before -> after dust:")
    for rad in (140.0, 200.0, 260.0):
        b = float((r_before < rad).mean())
        a = float((r_after < rad).mean())
        print(f"[casa-obs]   r < {rad:5.0f}\": {b:.4f} -> {a:.4f} "
              f"({100 * (a / b - 1):+.1f}%)")
    print(f"[casa-obs] wrote {h5_out}")
    return h5_out


def _offset_arcsec(ra, dec):
    """Angular distance from the Cas A centre [arcsec], for the aperture report."""
    d0, d = np.deg2rad(DEC0), np.deg2rad(dec)
    dr = np.deg2rad(ra - RA0)
    cosc = np.clip(np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(dr),
                   -1.0, 1.0)
    return np.rad2deg(np.arccos(cosc)) * 3600.0


def simulate_instrument(h5_events, args):
    """pyXSIM event list -> SIMPUT -> SOXS Chandra event file."""
    import pyxsim
    import soxs

    prefix = scratch_prefix(args)
    simput = f"{prefix}_simput"
    el = pyxsim.EventList(h5_events)
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


def subgrid_photon_events(state, args):
    """One event list per sub-grid phase, merged -- see :mod:`_subgrid`.

    A phase differs from its cell only in density and in how long it has been
    shocked, because the split is at fixed cell pressure. So each phase is the
    same state with ``rho``, ``time_since_shock`` and ``density_time`` scaled, and
    everything downstream -- temperature, T_e, n_e, the ionization age, the ion
    fractions -- is recomputed from those by the existing code path. There is no
    second physics implementation to keep in step, which is the whole reason the
    split was formulated at fixed pressure.

    **Calibrate chi with ``casa_xrism.py --subgrid-scan`` first.** A row there
    costs a couple of minutes and this costs 45, and the two observables (the
    electron temperature and the ionization age) pull in opposite directions, so
    guessing chi here wastes the expensive step.
    """
    import _subgrid

    chi, f_mass = args.subgrid_chi, args.subgrid_fmass
    _say(_subgrid.describe(chi, f_mass, net_mode=args.subgrid_net_mode))

    parts = []
    base = state["fields"]
    # the factors come from _subgrid, not from a copy here: density_time is
    # rho * t, so its factor is fixed once the other two are chosen, and a local
    # copy that broke that tie made two net_modes silently identical
    for name, (rho_f, t_f, net_f, vol) in _subgrid.phase_factors(
            chi, f_mass, args.subgrid_net_mode).items():
        _say(f"[casa-obs] --- sub-grid phase '{name}': rho x {rho_f:.3f}, "
             f"t_shock x {t_f:.3f}, n_e t x {net_f:.3f}, "
             f"volume fraction {vol:.3f} ---")
        fields = dict(base)
        fields["rho"] = base["rho"] * rho_f
        if "time_since_shock" in base:
            fields["time_since_shock"] = base["time_since_shock"] * t_f
        if "density_time" in base:
            fields["density_time"] = base["density_time"] * net_f
        phase_state = dict(state, fields=fields)
        parts.append(make_photon_events(phase_state, args,
                                        suffix=f"_{name}", em_scale=vol))

    return merge_event_lists(parts, f"{scratch_prefix(args)}_events.h5")


def merge_event_lists(inputs, out):
    """Concatenate pyXSIM event lists into one -- for multi-component sources.

    A pyXSIM event list is three arrays (``eobs``, ``xsky``, ``ysky``) plus a
    ``parameters`` group, so combining components is a concatenation and a sum of
    the fluxes. What is NOT optional is checking that the parameters agree:
    merging lists made with different collecting areas or exposure times would
    silently produce an event list whose count rate means nothing, and nothing
    downstream would notice.

    **And ``info/filenames`` must be rewritten.** ``pyxsim.EventList`` ignores
    the path it is handed and re-reads the filenames stored inside the file, so a
    merged list that still carries its first input's name serves SOXS the first
    component only -- the run completes and the answer looks like "the second
    component does nothing". That trap cost a full A/B cycle when the dust halo
    was added; see :func:`apply_dust_halo`, which does the same thing.
    """
    import h5py

    # These must match, or the merged rate is meaningless. NOT emin/emax: those
    # are OUTPUTS -- pyXSIM stores the actual observed-frame energy range of the
    # photons it drew, so two components at different temperatures legitimately
    # differ there (0.402 vs 0.438 keV for the two sub-grid phases). Including
    # them refused a perfectly valid merge after 80 minutes of photon
    # generation. The merged range is the union, set below.
    STRICT = ("area", "exp_time", "nH", "redshift")
    SPAN = ("emin", "emax")
    with h5py.File(inputs[0], "r") as f0:
        ref = {k: f0[f"parameters/{k}"][()] for k in STRICT}
        span = {k: f0[f"parameters/{k}"][()] for k in SPAN}
    for path in inputs[1:]:
        with h5py.File(path, "r") as f:
            for k in STRICT:
                v = f[f"parameters/{k}"][()]
                if not np.isclose(v, ref[k]):
                    raise SystemExit(
                        f"refusing to merge {path} into {inputs[0]}: "
                        f"{k} = {v} against {ref[k]}. Merging event lists made "
                        f"with different {k} gives a count rate that means "
                        f"nothing.")
            span["emin"] = min(span["emin"], f["parameters/emin"][()])
            span["emax"] = max(span["emax"], f["parameters/emax"][()])

    import shutil
    shutil.copyfile(inputs[0], out)
    with h5py.File(out, "r+") as fo:
        for path in inputs[1:]:
            with h5py.File(path, "r") as fi:
                for name in ("eobs", "xsky", "ysky"):
                    a, b = fo[f"data/{name}"], fi[f"data/{name}"]
                    n0 = a.shape[0]
                    a.resize((n0 + b.shape[0],))
                    a[n0:] = b[:]
                fo["parameters/flux"][()] = (fo["parameters/flux"][()]
                                             + fi["parameters/flux"][()])
        for k in SPAN:                          # the merged list spans the union
            fo[f"parameters/{k}"][()] = span[k]
        # pyxsim re-reads this, not the path it is given
        fo["info"].attrs["filenames"] = np.array([out], dtype=object)
        n = fo["data/eobs"].shape[0]
    with h5py.File(out, "r") as fo:                 # assert the round trip
        assert fo["data/eobs"].shape[0] == n
        assert str(np.asarray(fo["info"].attrs["filenames"])[0]) == out
    _say(f"[casa-obs] merged {len(inputs)} components -> {n:.3e} events")
    return out


def make_events(state, args):
    """The whole forward model: state -> photons -> dust -> Chandra event file.

    Only the FIRST step is parallel. pyXSIM distributes photon generation and
    projection over the grids; the dust halo and SOXS's ``instrument_simulator``
    are serial, and running them on every rank would apply the halo N times to
    the same file and have N processes write the same event list. So the
    non-root ranks stop at the barrier and return.

    With ``--subgrid-chi`` the emission is computed once per PHASE and the event
    lists merged. That is what a two-component thermal model in each cell means,
    and it is why the emission measure being explicit (:func:`make_yt_dataset`)
    matters: a phase filling part of a cell is the same fields with the emission
    measure scaled by its volume fraction, so no new plumbing is needed.
    """
    if args.pyxsim_events is None and args.subgrid_chi is not None:
        h5 = subgrid_photon_events(state, args)
    else:
        h5 = args.pyxsim_events or make_photon_events(state, args)
    if _MPI_SIZE > 1:
        _MPI.COMM_WORLD.Barrier()
        if not _is_root():
            return None
    if args.halo:
        h5 = apply_dust_halo(h5, args)
    return simulate_instrument(h5, args)


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


def halo_image_figure(panels, *, out_path, crop_arcsec=250.0, smooth_pix=3.0,
                      vmin=2e-6, vmax=2e-3):
    """The image on an ABSOLUTE, shared, logarithmic surface-brightness scale.

    :func:`comparison_figure` deliberately scales each panel to its own 99.6th
    percentile and crops to 200", which is right for judging morphology and
    useless for judging a halo: the scattered light is three decades below the
    shell and mostly outside the crop. This is the complementary view -- same
    counts/s/pixel scale on every panel, wide enough to hold r < 255", and
    logarithmic so three decades are visible at once.

    ``panels`` is a list of ``(image, exposure_s, title)``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from scipy.ndimage import gaussian_filter

    cmap = LinearSegmentedColormap.from_list("chandra_blue", [
        (0.00, "#000005"), (0.15, "#04102e"), (0.35, "#0a3f8f"),
        (0.60, "#1f7fd4"), (0.82, "#7fc4ef"), (1.00, "#f2fbff")])

    # the grid is only 1024 x 0.492" = 504" wide, so a crop wider than its half
    # would silently wrap the slice and give a black frame
    c = NPIX_COMPARE // 2
    half = min(int(crop_arcsec / PIXEL_ARCSEC), c)
    crop_arcsec = half * PIXEL_ARCSEC
    sl = slice(c - half, c + half)
    ext = [crop_arcsec, -crop_arcsec, -crop_arcsec, crop_arcsec]
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6.8),
                             facecolor="black")
    for ax, (img, exp, title) in zip(np.atleast_1d(axes), panels):
        a = gaussian_filter(np.asarray(img, float)[sl, sl] / exp, smooth_pix)
        im = ax.imshow(np.maximum(a, vmin), origin="lower", extent=ext, cmap=cmap,
                       norm=norm, interpolation="bilinear")
        ax.add_patch(plt.Circle((0, 0), R_FS_ARCSEC, fill=False, color="tab:red",
                                ls=":", lw=1.0, alpha=0.8))
        ax.set_title(title, color="white", fontsize=11)
        ax.set_facecolor("black")
        ax.tick_params(colors="0.6", labelsize=8)
        ax.set_xlabel("offset [arcsec]", color="0.6", fontsize=9)
    cb = fig.colorbar(im, ax=np.atleast_1d(axes), fraction=0.03, pad=0.01)
    cb.set_label("surface brightness [counts s$^{-1}$ pixel$^{-1}$], 0.5-7 keV",
                 color="0.8", fontsize=9)
    cb.ax.tick_params(colors="0.7", labelsize=8)
    fig.savefig(out_path, dpi=150, facecolor="black", bbox_inches="tight")
    print(f"[casa-obs] saved {out_path}")


#: Annuli the radial comparison is quoted in [arcsec]. The first two straddle
#: the bright shell, the rest are outside the forward shock, which is where the
#: model and Chandra part company.
RADIAL_ANNULI = ((60, 100), (100, 140), (140, 160), (160, 180), (180, 220),
                 (220, 260))

#: Cas A's observed forward-shock radius, 2.52 pc at 3.4 kpc [arcsec]. The
#: "outside the remnant" fraction is quoted against THIS, not against the
#: nearest annulus edge -- the two differ by a lot, because the profile is
#: falling by an order of magnitude across the shock (13.5 % beyond 140",
#: 8.8 % beyond 153", 7.0 % beyond 160").
R_FS_ARCSEC = np.rad2deg(2.52 / (DISTANCE_KPC * 1e3)) * 3600.0


def radial_profile(img, exposure_s, bins):
    """Azimuthally averaged surface brightness [counts/s/pixel] in ``bins``."""
    c = NPIX_COMPARE // 2
    yy, xx = np.mgrid[:NPIX_COMPARE, :NPIX_COMPARE]
    rr = np.hypot(xx - c, yy - c) * PIXEL_ARCSEC
    idx = np.digitize(rr.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    tot = np.bincount(idx[ok], weights=img.ravel()[ok], minlength=len(bins) - 1)
    area = np.bincount(idx[ok], minlength=len(bins) - 1)
    return np.where(area > 0, tot / np.maximum(area, 1), 0.0) / exposure_s


def report_radial(syn, syn_exp, real, real_exp):
    """Surface brightness annulus by annulus -- the measurement outside r_FS.

    Quoted in counts/s/pixel, NOT normalised: the deficit beyond the forward
    shock is an amplitude, and a profile scaled to its own peak hides it.
    """
    edges = np.array(sorted({0.0} | {float(e) for a in RADIAL_ANNULI for e in a}))
    s = radial_profile(syn, syn_exp, edges)
    r = radial_profile(real, real_exp, edges)
    print("[casa-obs] surface brightness [counts/s/pixel], 0.5-7 keV:")
    print("[casa-obs]   radius        model      Chandra    real/model")
    for lo, hi in RADIAL_ANNULI:
        i = int(np.searchsorted(edges, lo))
        ratio = r[i] / s[i] if s[i] > 0 else np.inf
        print(f"[casa-obs]   {lo:3d}-{hi:3d}\"   {s[i]:.3e}    {r[i]:.3e}    {ratio:8.1f}")
    # the headline number, straight off the images rather than off the binned
    # profile, so nothing is extrapolated over the corners the grid does not
    # cover past r = 252"
    c = NPIX_COMPARE // 2
    yy, xx = np.mgrid[:NPIX_COMPARE, :NPIX_COMPARE]
    rr = np.hypot(xx - c, yy - c) * PIXEL_ARCSEC
    inside, outside = rr < 260.0, (rr >= R_FS_ARCSEC) & (rr < 260.0)
    for name, img in (("model", syn), ("Chandra", real)):
        frac = img[outside].sum() / max(img[inside].sum(), 1e-30)
        print(f"[casa-obs]   fraction of the r < 260\" flux beyond "
              f"r_FS = {R_FS_ARCSEC:.0f}\": {name:8s} {100 * frac:5.2f}%")


def radial_profile_figure(syn, real, *, out_path, label, syn_exposure_s,
                          real_exposure_s, overlay=None):
    """Azimuthally averaged surface-brightness profiles -- the quantitative test.

    In counts/s/pixel through the same response on the same grid, so the two
    curves are directly comparable in amplitude as well as in shape. The forward
    shock is the outer break; the flux beyond it is the sightline rather than
    the remnant (dust scattering, plus the far PSF wings SOXS does not model --
    it convolves with the core PSF image only). Without ``--halo`` the model
    puts nothing out there at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.arange(0, 264, 4.0)
    mid = 0.5 * (bins[:-1] + bins[1:])

    curves = [(syn, syn_exposure_s, "astronomix (synthetic)", "-", None)]
    if overlay is not None:
        o = np.load(overlay)
        curves.append((np.asarray(o["counts"], float), float(o["exposure"]),
                       Path(overlay).stem.replace("_synimg", ""), "-", "0.55"))
    curves.append((real, real_exposure_s, f"Chandra {label}", "--", None))

    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    for img, exp, lbl, style, col in curves:
        prof = radial_profile(img, exp, bins)
        ax.semilogy(mid, np.maximum(prof, 1e-9), style, label=lbl, color=col)
    # observed shock radii at 3.4 kpc
    for r_pc, name, col in ((2.52, "$r_{FS}$", "tab:red"), (1.58, "$r_{RS}$", "tab:orange")):
        arcsec = np.rad2deg(r_pc / (DISTANCE_KPC * 1e3)) * 3600.0
        ax.axvline(arcsec, color=col, ls=":", lw=1.2)
        ax.text(arcsec, 0.96, name, color=col, ha="center", va="top", fontsize=9,
                transform=ax.get_xaxis_transform())
    ax.set(xlabel="radius [arcsec]",
           ylabel="surface brightness [counts s$^{-1}$ pixel$^{-1}$]",
           ylim=(3e-6, 5e-3))
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
    ap.add_argument("--subgrid-chi", type=float, default=None, metavar="CHI",
                    help="re-read every cell as a two-phase medium of density "
                         "contrast CHI and observe both phases (see _subgrid). "
                         "This is an INTERPRETATION LAYER, not simulated "
                         "structure. CALIBRATE IT WITH casa_xrism.py "
                         "--subgrid-scan FIRST: a row there costs minutes and "
                         "this costs 45")
    ap.add_argument("--subgrid-fmass", type=float, default=0.5, metavar="F",
                    help="mass fraction of the dense phase")
    ap.add_argument("--subgrid-net-mode", default="unchanged",
                    choices=("density", "unchanged", "crossing"),
                    help="how the dense phase's ionization age follows from the "
                         "cell's; n_e t is already at the top of the observed "
                         "range with no boost, which bounds this")
    ap.add_argument("--tracer-split", default="hwang_laming",
                    choices=sorted(TRACER_SPLIT_PRESETS),
                    help="how the Si and O tracers divide among the elements "
                         "they stand for. HWANG_LAMING (default) reproduces the "
                         "remnant-integrated shocked masses but puts Ar/Si and "
                         "Ca/Si at 1.7-2.7x solar everywhere; XRISM_BULK "
                         "matches the per-pixel line ratios XRISM measures "
                         "instead. The two disagree because the real "
                         "enhancement is confined to the jets and one tracer "
                         "cannot say so -- see _plasma.TRACER_SPLIT_PRESETS")
    ap.add_argument("--te-model", default="ghavamian", choices=TE_MODELS,
                    help="electron-heating prescription at the shock. The "
                         "default is Ghavamian et al. (2007)'s constant 0.3 keV, "
                         "which was calibrated on HYDROGEN-dominated ISM shocks "
                         "and is applied here to a reverse shock in "
                         "METAL-dominated ejecta -- the single largest "
                         "assumption in the forward model. 'beta' scales with "
                         "the local post-shock temperature instead; "
                         "'equilibrated' and 'minimal' are the hot and cold "
                         "bounds. See _plasma.shock_electron_temperature")
    ap.add_argument("--kt-e-shock", type=float, default=0.3, metavar="KEV",
                    help="the constant, for --te-model ghavamian (keV)")
    ap.add_argument("--beta-shock", type=float, default=0.05, metavar="B",
                    help="T_e / T at the shock, for --te-model beta")
    ap.add_argument("--no-background", action="store_true", help="no instrumental/sky background")
    ap.add_argument("--scratch", default="/export/data/lstorcks/supernova_showcase/xray_scratch",
                    help="where the (large) photon/event/SIMPUT intermediates go")
    ap.add_argument("--compare", default=None,
                    help="also bin onto the real-data grid and compare with this epoch "
                         "(e.g. 2004)")
    ap.add_argument("--events", default=None,
                    help="skip the simulation and re-bin/compare an existing event file")
    ap.add_argument("--pyxsim-events", default=None,
                    help="reuse an existing pyXSIM ``*_events.h5`` (the projected, "
                         "absorbed photon list) instead of regenerating it. That is "
                         "the expensive stage and it is untouched by the dust "
                         "options below, so a halo A/B costs only the SOXS run")

    # ---- the sightline, not the remnant ------------------------------------
    ap.add_argument("--halo", action="store_true",
                    help="scatter the photons off interstellar dust before they "
                         "reach the telescope. 8.8%% of Chandra's r < 260\" counts "
                         "lie outside the forward shock and the model puts 0.1%% "
                         "there; a scattering halo at N_H = 1.2e22 is the only "
                         "candidate with the right radial shape. Adds no flux -- "
                         "it moves photons, and loses the ones it moves off the "
                         "detector. See :mod:`_dusthalo`")
    ap.add_argument("--halo-profile", default="mie", choices=["mie", "draine03"],
                    help="angular distribution of a scattering. 'mie' (default) "
                         "tabulates it from newdust's Mie calculation on an MRN "
                         "grain population; 'draine03' uses the published "
                         "analytic approximation (Draine 2003, Eqs. 9 and 11), "
                         "whose median angle is a third smaller because WD01 is "
                         "not MRN. Running both measures the grain-size systematic")
    ap.add_argument("--halo-screen", type=float, default=None, metavar="X",
                    help="put all the dust in one screen at x = 1 - d/D instead "
                         "of spreading it uniformly along the sightline. Cas A "
                         "sits just beyond the Perseus arm, so x = 0.41 (dust at "
                         "2 kpc of 3.4) is the physically motivated alternative "
                         "to the default uniform column")
    ap.add_argument("--halo-seed", type=int, default=1234,
                    help="RNG seed for the scattering draw")
    ap.add_argument("--overlay-synimg", default=None, metavar="NPZ",
                    help="also draw this saved ``*_synimg.npz`` on the radial "
                         "figure. Meant for a before/after: point a --halo run "
                         "at the no-halo run's image and the sightline's effect "
                         "is one plot instead of two")
    args = ap.parse_args()

    if args.out is None:
        args.out = str(FIGURES_DIR.parent / Path(args.state).stem)
    if args.instrument is None:
        args.instrument = instrument_for_epoch(args.compare)

    # BEFORE the state is read: element_mass_fractions looks TRACER_SPLIT up at
    # call time, so setting it any later would mix two conventions in one run.
    set_tracer_split(args.tracer_split)
    _say(f"[casa-obs] tracer split '{args.tracer_split}':\n"
         f"{tracer_split_report()}")

    state = load_state(args.state)
    print(f"[casa-obs] {args.state}: {state['num_cells']}^3, box {state['box_pc']} pc, "
          f"age {state['age_yr']:.0f} yr, "
          f"scalars {sorted(k for k in state['fields'] if k.startswith('C_') or k in ('shocked_fraction', 'time_since_shock', 'density_time'))}")

    evtfile = args.events or make_events(state, args)
    if evtfile is None:            # a non-root MPI rank: its work is done
        return

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
                              label=args.compare, syn_exposure_s=syn_exp,
                              real_exposure_s=real_exp, overlay=args.overlay_synimg)
        report_radial(syn, syn_exp, real, real_exp)

        # the absolute-scale view, which is the only one a halo shows up in
        panels = [(syn, syn_exp, f"astronomix{' + dust halo' if args.halo else ''}"
                                 f"\nsynthetic ACIS-S, {args.exposure:.0f} ks")]
        if args.overlay_synimg:
            o = np.load(args.overlay_synimg)
            panels.insert(0, (np.asarray(o["counts"], float), float(o["exposure"]),
                              f"{Path(args.overlay_synimg).stem.replace('_synimg', '')}"
                              f"\n(no sightline model)"))
        panels.append((real, real_exp, f"Chandra, real (epoch {args.compare})"
                                       f"\n{real_exp / 1e3:.0f} ks"))
        halo_image_figure(panels, out_path=FIGURES_DIR /
                          f"{Path(args.out).name}_absolute_{args.compare}.png")

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
