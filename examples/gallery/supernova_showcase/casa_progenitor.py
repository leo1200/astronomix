"""
A real presupernova star instead of a fitted ejecta profile.

``casa_calibrate_1d.py`` builds its ejecta from an analytic broken power law
whose inner slope, outer slope, core radius and total mass are all *fitted* to
the observed shock radii. That works -- the shock radii come out right -- but it
means the ejecta density profile, which is what sets where the reverse shock
sits and how much mass it has swept, is an input we tuned rather than a
prediction. The inner-slope table in ``CALIBRATION.md`` (delta = 0, 0.5, 1, 1.5,
2 scored against the unshocked mass) exists only because we are guessing it.

This module replaces the guess with a real star: the KEPLER solar-metallicity
presupernova models of Sukhbold, Ertl, Woosley, Brown & Janka (2016), 200 models
from 9 to 120 Msun, which the Garching core-collapse archive distributes openly
at

    https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/data/SEWBJ_2015/

Each file is a KEPLER dump with, per zone, the enclosed mass, radius, velocity,
density, temperature, pressure, specific energy and entropy, plus mass fractions
for 18 species from neutrons to Fe56. That is everything a 1D explosion needs.

**Cassiopeia A was a Type IIb**, so its progenitor had lost all but ~0.1 Msun of
its hydrogen envelope, and the s-series models are single stars that still carry
theirs (s16.0 keeps 5.3 Msun of H in an 887 Rsun red supergiant). Stripping is
therefore not a cosmetic step, it is what makes the model the right kind of
star, and it is what Orlando et al. do too -- their W15-IIb is a 15 Msun model
modified the same way. ``--strip`` removes the envelope down to a target
hydrogen mass, conserving nothing (the mass genuinely left the star through a
wind or to a companion).

The other required surgery is the **mass cut**: the iron core collapses to the
compact object and does not participate in the explosion. It is excised, and
what remains is the ejecta.

Usage (CPU, seconds)::

    ./run.sh casa_progenitor.py --model s16.0 --report
    ./run.sh casa_progenitor.py --model s16.0 --strip 0.1 --save casa_prog_s16_IIb.npz
"""

# general
import argparse
from pathlib import Path

# numerics
import numpy as np

# units and constants
from astropy import units as u
import astropy.constants as const

#: Where the unpacked Garching archive lives (see the module docstring).
PROGENITOR_DIR = Path("/export/data/lstorcks/progenitors/progenitor_models")

#: KEPLER dump columns, in file order, after the leading ``grid`` index.
SCALAR_COLUMNS = ("mass", "radius", "velocity", "density", "temperature",
                  "pressure", "specific_energy", "specific_entropy",
                  "angular_velocity", "a_bar", "y_e")

#: Species columns, in file order, after the ``stability`` and ``NETWORK`` words.
SPECIES = ("n", "H1", "He3", "He4", "C12", "N14", "O16", "Ne20", "Mg24",
           "Si28", "S32", "Ar36", "Ca40", "Ti44", "Cr48", "Fe52", "Fe54",
           "Ni56", "Fe56", "Fe")

MSUN = float((1.0 * u.Msun).to(u.g).value)
RSUN = float((1.0 * u.Rsun).to(u.cm).value)
G_CGS = float(const.G.cgs.value)


# =============================================================================
# ============ ↓ Reading a KEPLER presupernova dump ↓ =========================
# =============================================================================
def load_presn(model, directory=PROGENITOR_DIR):
    """Read one ``s<M>_presn`` KEPLER dump into a dict of CGS arrays.

    The zone quantities are OUTER-edge values (mass, radius and velocity) mixed
    with cell averages (density, temperature, ...), which is how KEPLER writes
    them; ``dm`` is differenced from the enclosed mass accordingly. Missing
    entries are written as ``---`` and become zero, not NaN -- an absent species
    is an abundance of zero, and letting it stay NaN quietly poisons every mass
    integral downstream.
    """
    path = Path(directory) / f"{model}_presn"
    if not path.exists():
        raise SystemExit(
            f"no progenitor {path}. Fetch and unpack the Garching archive:\n"
            f"  curl -O https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/"
            f"data/SEWBJ_2015/data/progenitor_models.tar.gz")

    rows = []
    with open(path) as fh:
        fh.readline()                                   # VERSION banner
        fh.readline()                                   # column names
        for line in fh:
            parts = line.replace("---", "0.0").split()
            if not parts or not parts[0].endswith(":"):
                continue
            # [1:12] are the scalars, [12:14] are the words 'stability' and
            # 'NETWORK', and the rest are the species mass fractions
            rows.append([float(x) for x in parts[1:12]]
                        + [float(x) for x in parts[14:]])
    a = np.asarray(rows, dtype=np.float64)
    if a.shape[1] != len(SCALAR_COLUMNS) + len(SPECIES):
        raise SystemExit(f"{path}: {a.shape[1]} columns, expected "
                         f"{len(SCALAR_COLUMNS) + len(SPECIES)}")

    star = {name: a[:, i] for i, name in enumerate(SCALAR_COLUMNS)}
    star["X"] = {sp: a[:, len(SCALAR_COLUMNS) + i] for i, sp in enumerate(SPECIES)}
    star["dm"] = np.diff(np.concatenate([[0.0], star["mass"]]))
    star["model"] = model
    return star


def species_mass(star, sp):
    """Total mass of species ``sp`` in the star, in solar masses."""
    return float(np.sum(star["X"][sp] * star["dm"]) / MSUN)


def core_boundary(star, species, threshold=0.01):
    """Mass coordinate of the core DEPLETED in ``species`` (grams).

    A "core" is named for what it has burned away, not for what it contains: the
    He core is everything inside the point where hydrogen vanishes, the CO core
    inside where helium does. So the boundary is the OUTERMOST zone still below
    the threshold, scanning from the surface inwards.

    Scanning the other way -- first zone above the threshold -- looks equivalent
    and is not: alpha-rich freezeout leaves percent-level helium in the deep
    core, so an inside-out scan reports a "CO core" of 0.35 Msun sitting inside
    a 1.5 Msun silicon core, which is impossible and was exactly the bug this
    docstring exists to prevent.
    """
    below = np.where(star["X"][species] < threshold)[0]
    return float(star["mass"][below[-1]]) if below.size else 0.0


def silicon_core_mass(star):
    """Outer edge of the silicon-dominated core, in grams.

    Silicon is not monotonic -- it is depleted both in the iron core below and
    in the oxygen shell above -- so :func:`core_boundary` cannot find it. Use
    the composition crossover instead: the outermost zone where silicon still
    outweighs oxygen.
    """
    inside_co = star["mass"] <= core_boundary(star, "He4")
    dominant = np.where((star["X"]["Si28"] > star["X"]["O16"]) & inside_co)[0]
    return float(star["mass"][dominant[-1]]) if dominant.size else 0.0


def binding_energy(star, mass_cut):
    """Gravitational binding energy of the material above ``mass_cut`` (erg).

    ``E_bind = int (G m / r - e_int) dm``, positive when the envelope is bound.
    This is the energy the explosion has to pay before any of it appears as
    ejecta kinetic energy, so it is the number that decides whether gravity can
    be neglected while the shock crosses the star.
    """
    sel = star["mass"] > mass_cut
    m, r, dm = star["mass"][sel], star["radius"][sel], star["dm"][sel]
    e_int = star["specific_energy"][sel]
    return float(np.sum((G_CGS * m / r - e_int) * dm))


# =============================================================================
# ============ ↑ Reading a KEPLER presupernova dump ↑ =========================
# =============================================================================


# =============================================================================
# ============ ↓ Making it the right kind of star ↓ ===========================
# =============================================================================
def strip_envelope(star, hydrogen_mass_msun):
    """Remove the hydrogen envelope down to ``hydrogen_mass_msun`` remaining.

    Cas A was a Type IIb: its progenitor reached core collapse with only ~0.1
    Msun of hydrogen left, having lost the rest to a wind or a companion. The
    s-series models are single stars that keep theirs, so this is the step that
    turns a II-P progenitor into a IIb one. Mass is deliberately NOT conserved
    -- the envelope really is gone.

    The cut is placed where the hydrogen ABOVE it integrates to the target, so
    the He core and every interface below it are untouched.
    """
    x_h, dm = star["X"]["H1"], star["dm"]
    # Hydrogen mass RETAINED if the star is truncated at each zone, i.e. the
    # hydrogen INTERIOR to it. (Using the hydrogen exterior instead cuts near
    # the surface and removes almost nothing -- the star stays a red supergiant
    # with 5.3 Msun of H and the run silently models the wrong supernova.)
    h_kept = np.cumsum(x_h * dm) / MSUN
    keep = np.where(h_kept <= hydrogen_mass_msun)[0]
    if keep.size == 0:
        raise SystemExit(f"the star has only {species_mass(star, 'H1'):.3f} "
                         f"Msun of hydrogen, less than the {hydrogen_mass_msun} "
                         f"asked for: it is already stripped")
    cut = int(keep[-1])

    out = dict(star)
    out.update({k: star[k][:cut + 1] for k in SCALAR_COLUMNS})
    out["dm"] = star["dm"][:cut + 1]
    out["X"] = {sp: v[:cut + 1] for sp, v in star["X"].items()}
    return out


def excise_core(star, mass_cut_msun):
    """Drop everything inside ``mass_cut_msun``; that material becomes the star.

    The iron core collapses to the neutron star and never joins the ejecta. The
    remaining zones keep their absolute mass coordinate, because the enclosed
    mass below still sets the gravity they feel.
    """
    cut_g = mass_cut_msun * MSUN
    sel = star["mass"] > cut_g
    out = dict(star)
    out.update({k: star[k][sel] for k in SCALAR_COLUMNS})
    out["dm"] = star["dm"][sel]
    out["X"] = {sp: v[sel] for sp, v in star["X"].items()}
    out["mass_cut"] = cut_g
    return out


def iron_core_mass(star):
    """Mass of the iron core, as the outer edge of the Fe-group-dominated zone.

    Used as the DEFAULT mass cut, since that is roughly what collapses. It is
    only a starting guess -- the real mass cut is set by the explosion mechanism
    we are not simulating -- so it is reported, not hidden.
    """
    fe_group = (star["X"]["Fe56"] + star["X"]["Fe54"]
                + star["X"]["Ni56"] + star["X"]["Fe"])
    dominant = np.where(fe_group > 0.5)[0]
    return float(star["mass"][dominant[-1]]) / MSUN if dominant.size else 0.0


# =============================================================================
# ============ ↑ Making it the right kind of star ↑ ===========================
# =============================================================================


def report(star, mass_cut_msun=None):
    """Print the structure that matters for the explosion, with nothing hidden."""
    m_tot = star["mass"][-1] / MSUN
    print(f"[prog] {star['model']}: {len(star['mass'])} zones, "
          f"M = {m_tot:.3f} Msun, R = {star['radius'][-1] / RSUN:.1f} Rsun "
          f"({star['radius'][-1]:.3e} cm)")

    m_he = core_boundary(star, "H1") / MSUN
    m_co = core_boundary(star, "He4") / MSUN
    m_si = silicon_core_mass(star) / MSUN
    m_fe = iron_core_mass(star)
    print(f"[prog] cores: Fe {m_fe:.3f} | Si/O {m_si:.3f} | C/O {m_co:.3f} | "
          f"He {m_he:.3f} | total {m_tot:.3f} Msun")

    masses = {sp: species_mass(star, sp) for sp in SPECIES}
    shown = ", ".join(f"{sp} {masses[sp]:.3f}" for sp in
                      ("H1", "He4", "C12", "O16", "Ne20", "Si28", "S32", "Fe56")
                      if masses[sp] > 1e-3)
    print(f"[prog] composition (Msun): {shown}")

    cut = m_fe if mass_cut_msun is None else mass_cut_msun
    e_bind = binding_energy(star, cut * MSUN)
    print(f"[prog] above a {cut:.3f} Msun mass cut: ejecta {m_tot - cut:.3f} Msun, "
          f"binding energy {e_bind:.3e} erg")
    print(f"[prog]   = {100 * e_bind / 2.09e51:.2f}% of Cas A's calibrated "
          f"2.09e51 erg, so gravity is "
          f"{'negligible' if abs(e_bind) < 0.02 * 2.09e51 else 'NOT negligible'} "
          f"for the ejecta profile")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="s16.0", help="KEPLER model stem, e.g. s16.0")
    ap.add_argument("--dir", default=str(PROGENITOR_DIR), help="progenitor directory")
    ap.add_argument("--strip", type=float, default=None, metavar="MSUN",
                    help="strip the hydrogen envelope to this remaining H mass "
                         "(Cas A was a IIb: ~0.1). Omit to keep the star as-is")
    ap.add_argument("--mass-cut", type=float, default=None, metavar="MSUN",
                    help="excise everything inside this mass (default: the iron "
                         "core, which is what collapses)")
    ap.add_argument("--report", action="store_true", help="print the structure")
    ap.add_argument("--save", default=None, help="write the prepared star to npz")
    args = ap.parse_args()

    star = load_presn(args.model, args.dir)
    if args.report:
        print("--- as evolved ---")
        report(star)

    if args.strip is not None:
        star = strip_envelope(star, args.strip)
        if args.report:
            print(f"--- stripped to {args.strip} Msun of hydrogen (Type IIb) ---")
            report(star, args.mass_cut)

    if args.save:
        cut = iron_core_mass(star) if args.mass_cut is None else args.mass_cut
        ej = excise_core(star, cut)
        np.savez_compressed(
            args.save, model=star["model"], mass_cut=cut,
            **{k: ej[k] for k in SCALAR_COLUMNS},
            dm=ej["dm"], **{f"X_{sp}": v for sp, v in ej["X"].items()})
        print(f"[prog] saved {args.save}: {len(ej['mass'])} zones, "
              f"{(ej['mass'][-1] - cut * MSUN) / MSUN:.3f} Msun of ejecta "
              f"above a {cut:.3f} Msun mass cut")


if __name__ == "__main__":
    main()
