"""
Non-equilibrium plasma diagnostics from the shock-history scalars.

``casa_orlando.py --composition`` carries, per parcel, the time since it was
shocked and the integral of density over that time. Those two numbers are what
turn a hydrodynamic state into something an X-ray spectrum can be predicted
from, without integrating an ionization network (Dwarkadas, Dewey & Bauer 2010;
Orlando et al. 2015):

* **ionization age** ``n_e t`` = the electron column a parcel has swept since
  being shocked. Cas A's bulk shocked plasma sits at ``n_e t ~ 1e11 cm^-3 s`` at
  ``kT_e ~ 2 keV``, with the shocked Fe more ionized at ``~1e12`` (Hwang &
  Laming 2003, 2012). Below ``~1e12`` the plasma is under-ionized relative to
  equilibrium, so assuming collisional ionization equilibrium gets the He- and
  H-like line ratios wrong — which is the single largest remaining error in the
  synthetic spectra.

* **composition.** Every one of these conversions depends on it, and supernova
  ejecta are nothing like cosmic: the fully ionized oxygen layer that carries
  most of Cas A's ejecta mass has ``mu = 1.78`` against cosmic 0.61 and
  ``mu_e = 2.0`` against 1.18. The physics lives in :mod:`_plasma`, which
  ``casa_observe.py`` shares, so the diagnostics and the forward model cannot
  disagree about the temperature of the same cell.

* **electron and ion temperatures.** Behind a > 1000 km/s shock the electrons
  are heated to only ``kT_e ~ 0.3 keV`` almost independently of Mach number
  (Ghavamian, Laming & Rakowski 2007), while the ions take nearly all of the
  shock energy; the two then relax by Coulomb collisions over the time since
  shocking. At Cas A's densities and age the relaxation is incomplete, so
  ``T_e < T`` and using the single-fluid temperature over-predicts the X-ray
  emission from the hottest gas.

The main diagnostic here is the **emission-measure distribution in
(kT_e, n_e t)**, which is exactly the plane Hwang & Laming fit the real
observations in, so the model can be compared against their result rather than
against a picture.

Usage (CPU)::

    CUDA_VISIBLE_DEVICES= ./run.sh casa_plasma.py \\
        /export/data/lstorcks/supernova_showcase/orl_n256_comp.npz
"""

# general
import argparse
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# units and constants
from astropy import constants as const

# the shared plasma physics (also used by the X-ray forward model)
from _plasma import (
    CODE_DENSITY,
    CODE_LENGTH,
    KEV_IN_K,
    TRACER_SPLIT_PRESETS,
    _self_check,
    load_diagnostic_state,
    plasma_state,
    set_tracer_split,
    tracer_split_report,
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# observational targets (Hwang & Laming 2003, 2012)
TARGET_KT_KEV = 2.0
TARGET_NET_BULK = 1.0e11    # cm^-3 s, bulk shocked plasma
TARGET_NET_FE = 1.0e12      # cm^-3 s, shocked Fe


def emission_measure_distribution(n_e, T_e, net, *, weight=None, bins=80):
    """``EM`` binned in ``(kT_e, n_e t)`` -- the plane Hwang & Laming fit.

    The weight is ``n_e^2`` rather than the conventional ``n_e n_H``: in ejecta
    that are hydrogen-free by construction ``n_H`` is not a measure of anything,
    and both the free-free continuum and the line emission scale with the
    electron density times the emitting-ion density, for which ``n_e`` is the
    available proxy. (Hwang & Laming quote ``n_e n_H`` emission measures from
    fits that assume a hydrogen-normalised APEC model, so the comparison is of
    the DISTRIBUTION's location in the plane, not of the absolute normalisation.)
    """
    kT = T_e / KEV_IN_K
    em = n_e ** 2
    if weight is not None:
        em = em * weight

    ok = (net > 1e6) & (kT > 1e-3) & np.isfinite(kT) & np.isfinite(net)
    kt_bins = np.logspace(-2, 1.3, bins)
    net_bins = np.logspace(8, 13, bins)
    H, _, _ = np.histogram2d(kT[ok].ravel(), net[ok].ravel(),
                             bins=[kt_bins, net_bins], weights=em[ok].ravel())
    return H, kt_bins, net_bins


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", help="a --save-state npz carrying the shock history")
    ap.add_argument("--tracer-split", default="hwang_laming",
                    choices=sorted(TRACER_SPLIT_PRESETS),
                    help="see _plasma.TRACER_SPLIT_PRESETS; the two presets\n"
                         "disagree, and every mass and abundance below depends\n"
                         "on which is used")
    ap.add_argument("--out", default="casa_plasma", help="figure name stem")
    args = ap.parse_args()

    _self_check()

    # BEFORE anything reads a composition: element_mass_fractions looks
    # TRACER_SPLIT up at call time, so setting it late would silently mix
    # two conventions in one report.
    set_tracer_split(args.tracer_split)
    print(f"[plasma] tracer split '{args.tracer_split}':\n"
          f"{tracer_split_report()}")

    fields, meta = load_diagnostic_state(args.state)
    state = fields
    age, box, n = meta["age"], meta["box"], meta["num_cells"]

    # all of the physics, with the composition the run actually carried
    ps = plasma_state(state)
    T, T_e, T_i, net = ps["T"], ps["T_e"], ps["T_i"], ps["net"]
    n_e, shocked = ps["n_e"], ps["shocked"]
    if not ps["info"]["composition_tracked"]:
        print("[plasma] WARNING: no composition scalars in this state -- falling "
              "back to cosmic abundances everywhere, which understates the "
              "ejecta temperature by ~3x")
    cell_vol_cm3 = (box / n * CODE_LENGTH) ** 3
    m_sun = const.M_sun.cgs.value
    mass = state["rho"] * CODE_DENSITY * cell_vol_cm3 / m_sun

    print(f"\n[plasma] {args.state}: {n}^3, age {age:.0f} yr")
    print(f"    shocked volume fraction {shocked.mean():.3f}")
    if "C_ej" in state:
        m_ej_shocked = float(np.sum(mass * state["C_ej"] * shocked))
        m_ej_total = float(np.sum(mass * state["C_ej"]))
        print(f"    ejecta mass {m_ej_total:.3f} Msun, of which shocked "
              f"{m_ej_shocked:.3f} ({100 * m_ej_shocked / max(m_ej_total, 1e-30):.0f}%)"
              f"   [Hwang & Laming 2012: 2.8-3.7 Msun shocked, 0.3-0.4 unshocked]")
        # per ELEMENT, with each tracer divided by _plasma.TRACER_SPLIT: the
        # carried "Si" is the Si/S/Ar/Ca layer and the carried "O" the O/Ne/Mg
        # layer, so comparing the raw tracer against a single measured element
        # over-counts it (it read 1.6x high for Si that way)
        for el, target in (("O", 2.0), ("Ne", 0.03), ("Mg", 0.03), ("Si", 0.08),
                           ("S", 0.06), ("Ar", 0.02), ("Ca", 0.02), ("Fe", 0.14)):
            if el not in ps["X"] or np.ndim(ps["X"][el]) == 0:
                continue
            m_el = float(np.sum(mass * ps["X"][el] * state["C_ej"] * shocked))
            print(f"    shocked {el:2s}: {m_el:.4f} Msun   [Hwang & Laming "
                  f"2012: {target}]")

    if np.any(shocked):
        em_w = (n_e ** 2)[shocked]
        avg = lambda f: float(np.average(f[shocked], weights=em_w))    # noqa: E731
        mom = ps["moments"]
        print(f"    EM-weighted mu = {avg(mom['mu']):.3f}, mu_e = "
              f"{avg(mom['mu_e']):.3f}   [cosmic 0.62 / 1.19; the shocked gas is "
              "a mix of wind and metal ejecta]")
        print(f"    EM-weighted kT_e = {avg(T_e) / KEV_IN_K:.2f} keV   "
              f"[observed ~{TARGET_KT_KEV}]")
        print(f"    EM-weighted kT_i = {avg(T_i) / KEV_IN_K:.2f} keV, single-fluid "
              f"kT = {avg(T) / KEV_IN_K:.2f} keV")
        print(f"    EM-weighted n_e t = {avg(net):.2e} cm^-3 s   "
              f"[observed ~{TARGET_NET_BULK:.0e}]")
        print(f"    EM-weighted T_e/T = {avg(T_e / T):.3f} "
              "(1 = full equilibration)")

    # ---- figure -----------------------------------------------------------
    H, kt_bins, net_bins = emission_measure_distribution(n_e, T_e, net)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.set_facecolor("black")          # so the empty regions match the colormap
    pos = H[H > 0]
    if pos.size:
        m = ax.pcolormesh(kt_bins[:-1], net_bins[:-1], H.T,
                          norm=LogNorm(vmin=pos.max() * 1e-4, vmax=pos.max()),
                          cmap="magma", shading="auto")
        fig.colorbar(m, ax=ax, label="emission measure [arb.]")
    ax.plot([TARGET_KT_KEV], [TARGET_NET_BULK], "c*", ms=16,
            label="bulk (Hwang & Laming)")
    ax.plot([TARGET_KT_KEV], [TARGET_NET_FE], "w*", ms=13, label="shocked Fe")
    ax.set(xscale="log", yscale="log", xlabel="$kT_e$ [keV]",
           ylabel=r"$n_e t$ [cm$^{-3}$ s]",
           title="emission measure vs electron temperature and ionization age")
    ax.legend(fontsize=8, loc="lower left")

    # radial profile of the two temperatures
    ax = axes[1]
    c = (np.arange(n) + 0.5) / n * box - box / 2
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    edges = np.linspace(0, box / 2, 60)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, len(edges) - 2)
    w = (n_e ** 2).ravel()
    for field, lbl, style in ((T_e, "$T_e$", "-"), (T_i, "$T_i$", "--"),
                              (T, "single-fluid $T$", ":")):
        num = np.bincount(idx, weights=(field.ravel() * w), minlength=len(edges) - 1)
        den = np.bincount(idx, weights=w, minlength=len(edges) - 1)
        prof = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan) / KEV_IN_K
        ax.semilogy(0.5 * (edges[:-1] + edges[1:]), prof, style, label=lbl)
    ax.axhline(TARGET_KT_KEV, color="0.5", lw=0.8)
    ax.text(0.05, TARGET_KT_KEV * 1.1, "observed bulk $kT_e$", fontsize=8, color="0.4")
    ax.set(xlabel="r [pc]", ylabel="temperature [keV]", ylim=(1e-2, 1e2),
           title="emission-measure-weighted temperatures")
    ax.legend(fontsize=9)

    out = FIGURES_DIR / f"{args.out}.png"
    fig.savefig(out, dpi=150)
    print(f"\n[plasma] saved {out}")


if __name__ == "__main__":
    main()
