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
from astropy import units as u

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# the showcase code units: pc / Msun / 1000 km s^-1
CODE_LENGTH = (1.0 * u.pc).to(u.cm).value
CODE_MASS = (1.0 * u.Msun).to(u.g).value
CODE_VELOCITY = (1000.0 * u.km / u.s).to(u.cm / u.s).value
CODE_TIME = CODE_LENGTH / CODE_VELOCITY
CODE_DENSITY = CODE_MASS / CODE_LENGTH ** 3
CODE_PRESSURE = CODE_DENSITY * CODE_VELOCITY ** 2

MU_PARTICLE = 0.61          # mean mass per particle, ionized cosmic composition
MU_ELECTRON = 1.18          # mean mass per electron, in m_p
MASS_PER_NUCLEUS = 1.4

# observational targets (Hwang & Laming 2003, 2012)
TARGET_KT_KEV = 2.0
TARGET_NET_BULK = 1.0e11    # cm^-3 s, bulk shocked plasma
TARGET_NET_FE = 1.0e12      # cm^-3 s, shocked Fe


def ionization_age(density_time_code):
    """``n_e t`` in cm^-3 s from the accumulated ``integral of rho dt`` (code units).

    The solver accumulates ``rho dt`` in code units while a parcel is shocked;
    converting to an electron column is one exact factor,
    ``n_e = rho / (mu_e m_p)``.
    """
    return (density_time_code * CODE_DENSITY * CODE_TIME
            / (MU_ELECTRON * const.m_p.cgs.value))


def coulomb_equilibration_time(T_e, T_i, n_e, coulomb_log=31.0):
    """Spitzer electron-ion equipartition time (s), for hydrogenic ions.

    ``t_eq = 3 m_e m_p / (8 sqrt(2 pi) n_e e^4 lnLambda) * (kT_e/m_e + kT_i/m_p)^{3/2}``

    Written out from the constants rather than quoted from a fitting formula, so
    the units are checkable rather than remembered; :func:`_self_check` prints it
    at benchmark points. It gives 8.2e3 yr at ``T = 1e7 K``, ``n_e = 1 cm^-3``,
    and ~3e3 yr at Cas A's post-shock conditions — an order of magnitude longer
    than the remnant's age.
    """
    m_e = const.m_e.cgs.value
    m_p = const.m_p.cgs.value
    e = const.e.esu.value
    k = const.k_B.cgs.value
    num = 3.0 * m_e * m_p
    den = 8.0 * np.sqrt(2.0 * np.pi) * np.maximum(n_e, 1e-30) * e ** 4 * coulomb_log
    return (num / den) * (k * T_e / m_e + k * T_i / m_p) ** 1.5


def electron_ion_temperatures(temperature, density, time_since_shock_code,
                              kT_e_shock_keV=0.3, n_substeps=8):
    """Split the single-fluid temperature into ``(T_e, T_i)``.

    The single-fluid solver evolves one temperature, which physically is the
    mass-weighted mean: pressure balance requires
    ``n_e T_e + n_i T_i = (n_e + n_i) T``. So ``T`` fixes one combination, and
    the Coulomb relaxation started from the Ghavamian et al. (2007) initial
    condition fixes the other:

    * at the shock the electrons receive ``kT_e ~ 0.3 keV`` regardless of Mach
      number (valid above ~1000 km/s), the ions the rest;
    * downstream the two relax towards each other on
      :func:`coulomb_equilibration_time`, integrated over the parcel's own time
      since shocking.

    The relaxation is integrated with a few explicit substeps in log time rather
    than analytically, because ``t_eq`` depends on ``T_e`` itself.

    Args:
        temperature: The single-fluid temperature (K).
        density: The density (code units).
        time_since_shock_code: Time since shocking (code units).
        kT_e_shock_keV: Post-shock electron temperature.
        n_substeps: Substeps for the relaxation integral.

    Returns:
        ``(T_e, T_i)`` in K. Unshocked parcels (zero time since shock) are
        returned with ``T_e = T_i = T``: they were never shocked, so there is no
        two-temperature state to describe.
    """
    n_H = density * CODE_DENSITY / (MASS_PER_NUCLEUS * const.m_p.cgs.value)
    n_e = 1.2 * n_H
    n_i = n_H                        # one ion per nucleus, near enough
    dt_total = time_since_shock_code * CODE_TIME

    keV = (1.0 * u.keV / const.k_B).to(u.K).value
    T_e = np.full_like(temperature, kT_e_shock_keV * keV)
    # the ions carry the rest of the thermal energy at the shock
    T_i = np.maximum(((n_e + n_i) * temperature - n_e * T_e) / np.maximum(n_i, 1e-30),
                     T_e)

    shocked = dt_total > 0.0
    dt = dt_total / n_substeps
    for _ in range(n_substeps):
        t_eq = coulomb_equilibration_time(T_e, T_i, n_e)
        # relax towards the common mean, conserving the total thermal energy
        T_mean = (n_e * T_e + n_i * T_i) / (n_e + n_i)
        f = np.where(shocked, 1.0 - np.exp(-np.clip(dt / t_eq, 0.0, 50.0)), 0.0)
        T_e = T_e + f * (T_mean - T_e)
        T_i = T_i + f * (T_mean - T_i)

    T_e = np.where(shocked, T_e, temperature)
    T_i = np.where(shocked, T_i, temperature)
    return T_e, T_i


def _self_check():
    """Print the equipartition time at a benchmark point, so the units are visible."""
    keV = (1.0 * u.keV / const.k_B).to(u.K).value
    for T, n in ((1e7, 1.0), (1e7, 10.0), (2.0 * keV, 10.0)):
        t = coulomb_equilibration_time(T, T, n)
        print(f"    t_eq(T = {T:.2e} K, n_e = {n:g}) = {t:.3e} s "
              f"= {t / 3.156e7:.3e} yr")
    print("    At Cas A's post-shock density (n_e ~ 10) and kT_e ~ 2 keV this is "
          "~3e3 yr,\n    an order of magnitude longer than the remnant's 350 yr "
          "age: electron-ion\n    equilibration is far from complete, which is "
          "why T_e must be modelled\n    rather than assumed equal to the "
          "single-fluid temperature.")


def emission_measure_distribution(state, T_e, net, *, weight=None, bins=80):
    """``EM`` binned in ``(kT_e, n_e t)`` -- the plane Hwang & Laming fit."""
    keV = (1.0 * u.keV / const.k_B).to(u.K).value
    kT = T_e / keV
    n_H = state["rho"] * CODE_DENSITY / (MASS_PER_NUCLEUS * const.m_p.cgs.value)
    em = 1.2 * n_H ** 2                       # n_e n_H, per unit volume
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
    ap.add_argument("--out", default="casa_plasma", help="figure name stem")
    args = ap.parse_args()

    print("[plasma] Coulomb equilibration time, unit check:")
    _self_check()

    d = np.load(args.state)
    if "density_time" not in d:
        raise SystemExit(
            f"{args.state} carries no shock history -- rerun casa_orlando.py "
            "with --composition")
    state = {k: np.asarray(d[k], dtype=np.float64) for k in
             ("rho", "press", "density_time", "time_since_shock")}
    # The shocked FRACTION, not "has any accumulated time": advection leaks an
    # infinitesimal amount of the record into every cell, so a > 0 test would
    # call the whole box shocked (this is exactly the trap the solver's own
    # latch had to be rewritten to avoid).
    state["shocked_fraction"] = (np.asarray(d["shocked_fraction"], dtype=np.float64)
                                 if "shocked_fraction" in d else
                                 (state["time_since_shock"] > 0).astype(np.float64))
    for k in ("C_ej", "C_Fe", "C_Si", "C_O"):
        if k in d:
            state[k] = np.asarray(d[k], dtype=np.float64)
    # refuse a blown-up state: analysing one produces confident nonsense (a
    # previous aborted run reported an ejecta mass of 2.5e13 Msun)
    bad = []
    if not np.all(np.isfinite(state["rho"])):
        bad.append(f"{int(np.sum(~np.isfinite(state['rho'])))} non-finite density cells")
    if float(state["rho"].max()) > 1e6:
        bad.append(f"max density {float(state['rho'].max()):.3e}")
    if not np.all(np.isfinite(state["press"])) or float(state["press"].min()) < 0:
        bad.append("non-finite or negative pressure")
    # an aborted run can leave the density looking innocuous while the ENERGY has
    # collapsed; a total ejecta mass near zero or a wildly super-unity T_e/T are
    # the tells (one aborted run reported 2.5e13 Msun, another 0.000)
    if "C_ej" in d:
        m_ej = float(np.sum(np.asarray(d["C_ej"]) * state["rho"])
                     * (float(d["box"]) / int(d["num_cells"])) ** 3)
        if not (0.1 < m_ej < 100.0):
            bad.append(f"total ejecta mass {m_ej:.3e} Msun")
    if bad:
        raise SystemExit(
            f"{args.state} looks unphysical ({'; '.join(bad)}) -- the run "
            "probably aborted. Check its log for an ABORT before analysing.")

    age = float(d["age"])
    box = float(d["box"])
    n = int(d["num_cells"])

    T = (MU_PARTICLE * const.m_p.cgs.value / const.k_B.cgs.value) * (
        state["press"] * CODE_PRESSURE) / (state["rho"] * CODE_DENSITY)
    net = ionization_age(state["density_time"])
    T_e, T_i = electron_ion_temperatures(T, state["rho"], state["time_since_shock"])
    keV = (1.0 * u.keV / const.k_B).to(u.K).value

    shocked = state["shocked_fraction"] > 0.5
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
        for el, target in (("Fe", 0.14), ("Si", 0.08)):
            key = f"C_{el}"
            if key in state:
                m_el = float(np.sum(mass * state[key] * state["C_ej"] * shocked))
                print(f"    shocked {el}: {m_el:.4f} Msun   [observed {target}]")

    if np.any(shocked):
        em_w = 1.2 * (state["rho"] * CODE_DENSITY
                      / (MASS_PER_NUCLEUS * const.m_p.cgs.value)) ** 2 * shocked
        kt_med = float(np.average((T_e / keV)[shocked], weights=em_w[shocked]))
        net_med = float(np.average(net[shocked], weights=em_w[shocked]))
        print(f"    EM-weighted kT_e = {kt_med:.2f} keV   [observed ~{TARGET_KT_KEV}]")
        print(f"    EM-weighted n_e t = {net_med:.2e} cm^-3 s   "
              f"[observed ~{TARGET_NET_BULK:.0e}]")
        print(f"    EM-weighted T_e/T = "
              f"{float(np.average((T_e / T)[shocked], weights=em_w[shocked])):.3f} "
              "(1 = full equilibration)")

    # ---- figure -----------------------------------------------------------
    H, kt_bins, net_bins = emission_measure_distribution(state, T_e, net)
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
    w = (1.2 * (state["rho"] * CODE_DENSITY
                / (MASS_PER_NUCLEUS * const.m_p.cgs.value)) ** 2).ravel()
    for field, lbl, style in ((T_e, "$T_e$", "-"), (T_i, "$T_i$", "--"),
                              (T, "single-fluid $T$", ":")):
        num = np.bincount(idx, weights=(field.ravel() * w), minlength=len(edges) - 1)
        den = np.bincount(idx, weights=w, minlength=len(edges) - 1)
        prof = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan) / keV
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
