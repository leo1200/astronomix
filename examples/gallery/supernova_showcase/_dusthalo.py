"""The dust-scattering halo: what the SIGHTLINE adds to the image.

Everything else in this showcase models the remnant. This module models the
3.4 kpc of Galactic dust between it and Chandra, which takes a fraction of every
photon and puts it back on the detector arcminutes away from where it started.
It is not a correction to the hydrodynamics and it cannot be: it is a property
of the column, not of the explosion.

The measurement that motivates it. Binned onto the same sky grid, 8.8 % of
Chandra's r < 260" counts sit OUTSIDE the forward shock, where the simulation
puts 0.1 % -- and that 0.1 % is the instrumental and Galactic background SOXS
adds, i.e. the model contributes nothing there at all. Inside 140" the two agree
to 10-20 %. The deficit is entirely an *outside* problem, and its radial shape
(declining as roughly theta^-1.5 to theta^-2 past 160") is the shape of a
scattering halo.

**A halo REDISTRIBUTES photons, it does not create them.** Nothing here adds
flux. Photons are moved, and the ones moved off the detector are gone -- which
is the whole point for the count-rate comparison, and why the sign of that
correction cannot be guessed (see the note in ``scatter_sky_positions``).

The physics, and where each piece comes from
--------------------------------------------
* **Cross-sections: computed, not fitted.** ``newdust`` (Corrales et al. 2016;
  installed as ``xdust``) runs Mie theory on an MRN grain population -- silicate
  plus graphite, ``dn/da ~ a^-3.5`` between 0.005 and 0.25 micron, with MRN's own
  normalisation constants. Nothing in the halo is tuned to our residual.
* **The dust column follows N_H.** MRN's normalisation implies 1.564e-26 g of
  dust per hydrogen (a dust-to-H mass ratio of 0.0093), so the same ``--nh``
  that sets the photoelectric absorption sets the scattering column. There is no
  free normalisation.
* **That is checkable, and it checks out.** The model gives
  ``tau_sca(1 keV) = 0.59 (N_H / 1e22)``; Predehl & Schmitt (1995), from 25
  measured ROSAT halos, give a mean ``S ~ 0.5``. 19 % agreement, with nothing
  fitted, is as good as this quantity is known.
* **The angular profile is Mie too**, and it reproduces the published analytic
  approximation. Draine (2003), Eqs. 9 and 11, gives a median scattering angle
  ``theta_50 ~ 360" (keV/E)`` and a cumulative
  ``sigma(<theta)/sigma = s^2/(1+s^2)`` in ``s = theta/theta_50``. Our MRN
  population gives ``theta_50 ~ 490" (keV/E)`` -- a third wider, because MRN is
  not Draine's WD01 size distribution -- and a cumulative shape that matches
  Eq. 11 to a few percent at every ``s``. ``--halo-profile draine03`` swaps in
  the analytic form so the size-distribution systematic can be measured rather
  than argued about.

Two things are worth stating plainly with any figure.

* **``tau_sca`` is not ``E^-2`` where it matters.** It is asymptotically, and it
  is to better than 5 % above 4 keV, but through the 0.5-2 keV band the local
  index wanders between -1.4 and -2.2 and is not even monotonic, because grains
  stop being small compared with the wavelength. The band dependence is still
  steep -- ``tau`` falls 7.0x from 0.5 to 2 keV -- so the qualitative statement
  (scattering is a soft-band effect) survives, but ``E^-2`` should not be used
  to scale it.
* **The sightline distribution is the one real assumption.** The default is dust
  spread uniformly over the 3.4 kpc, which is the assumption-free choice and
  needs no parameter. Cas A actually sits just beyond the Perseus arm, so a
  single screen at ~2 kpc (``--halo-screen 0.41``) is the physically motivated
  alternative; running both is how its size is measured.

Multiple scattering is included, and is not optional at these depths: at
``N_H = 1.2e22`` the 1 keV depth is 0.71, so a quarter of the scattered photons
scatter twice. The number of scatterings is drawn from ``Poisson(tau_sca(E))``
and the small-angle deflections add as vectors, each with its own position along
the sightline. Single-scattering would be a 25 % error at 1 keV and a factor of
two at 0.5.

Needs ``newdust``/``xdust`` only to BUILD the table
(``pip install git+https://github.com/eblur/newdust.git``); the table is cached
and the forward model reads it without the dust package.
"""

# general
import os
from pathlib import Path

# numerics
import numpy as np

#: where the cached scattering table lives (next to the other big data)
TABLE_DIR = Path(os.environ.get(
    "CASA_DUST_TABLE_DIR", "/export/data/lstorcks/supernova_showcase"))

# =============================================================================
# ============ ↓ The grain population ↓ =======================================
# =============================================================================
#: MRN (Mathis, Rumpl & Nordsieck 1977) grain size limits, micron.
MRN_AMIN, MRN_AMAX = 0.005, 0.25

#: MRN power-law index of ``dn/da``.
MRN_P = 3.5

#: MRN's own normalisation constants ``A`` in ``dn/da = A n_H a^-3.5``,
#: [cm^2.5 per H], paired with the bulk grain density [g cm^-3]. These fix the
#: dust mass per hydrogen, which is what makes the scattering column follow
#: ``--nh`` with nothing left to tune.
MRN_NORM = ((10 ** -25.13, 2.2),      # graphite
            (10 ** -25.11, 3.5))      # silicate

#: fraction of the dust mass in silicate, for ``newdust``'s MRN builder. The
#: value implied by MRN_NORM above (silicate mass / total) is 0.625; 0.6 is
#: newdust's default and the difference is far below the model uncertainty.
MRN_FSIL = 0.6


def dust_mass_per_hydrogen(amin=MRN_AMIN, amax=MRN_AMAX):
    """Dust mass column per hydrogen atom [g], from MRN's own normalisation.

    ``int (4/3) pi rho a^3 * A a^-3.5 da`` over the two grain materials. The
    result, 1.564e-26 g/H = 0.0093 by mass, is the standard interstellar
    dust-to-gas ratio -- which is the point: it is not a knob.
    """
    total = 0.0
    for norm, rho in MRN_NORM:
        a0, a1 = amin * 1e-4, amax * 1e-4          # micron -> cm
        total += (4.0 / 3.0) * np.pi * rho * norm * 2.0 * (np.sqrt(a1) - np.sqrt(a0))
    return total


# =============================================================================
# ============ ↑ The grain population ↑ =======================================
# =============================================================================

# =============================================================================
# ============ ↓ Draine (2003) analytic profile ↓ =============================
# =============================================================================
#: Draine (2003) Eq. 9: median scattering angle of the WD01 R_V = 3.1 mixture,
#: in arcsec at 1 keV, scaling as 1/E. Used by ``--halo-profile draine03`` and,
#: more usefully, as the variable the tabulated Mie profile is stored against --
#: expressing the Mie inverse-CDF as a slowly varying correction to this keeps
#: the heavy tail accurate without a huge table.
DRAINE_THETA50_1KEV = 360.0


def draine_theta50(energy_keV):
    """Median scattering angle [arcsec], Draine (2003) Eq. 9."""
    return DRAINE_THETA50_1KEV / np.asarray(energy_keV, dtype=np.float64)


def draine_inverse_cdf(u):
    """Invert Draine (2003) Eq. 11, ``F = s^2/(1+s^2)``, for ``s = theta/theta_50``."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0 - 1e-12)
    return np.sqrt(u / (1.0 - u))


# =============================================================================
# ============ ↑ Draine (2003) analytic profile ↑ =============================
# =============================================================================

# =============================================================================
# ============ ↓ Building the Mie table ↓ =====================================
# =============================================================================
#: Energy grid of the cached table [keV]. Spans the pyXSIM source model's range
#: with ~7 % spacing, which is far finer than the profile varies.
ENERGY_GRID = np.logspace(np.log10(0.3), np.log10(12.0), 40)

#: Scattering-angle grid the differential cross-section is integrated on
#: [arcsec]. The top end is 8 degrees: at 1 keV that is 74 theta_50, enclosing
#: 99.98 % of the scattered flux, and everything past it is off any detector.
THETA_GRID = np.logspace(np.log10(0.3), np.log10(2.88e4), 384)

#: The inverse CDF is tabulated against ``s = theta / theta_50(Draine)`` rather
#: than against the uniform deviate, because ``theta(u)`` diverges at ``u -> 1``
#: while ``theta(s)`` does not: the stored quantity is an O(1) ratio even deep
#: in the tail. 1e-3 to 1e3 covers the whole grid at every energy.
S_GRID = np.logspace(-3.0, 3.0, 256)

#: grain-size grid points for the Mie calculation
NA_MIE = 48


def table_path(amax=MRN_AMAX):
    """Cache file for one grain population.

    Not keyed on the column: the dust is optically thin per scattering, so
    ``tau_sca`` is strictly linear in N_H and the ANGULAR distribution does not
    depend on it at all. The table is therefore built once, at N_H = 1e22, and
    the depth scaled at load time -- which matters, because the Mie calculation
    takes about an hour.
    """
    return TABLE_DIR / f"dust_halo_mrn_amax{amax:.3f}.npz"


#: the reference column the table is tabulated at [1e22 cm^-2]
TABLE_NH = 1.0


def _mie_chunk(job):
    """One slab of the energy grid. Module-level so it can be pickled to a Pool.

    Mie theory is independent per (energy, grain size), so the only thing
    splitting the energy grid costs is a few duplicated grain populations --
    and it turns an hour of wall clock into a couple of minutes.
    """
    from xdust import grainpop            # newdust, installed as ``xdust``
    import astropy.units as u_

    energies, amax, md = job
    gp = grainpop.make_MRN(amin=MRN_AMIN, amax=amax, p=MRN_P, md=md,
                           fsil=MRN_FSIL, na=NA_MIE)
    gp.calculate_ext(energies * u_.keV, theta=THETA_GRID * u_.arcsec)
    # ``int_diff`` is d tau_sca / d Omega per grain population; summing the
    # silicate and the two graphite orientations gives the mixture, and
    # integrating it over solid angle must return tau_sca (checked by the
    # caller).
    dtau = sum(np.asarray(s.int_diff.to("arcsec^-2").value, dtype=np.float64)
               for s in gp.gpoplist)
    return np.asarray(gp.tau_sca, dtype=np.float64), dtau


def build_table(amax=MRN_AMAX, path=None, verbose=True, processes=None):
    """Mie ``tau_sca(E)`` and the scattering-angle inverse CDF, cached to npz.

    Stores ``theta_of_s[E, s]``: the scattering angle whose cumulative
    cross-section equals that of ``s`` under Draine's Eq. 11. Sampling is then
    ``u ~ U(0,1) -> s = sqrt(u/(1-u)) -> theta``, a bilinear interpolation.
    """
    import multiprocessing as mp

    from scipy.integrate import cumulative_trapezoid

    nh = TABLE_NH
    md = dust_mass_per_hydrogen(amax=amax) * nh * 1e22
    nproc = min(processes or ENERGY_GRID.size, ENERGY_GRID.size, mp.cpu_count())
    if verbose:
        print(f"[dust-halo] MRN {MRN_AMIN}-{amax} um, N_H = {nh:.2f}e22 cm^-2 "
              f"-> dust column {md:.3e} g cm^-2")
        print(f"[dust-halo] running Mie on {ENERGY_GRID.size} energies x "
              f"{THETA_GRID.size} angles x {NA_MIE} sizes, on {nproc} processes")

    jobs = [(chunk, amax, md) for chunk in np.array_split(ENERGY_GRID, nproc)]
    with mp.Pool(nproc) as pool:
        out = pool.map(_mie_chunk, jobs)
    tau_sca = np.concatenate([o[0] for o in out])                       # NE
    dtau = np.concatenate([o[1] for o in out], axis=0)                  # NE x NTH

    enclosed = cumulative_trapezoid(dtau * 2.0 * np.pi * THETA_GRID[None, :],
                                    THETA_GRID, axis=1, initial=0.0)
    closure = enclosed[:, -1] / tau_sca
    if verbose:
        print(f"[dust-halo] solid-angle closure int(dtau/dOmega)dOmega / tau_sca: "
              f"{closure.min():.4f} to {closure.max():.4f} (1.0 is exact; the "
              f"shortfall is flux scattered past {THETA_GRID[-1] / 3600:.1f} deg)")
    cdf = enclosed / enclosed[:, -1:]

    # invert onto the s-grid, energy by energy
    theta_of_s = np.empty((ENERGY_GRID.size, S_GRID.size), dtype=np.float64)
    u_of_s = S_GRID ** 2 / (1.0 + S_GRID ** 2)
    for i in range(ENERGY_GRID.size):
        # cdf is monotone but can repeat at the ends; np.interp needs increasing
        c = np.maximum.accumulate(cdf[i])
        theta_of_s[i] = np.interp(u_of_s, c, THETA_GRID)

    if verbose:
        t50 = np.array([np.interp(0.5, np.maximum.accumulate(cdf[i]), THETA_GRID)
                        for i in range(ENERGY_GRID.size)])
        e1 = int(np.argmin(np.abs(ENERGY_GRID - 1.0)))
        print(f"[dust-halo] tau_sca(1 keV) = {tau_sca[e1]:.3f} "
              f"-> S = tau/(N_H/1e22) = {tau_sca[e1] / nh:.3f} "
              f"(Predehl & Schmitt 1995 measured mean S ~ 0.5)")
        print(f"[dust-halo] median scattering angle {t50[e1] * ENERGY_GRID[e1]:.0f}\" "
              f"(keV/E) vs Draine (2003) Eq. 9 {DRAINE_THETA50_1KEV:.0f}\" (keV/E)")

    path = Path(path or table_path(amax))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, energy=ENERGY_GRID, s=S_GRID,
                        tau_per_nh22=tau_sca / nh, theta_of_s=theta_of_s,
                        amax=amax,
                        theta50=np.array([np.interp(0.5, np.maximum.accumulate(cdf[i]),
                                                    THETA_GRID)
                                          for i in range(ENERGY_GRID.size)]))
    if verbose:
        print(f"[dust-halo] wrote {path}")
    return path


def load_table(nh, amax=MRN_AMAX, verbose=True):
    """Read the cached table and scale the depth to this column."""
    path = table_path(amax)
    if not path.exists():
        build_table(amax=amax, path=path, verbose=verbose)
    d = np.load(path)
    t = {k: d[k] for k in d.files}
    t["nh"] = nh
    t["tau_sca"] = t["tau_per_nh22"] * nh
    return t


# =============================================================================
# ============ ↑ Building the Mie table ↑ =====================================
# =============================================================================

# =============================================================================
# ============ ↓ Scattering the photons ↓ =====================================
# =============================================================================
def _sample_scattering_angle(energy, u, table, profile):
    """Scattering angles [arcsec] for photons of ``energy`` [keV], deviates ``u``."""
    s = draine_inverse_cdf(u)
    if profile == "draine03":
        return s * draine_theta50(energy)

    # bilinear interpolation of the tabulated inverse CDF in (log E, log s)
    le, ls = np.log(table["energy"]), np.log(table["s"])
    x = np.clip(np.interp(np.log(energy), le, np.arange(le.size)), 0, le.size - 1)
    y = np.clip(np.interp(np.log(s), ls, np.arange(ls.size)), 0, ls.size - 1)
    i0 = np.minimum(x.astype(np.intp), le.size - 2)
    j0 = np.minimum(y.astype(np.intp), ls.size - 2)
    fx, fy = x - i0, y - j0
    t = np.log(table["theta_of_s"])
    return np.exp((1 - fx) * ((1 - fy) * t[i0, j0] + fy * t[i0, j0 + 1])
                  + fx * ((1 - fy) * t[i0 + 1, j0] + fy * t[i0 + 1, j0 + 1]))


def scatter_sky_positions(ra, dec, energy, *, nh, profile="mie", screen_x=None,
                          amax=MRN_AMAX, seed=1234, verbose=True):
    """Move each photon to where interstellar dust actually put it.

    ``ra``/``dec`` in degrees, ``energy`` in keV, ``nh`` in 1e22 cm^-2. Returns
    ``(ra', dec', report)``; the photon COUNT is unchanged, by construction.

    The number of scatterings is ``Poisson(tau_sca(E))`` and the small-angle
    deflections add as vectors. Each one happens at its own place along the
    sightline: a photon deflected by ``theta_s`` at a fractional distance
    ``x = 1 - d/D`` from the observer appears displaced by ``x theta_s``, so
    dust at the observer deflects fully and dust at the remnant not at all.
    ``screen_x=None`` spreads the dust uniformly (``x ~ U(0,1]``); a float puts
    it all in one screen there.

    On the count rate, which this settles rather than assumes. Cas A is
    EXTENDED, so most scattered photons land back on the remnant and the halo is
    not separable from the source. What leaves an r < 200" aperture is only the
    tail, and the real observation loses it exactly as this model now does --
    so the model and the data are finally being measured through the same
    aperture. The report returns the fraction that leaves any given radius; do
    not quote a correction without it.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    rng = np.random.default_rng(seed)

    table = load_table(nh, amax=amax, verbose=verbose)
    # log-log, because tau_sca is a steep power law and linear interpolation
    # across a decade of it is a percent-level error for free
    tau = np.exp(np.interp(np.log(energy), np.log(table["energy"]),
                           np.log(table["tau_sca"])))

    n_scat = rng.poisson(tau)
    # accumulate the deflection in the tangent plane, in arcsec
    dxi = np.zeros(ra.size)
    deta = np.zeros(ra.size)
    for k in range(1, int(n_scat.max()) + 1 if n_scat.size else 1):
        m = n_scat >= k
        if not m.any():
            break
        n = int(m.sum())
        theta_s = _sample_scattering_angle(energy[m], rng.random(n), table, profile)
        x = np.full(n, screen_x) if screen_x is not None else rng.random(n)
        seen = x * theta_s                       # what this deflection looks like
        az = rng.uniform(0.0, 2.0 * np.pi, n)
        dxi[m] += seen * np.cos(az)
        deta[m] += seen * np.sin(az)

    alpha = np.hypot(dxi, deta) / 3600.0                    # degrees
    phi = np.arctan2(dxi, deta)                             # position angle from N
    # exact offset on the sphere, so the arcminute-to-degree tail is not skewed
    a, p = np.deg2rad(alpha), phi
    d0, r0 = np.deg2rad(dec), np.deg2rad(ra)
    sd = np.sin(d0) * np.cos(a) + np.cos(d0) * np.sin(a) * np.cos(p)
    dec_out = np.rad2deg(np.arcsin(np.clip(sd, -1.0, 1.0)))
    ra_out = np.rad2deg(r0 + np.arctan2(np.sin(p) * np.sin(a) * np.cos(d0),
                                        np.cos(a) - np.sin(d0) * sd)) % 360.0

    scattered = n_scat > 0
    report = {
        "profile": profile,
        "screen_x": screen_x,
        "n_photons": int(ra.size),
        "frac_scattered": float(scattered.mean()),
        "mean_scatterings": float(n_scat.mean()),
        "max_scatterings": int(n_scat.max()) if n_scat.size else 0,
        "median_offset_arcsec": float(np.median(alpha[scattered]) * 3600.0)
                                if scattered.any() else 0.0,
        "tau_1keV": float(np.interp(1.0, table["energy"], table["tau_sca"])),
    }
    if verbose:
        print(f"[dust-halo] tau_sca(1 keV) = {report['tau_1keV']:.3f}; "
              f"{100 * report['frac_scattered']:.1f}% of photons scatter at least "
              f"once (mean {report['mean_scatterings']:.2f}, max "
              f"{report['max_scatterings']})")
        print(f"[dust-halo] median displacement of a scattered photon: "
              f"{report['median_offset_arcsec']:.0f}\"")
    return ra_out, dec_out, report


# =============================================================================
# ============ ↑ Scattering the photons ↑ =====================================
# =============================================================================


def selftest(nh=1.2, amax=MRN_AMAX, n=400_000, seed=7):
    """Check the sampler against the table, and predict what the halo will do.

    Three things are worth verifying before the halo is let anywhere near a
    science figure: that the drawn angles really follow the tabulated
    cross-section, that the count is conserved, and how much of a point source's
    flux the sightline moves past a given radius -- the last is the whole
    measurement, and it can be read off here without running pyXSIM.
    """
    table = load_table(nh, amax=amax)
    rng = np.random.default_rng(seed)

    # Test on grid energies, so a disagreement is the SAMPLER and not the
    # interpolation between two tabulated energies.
    print("\n--- sampled angles vs the tabulated cross-section ---")
    print("  E [keV]     quantile      sampled       table       ratio")
    for i in np.searchsorted(table["energy"], (0.7, 1.0, 2.0, 4.0)):
        e = float(table["energy"][i])
        th = _sample_scattering_angle(np.full(n, e), rng.random(n), table, "mie")
        for q in (0.1, 0.5, 0.9):
            want = np.interp(np.sqrt(q / (1 - q)), table["s"], table["theta_of_s"][i])
            got = np.quantile(th, q)
            print(f"  {e:7.2f}   {q:10.2f}   {got:10.1f}\"  {want:10.1f}\"   "
                  f"{got / want:8.3f}")

    print("\n--- what the sightline does to a POINT source at the centre ---")
    print("  (an upper bound on the effect: Cas A is 150\" across, so much of "
          "this\n   lands back on the remnant rather than outside it)")
    energies = {"0.5-1.5 keV": 1.0, "1.5-3 keV": 2.1, "4-6 keV": 4.9}
    for geom, x in (("uniform", None), ("Perseus screen x=0.41", 0.41)):
        print(f"  {geom}:")
        for lbl, e in energies.items():
            ra, dec, rep = scatter_sky_positions(
                np.zeros(n), np.full(n, 60.0), np.full(n, e), nh=nh,
                screen_x=x, seed=seed, verbose=False)
            off = np.rad2deg(np.arccos(np.clip(
                np.sin(np.deg2rad(60.0)) * np.sin(np.deg2rad(dec))
                + np.cos(np.deg2rad(60.0)) * np.cos(np.deg2rad(dec))
                * np.cos(np.deg2rad(ra)), -1, 1))) * 3600.0
            tau = np.exp(np.interp(np.log(e), np.log(table["energy"]),
                                   np.log(table["tau_sca"])))
            print(f"    {lbl:12s} tau={tau:5.2f} "
                  f"scattered {100 * rep['frac_scattered']:5.1f}%  "
                  f"| beyond 140\" {100 * (off > 140).mean():5.2f}%  "
                  f"200\" {100 * (off > 200).mean():5.2f}%  "
                  f"260\" {100 * (off > 260).mean():5.2f}%")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="build/inspect the dust-scattering table")
    ap.add_argument("--nh", type=float, default=1.2, help="N_H / 1e22 cm^-2")
    ap.add_argument("--amax", type=float, default=MRN_AMAX,
                    help="max grain radius (micron)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if cached")
    ap.add_argument("--selftest", action="store_true",
                    help="verify the sampler and predict the halo's reach")
    a = ap.parse_args()

    if a.rebuild or not table_path(a.amax).exists():
        build_table(amax=a.amax)
    t = load_table(a.nh, amax=a.amax)

    print("\n  E [keV]   tau_sca   theta_50 [\"]   E x theta_50   Draine03 Eq.9")
    for e, tau, t50 in zip(t["energy"], t["tau_sca"], t["theta50"]):
        if 0.4 < e < 9.0:
            print(f"  {e:7.2f}   {tau:7.4f}   {t50:11.1f}   {e * t50:12.0f}   "
                  f"{DRAINE_THETA50_1KEV:13.0f}")

    if a.selftest:
        selftest(a.nh, a.amax)
