"""The plasma state behind the hydrodynamics: composition, ionization, T_e.

The solver evolves density, momentum and total energy plus a set of passive
scalars. Everything an X-ray observation actually responds to -- the electron
temperature, the electron density, the ionization age -- has to be reconstructed
from those, and every one of them depends on the COMPOSITION, which in a
supernova remnant is nothing like cosmic:

* the **mean molecular weight** ``mu`` converts pressure into temperature. In
  fully ionized cosmic gas ``mu = 0.61``; in the fully ionized oxygen layer that
  carries most of Cas A's ejecta mass it is ``16/9 = 1.78``. Using the cosmic
  value there under-predicts the temperature of the brightest ejecta by a factor
  of nearly three, and the temperature is what sets the spectrum.
* the **electrons per unit mass** ``1/mu_e`` set the emission measure and the
  ionization age. Cosmic gas gives ``mu_e = 1.18``; metal ejecta gives ``~2``,
  so the same mass of ejecta carries 1.7x FEWER electrons than the cosmic
  conversion assumes -- directly relevant, since the model's emission-weighted
  ``n_e t`` sits above Hwang & Laming's measurement.
* the **Coulomb equilibration rate** depends on ``sum n_i Z_i^2 / m_i``, which
  is not the same function of density in an oxygen plasma as in a hydrogen one.

This module is deliberately free of JAX and of any solver import, so both the
``astx`` environment (``casa_plasma.py``) and the CPU-only ``xrayobs``
environment (``casa_observe.py``) can use exactly the same physics -- the two
used to carry separate, and inconsistent, copies of it.

The one assumption everything here shares is FULL IONIZATION. At Cas A's
ionization ages that is good for H, He and O (``n_e t ~ 1e11`` leaves oxygen
He- and H-like) and optimistic for Fe, whose mean charge is nearer 17 than 26 at
``n_e t ~ 1e12``. The effect on ``mu_e`` is at the 10-20 % level in Fe-rich gas
only, and it goes the same way as the NEI correction to the spectrum, which is
the larger of the two. :func:`mean_charge_fully_ionized` is the hook to replace
when an ionization balance is available.
"""

# numerics
import numpy as np

# units and constants
from astropy import constants as const
from astropy import units as u

# =============================================================================
# ============ ↓ Code units and atomic data ↓ =================================
# =============================================================================
#: the showcase code units: pc / Msun / 1000 km s^-1
CODE_LENGTH = (1.0 * u.pc).to(u.cm).value
CODE_MASS = (1.0 * u.Msun).to(u.g).value
CODE_VELOCITY = (1000.0 * u.km / u.s).to(u.cm / u.s).value
CODE_TIME = CODE_LENGTH / CODE_VELOCITY
CODE_DENSITY = CODE_MASS / CODE_LENGTH ** 3
CODE_PRESSURE = CODE_DENSITY * CODE_VELOCITY ** 2

#: element -> (mass number A, nuclear charge Z), for every element the tracers
#: stand for. The four species ``casa_orlando.py --composition`` carries are
#: TRACKED_SPECIES; hydrogen is the remainder.
ATOMIC = {"H": (1.008, 1), "He": (4.003, 2), "C": (12.011, 6), "N": (14.007, 7),
          "O": (15.999, 8), "Ne": (20.180, 10), "Mg": (24.305, 12),
          "Si": (28.085, 14), "S": (32.06, 16), "Ar": (39.948, 18),
          "Ca": (40.078, 20), "Fe": (55.845, 26), "Ni": (58.693, 28)}

#: species carried as passive scalars (hydrogen is 1 - sum of these)
TRACKED_SPECIES = ("Fe", "Si", "O", "He")

#: Solar abundances by NUMBER relative to hydrogen (Anders & Grevesse, the
#: "angr" table SOXS/pyXSIM default to). An abundance "in solar units" is
#: ``(n_el/n_H) / (n_el/n_H)_sun``, so these convert simulated number densities
#: into what an APEC-based model wants.
SOLAR_NUMBER_RATIO_TO_H = {
    "He": 9.77e-2, "C": 3.63e-4, "N": 1.12e-4, "O": 8.51e-4, "Ne": 1.23e-4,
    "Mg": 3.80e-5, "Si": 3.55e-5, "S": 1.62e-5, "Ar": 3.63e-6, "Ca": 2.29e-6,
    "Fe": 4.68e-5, "Ni": 1.78e-6,
}

#: How each carried tracer divides among the elements it actually stands for.
#:
#: The run carries four species because passive scalars cost ~1 % a step each,
#: but the layers they label are nucleosynthetic layers, not single elements:
#: ``_common.IIB_LAYERS`` defines the "Si" layer as Si+S and the "O" layer as
#: O/Ne/Mg. Treating them as pure Si and pure O puts the S, Ne and Mg emission
#: into the wrong lines -- and S is a major Cas A emitter, which is why Hwang &
#: Laming quote Si and S together.
#:
#: The split is by MASS, from Hwang & Laming (2012)'s measured shocked masses
#: (O 2, Ne 0.03, Mg 0.03; Si 0.08, S 0.06, Ar 0.02, Ca 0.02) -- from the
#: observation rather than a yield model, since being comparable to that
#: observation is the point of this pipeline. A fixed ratio within each layer is
#: exactly as much as co-located tracers can say; resolving it would need one
#: scalar per element.
TRACER_SPLIT = {
    "O": {"O": 0.971, "Ne": 0.0145, "Mg": 0.0145},
    "Si": {"Si": 0.444, "S": 0.333, "Ar": 0.111, "Ca": 0.111},
    "Fe": {"Fe": 1.0},
    "He": {"He": 1.0},
}


def element_mass_fractions(X):
    """Expand the tracer mass fractions into per-ELEMENT mass fractions."""
    out = {}
    for tracer, parts in TRACER_SPLIT.items():
        if tracer not in X:
            continue
        for el, frac in parts.items():
            out[el] = out.get(el, 0.0) + X[tracer] * frac
    if "H" in X:
        out["H"] = X["H"]
    return out

#: fallback composition for states written before the scalars existed
COSMIC = {"H": 0.70, "He": 0.28, "O": 0.01, "Si": 0.005, "Fe": 0.005}

M_P = const.m_p.cgs.value
M_E = const.m_e.cgs.value
K_B = const.k_B.cgs.value
E_ESU = const.e.esu.value
KEV_IN_K = (1.0 * u.keV / const.k_B).to(u.K).value
# =============================================================================
# ============ ↑ Code units and atomic data ↑ =================================
# =============================================================================


def mean_charge_fully_ionized(element):
    """Mean charge per nucleus. Full stripping -- the one place to change.

    Replacing this with a table in ``(kT_e, n_e t)`` is what turns every
    ionization-dependent quantity here non-equilibrium at once.
    """
    return ATOMIC[element][1]


def mass_fractions(fields, *, species=TRACKED_SPECIES, fallback=COSMIC):
    """Per-cell mass fractions including hydrogen, from the tracked scalars.

    ``fields`` is anything indexable by ``"C_<element>"`` (an ``npz``, a dict).
    Hydrogen is the remainder, which is how the layered composition was
    constructed: the scalars are normalised to sum to one WITH hydrogen, so
    ``X_H = 1 - sum`` is the H envelope plus the circumstellar hydrogen rather
    than a leftover.

    Returns ``(X, tracked)`` -- the dict of mass fractions and whether the state
    actually carried the scalars.
    """
    have = [s for s in species if f"C_{s}" in fields]
    if not have:
        return {el: np.float64(x) for el, x in fallback.items()}, False
    X = {el: np.asarray(fields[f"C_{el}"], dtype=np.float64) for el in have}
    # the advected scalars are bounded to [0, 1] individually but their SUM can
    # drift slightly above 1 in a mixing cell; clip the metals collectively so
    # hydrogen never goes negative
    total_metals = sum(X.values())
    over = total_metals > 1.0
    if np.any(over):
        scale = np.where(over, 1.0 / np.maximum(total_metals, 1e-30), 1.0)
        X = {el: x * scale for el, x in X.items()}
        total_metals = np.minimum(total_metals, 1.0)
    X["H"] = 1.0 - total_metals
    return X, True


def composition_moments(X):
    """The four sums every ionized-plasma quantity is built from.

    Returns a dict with

    * ``mu``    -- mass per free particle (electrons + ions), in ``m_p``;
      ``p = rho k T / (mu m_p)``;
    * ``mu_e``  -- mass per electron, ``n_e = rho / (mu_e m_p)``;
    * ``mu_i``  -- mass per ion (= per nucleus), ``n_i = rho / (mu_i m_p)``;
    * ``z2_a2`` -- ``sum_i X_i Z_i^2 / A_i^2``, which is ``sum_i n_i Z_i^2 / m_i``
      per unit mass density and is what the Coulomb equilibration rate scales
      with. It is 1 for hydrogen and 0.25 for oxygen: heavy ions couple to the
      electrons four times more weakly per gram.
    """
    inv_mu_e = inv_mu_i = z2_a2 = 0.0
    for el, x in X.items():
        A, _ = ATOMIC[el]
        Z = mean_charge_fully_ionized(el)
        inv_mu_e = inv_mu_e + x * Z / A
        inv_mu_i = inv_mu_i + x / A
        z2_a2 = z2_a2 + x * Z ** 2 / A ** 2
    inv_mu = inv_mu_e + inv_mu_i
    tiny = 1e-30
    return dict(mu=1.0 / np.maximum(inv_mu, tiny),
                mu_e=1.0 / np.maximum(inv_mu_e, tiny),
                mu_i=1.0 / np.maximum(inv_mu_i, tiny),
                z2_a2=z2_a2)


def number_densities(rho_code, X):
    """``n_e``, ``n_i``, ``n_H`` and the nucleus density (cm^-3) from the composition."""
    m = composition_moments(X)
    rho = np.asarray(rho_code, dtype=np.float64) * CODE_DENSITY
    return dict(n_e=rho / (m["mu_e"] * M_P),
                n_i=rho / (m["mu_i"] * M_P),
                n_H=rho * X["H"] / (ATOMIC["H"][0] * M_P),
                moments=m)


def temperature(rho_code, press_code, X):
    """Single-fluid temperature (K) with the LOCAL mean molecular weight.

    The hydrodynamics evolves the pressure, so this is a pure diagnostic
    conversion -- but it is not a small one. At fixed pressure and density, a
    plasma with fewer particles per gram is hotter, so the metal ejecta are
    ~3x hotter than the cosmic-``mu`` conversion says.
    """
    mu = composition_moments(X)["mu"]
    return (mu * M_P / K_B) * (np.asarray(press_code, dtype=np.float64) * CODE_PRESSURE) \
        / (np.asarray(rho_code, dtype=np.float64) * CODE_DENSITY)


def ionization_age(density_time_code, X):
    """``n_e t`` (cm^-3 s) from the accumulated ``integral of rho dt``.

    Exact for a Lagrangian parcel: the solver accumulates ``rho dt`` while the
    parcel is shocked and the composition rides along with it, so the electron
    column is that integral divided by the parcel's own ``mu_e m_p``. Note this
    is where the metal ejecta correction enters most directly -- the same
    ``rho dt`` is 1.7x fewer electrons in the oxygen layer than the cosmic
    conversion returns.
    """
    mu_e = composition_moments(X)["mu_e"]
    return (np.asarray(density_time_code, dtype=np.float64) * CODE_DENSITY * CODE_TIME
            / (mu_e * M_P))


def coulomb_logarithm(T_e, n_e):
    """Electron-ion Coulomb logarithm (NRL formulary, hot-electron branch).

    ``24 - ln(sqrt(n_e) / T_e[eV])``, valid for ``T_e > 10 Z^2 eV``. For iron
    that threshold is 6.8 keV, above Cas A's electron temperatures, so the
    Fe-dominated cells formally belong on the other branch; the two differ by
    about one unit in ~31, i.e. 3 % in the equilibration rate. Clipped to a sane
    range so the cold, unshocked interior cannot produce a negative logarithm.
    """
    T_ev = np.maximum(np.asarray(T_e, dtype=np.float64), 1.0) / 1.16045e4
    return np.clip(24.0 - np.log(np.sqrt(np.maximum(n_e, 1e-30)) / T_ev), 5.0, 40.0)


def equipartition_time(T_e, T_i, rho_code, X, coulomb_log=None):
    """Spitzer electron-ion equipartition time (s) for a multi-species plasma.

    Written out from the constants rather than quoted from a fitting formula, so
    the units are checkable. For a single ion species,

    ``t_eq = 3 m_e m_i / (8 sqrt(2 pi) n_i Z^2 e^4 lnL) * (kT_e/m_e + kT_i/m_i)^{3/2}``

    with ``dT_e/dt = (T_i - T_e) / t_eq``. It reproduces the NRL energy-exchange
    rate to 3 %. For a MIXTURE the rates add: ``1/t_eq = sum_i 1/t_eq,i``, and
    since ``T_e/m_e`` dominates the bracket by two orders of magnitude the ion
    mass drops out of it, leaving the composition dependence entirely in
    ``sum_i n_i Z_i^2/m_i = rho * z2_a2 / m_p^2``.
    """
    m = composition_moments(X)
    rho = np.asarray(rho_code, dtype=np.float64) * CODE_DENSITY
    n_e = rho / (m["mu_e"] * M_P)
    if coulomb_log is None:
        coulomb_log = coulomb_logarithm(T_e, n_e)
    sum_nz2_m = rho * m["z2_a2"] / M_P ** 2          # sum_i n_i Z_i^2 / m_i
    bracket = (K_B * T_e / M_E + K_B * T_i / (m["mu_i"] * M_P)) ** 1.5
    return (3.0 * M_E / (8.0 * np.sqrt(2.0 * np.pi) * E_ESU ** 4 * coulomb_log)) \
        * bracket / np.maximum(sum_nz2_m, 1e-60)


#: Electron-heating prescriptions at the collisionless shock. See
#: :func:`shock_electron_temperature` for what each one asserts and why the
#: default is the weakest link in the whole forward model.
TE_MODELS = ("ghavamian", "beta", "equilibrated", "minimal")


def shock_electron_temperature(T, f_e, *, model="ghavamian", kT_e_shock_keV=0.3,
                               beta_shock=0.05):
    """Electron temperature IMMEDIATELY behind the shock, before relaxation.

    This one choice sets the spectrum, and the default is an extrapolation.
    Ghavamian, Laming & Rakowski (2007) measured ``kT_e ~ 0.3 keV``,
    approximately independent of Mach number above ~1000 km/s, at Balmer-
    dominated shocks in **hydrogen-dominated ISM gas**. Cas A's X-rays come
    from the *reverse* shock in **metal-dominated ejecta**, where the mean ion
    mass is 16-56 times the proton mass and the electron-to-ion number ratio is
    8 rather than 1.2. Nothing in that calibration transfers automatically.

    The residual the model actually shows -- too little soft line emission
    (0.46-0.62x) together with too much hard continuum (1.56-1.71x) and Fe-K
    (2.42x) -- is the signature of *too much hot gas*, which is a statement
    about this function and not about the hydrodynamics. So it is made
    switchable, and the alternatives bracket the physics rather than
    interpolating between fits:

    ``ghavamian``
        The published constant. ``kT_e = min(kT_e_shock_keV, T)``.
    ``beta``
        A fixed FRACTION of the local post-shock temperature,
        ``T_e = beta_shock * T``. The physical difference from ``ghavamian`` is
        that it scales with the local shock strength instead of being pinned to
        an absolute energy calibrated on other shocks -- so a slow reverse
        shock in dense ejecta heats its electrons less, which the constant
        cannot express.
    ``equilibrated``
        ``T_e = T``: instantaneous equipartition. The HOT bound. Physically
        excluded at Cas A's ionization age, and included precisely because it
        brackets the answer from the wrong side.
    ``minimal``
        ``T_e / T_i = m_e / m_p``: adiabatic compression of the electrons and
        no collisionless heating at all. The COLD bound.

    Args:
        T: Single-fluid (mass-weighted mean) temperature, K.
        f_e: Electron share of the particles, ``n_e / (n_e + n_i)``.
        model: One of :data:`TE_MODELS`.
        kT_e_shock_keV: The constant, for ``ghavamian``.
        beta_shock: The fraction, for ``beta``.

    Returns:
        ``(T_e, T_i)`` immediately post-shock, satisfying
        ``f_e T_e + (1 - f_e) T_i = T`` exactly, so no energy is created or
        destroyed by the choice.
    """
    if model not in TE_MODELS:
        raise SystemExit(f"unknown --te-model {model!r}; choose from {TE_MODELS}")

    if model == "equilibrated":
        return np.array(T, dtype=np.float64), np.array(T, dtype=np.float64)

    if model == "minimal":
        # T_e = eps * T_i with eps = m_e/m_p, and f_e T_e + (1-f_e) T_i = T
        eps = M_E / M_P
        T_i = T / np.maximum(f_e * eps + (1.0 - f_e), 1e-30)
        return eps * T_i, T_i

    if model == "beta":
        T_e = beta_shock * np.asarray(T, dtype=np.float64)
    else:                                       # ghavamian
        # never more than the gas has: in gas cooler than 0.3 keV the constant
        # would hand the electrons more energy than the cell contains, and the
        # ions would have to go below the electrons to pay for it
        T_e = np.minimum(np.full_like(T, kT_e_shock_keV * KEV_IN_K), T)
    T_i = (T - f_e * T_e) / np.maximum(1.0 - f_e, 1e-30)
    return T_e, T_i


def electron_ion_temperatures(T, rho_code, time_since_shock_code, X,
                              kT_e_shock_keV=0.3, n_substeps=16,
                              te_model="ghavamian", beta_shock=0.05):
    """Split the single-fluid temperature into ``(T_e, T_i)``.

    The solver evolves one temperature, which is the mass-weighted mean: the
    pressure it carries is ``(n_e + n_i) k T``, so ``T`` fixes one combination
    of the two and the shock physics has to fix the other.

    * **At the shock** the electrons receive ``kT_e ~ 0.3 keV`` almost
      independently of Mach number above ~1000 km/s (Ghavamian, Laming &
      Rakowski 2007), and the ions take the rest. In heavy-element ejecta the
      "rest" is a great deal: with ``n_e/n_i = 8`` in fully ionized oxygen the
      ions must carry nine times the single-fluid temperature, which is the
      physical statement that shock heating is mass-proportional and oxygen ions
      are sixteen times heavier than protons. (Ghavamian et al. calibrated this
      on H-dominated ISM shocks; applying it to the reverse shock in metal
      ejecta is an extrapolation, and the single largest assumption here.)
    * **Downstream** the two relax by Coulomb collisions. Conserving thermal
      energy, ``n_e dT_e = -n_i dT_i``, so the DIFFERENCE decays not on
      ``t_eq`` but on ``t_eq * n_i / (n_e + n_i)`` -- faster by a factor 2.1 in
      cosmic gas and by 9 in fully ionized oxygen, because the electrons hold
      most of the heat capacity there. Relaxing on ``t_eq`` itself (what an
      earlier version of this did) leaves the electrons systematically too cold.

    The relaxation is integrated in TIME with the parcel's PRESENT density and
    temperature, and for an adiabatic parcel that is not an approximation but
    exact. The Spitzer time goes as ``t_eq ~ T^{3/2} / n_e``, and adiabatic
    expansion carries ``T ~ rho^{2/3}``, so ``T^{3/2}/n_e`` is INVARIANT along
    the trajectory: a parcel that has expanded since being shocked is both
    cooler and thinner in exactly the proportion that leaves ``t_eq``
    unchanged. Hence ``int dt / t_eq = t / t_eq(now)``.

    That invariance is what makes the present-day snapshot sufficient, and it is
    only available because this remnant is adiabatic to measurement precision
    (t_cool ~ 2.8 Myr in the shocked gas). Integrating instead in the electron
    column ``int n_e dt`` -- tempting, since the solver carries it for the
    ionization age -- assumes ``t_eq n_e`` is the invariant, which would require
    a constant temperature; it over-equilibrates the ejecta by ~40 %.

    Args:
        T: Single-fluid temperature (K).
        rho_code: Density (code units), for the local rate.
        time_since_shock_code: Time since the parcel was shocked (code units).
        X: Per-element mass fractions.
        kT_e_shock_keV: Post-shock electron temperature (Ghavamian et al.).
        n_substeps: Substeps for the relaxation integral.

    Unshocked parcels are returned with ``T_e = T_i = T``: they were never
    shocked, so there is no two-temperature state to describe.
    """
    T = np.asarray(T, dtype=np.float64)
    m = composition_moments(X)
    rho = np.asarray(rho_code, dtype=np.float64) * CODE_DENSITY
    n_e = rho / (m["mu_e"] * M_P)
    n_i = rho / (m["mu_i"] * M_P)
    f_e = n_e / (n_e + n_i)                     # electron share of the particles

    T_e, T_i = shock_electron_temperature(
        T, f_e, model=te_model, kT_e_shock_keV=kT_e_shock_keV,
        beta_shock=beta_shock)

    dt_total = np.asarray(time_since_shock_code, dtype=np.float64) * CODE_TIME
    shocked = dt_total > 0.0
    dt = dt_total / n_substeps
    for _ in range(n_substeps):
        # the temperature DIFFERENCE decays on t_eq * n_i/(n_e + n_i); both
        # temperatures therefore approach the (energy-conserving) mean on it
        tau = equipartition_time(T_e, T_i, rho_code, X) * (1.0 - f_e)
        frac = np.where(shocked, -np.expm1(-np.clip(dt / tau, 0.0, 50.0)), 0.0)
        T_mean = f_e * T_e + (1.0 - f_e) * T_i
        T_e = T_e + frac * (T_mean - T_e)
        T_i = T_i + frac * (T_mean - T_i)

    return np.where(shocked, T_e, T), np.where(shocked, T_i, T)


def plasma_state(fields, *, kT_e_shock_keV=0.3, two_temperature=True,
                 te_model="ghavamian", beta_shock=0.05):
    """Everything the X-ray model needs, from a ``--save-state`` npz.

    One entry point so that the diagnostics (``casa_plasma.py``) and the forward
    model (``casa_observe.py``) cannot drift apart -- they were computing
    different temperatures from the same file.

    Returns a dict of cgs fields plus ``info``, a summary of what the state
    actually carried (composition? shock history?) for the caller to report.
    """
    X_tracer, tracked = mass_fractions(fields)
    # everything downstream works in ELEMENTS, not tracers: the "Si" scalar
    # stands for Si+S+Ar+Ca and the "O" scalar for O/Ne/Mg (see TRACER_SPLIT)
    X = element_mass_fractions(X_tracer)
    nd = number_densities(fields["rho"], X)
    T = temperature(fields["rho"], fields["press"], X)

    net = (ionization_age(fields["density_time"], X)
           if "density_time" in fields else None)
    has_history = "time_since_shock" in fields
    if two_temperature and has_history:
        T_e, T_i = electron_ion_temperatures(
            T, fields["rho"], fields["time_since_shock"], X,
            kT_e_shock_keV=kT_e_shock_keV, te_model=te_model,
            beta_shock=beta_shock)
    else:
        T_e = T_i = T

    # The shocked FRACTION, never "has any accumulated time": advection leaks an
    # infinitesimal amount of the record into every cell, so a > 0 test calls
    # the whole box shocked.
    shocked = (np.asarray(fields["shocked_fraction"], dtype=np.float64) > 0.5
               if "shocked_fraction" in fields else np.ones_like(T, dtype=bool))

    return dict(X=X, X_tracer=X_tracer, T=T, T_e=T_e, T_i=T_i, n_e=nd["n_e"],
                n_i=nd["n_i"], n_H=nd["n_H"], moments=nd["moments"],
                net=net, shocked=shocked,
                info=dict(composition_tracked=tracked, shock_history=has_history,
                          two_temperature=bool(two_temperature and has_history),
                          te_model=te_model, kT_e_shock_keV=kT_e_shock_keV,
                          beta_shock=beta_shock))


def _self_check():
    """Print the benchmark numbers, so the units and the rates are visible."""
    yr = 3.155693e7
    print("[plasma] composition moments (fully ionized):")
    for name, X in (("cosmic", COSMIC), ("pure O", {"O": 1.0}),
                    ("pure Fe", {"Fe": 1.0}), ("pure Si", {"Si": 1.0})):
        m = composition_moments(X)
        print(f"    {name:8s} mu = {m['mu']:.3f}  mu_e = {m['mu_e']:.3f}  "
              f"mu_i = {m['mu_i']:.3f}  n_e/n_i = {m['mu_i'] / m['mu_e']:.2f}  "
              f"z2/a2 = {m['z2_a2']:.3f}")
    print("[plasma] equipartition time and the difference-decay time it implies:")
    for T, n_e_target, X, name in ((1e7, 1.0, {"H": 1.0}, "H"),
                                   (1e8, 10.0, COSMIC, "cosmic"),
                                   (1e8, 10.0, {"O": 1.0}, "pure O")):
        m = composition_moments(X)
        rho_code = n_e_target * m["mu_e"] * M_P / CODE_DENSITY
        t_eq = equipartition_time(T, T, rho_code, X)
        tau = t_eq * (1.0 - 1.0 / (1.0 + m["mu_e"] / m["mu_i"]))
        print(f"    {name:7s} T = {T:.0e} K, n_e = {n_e_target:g}: "
              f"t_eq = {t_eq / yr:.2e} yr, tau_diff = {tau / yr:.2e} yr")
    print("    Cas A is 350 yr old, so equilibration is far from complete and "
          "T_e < T:\n    the electron temperature has to be modelled, not "
          "assumed equal to T.")
    _assert_physics()


def _assert_physics():
    """Check the two properties the reconstruction must have, on every run.

    These are cheap and they are exactly the two things that went wrong before:
    a mean molecular weight that did not match the composition it came from, and
    a relaxation that did not conserve the thermal energy it was redistributing
    (which shows up as a temperature the hydrodynamics never had).
    """
    # 1. the moments are what hand-calculation says
    m = composition_moments({"O": 1.0})
    assert abs(m["mu"] - 16.0 / 9.0) < 1e-3, m["mu"]        # 8 electrons + 1 ion
    assert abs(m["mu_e"] - 2.0) < 1e-3, m["mu_e"]
    m = composition_moments({"H": 1.0})
    assert abs(m["mu"] - 0.504) < 1e-3, m["mu"]      # A_H/2, one electron each
    assert abs(m["mu_e"] - 1.008) < 1e-3, m["mu_e"]

    # 2. the relaxation conserves (n_e + n_i) T exactly, for any elapsed time,
    #    in both a hydrogen and a metal plasma
    for X in ({"H": 0.7, "He": 0.28, "O": 0.02}, {"O": 1.0}):
        mom = composition_moments(X)
        f_e = 1.0 / (1.0 + mom["mu_e"] / mom["mu_i"])
        rho_code = 10.0 * mom["mu_e"] * M_P / CODE_DENSITY      # n_e = 10 cm^-3
        T = np.array([1e7, 1e8, 1e9])
        for age_yr in (1.0, 350.0, 1e5):
            dt = age_yr * 3.155693e7 / CODE_TIME
            T_e, T_i = electron_ion_temperatures(
                T, rho_code, np.full_like(T, dt), X)
            err = np.max(np.abs((f_e * T_e + (1.0 - f_e) * T_i) / T - 1.0))
            assert err < 1e-10, (X, age_yr, err)
            assert np.all(T_e <= T_i + 1e-6 * T), (X, age_yr)
    print("[plasma] self-check passed (moments, energy conservation in the "
          "T_e/T_i relaxation)")


if __name__ == "__main__":
    _self_check()
