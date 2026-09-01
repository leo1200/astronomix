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
#: The split is by MASS. A fixed ratio within each layer is exactly as much as
#: co-located tracers can say; resolving it would need one scalar per element.
#:
#: TWO OBSERVATIONS DISAGREE ABOUT IT, AND THE DISAGREEMENT IS PHYSICAL.
#:
#: ``hwang_laming`` reproduces Hwang & Laming (2012)'s measured shocked masses
#: (O 2, Ne 0.03, Mg 0.03; Si 0.08, S 0.06, Ar 0.02, Ca 0.02), integrated over
#: the WHOLE remnant. It was chosen because being comparable to that measurement
#: is the point of this pipeline. But it puts, relative to solar,
#:
#:     S/Si = 1.44x     Ar/Si = 1.72x     Ca/Si = 2.72x
#:
#: everywhere -- and XRISM/Resolve (Vink et al. 2026) measures S/Si = 0.88-1.12
#: solar per 30" pixel, with the Ar and Ca enhancement confined to the NE and SW
#: jet bases. Both can be true at once: the remnant-integrated Ar mass can be
#: 0.02 Msun with most of it in the jets while the ratio is solar in the bulk.
#:
#: This model cannot express that. One "Si" tracer stands for the whole
#: Si/S/Ar/Ca layer, so any enhancement is spread over ALL of it -- including the
#: brightest smooth Si, which is where the emission is. That is exactly the
#: recorded residual "Ar and Ca lines ~2x too strong while Si is 0.74x": not a
#: bug and not a physical effect, but the cost of four tracers standing for nine
#: elements.
#:
#: So the choice is exposed rather than hidden, and the two presets fix different
#: things. ``xrism_bulk`` matches the per-pixel LINE RATIOS where the emission
#: is, at the price of under-predicting the remnant-integrated Ar and Ca masses
#: (the jet enhancement is then absent rather than mislocated).
#: Select with :func:`set_tracer_split`; anything that reports masses or
#: abundances must say which was used.
#: Hwang & Laming (2012) shocked masses [Msun], remnant-integrated. The presets
#: are DERIVED from these rather than typed in as fractions: the hand-typed
#: version summed to 0.999 in the Si layer, which quietly lost 0.1 % of it.
HWANG_LAMING_SHOCKED_MASSES = {
    "O": 2.0, "Ne": 0.03, "Mg": 0.03,
    "Si": 0.08, "S": 0.06, "Ar": 0.02, "Ca": 0.02,
    "Fe": 0.14,
}


def _mass_layer_split(elements, masses):
    """Normalise measured masses into mass fractions within one layer."""
    total = sum(masses[el] for el in elements)
    return {el: masses[el] / total for el in elements}


TRACER_SPLIT_PRESETS = {
    "hwang_laming": {
        "O": _mass_layer_split(("O", "Ne", "Mg"), HWANG_LAMING_SHOCKED_MASSES),
        "Si": _mass_layer_split(("Si", "S", "Ar", "Ca"),
                                HWANG_LAMING_SHOCKED_MASSES),
        "Fe": {"Fe": 1.0},
        "He": {"He": 1.0},
    },
    # solar S/Si, Ar/Si and Ca/Si by number, i.e. XRISM's bulk line ratios.
    # Derived from SOLAR_NUMBER_RATIO_TO_H, not typed in -- see _solar_layer_split.
    "xrism_bulk": None,             # filled in below
}

#: which preset is active
TRACER_SPLIT_NAME = "hwang_laming"


def _solar_layer_split(elements):
    """Mass fractions within a layer that give exactly SOLAR number ratios.

    Computed from :data:`SOLAR_NUMBER_RATIO_TO_H` so the preset cannot drift out
    of step with the abundance table the forward model normalises to -- typing
    the numbers in is how a "solar" ratio silently stops being one.
    """
    w = {el: SOLAR_NUMBER_RATIO_TO_H[el] * ATOMIC[el][0] for el in elements}
    total = sum(w.values())
    return {el: v / total for el, v in w.items()}


# ONLY the Si layer changes. XRISM/Resolve's band does not constrain Ne/O or
# Mg/O, and forcing those to solar would be nucleosynthetic nonsense: it puts
# 15 % of the oxygen layer's mass into neon, whereas Hwang & Laming measure
# Ne/O and Mg/O far BELOW solar (0.03 against 2.0 Msun), which is what an
# oxygen-burning layer should look like. Changing a ratio no observation
# constrains, in the same commit as one it does, is how a calibration stops
# being traceable.
TRACER_SPLIT_PRESETS["xrism_bulk"] = dict(
    TRACER_SPLIT_PRESETS["hwang_laming"],
    Si=_solar_layer_split(("Si", "S", "Ar", "Ca")),
)

TRACER_SPLIT = TRACER_SPLIT_PRESETS[TRACER_SPLIT_NAME]


def set_tracer_split(name):
    """Choose a :data:`TRACER_SPLIT_PRESETS` entry, process-wide.

    Module-level state, deliberately: :func:`element_mass_fractions` is called
    from deep inside the forward model and threading the choice through every
    caller would be a large change for a switch that must be set once per run.
    The cost is that it MUST be set before any composition is computed, and that
    every consumer has to report :data:`TRACER_SPLIT_NAME` alongside its numbers.
    """
    global TRACER_SPLIT, TRACER_SPLIT_NAME
    if name not in TRACER_SPLIT_PRESETS:
        raise SystemExit(f"unknown tracer split {name!r}; choose from "
                         f"{sorted(TRACER_SPLIT_PRESETS)}")
    TRACER_SPLIT_NAME = name
    TRACER_SPLIT = TRACER_SPLIT_PRESETS[name]
    return TRACER_SPLIT


def tracer_split_report(name=None):
    """One line per layer element: mass fraction and ratio to solar."""
    split = TRACER_SPLIT_PRESETS[name] if name else TRACER_SPLIT
    lines = []
    for tracer, parts in split.items():
        if len(parts) == 1:
            continue
        ref = tracer
        for el, frac in parts.items():
            solar = (SOLAR_NUMBER_RATIO_TO_H[el] / SOLAR_NUMBER_RATIO_TO_H[ref]
                     * ATOMIC[el][0] / ATOMIC[ref][0])
            lines.append(f"    {el:3s} {frac:.4f} by mass, "
                         f"{el}/{ref} = {frac / parts[ref] / solar:.2f}x solar")
    return "\n".join(lines)


def element_mass_fractions(X):
    """Expand the tracer mass fractions into per-ELEMENT mass fractions.

    Reads :data:`TRACER_SPLIT` at call time, so :func:`set_tracer_split` takes
    effect for callers that imported this function rather than the dict.
    """
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


def species_ion_temperature(T_i, mu_i, element):
    """Ion temperature of ONE species, from the model's mean ion temperature.

    The solver carries one temperature and :func:`electron_ion_temperatures`
    splits it into one electron and one *mean* ion temperature. A collisionless
    shock does not do that: it heats each ion species to
    ``T_s = (3/16) m_s v_s^2 / k``, i.e. **in proportion to its mass**, so the
    oxygen and the iron behind the same shock differ by a factor 3.5. The mean
    is what the mean is::

        T_i = sum_s n_s T_s / sum_s n_s = (T_ref / m_ref) <m>_i

    so inverting it for one species costs nothing but the composition already
    carried::

        T_s = T_i * m_s / (mu_i m_p)

    XRISM/Resolve now measures this difference directly from the *thermal* part
    of the line widths -- ``T_i(Fe) - T_i(Si) = 300 +- 180 keV`` (NW) and
    ``150 +- 60 keV`` (SE) (Vink et al. 2026) -- so it is a prediction of the
    calibrated dynamics that can be checked rather than an assumption. Note that
    it is essentially ``(3/16)(m_Fe - m_Si) v_s^2`` at the reverse shock, which
    is 176 keV at 1800 km/s: the measurement is a shock-velocity measurement in
    disguise, and getting it right is a statement about the hydrodynamics.

    **Two caveats, and they point in opposite directions.**

    *Upper bound:* nothing here relaxes the species towards each other. Ion-ion
    Coulomb collisions do, so the true difference is smaller than this by an
    amount the model does not track. That XRISM still measures a difference
    means the relaxation is incomplete, not absent.

    *Robust to expansion:* adiabatic expansion cools every species by the same
    ``rho^(2/3)``, so the RATIO ``T_s / T_i`` is preserved along a parcel's
    trajectory. The formula is therefore valid at the observation epoch and not
    only at the shock -- unlike almost everything else in this module, it needs
    no history integral.

    Args:
        T_i: Mean ion temperature [K], from :func:`electron_ion_temperatures`.
        mu_i: Mean ion mass in units of ``m_p`` (``moments["mu_i"]``).
        element: Symbol in :data:`ATOMIC`.

    Returns:
        The species' ion temperature [K], same shape as ``T_i``.
    """
    if element not in ATOMIC:
        raise KeyError(f"unknown element {element!r}; have {sorted(ATOMIC)}")
    A = ATOMIC[element][0]
    return np.asarray(T_i, dtype=np.float64) * A / np.maximum(mu_i, 1e-30)


def shock_speed_from_pressure(press_code, rho_code):
    """Shock speed [cm/s] implied by the post-shock state. **Composition-free.**

    From ``p = n k T`` with ``n = rho / (mu m_p)`` and the strong-shock jump
    ``T_2 = (3/16) mu m_p v^2 / k``, the mean molecular weight CANCELS::

        v = sqrt(16 p / (3 rho))

    That is what makes this the right quantity to put in a guardrail: it compares
    directly to a measured shock speed without assuming an ejecta composition, it
    needs no electron/ion split, and it is identical in the 1D and 3D solvers so
    the two can be checked against each other. (It equals ``1.79 c_s`` at
    ``gamma = 5/3``, as it must.)

    **Why it exists.** The CSM shell was found to raise the temperature of the gas
    just inside the contact discontinuity by a factor 1.8 while moving r_RS by
    0.07 sigma -- a trade invisible for a month because every guardrail in this
    project was a radius, a mass, a band ratio or a structure statistic, and none
    of them was thermodynamic. A factor of two in the temperature of the emitting
    gas should not be able to hide.

    **Report it BOTH ways.** Averaged over a shell just outside r_RS it varies by
    2.1x between configurations (1213-2576 km/s); emission-weighted over all the
    shocked gas the same states differ by only 7 % (2740-3178). The global average
    hid the local factor of two, so a guardrail that quotes one number quotes the
    wrong one.
    """
    return np.sqrt(16.0 / 3.0 * np.asarray(press_code, dtype=np.float64)
                   / np.maximum(np.asarray(rho_code, dtype=np.float64), 1e-300)
                   ) * CODE_VELOCITY


def thermal_line_width(T_s, element):
    """1-sigma thermal Doppler width of a line of ``element`` [cm/s].

    ``sigma_v = sqrt(k T_s / m_s)``. Worth having next to
    :func:`species_ion_temperature` because of what the pair implies: under
    *exactly* mass-proportional heating ``T_s / m_s`` is the same for every
    species, so **every line has the same thermal width** and none of an
    observed difference between species is thermal. XRISM measures the Fe-group
    lines broader than the intermediate-mass ones by ~1500 km/s, so either the
    heating is super-mass-proportional for iron or -- their reading -- the iron
    was shocked by a faster shock, ``v_Fe^2 = (2300 km/s)^2 + v_IME^2``.
    Either way it is a constraint on the model's *dynamics*.
    """
    A = ATOMIC[element][0]
    return np.sqrt(K_B * np.asarray(T_s, dtype=np.float64) / (A * M_P))


#: fields a diagnostic needs from a ``--save-state`` npz, beyond the optional
#: composition tracers
_REQUIRED_STATE_FIELDS = ("rho", "press", "density_time", "time_since_shock")


def load_diagnostic_state(path):
    """Read a ``--save-state`` npz for a diagnostic, and REFUSE a blown-up one.

    Shared by ``casa_plasma.py`` and ``casa_xrism.py`` because the guard is the
    interesting part and it must not drift between them. Analysing an aborted
    run does not fail -- it produces confident nonsense, and it has: one aborted
    state reported an ejecta mass of 2.5e13 Msun and another 0.000 with
    ``T_e/T = 79.9``. A density-only guard let the second one through, so all
    four of density, pressure, ionization age and total ejecta mass are checked.

    Returns ``(fields, meta)`` where ``meta`` carries ``age``, ``box``,
    ``num_cells`` and, for states written after the provenance fix, ``argv`` and
    ``git_commit``.
    """
    d = np.load(path)
    if "density_time" not in d:
        raise SystemExit(
            f"{path} carries no shock history -- rerun casa_orlando.py with "
            "--composition")
    fields = {k: np.asarray(d[k], dtype=np.float64)
              for k in _REQUIRED_STATE_FIELDS}

    # The shocked FRACTION, not "has any accumulated time": advection leaks an
    # infinitesimal amount of the record into every cell, so a > 0 test would
    # call the whole box shocked (exactly the trap the solver's own latch had to
    # be rewritten to avoid).
    fields["shocked_fraction"] = (
        np.asarray(d["shocked_fraction"], dtype=np.float64)
        if "shocked_fraction" in d
        else (fields["time_since_shock"] > 0).astype(np.float64))

    # C_He matters as much as the others: it is a quarter of the ejecta mass, and
    # leaving it out would put that mass into hydrogen and halve the local mu.
    for k in ("C_ej", "C_Fe", "C_Si", "C_O", "C_He"):
        if k in d:
            fields[k] = np.asarray(d[k], dtype=np.float64)
    for k in ("vx", "vy", "vz"):
        if k in d:
            fields[k] = np.asarray(d[k], dtype=np.float64)

    bad = []
    if not np.all(np.isfinite(fields["rho"])):
        bad.append(f"{int(np.sum(~np.isfinite(fields['rho'])))} non-finite "
                   "density cells")
    if float(fields["rho"].max()) > 1e6:
        bad.append(f"max density {float(fields['rho'].max()):.3e}")
    if not np.all(np.isfinite(fields["press"])) or float(fields["press"].min()) < 0:
        bad.append("non-finite or negative pressure")
    if "C_ej" in fields:
        dx = float(d["box"]) / int(d["num_cells"])
        m_ej = float(np.sum(fields["C_ej"] * fields["rho"])) * dx ** 3
        if not (0.1 < m_ej < 100.0):
            bad.append(f"total ejecta mass {m_ej:.3e} Msun")
    if bad:
        raise SystemExit(
            f"{path} looks unphysical ({'; '.join(bad)}) -- the run probably "
            "aborted. Check its log for an ABORT before analysing.")

    meta = dict(age=float(d["age"]), box=float(d["box"]),
                num_cells=int(d["num_cells"]))
    for k in ("argv", "git_commit"):
        if k in d:
            meta[k] = str(np.asarray(d[k]).item())
    return fields, meta


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

    # 3. the per-species split reproduces the mean it came from. This is the one
    #    property that makes species_ion_temperature more than a rescaling: the
    #    number-weighted mean of the per-species temperatures must be the mean
    #    ion temperature the solver actually carries, for ANY composition.
    for X in ({"O": 1.0}, {"O": 0.6, "Si": 0.2, "Fe": 0.2}, COSMIC):
        mom = composition_moments(X)
        T_i = 1e9
        num = den = 0.0
        for el, X_el in X.items():
            if X_el <= 0:
                continue
            n_s = X_el / ATOMIC[el][0]                  # per unit mass
            num += n_s * species_ion_temperature(T_i, mom["mu_i"], el)
            den += n_s
        assert abs(num / den / T_i - 1.0) < 1e-10, (X, num / den / T_i)

    # 3b. the composition-free shock speed really is composition-free, and really
    #     is 1.79 c_s. Both are one-line identities and both are the reason it is
    #     the quantity a thermodynamic guardrail should use.
    rho_t = np.array([1.0, 4.0, 0.25])
    press_t = np.array([1.0, 2.0, 0.5])
    v = shock_speed_from_pressure(press_t, rho_t)
    c_s = np.sqrt(5.0 / 3.0 * press_t / rho_t) * CODE_VELOCITY
    assert np.allclose(v / c_s, np.sqrt(16.0 / 5.0), rtol=1e-12), v / c_s
    # and it inverts the strong-shock jump for ANY mu
    for X in ({"H": 1.0}, {"O": 1.0}, COSMIC):
        mom = composition_moments(X)
        v0 = 2.0e8 / CODE_VELOCITY                  # code units
        T2 = 3.0 / 16.0 * mom["mu"] * M_P * (v0 * CODE_VELOCITY) ** 2 / K_B
        rho0 = 1.0
        p0 = rho0 * CODE_DENSITY / (mom["mu"] * M_P) * K_B * T2 / CODE_PRESSURE
        assert abs(shock_speed_from_pressure(p0, rho0) / (v0 * CODE_VELOCITY)
                   - 1.0) < 1e-10, X

    # 4. mass-proportional heating gives every species the SAME thermal line
    #    width, which is why an observed difference between species cannot be
    #    thermal (see thermal_line_width)
    w = [thermal_line_width(species_ion_temperature(1e9, 16.0, el), el)
         for el in ("O", "Si", "Fe")]
    assert max(w) / min(w) - 1.0 < 1e-12, w

    # 5. every preset is a normalised split of every layer it names, and the
    #    two differ ONLY where an observation says they should. A preset that
    #    silently renormalises, or that moves a ratio nothing constrains, is the
    #    failure mode this guard exists for.
    base = TRACER_SPLIT_PRESETS["hwang_laming"]
    for name, split in TRACER_SPLIT_PRESETS.items():
        assert set(split) == set(base), (name, sorted(split))
        for tracer, parts in split.items():
            assert abs(sum(parts.values()) - 1.0) < 1e-12, (name, tracer)
            assert all(v >= 0.0 for v in parts.values()), (name, tracer)
    differ = {t for t in base if TRACER_SPLIT_PRESETS["xrism_bulk"][t] != base[t]}
    assert differ == {"Si"}, differ
    # and xrism_bulk really is solar in the ratios XRISM measured
    xb = TRACER_SPLIT_PRESETS["xrism_bulk"]["Si"]
    for el in ("S", "Ar", "Ca"):
        solar = (SOLAR_NUMBER_RATIO_TO_H[el] / SOLAR_NUMBER_RATIO_TO_H["Si"]
                 * ATOMIC[el][0] / ATOMIC["Si"][0])
        assert abs(xb[el] / xb["Si"] / solar - 1.0) < 1e-12, el

    # 6. the reverse-shock arithmetic XRISM measures: (3/16)(m_Fe - m_Si) v^2 at
    #    1800 km/s is ~176 keV, inside the observed 150 +- 60 to 300 +- 180 keV.
    #    A single hand-checkable number tying the formula to the measurement.
    v = 1.8e8
    dT_keV = (3.0 / 16.0) * (ATOMIC["Fe"][0] - ATOMIC["Si"][0]) * M_P * v ** 2 \
        / K_B / KEV_IN_K
    assert 150.0 < dT_keV < 210.0, dT_keV

    print("[plasma] self-check passed (moments, energy conservation in the "
          f"T_e/T_i relaxation, per-species ion temperatures; the reverse-shock "
          f"Fe-Si ion temperature difference at 1800 km/s is {dT_keV:.0f} keV "
          f"against XRISM's 150-300)")


if __name__ == "__main__":
    _self_check()
