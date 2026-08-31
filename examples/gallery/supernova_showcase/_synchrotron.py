"""
Non-thermal synchrotron: the half of Cas A's hard X-ray flux the model omits.

WHY THIS IS NOT OPTIONAL, AND WHY IT IS NOT A "REFINEMENT"
----------------------------------------------------------
Helder & Vink (2008) measure **~54 % of Cas A's 4.2-6 keV flux as non-thermal**
integrated over the remnant, and Vink et al. (2026) find 47-90 % per XRISM
pixel. The forward model in this directory is purely thermal. So the recorded
residual "our 4.2-6 keV band is 1.71x the observed" is **not a comparison
between the same quantity**: the observed thermal flux in that band is ~0.46x
the observed total, which makes the model's thermal continuum ~3.7x too bright,
not 1.7x.

That reframing is the reason this module exists, and it is worth being blunt
about the direction: **adding synchrotron makes the hard-band residual worse,
not better.** It does not rescue the model. What it does is make the band a
measurement of something, and turn a 3-5x deficit at the 140-180" rim into a
component with a shape that can be checked.

HOW FAR THIS CAN BE PUSHED WITHOUT FITTING -- READ THIS FIRST
------------------------------------------------------------
The obvious implementation has two free parameters, an electron injection
efficiency and a magnetic-field amplification factor, and with both free the
X-ray rim brightness is fitted and the exercise says nothing. It turns out most
of that freedom can be removed, and what remains is ONE GEOMETRIC parameter with
a direct measurement behind it.

**Step 1: the radio flux fixes the electrons.** Cas A's 1 GHz flux is one of the
best-measured numbers in astronomy -- 2720 Jy, alpha = 0.77 (Baars et al. 1977) --
and it comes from the same relativistic electrons. Radio and X-ray synchrotron
emissivity both scale as ``K B^((s+1)/2)``, so normalising the summed radio
emission to the measured flux fixes that product and cancels it from the ratio::

    X-ray / radio = (E_X/E_radio)^-alpha * exp(-sqrt(E_X/E_cut))

**Step 2: the cutoff is field-independent.** For loss-limited acceleration
``E_cut`` depends only on the shock speed and the gyrofactor (next section), so
the field amplification cancels here too.

**Step 3: the field returns, through the emitting VOLUME.** This is the part the
first version of this module got wrong, and it is worth stating because the
mistake is natural. The electrons radiating 5 keV synchrotron live
``t_sync = 0.9 yr`` at 100 uG and ``0.08 yr`` at Cas A's ~500 uG rim field,
against a 350 yr remnant. They exist in a thin layer at the shock, while the
radio electrons fill the whole shell -- so assuming the two are co-spatial
overstates the X-ray volume. Carried out (:func:`_self_check` reproduces every
number):

===============================================  ==========================
emitting volume assumed                          4.2-6 keV vs observed
===============================================  ==========================
co-spatial with the radio (whole shell)          **9.3x too bright**
purely advected loss layer, 1.0e-4 pc            **0.003x** (300x too faint)
**observed filament width, 1-3 arcsec**          **0.51x - 1.53x**
===============================================  ==========================

**So the picture is quantitatively consistent.** Anchored on the measured radio
flux, with a loss-limited cutoff and the observed non-thermal filament width as
the emitting thickness, the predicted 4.2-6 keV non-thermal flux lands within a
factor of two of the measured one -- across the whole observed range of filament
widths, with no efficiency and no amplification factor fitted.

**What is NOT predicted is the filament width itself.** The advected loss layer
is 2936x thinner than the emitting shell, so the observed 1-3 arcsec filaments
are 160-480x thicker than pure advection allows. That is a known result and it
requires diffusive transport or magnetic damping downstream -- an open problem
this model does not contain. The width therefore enters as a measured input, and
the honest description of this component is **"one geometric parameter, taken
from an observation", not "two free physics parameters"** and not "a
prediction".

WHAT IS PREDICTED, WITH NOTHING ADJUSTABLE AT ALL
-------------------------------------------------
The *shape* depends only on the cutoff, so two things are free of even that one
parameter:

1. **The cutoff energy is a shock-speed measurement.** Inverting XRISM's fitted
   photon index Gamma = 2.94-3.43 through the loss-limited relation at eta = 1
   gives a shock at **1708-2423 km/s** -- Cas A's *reverse* shock (1800-2000),
   not its forward shock (~5000). So the non-thermal continuum in those pointings
   is reverse-shock emission, which agrees with the published association of the
   non-thermal continuum with the reverse shock in the (south)west, and which
   nothing here was tuned to produce.
2. **The spatial distribution of the cutoff** follows from the simulated
   shock-velocity field, so the *radial profile* and the *hardness gradient* of
   the non-thermal component are predicted up to one overall factor -- and those
   are exactly the quantities the recorded 3-5x rim deficit is about.

AND THE DIRECTION IT MOVES THE RESIDUAL IS STILL UNFAVOURABLE
-------------------------------------------------------------
Worth repeating, because the section above is encouraging and the consequence is
not: **adding synchrotron makes the hard-band thermal residual worse.** If half
the observed 4-6 keV flux is non-thermal, the model's purely thermal 1.71x
becomes ~3.7x against the thermal part alone. This component does not rescue the
model; it makes the band mean something, and hands the problem to the ejecta
density structure (``_subgrid.py``).

THE ONE ROBUST RESULT THIS RESTS ON
-----------------------------------
For a shock where acceleration is limited by synchrotron losses rather than by
age or escape -- which is Cas A's regime -- the cutoff photon energy is

    h nu_cut ~ 1.4 keV * (v_s / 3000 km/s)^2 / eta

and is **independent of the magnetic field** (Zirakashvili & Aharonian 2007;
Vink 2012 for the review). The B-dependences of the acceleration rate and of the
loss rate cancel exactly. That is what lets the field amplification drop out of
the cutoff -- though NOT out of the emitting volume, which is the whole content
of the section above.

The spectral shape near the cutoff is taken as ``nu^-alpha exp(-sqrt(nu/nu_cut))``,
the loss-limited form from the same work. Its consequence is checkable and
sharp: the *local* photon index steepens through the band, which is why XRISM
fits Gamma = 2.94-3.43 while the radio index is only 0.77 -- and matching that
steepening is a test the normalisation cannot be tuned to pass.

WHERE THE SHOCK VELOCITY COMES FROM
-----------------------------------
Not from a shock finder. The ion temperature IS the shock velocity:
``T_i = (3/16) mu_i m_p v_s^2 / k`` (see :func:`_plasma.species_ion_temperature`,
where the same relation is checked against XRISM's measured Fe-Si difference).
Adiabatic expansion since shocking lowers it, so this is used only on
freshly-shocked cells -- which is also the only place X-ray synchrotron comes
from, because the electrons that make it lose their energy in decades. The radio
electrons are long-lived and the whole shocked volume contributes, so the two
components are weighted differently and deliberately.

WHAT IS STILL A FIT, AND MUST BE SAID
-------------------------------------
* **The non-thermal filament width**, taken from the observation (1-3 arcsec).
  It is the one geometric parameter and it is what the factor-of-two agreement
  above rests on.
* ``eta``. Bohm diffusion is ``eta = 1``; Cas A's rim filaments are usually
  fitted with ``eta`` of a few. It is scanned, not chosen.
* **The radio normalisation is global, so the radio MORPHOLOGY is not
  predicted** -- only its total. If the model puts its relativistic electrons in
  the wrong places, this hides it. Checking the synthetic radio image against
  the VLA map is the natural next test and is not done here.
* Cas A's radio flux **decreases secularly**, ~0.6-0.8 %/yr, which is why the
  epoch of the 2720 Jy measurement is recorded with it.
* No inverse Compton, no synchrotron self-absorption, no CR back-reaction on the
  dynamics (measured null for the shock radii -- ``CALIBRATION.md`` Result 3).
"""

# numerics
import numpy as np

# the shared constants, so this module cannot disagree with the plasma model
from _plasma import ATOMIC, K_B, KEV_IN_K, M_P

# =============================================================================
# ============ ↓ Measurements this module is anchored on ↓ =====================
# =============================================================================
#: Cas A's integrated radio flux density and spectral index (Baars et al. 1977).
#: The flux decreases ~0.6-0.8 %/yr, so the epoch matters and is recorded.
RADIO_FLUX_JY = 2720.0
RADIO_FREQ_GHZ = 1.0
RADIO_EPOCH = 1965.0
RADIO_ALPHA = 0.77          # S_nu ~ nu^-alpha, so the electron index s = 2a+1
RADIO_SECULAR_DECLINE_PER_YR = 0.007

#: Zirakashvili & Aharonian (2007): loss-limited cutoff photon energy at
#: 3000 km/s and Bohm diffusion. Independent of B -- see the module docstring.
CUTOFF_KEV_AT_3000 = 1.4

#: measured non-thermal share of the 4.2-6 keV band, to compare a prediction
#: against. Helder & Vink (2008) remnant-integrated; Vink et al. (2026) per
#: XRISM pixel. The remnant-integrated value is the one matching our aperture.
NONTHERMAL_FRACTION_4_6 = 0.54
NONTHERMAL_FRACTION_4_6_RANGE = (0.47, 0.90)
#: XRISM's fitted power-law photon index, per pixel
PHOTON_INDEX_OBSERVED = (2.94, 3.43)

#: observed width of Cas A's non-thermal X-ray filaments [arcsec]. This is the
#: ONE geometric parameter the X-ray normalisation rests on -- see the docstring.
FILAMENT_WIDTH_ARCSEC = (1.0, 3.0)
#: thickness of the X-ray emitting shell [pc], for the volume ratio
SHELL_THICKNESS_PC = 0.3
#: one arcsec at 3.4 kpc
ARCSEC_IN_PC = 1.0 / (206264.806 / 3400.0)
# =============================================================================
# ============ ↑ Measurements this module is anchored on ↑ =====================
# =============================================================================


def electron_index(alpha=RADIO_ALPHA):
    """Electron energy index ``s`` from the radio spectral index: ``s = 2a + 1``.

    Test-particle diffusive shock acceleration at a strong (r = 4) shock predicts
    ``s = 2``, i.e. ``alpha = 0.5``. Cas A's 0.77 is steeper than that, which is
    a real and long-standing discrepancy; taking ``s`` from the OBSERVED radio
    index rather than from the theory is the choice that keeps the X-ray
    prediction anchored to the same electrons that make the radio.
    """
    return 2.0 * alpha + 1.0


def shock_velocity_from_ion_temperature(T_i, mu_i):
    """``v_s`` [cm/s] from the mean ion temperature: the inverse of ``(3/16) mu m v^2``.

    Valid on FRESHLY shocked gas only. Adiabatic expansion cools a parcel after
    shocking, so on old gas this underestimates the shock that made it -- which is
    why :func:`emissivity_fields` gates on ``time_since_shock``.
    """
    T_i = np.asarray(T_i, dtype=np.float64)
    return np.sqrt(16.0 / 3.0 * K_B * T_i / (np.maximum(mu_i, 1e-30) * M_P))


def cutoff_photon_energy_keV(v_shock_cgs, eta=1.0):
    """Loss-limited synchrotron cutoff [keV]. **Independent of B.**

    ``1.4 keV (v_s/3000 km/s)^2 / eta``. The field-independence is not an
    approximation but a cancellation: the acceleration time and the synchrotron
    loss time both scale as ``1/B^2`` at fixed ``E``, so ``E_max`` depends only
    on ``v_s`` and the gyrofactor. It is the single fact that makes the X-ray
    prediction here independent of the field amplification.
    """
    v = np.asarray(v_shock_cgs, dtype=np.float64) / 3.0e8      # in 3000 km/s
    return CUTOFF_KEV_AT_3000 * v ** 2 / np.maximum(eta, 1e-30)


def spectral_shape(E_keV, E_cut_keV, alpha=RADIO_ALPHA):
    """``nu^-alpha exp(-sqrt(nu/nu_cut))``, normalised to 1 at 1 keV.

    The loss-limited form. Returns the shape only: every absolute factor
    (``K``, ``B^((s+1)/2)``, the emissivity constant) cancels when the summed
    radio emission is normalised to the measured flux, so they are deliberately
    absent rather than approximated.
    """
    E = np.asarray(E_keV, dtype=np.float64)
    Ec = np.maximum(np.asarray(E_cut_keV, dtype=np.float64), 1e-30)
    return E ** (-alpha) * np.exp(-np.sqrt(E / Ec))


def band_shape_integral(E_lo_keV, E_hi_keV, E_cut_keV, *, alpha=RADIO_ALPHA,
                        n=64):
    """``int E * shape dE`` over a band -- an ENERGY flux, per cell.

    pyXSIM's ``PowerLawSourceModel`` wants a luminosity in the band, so the
    integrand carries the extra factor of ``E``. Integrated on a log grid because
    the cutoff makes the integrand span orders of magnitude within one band.
    """
    Ec = np.asarray(E_cut_keV, dtype=np.float64)
    e = np.logspace(np.log10(E_lo_keV), np.log10(E_hi_keV), n)
    # shape (n,) x Ec.shape -> integrate over axis 0
    grid = e.reshape((-1,) + (1,) * Ec.ndim)
    integrand = grid * spectral_shape(grid, Ec[None, ...], alpha=alpha)
    return np.trapezoid(integrand, e, axis=0)


def local_photon_index(E_keV, E_cut_keV, alpha=RADIO_ALPHA):
    """Logarithmic slope of the PHOTON spectrum at ``E``, i.e. Gamma.

    Analytic, from ``dN/dE ~ E^-(alpha+1) exp(-sqrt(E/E_cut))``::

        Gamma = alpha + 1 + (1/2) sqrt(E / E_cut)

    This is the quantity XRISM fits (2.94-3.43) and it is where the model can be
    falsified without touching the normalisation: at the radio index alpha = 0.77
    the underlying power law has Gamma = 1.77, so an observed Gamma near 3
    requires ``sqrt(E/E_cut) ~ 2.5``, i.e. ``E_cut ~ E/6`` -- a cutoff well below
    the band, which is what "loss-limited at a few thousand km/s" means.
    """
    E = np.asarray(E_keV, dtype=np.float64)
    Ec = np.maximum(np.asarray(E_cut_keV, dtype=np.float64), 1e-30)
    return alpha + 1.0 + 0.5 * np.sqrt(E / Ec)


def cutoff_for_photon_index(gamma, E_keV, alpha=RADIO_ALPHA):
    """Inverse of :func:`local_photon_index` -- the cutoff a fitted Gamma implies."""
    x = 2.0 * (np.asarray(gamma, dtype=np.float64) - alpha - 1.0)
    return np.asarray(E_keV, dtype=np.float64) / np.maximum(x, 1e-30) ** 2


def eta_for_photon_index(gamma, E_keV, v_shock_cgs, alpha=RADIO_ALPHA):
    """The gyrofactor consistent with an observed Gamma at a known shock speed.

    Turns the one free parameter into a measurement, given a shock velocity. Both
    inputs are available: XRISM fits Gamma per pixel and the simulation supplies
    ``v_s`` per cell, so ``eta`` is over-determined rather than chosen -- and if
    the implied values are not of order unity to ten, the loss-limited premise
    is wrong and the model should say so.
    """
    Ec = cutoff_for_photon_index(gamma, E_keV, alpha=alpha)
    v = np.asarray(v_shock_cgs, dtype=np.float64) / 3.0e8
    return CUTOFF_KEV_AT_3000 * v ** 2 / np.maximum(Ec, 1e-30)


def electron_lorentz_factor(E_keV, B_gauss):
    """``gamma`` of the electrons radiating ``E_keV`` at field ``B``.

    From ``nu_c = 4.2e6 gamma^2 B[G]`` Hz. Needed because the SYNCHROTRON
    LIFETIME depends on it, and the lifetime is what sets the emitting volume --
    the one place the field amplification does not cancel.
    """
    nu = np.asarray(E_keV, dtype=np.float64) / 4.135667696e-18      # keV -> Hz
    return np.sqrt(nu / (4.2e6 * np.maximum(B_gauss, 1e-30)))


def synchrotron_lifetime_yr(E_keV, B_gauss):
    """Cooling time [yr] of the electrons radiating ``E_keV``.

    ``t = 24.5 yr / ((B/100 uG)^2 (E_e/TeV))``. At Cas A's ~500 uG rim field the
    5 keV electrons last **0.08 yr** against a 350 yr remnant, which is why the
    X-ray synchrotron comes from a thin layer at the shock while the radio comes
    from the whole shell -- and therefore why normalising to the radio flux
    predicts the X-ray flux only once an emitting thickness is supplied. See the
    module docstring.
    """
    B = np.asarray(B_gauss, dtype=np.float64)
    gamma = electron_lorentz_factor(E_keV, B)
    E_TeV = gamma * 0.511e-3 * 1e-3             # m_e c^2 in MeV -> TeV
    return 24.5 / np.maximum((B / 1e-4) ** 2 * E_TeV, 1e-30)


def loss_layer_thickness_pc(E_keV, B_gauss, v_shock_cgs):
    """Thickness of a purely ADVECTED synchrotron loss layer [pc].

    ``(v_s / 4) * t_sync``, the compression-4 downstream speed times the cooling
    time: **1.0e-4 pc** at 5 keV, 500 uG and 5000 km/s, i.e. 2936x thinner than
    a ~0.3 pc emitting shell. Cas A's non-thermal filaments are observed at
    1-3 arcsec (0.016-0.05 pc), which is **160-480x thicker**, so the real
    transport is diffusive or the field is damped -- and the observed width, not
    this one, is what the emitting volume should be taken from.
    """
    t = synchrotron_lifetime_yr(E_keV, B_gauss) * 3.155693e7        # s
    return np.asarray(v_shock_cgs, dtype=np.float64) / 4.0 * t / 3.0857e18


def amplified_field_gauss(rho_cgs, v_shock_cgs, eps_B=0.01):
    """``B`` from a fixed fraction of the ram pressure: ``B^2/8pi = eps_B rho v_s^2``.

    The cutoff does not use it (:func:`cutoff_photon_energy_keV` is
    field-independent), but :func:`synchrotron_lifetime_yr` does, and through it
    the emitting volume -- so this is where the field amplification re-enters
    after appearing to cancel. Compare what it gives against Cas A's measured
    ~0.5 mG rim to see how much of the answer is being assumed.
    """
    rho = np.asarray(rho_cgs, dtype=np.float64)
    v = np.asarray(v_shock_cgs, dtype=np.float64)
    return np.sqrt(8.0 * np.pi * eps_B * rho * v ** 2)


def radio_flux_at(year, *, flux_jy=RADIO_FLUX_JY, epoch=RADIO_EPOCH,
                  decline=RADIO_SECULAR_DECLINE_PER_YR):
    """Cas A's 1 GHz flux at a later epoch, given the measured secular decline.

    ~0.7 %/yr compounds to a 25 % difference between the 1965 measurement and a
    2004 comparison, which is larger than most of the residuals in this study.
    """
    return flux_jy * (1.0 - decline) ** (year - epoch)


def nonthermal_band_fraction(E_cut_keV, weights, *, band=(4.2, 6.0),
                             radio_ghz=RADIO_FREQ_GHZ, alpha=RADIO_ALPHA):
    """Ratio of band synchrotron energy flux to radio, summed over cells.

    The whole point: this is the X-ray-to-radio ratio, in which every absolute
    factor has cancelled. Multiply by a measured radio flux to get an absolute
    X-ray flux -- no electron efficiency, no field.

    ``weights`` is the relative radio emissivity per cell (``K B^((s+1)/2)``),
    which need only be known up to one global constant.
    """
    E_radio_keV = radio_ghz * 1e9 * 4.135667696e-18      # h nu in keV
    w = np.asarray(weights, dtype=np.float64)
    num = np.sum(w * band_shape_integral(band[0], band[1], E_cut_keV,
                                         alpha=alpha))
    # the radio is far below any cutoff, so its shape is the bare power law
    den = np.sum(w * E_radio_keV * spectral_shape(E_radio_keV,
                                                  np.full_like(E_cut_keV, 1e30),
                                                  alpha=alpha))
    return float(num / max(den, 1e-300))


def _self_check():
    """Every number the module claims, checked against its published source."""
    # 1. the electron index from Cas A's radio index, and DSA's own value
    assert abs(electron_index(0.77) - 2.54) < 1e-9
    assert abs(electron_index(0.5) - 2.0) < 1e-9        # strong-shock DSA

    # 2. the anchor: 1.4 keV at 3000 km/s, Bohm. And the scaling is v^2/eta.
    assert abs(cutoff_photon_energy_keV(3.0e8, 1.0) - 1.4) < 1e-12
    assert abs(cutoff_photon_energy_keV(6.0e8, 1.0) - 5.6) < 1e-12
    assert abs(cutoff_photon_energy_keV(3.0e8, 4.0) - 0.35) < 1e-12

    # 3. the ion-temperature inverse round-trips against _plasma's forward form
    for A, T in ((28.085, 4.85e9), (55.845, 1.0e10)):
        v = shock_velocity_from_ion_temperature(T, A)
        T_back = 3.0 / 16.0 * A * M_P * v ** 2 / K_B
        assert abs(T_back / T - 1.0) < 1e-12, (A, T)
    # and 1800 km/s in silicon is the 176 keV _plasma._assert_physics checks
    T_si = 3.0 / 16.0 * ATOMIC["Si"][0] * M_P * (1.8e8) ** 2 / K_B
    assert 150.0 < T_si / KEV_IN_K < 210.0, T_si / KEV_IN_K

    # 4. Gamma and its inverse are consistent, and the radio limit is right
    for Ec in (0.1, 1.0, 10.0):
        g = local_photon_index(5.0, Ec)
        assert abs(cutoff_for_photon_index(g, 5.0) / Ec - 1.0) < 1e-9, Ec
    # far below the cutoff the photon index is alpha + 1
    assert abs(local_photon_index(1e-6, 1.0) - (RADIO_ALPHA + 1.0)) < 1e-3

    # 5. THE FALSIFIABLE ONE: what shock speed does XRISM's Gamma imply, at
    #    eta = 1? If the loss-limited picture holds it must be a few thousand
    #    km/s -- the range Cas A's shocks are actually measured at.
    for g in PHOTON_INDEX_OBSERVED:
        Ec = float(cutoff_for_photon_index(g, 5.0))
        v = 3.0e8 * np.sqrt(Ec / CUTOFF_KEV_AT_3000)
        assert 500.0 < v / 1e5 < 6000.0, (g, Ec, v / 1e5)

    # 6. the band integral is positive, monotone in the cutoff, and the
    #    non-thermal fraction machinery runs on an array
    Ec = np.array([0.05, 0.2, 1.0, 5.0])
    I = band_shape_integral(4.2, 6.0, Ec)
    assert np.all(I > 0) and np.all(np.diff(I) > 0), I
    w = np.array([1.0, 2.0, 3.0, 4.0])
    r = nonthermal_band_fraction(Ec, w)
    assert np.isfinite(r) and r > 0.0, r

    # 7. THE THREE-WAY BRACKET on the emitting volume, reproduced end to end.
    #    This is the module's central quantitative claim and every number in the
    #    docstring's table comes from here.
    E_radio_keV = RADIO_FREQ_GHZ * 1e9 * 4.135667696e-18
    assert E_radio_keV < 1e-8                # the radio is 9 decades below 5 keV
    obs_nt = 2.7e-11                         # observed non-thermal 4.2-6 keV
    ratio = nonthermal_band_fraction(np.array([0.6]), np.array([1.0]))
    nu_S_nu = RADIO_FREQ_GHZ * 1e9 * radio_flux_at(2004.0) * 1e-23
    pred_cospatial = ratio * nu_S_nu
    # (a) co-spatial with the radio: too bright, by about an order of magnitude
    assert 5.0 < pred_cospatial / obs_nt < 20.0, (ratio, pred_cospatial)
    # (b) the lifetime, months at Cas A's rim field, and monotone in B
    t500 = float(synchrotron_lifetime_yr(5.0, 500e-6))
    t100 = float(synchrotron_lifetime_yr(5.0, 100e-6))
    assert 0.05 < t500 < 0.12, t500
    assert 0.6 < t100 < 1.3, t100
    assert t100 > t500                       # weaker field, longer-lived
    # (c) a purely advected loss layer: far too faint
    d_adv = float(loss_layer_thickness_pc(5.0, 500e-6, 5.0e8))
    assert 5e-5 < d_adv < 5e-4, d_adv
    assert 1e3 < SHELL_THICKNESS_PC / d_adv < 1e4, SHELL_THICKNESS_PC / d_adv
    assert pred_cospatial * d_adv / SHELL_THICKNESS_PC / obs_nt < 0.01
    # (d) the OBSERVED filament width: within a factor of two, both ends
    for w_arcsec in FILAMENT_WIDTH_ARCSEC:
        p = pred_cospatial * (w_arcsec * ARCSEC_IN_PC) / SHELL_THICKNESS_PC
        assert 0.4 < p / obs_nt < 2.0, (w_arcsec, p / obs_nt)
    # and the observed filaments really are far thicker than advection allows
    assert 100.0 < FILAMENT_WIDTH_ARCSEC[0] * ARCSEC_IN_PC / d_adv < 1000.0

    # 8. the secular decline, in the direction and size it is measured
    f2004 = radio_flux_at(2004.0)
    assert f2004 < RADIO_FLUX_JY and 0.6 < f2004 / RADIO_FLUX_JY < 0.85, f2004

    print("[synchrotron] self-check passed.")
    print(f"    electron index from Cas A's radio alpha = {RADIO_ALPHA}: "
          f"s = {electron_index():.2f}  (strong-shock DSA predicts 2.0)")
    for v_kms in (2000.0, 3000.0, 5000.0):
        Ec = float(cutoff_photon_energy_keV(v_kms * 1e5))
        g = float(local_photon_index(5.0, Ec))
        print(f"    v_s = {v_kms:5.0f} km/s (eta=1): E_cut = {Ec:5.2f} keV, "
              f"Gamma at 5 keV = {g:.2f}")
    print(f"    XRISM fits Gamma = {PHOTON_INDEX_OBSERVED[0]}-"
          f"{PHOTON_INDEX_OBSERVED[1]}, which at 5 keV needs E_cut = "
          f"{float(cutoff_for_photon_index(PHOTON_INDEX_OBSERVED[1], 5.0)):.2f}-"
          f"{float(cutoff_for_photon_index(PHOTON_INDEX_OBSERVED[0], 5.0)):.2f}"
          " keV,")
    v_lo = 3.0e8 * np.sqrt(float(cutoff_for_photon_index(
        PHOTON_INDEX_OBSERVED[1], 5.0)) / CUTOFF_KEV_AT_3000) / 1e5
    v_hi = 3.0e8 * np.sqrt(float(cutoff_for_photon_index(
        PHOTON_INDEX_OBSERVED[0], 5.0)) / CUTOFF_KEV_AT_3000) / 1e5
    print(f"    i.e. a loss-limited shock at {v_lo:.0f}-{v_hi:.0f} km/s for "
          "eta = 1 -- Cas A's forward shock is\n    measured near 5000 km/s and "
          "its reverse shock near 1800-2000, so the\n    premise is self-"
          "consistent and eta is of order unity to a few.")
    print(f"    The 5 keV electrons live {t500:.2f} yr at 500 uG (350 yr "
          f"remnant), so the emitting\n    VOLUME, not the spectrum, sets the "
          f"X-ray normalisation. Radio-anchored 4.2-6 keV\n    flux against an "
          f"observed non-thermal ~{obs_nt:.1e} erg/cm2/s:")
    print(f"      co-spatial with the radio      "
          f"{pred_cospatial:.2e}   {pred_cospatial / obs_nt:6.2f}x")
    p_adv = pred_cospatial * d_adv / SHELL_THICKNESS_PC
    print(f"      advected loss layer, {d_adv:.1e} pc  "
          f"{p_adv:.2e}   {p_adv / obs_nt:6.4f}x")
    for w in FILAMENT_WIDTH_ARCSEC:
        p = pred_cospatial * (w * ARCSEC_IN_PC) / SHELL_THICKNESS_PC
        print(f"      OBSERVED filament width {w:.0f}\"      "
              f"{p:.2e}   {p / obs_nt:6.2f}x")
    print(f"    So the picture is CONSISTENT within a factor of two once the "
          f"observed filament\n    width supplies the thickness -- but that "
          f"width is {FILAMENT_WIDTH_ARCSEC[0] * ARCSEC_IN_PC / d_adv:.0f}-"
          f"{FILAMENT_WIDTH_ARCSEC[1] * ARCSEC_IN_PC / d_adv:.0f}x what pure "
          f"advection\n    allows, so it is a measured INPUT and not a "
          "prediction. One geometric parameter.")
    print(f"    Cas A radio: {RADIO_FLUX_JY:.0f} Jy at "
          f"{RADIO_FREQ_GHZ:.0f} GHz in {RADIO_EPOCH:.0f}, "
          f"{radio_flux_at(2004.0):.0f} Jy by 2004 at "
          f"{100 * RADIO_SECULAR_DECLINE_PER_YR:.1f} %/yr")


if __name__ == "__main__":
    _self_check()
