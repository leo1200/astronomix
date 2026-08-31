"""
Sub-grid density structure: what each cell would contain if it were resolved.

WHY THIS IS NECESSARY AND WHY IT CANNOT BE AVOIDED BY REFINING THE GRID
----------------------------------------------------------------------
Cas A's X-ray ejecta knots are ~1 arcsec across, which is 0.016 pc at 3.4 kpc.
A cell is 0.027 pc at 256^3 in the 7 pc box and 0.014 pc at 512^3. So the
structures that carry the emission are **at or below one cell** at the top of
the verified ladder, and resolving one at the 6-8 cells a shock interaction
needs would take ``N >~ 1500``: 3-6x in linear resolution, 30-200x in cost, and
past a memory wall that already stops this study at 512^3.

That is not a reason to give up on the physics, because the consequence is
measurable and one-sided. Emission goes as ``n^2``, so unresolved density
structure loses emission measure; and the temperature behind a shock goes as
``v^2``, so a shock that fails to slow down entering an unresolved dense knot
stays too hot. Both errors have the sign of the observed residual.

WHAT THE MEASUREMENT SAYS, WHICH IS NOT WHAT THE LITERATURE SAYS
---------------------------------------------------------------
``casa_xrism.py`` measures, on ``orl_n256_final``, an ion temperature in the
silicon-line-emitting gas of 418 keV, which is a reverse shock at **2758 km/s**
against the published ~1800. The correction needed is therefore
``chi = (2758/1800)^2 ~ 2.3`` -- a *modest* contrast.

Laming & Hwang (2003) and Vink et al. (2026) both infer ``chi ~ 10`` (iron
group) to ``~100`` (intermediate-mass elements). **These are not the same
quantity and the difference is the point.** Their inference is the offset of the
data from a model with a single uniform 1800 km/s reverse shock; ours is the
offset from a 3D calculation that already contains a distribution of shock
velocities, clumping at contrast 5, pistons and a CSM-shell reflected shock. Most
of what a one-zone analysis has to attribute to ``chi`` is already present as
resolved structure. Quoting ``chi = 100`` here would double-count it.

So ``chi`` is calibrated against ``casa_xrism.py``, and the value is reported
next to the one-zone literature value with this explanation, every time.

THE MODEL: TWO PHASES AT FIXED CELL MASS, VOLUME AND PRESSURE
-------------------------------------------------------------
Each cell is re-read as a dense phase and a diffuse phase. Three constraints fix
everything from two parameters (``chi``, ``f_mass``):

* **mass** -- the phases carry ``f_mass`` and ``1 - f_mass`` of the cell's mass,
  so no mass is invented (the failure that ``POSITIVITY_HARD_FLOOR`` taught this
  project to check first);
* **volume** -- the volume fractions sum to one, so no emitting volume is
  invented either;
* **pressure equilibrium** -- ``p`` is uniform across the cell. This is the
  standard post-crushing state (Klein, McKee & Colella 1994): a cloud hit by a
  shock is compressed until it matches the surrounding pressure, on
  ``t_cc ~ sqrt(chi) a / v_s``, which is short compared with the 200 yr since
  the reverse shock arrived. It also makes the temperature split *free of new
  parameters*: ``rho T`` equal across the phases gives ``T_dense = T / chi``
  immediately, which is the same ``v^2/chi`` a transmitted shock gives.

Everything else follows::

    f_vol,dense  = f_mass / chi
    rho_dense    = chi * rho_cell
    rho_diffuse  = rho_cell (1 - f_mass) / (1 - f_vol,dense)
    T_dense      = T_cell / chi
    T_diffuse    = T_cell * rho_cell / rho_diffuse
    clumping C   = <n^2>/<n>^2 = f_vol chi^2 + (1 - f_vol) (rho_diff/rho_cell)^2

There is **no free normalisation and nothing is fitted to an image.** The two
parameters are a contrast and a mass fraction, both with observational
counterparts, and ``chi = 1`` is an exact identity (:func:`_self_check` asserts
it), so the whole module is a verifiable no-op when off.

WHAT IS DELIBERATELY NOT MODELLED
---------------------------------
* **The ionization age of the dense phase.** ``n_e t`` scales up with the
  density but down with the elapsed time, because the transmitted shock is
  slower and arrives later, and the two cancel to an extent this model cannot
  compute. ``net_mode`` exposes the three defensible choices -- scale with
  density, leave unchanged, or scale by ``chi/sqrt(chi)`` for a shock that
  crossed the clump at ``v/sqrt(chi)`` -- and the answer is **already bounded by
  the data**: ``casa_xrism.py`` shows the model's ``n_e t`` at the top of XRISM's
  range before any sub-grid boost, so a mode that multiplies it by ``chi``
  overshoots. That bound is the reason the choice is exposed rather than picked.
* **Destruction.** Real knots ablate and mix (Cas A's have measured ablation
  tails, Fesen et al. 2025), so a fixed ``f_mass`` overstates how much mass is
  still in clumps at 350 yr.
* **Where the clumps are.** ``f_mass`` is uniform over the ejecta. XRISM's
  contrast differs between the iron group and the intermediate-mass elements,
  which this cannot express with one number -- :func:`two_phase` takes ``chi``
  per element group for that reason, but the mass fraction stays global.

THIS IS AN INTERPRETATION LAYER, AND MUST BE LABELLED ONE
---------------------------------------------------------
The simulation does not contain this structure; this is a statement about what
it would contain if it did. Any figure or number using it says so. That is the
standing rule in ``OVERVIEW.md`` for a sub-grid clumping factor and it is not
relaxed by the fact that the parameters now have measurements behind them.
"""

# numerics
import numpy as np

#: contrast calibrated against casa_xrism.py's silicon ion temperature on
#: orl_n256_final: (2758 km/s / 1800 km/s)^2. NOT the one-zone literature value
#: -- see the module docstring.
CHI_CALIBRATED = 2.3

#: fraction of the ejecta mass in the dense phase. Cas A's X-ray image is
#: knot-dominated, so most of the EMISSION is in clumps; this is the mass, and
#: 0.5 is a deliberately unaggressive placeholder pending the scan in
#: casa_xrism.py --subgrid-scan.
F_MASS_DEFAULT = 0.5

#: how the dense phase's ionization age is obtained from the cell's
NET_MODES = ("density", "unchanged", "crossing")


def phase_factors(chi, f_mass, net_mode="crossing"):
    """The multiplicative factors defining each phase -- THE single source.

    Returns ``{name: (rho_factor, time_factor, net_factor, volume_fraction)}``,
    all relative to the cell.

    **This function exists because reimplementing it drifted, and the drift was
    invisible in the output.** The first version of the scan composed the
    ionization-age factor as ``rho_factor * time_factor``, which made
    ``net_mode="unchanged"`` behave identically to ``"density"`` -- so a scan
    meant to BOUND the ionization-age treatment silently measured the same thing
    twice. The tell was ``unchanged`` giving a HIGHER ``n_e t`` than
    ``crossing``, which is impossible by construction. Both ``casa_xrism.py``
    and ``casa_observe.py`` now call this instead of writing their own.

    The factors are tied together by ``net = rho * t``, so choosing ``net_mode``
    FIXES the time factor rather than leaving it free::

        net_factor  = {density: chi, unchanged: 1, crossing: sqrt(chi)}
        time_factor = net_factor / rho_factor

    That constraint is what the drifted version broke: it set the time factor and
    the density factor independently, so the two described different clumps.
    """
    if net_mode not in NET_MODES:
        raise SystemExit(f"unknown net_mode {net_mode!r}, choose from {NET_MODES}")
    if not (0.0 < f_mass < 1.0):
        raise SystemExit(f"--subgrid-fmass must be in (0, 1), got {f_mass}")
    if chi < 1.0:
        raise SystemExit(f"--subgrid-chi must be >= 1 (the dense phase is "
                         f"dense), got {chi}")
    f_vol = f_mass / chi
    if f_vol >= 1.0:
        raise SystemExit(
            f"chi = {chi} and f_mass = {f_mass} need the dense phase to fill "
            f"{100 * f_vol:.0f}% of every cell, which leaves no diffuse phase. "
            f"f_mass < chi is required.")

    rho_d = float(chi)
    rho_u = (1.0 - f_mass) / (1.0 - f_vol)
    net_d = {"density": rho_d, "unchanged": 1.0,
             "crossing": float(np.sqrt(chi))}[net_mode]
    # The DIFFUSE phase is what the resolved solution already describes: its
    # elapsed time is unchanged, so its ionization age follows its own density.
    return {
        "dense": (rho_d, net_d / rho_d, net_d, f_vol),
        "diffuse": (rho_u, 1.0, rho_u, 1.0 - f_vol),
    }


def two_phase(rho, T, *, chi=CHI_CALIBRATED, f_mass=F_MASS_DEFAULT,
              net=None, net_mode="crossing"):
    """Split a cell into a dense and a diffuse phase.

    Args:
        rho: Cell mean density, any units (only ratios are used).
        T: Cell temperature, any units.
        chi: Density contrast of the dense phase relative to the CELL MEAN.
            ``chi = 1`` reproduces the input exactly.
        f_mass: Mass fraction in the dense phase, in ``(0, 1)``.
        net: Optional ionization age of the cell, scaled per ``net_mode``.
        net_mode: One of :data:`NET_MODES`. ``density`` scales ``n_e t`` with the
            phase density (right if the clump was shocked at the same time as its
            surroundings, which it was not); ``unchanged`` keeps the cell value
            (right if the density rise and the shorter elapsed time cancel);
            ``crossing`` scales by ``chi / sqrt(chi) = sqrt(chi)``, for a
            transmitted shock crossing the clump at ``v/sqrt(chi)`` so the dense
            gas has been shocked for ``1/sqrt(chi)`` as long. ``crossing`` is the
            default because it is the only one of the three derived from the same
            transmitted-shock argument that sets the temperature.

    Returns:
        A list of two dicts, dense first, each with ``rho``, ``T``,
        ``f_vol`` (volume fraction) and ``net`` (or ``None``).

    Raises:
        SystemExit: if the parameters are outside the range where the split
            exists. ``f_mass > ... `` would need the dense phase to occupy more
            than the whole cell, and a silent clip there would produce a
            plausible number from an impossible configuration.
    """
    factors = phase_factors(chi, f_mass, net_mode)
    rho = np.asarray(rho, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    net = None if net is None else np.asarray(net, dtype=np.float64)

    out = []
    for name, (rho_f, _t_f, net_f, f_vol) in factors.items():
        # pressure equilibrium: rho T is the same in both phases and equal to
        # the cell's, so each phase's temperature is fixed by its own density
        out.append(dict(rho=rho_f * rho, T=T / rho_f, f_vol=f_vol, name=name,
                        net=None if net is None else net * net_f))
    return out


def clumping_factor(chi, f_mass):
    """``<n^2>/<n>^2`` the split implies, for comparison with the measured one.

    ``casa_xrism.py`` reports the value the RESOLVED state already has (1.45 at
    256^3, 1.87 at 512^3). The two multiply: a sub-grid split raises the
    emissivity of gas that is itself already structured, so the total is the
    product, not the larger of the two.
    """
    f_vol = f_mass / chi
    if f_vol >= 1.0:
        return np.nan
    return f_vol * chi ** 2 + (1.0 - f_vol) * ((1.0 - f_mass) / (1.0 - f_vol)) ** 2


def shock_velocity_ratio(chi):
    """Transmitted-shock velocity in the dense phase, as a fraction of ``v_s``.

    ``1/sqrt(chi)`` -- the statement that makes ``chi`` measurable from an ion
    temperature rather than fitted to an image.
    """
    return 1.0 / np.sqrt(chi)


def chi_for_velocity_ratio(ratio):
    """The inverse: contrast needed to slow the shock to ``ratio * v_s``.

    This is how :data:`CHI_CALIBRATED` was obtained -- from the ion temperature
    ``casa_xrism.py`` measures, not from a fit.
    """
    return 1.0 / np.asarray(ratio, dtype=np.float64) ** 2


def describe(chi, f_mass, *, net_mode="crossing", resolved_clumping=None):
    """A block of text saying what the split does, for a run log."""
    f_vol = f_mass / chi
    C = clumping_factor(chi, f_mass)
    lines = [
        f"[subgrid] two-phase re-interpretation: chi = {chi:.2f}, "
        f"f_mass = {f_mass:.2f}, net_mode = {net_mode}",
        f"[subgrid]   dense phase: {100 * f_vol:.1f}% of the volume, "
        f"{100 * f_mass:.0f}% of the mass, T / {chi:.2f}, "
        f"transmitted shock at {100 * shock_velocity_ratio(chi):.0f}% of v_s",
        f"[subgrid]   clumping factor contributed: {C:.2f}",
    ]
    if resolved_clumping is not None:
        lines.append(
            f"[subgrid]   the resolved state already has {resolved_clumping:.2f}, "
            f"so the total is {C * resolved_clumping:.2f}")
    lines.append(
        "[subgrid]   THIS IS AN INTERPRETATION LAYER: the simulation does not "
        "contain this structure.")
    return "\n".join(lines)


def _self_check():
    """The conservation laws, the chi = 1 identity, and the calibration."""
    rng = np.random.default_rng(0)
    rho = rng.lognormal(0.0, 0.7, 5000)
    T = rng.lognormal(16.0, 0.5, 5000)
    net = rng.lognormal(25.0, 0.4, 5000)

    # 1. chi = 1 is an exact identity, so the module is a verifiable no-op
    for f_mass in (0.1, 0.5, 0.9):
        for ph in two_phase(rho, T, chi=1.0, f_mass=f_mass, net=net):
            assert np.allclose(ph["rho"], rho, rtol=1e-12), ph["name"]
            assert np.allclose(ph["T"], T, rtol=1e-12), ph["name"]
            assert np.allclose(ph["net"], net, rtol=1e-12), ph["name"]
        assert abs(clumping_factor(1.0, f_mass) - 1.0) < 1e-12

    # 2. mass, volume and pressure are conserved for any (chi, f_mass)
    for chi in (1.5, 2.3, 10.0, 100.0):
        for f_mass in (0.05, 0.3, 0.5, 0.9):
            if f_mass >= chi:
                continue
            phases = two_phase(rho, T, chi=chi, f_mass=f_mass, net=net)
            m = sum(p["f_vol"] * p["rho"] for p in phases)
            v = sum(p["f_vol"] for p in phases)
            assert np.allclose(m, rho, rtol=1e-12), (chi, f_mass, "mass")
            assert abs(v - 1.0) < 1e-12, (chi, f_mass, "volume")
            for p in phases:
                assert np.allclose(p["rho"] * p["T"], rho * T, rtol=1e-12), \
                    (chi, f_mass, p["name"], "pressure")
            # the dense phase really is denser and cooler
            assert np.all(phases[0]["rho"] > phases[1]["rho"])
            assert np.all(phases[0]["T"] < phases[1]["T"])
            # and it raises the emission measure
            assert clumping_factor(chi, f_mass) >= 1.0 - 1e-12, (chi, f_mass)

    # 2b. THE DRIFT GUARD. net = rho * t must hold for every mode, and the three
    #     modes must give DIFFERENT ionization ages -- the bug was that two of
    #     them gave the same one while claiming to bound the choice.
    for chi in (1.5, 2.3, 4.0, 16.0):
        seen = {}
        for mode in NET_MODES:
            f = phase_factors(chi, 0.5, mode)
            for name, (rho_f, t_f, net_f, _v) in f.items():
                assert abs(rho_f * t_f - net_f) < 1e-12, (chi, mode, name)
            seen[mode] = f["dense"][2]
        assert len(set(seen.values())) == 3, (chi, seen)
        # and they are ordered: unchanged < crossing < density
        assert seen["unchanged"] < seen["crossing"] < seen["density"], (chi, seen)
    # at chi = 1 all three coincide, because there is nothing to choose
    one = {m: phase_factors(1.0, 0.5, m)["dense"][2] for m in NET_MODES}
    assert all(abs(v - 1.0) < 1e-12 for v in one.values()), one

    # 3. the velocity relation and its inverse are consistent, and the
    #    calibration is the number the docstring claims
    assert abs(chi_for_velocity_ratio(shock_velocity_ratio(7.0)) - 7.0) < 1e-10
    chi_cal = float(chi_for_velocity_ratio(1800.0 / 2758.0))
    assert abs(chi_cal - CHI_CALIBRATED) < 0.05, chi_cal

    # 4. an impossible configuration is refused, not clipped
    for bad in (dict(chi=2.0, f_mass=2.5), dict(chi=0.5, f_mass=0.5),
                dict(chi=2.0, f_mass=0.0)):
        try:
            two_phase(rho, T, **bad)
        except SystemExit:
            pass
        else:                                       # pragma: no cover
            raise AssertionError(f"accepted an impossible split: {bad}")

    print(f"[subgrid] self-check passed. chi from the measured Si ion "
          f"temperature (2758 km/s vs 1800) is {chi_cal:.2f}; at "
          f"f_mass = {F_MASS_DEFAULT} that is a clumping factor of "
          f"{clumping_factor(chi_cal, F_MASS_DEFAULT):.2f} and a dense-phase "
          f"temperature of T / {chi_cal:.2f}.")
    print(describe(CHI_CALIBRATED, F_MASS_DEFAULT, resolved_clumping=1.45))


if __name__ == "__main__":
    _self_check()
