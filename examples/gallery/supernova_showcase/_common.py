"""
Shared helpers for the supernova showcase.

Three high-order finite-difference (WENO, RK4-SSP) supernova-remnant setups
share the same solver configuration, code-unit conventions and initial-condition
builders, which live here:

  * :func:`fd_positivity` / :func:`make_fd_config`  -- the FD/WENO solver config
    with the positivity-preserving flux limiter that keeps a Sedov-strength or
    cold-ejecta blast stable in single precision (see the module note below).
  * :func:`athena_code_units`   -- the Guo, Kim & Stone (2025) ``snr.athinput``
    code units, used by ``snr_sedov.py``.
  * :func:`snr_code_units`      -- pc / Msun / 1000 km s^-1, used by the
    ejecta-driven remnants (``cassiopeia.py`` / ``young_snr_ism.py``).
  * :func:`tapered_sphere_weight`      -- smooth, renormalisable injection mask.
  * :func:`freely_expanding_ejecta`    -- cold, homologously-expanding ejecta
    (flat core + power-law envelope) laid on top of an arbitrary ambient,
    renormalised so the ejecta mass and kinetic energy hit their targets exactly.

Numerics note (why the positivity-preserving flux limiter)
-----------------------------------------------------------
A supernova blast is a Sedov-strength (thermal bomb) or cold-high-Mach (freely
expanding ejecta) flow. A pure high-order WENO scheme with only a hard
density/pressure floor NaNs within a handful of steps -- in *both* single and
double precision, so it is not a floating-point issue but a missing
positivity mechanism. Enabling the Hu-Adams-Shu / Zalesak FCT
positivity-preserving *flux* limiter (``PositivityConfig(preserving_flux=True)``)
blends each WENO interface flux toward the first-order Lax-Friedrichs flux by the
minimal amount that keeps density and pressure positive; it is a high-order
technique (no finite-volume / first-order fallback) and makes every setup here
stable and energy-conserving in float32. Two further requirements: a
well-resolved, tanh-tapered injection region (a single-cell top-hat NaNs
regardless), and an exact mass/energy renormalisation of that region.
"""

# general
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# units and constants
from astropy import units as u
import astropy.constants as const

# astronomix constants
from astronomix import (
    CARTESIAN,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import (
    POSITIVITY_HARD_FLOOR,
    POSITIVITY_REDISTRIBUTE,
    POSITIVITY_CONSERVATIVE,
)

# astronomix containers
from astronomix import (
    CodeUnits,
    SimulationConfig,
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
)

# radiative cooling
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    PIECEWISE_POWER_LAW,
    IMPLICIT_COOLING,
    EXPLICIT_COOLING,
)
from astronomix._modules._cooling._cooling_tables import schure_cooling

# turbulent initial-condition field generator
from astronomix.initial_condition_generation.turbulent_ic_generator import create_turb_field


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

GAMMA = 5.0 / 3.0


# =============================================================================
# ============ ↓ Solver configuration (high-order FD/WENO) ↓ ==================
# =============================================================================
def fd_positivity(tfloor=False, tfloor_stage=False, coldcrush_factor=8.0,
                  mode=POSITIVITY_HARD_FLOOR):
    """The positivity configuration that keeps the blast stable in float32.

    The positivity-preserving flux limiter (``preserving_flux``) is the key
    ingredient; a plain hard floor alone NaNs a strong blast in ~3 steps. The
    remaining flags are cheap backstops (``nan_safe``) and a vacuum handling that
    stops recovered velocities from spiking in floored cells (``vacuum_rest``).

    ``tfloor=True`` additionally upgrades the per-STEP pressure floor to the
    density-scaled temperature floor (``p >= rho * minimum_specific_pressure``,
    Athena tfloor) -- required for runs with REAL radiative cooling, where the
    cooled shock layer compresses to the isothermal jump and the constant floor
    leaves it pressureless against ram crushing. Keep it off for adiabatic runs
    (the proven hero recipe).
    """
    return PositivityConfig(
        # HARD_FLOOR is documented as NON-CONSERVATIVE and it is how these
        # runs manufacture mass: a drained cell is refilled from nothing, its
        # neighbour drains it again, and rho runs to 1e16+ while the box goes
        # from 17 Msun to 1e12 Msun. Selectable so the alternatives can be
        # tested against that.
        per_stage_mode=mode,
        per_step_mode=mode,
        per_step_specific_floor=tfloor,
        per_stage_specific_floor=tfloor_stage,
        preserving_flux=True,
        # First-order (LLF) blending at radiatively cooled, ram-pressure-crushed
        # cells (interface T within 8x of the minimum_specific_pressure
        # temperature floor). Without it, once the grid resolves the cooling
        # layer (N >= 512 for the blast/shell, earlier in the jet cone) the
        # crush has no pressure support and collapses without bound (rho ~ 1e16,
        # dt -> 0). Inert when params.minimum_specific_pressure == 0.
        coldcrush_blend=True,
        # 8 is the proven value for the adiabatic hero runs. With genuine
        # radiative cooling the crushing band is wider, so this is exposed:
        # the failure it guards against (rho running to 1e10+ and dt -> 0 in
        # the piston wakes) reappears at 8 once cooling is actually solved.
        coldcrush_blend_factor=float(coldcrush_factor),
        nan_safe=True,
        vacuum_rest=True,
    )


def periodic_box():
    """Triply-periodic boundaries (all three showcase setups use a periodic box)."""
    return BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    )


def make_fd_config(box_size, num_cells, mhd=False, cooling_config=None,
                   snapshot_settings=None, **extra):
    """Assemble the high-order FD/WENO :class:`SimulationConfig` for the showcase.

    Args:
        box_size: Cubic box side length (code units).
        num_cells: Cells per axis.
        mhd: Run the FD backend in MHD mode (with B = 0). Both hydro and MHD are
            stable here; hydro is the default.
        cooling_config: Optional :class:`CoolingConfig` (radiative losses).
        snapshot_settings: Optional :class:`SnapshotSettings`.
        **extra: Extra ``SimulationConfig`` overrides.
    """
    kwargs = dict(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        geometry=CARTESIAN,
        mhd=mhd,
        first_order_fallback=False,          # pure high-order flux (no FOFC)
        box_size=box_size,
        num_cells=num_cells,
        boundary_settings=periodic_box(),
        positivity_config=fd_positivity(),
        progress_bar=True,                   # live progress (t / t_end)
    )
    if cooling_config is not None:
        kwargs["cooling_config"] = cooling_config
    if snapshot_settings is not None:
        kwargs["return_snapshots"] = True
        kwargs["snapshot_settings"] = snapshot_settings
    kwargs.update(extra)
    return SimulationConfig(**kwargs)
# =============================================================================
# ============ ↑ Solver configuration (high-order FD/WENO) ↑ ==================
# =============================================================================


# =============================================================================
# ============ ↓ Code-unit systems ↓ ==========================================
# =============================================================================
def athena_code_units():
    """Guo, Kim & Stone (2025) ``snr.athinput`` code units.

    1 pc / (mass of n = 1 cm^-3 over 1 pc^3) / 1 Myr, expressed as
    ``CodeUnits(length, mass, velocity)`` with velocity = length / time.
    """
    length_cgs = 3.0856775809623245e18       # 1 pc
    mass_cgs = 3.015012867156497e31           # 1 cm^-3 over 1 pc^3
    time_cgs = 3.15576e13                      # 1 Myr
    return CodeUnits(length_cgs * u.cm, mass_cgs * u.g,
                     (length_cgs / time_cgs) * u.cm / u.s)


def snr_code_units():
    """Code units for the ejecta-driven remnants: pc / Msun / 1000 km s^-1."""
    return CodeUnits(1.0 * u.pc, 1.0 * u.Msun, 1000.0 * u.km / u.s)
# =============================================================================
# ============ ↑ Code-unit systems ↑ ==========================================
# =============================================================================


# =============================================================================
# ============ ↓ Initial-condition builders ↓ =================================
# =============================================================================
def centered_radius(helper_data, box_size, num_cells):
    """Radius (and centered X, Y, Z) from the box centre, in code length units."""
    c = helper_data.geometric_centers
    c0 = box_size / 2.0
    X, Y, Z = c[..., 0] - c0, c[..., 1] - c0, c[..., 2] - c0
    r = jnp.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    return r, X, Y, Z


def tapered_sphere_weight(r, radius, dx, taper_cells=3.0):
    """Smooth spherical injection weight in [0, 1] (≈1 for r < radius).

    A tanh taper of ``taper_cells`` cells replaces the sharp top-hat edge that a
    high-order scheme cannot resolve. Callers renormalise the weight so the
    injected mass / energy is independent of the taper.
    """
    return 0.5 * (1.0 - jnp.tanh((r - radius) / (taper_cells * dx)))


def ejecta_radial_shape(r, r_core, edge_radius, dx, *, envelope_slope=9.0,
                        taper_cells=3.0, inner_slope=0.0):
    """Dimensionless core-collapse ejecta density profile (flat core + envelope).

    ``(r / r_core)^-inner_slope`` inside ``r_core``, falling as
    ``(r / r_core)^-envelope_slope`` outside it, tanh-tapered to zero at
    ``edge_radius`` over ``taper_cells`` cells. ``inner_slope = 0`` is the flat
    core and reproduces the previous behaviour exactly.

    The inner slope is what sets how much ejecta is still UNSHOCKED at the
    remnant's age, and it is the parameter the original calibration had no
    handle on. A flat core puts ``M(<r) ~ r^3``, i.e. almost no mass at low
    velocity, so the reverse shock reaches the centre in mass long before it
    does in radius: the calibrated model leaves only 0.10 M_sun unshocked
    against the 0.3-0.4 observed. A centrally peaked core (``delta ~ 1-2``, the
    range standard ejecta models use) holds mass back at low velocity. The
    amplitude is left to the caller (who renormalises to the target ejecta
    mass), so the same profile can be laid down on a 1D radial grid (the
    calibration runs in ``casa_calibrate_1d.py``) and on the 3D Cartesian grid
    (:func:`freely_expanding_ejecta`) without the two drifting apart.

    Args:
        r: Radius field (code length).
        r_core: Flat-core radius (code length).
        edge_radius: Outer ejecta radius; may be an angle-dependent field.
        dx: Cell width (sets the taper length).
        envelope_slope: Power-law index n of the outer envelope (rho ∝ r^-n).
        taper_cells: tanh taper width of the ejecta edge, in cells.

    Returns:
        The dimensionless shape function, clipped to [0, 1].
    """
    r_safe = jnp.maximum(r, 0.5 * dx)
    x = r_safe / r_core
    core = jnp.where(r < r_core, x ** (-inner_slope), x ** (-envelope_slope))
    edge = tapered_sphere_weight(r, edge_radius, dx, taper_cells)
    # clip BELOW only. An upper clip at 1 is harmless for a flat core (where
    # the shape never exceeds 1 anyway) but silently flattens any centrally
    # peaked profile, which made ``inner_slope`` a no-op. The amplitude is set
    # by the caller's mass renormalisation, so the shape needs no upper bound.
    return jnp.clip(core * edge, 0.0, None)


def freely_expanding_ejecta(
    helper_data, code_units, box_size, num_cells, *,
    explosion_energy_erg, ejecta_mass_msun, ejecta_radius,
    rho_ambient, p_ambient,
    core_fraction=0.5, envelope_slope=9.0, taper_cells=3.0,
    ejecta_temperature=100.0 * u.K, mass_per_nucleus=1.4,
    clump_field=None, edge_radius_field=None,
):
    """Cold, freely-expanding (homologous) ejecta laid on top of an ambient medium.

    The ejecta density is a flat inner core out to ``core_fraction * ejecta_radius``
    joined to a steep ``rho ∝ r^-envelope_slope`` outer envelope (the standard
    core-collapse ejecta structure; ``envelope_slope`` ~ 9), tanh-tapered at
    ``ejecta_radius``. The velocity is homologous (v ∝ r). The density amplitude
    is renormalised so the ejecta mass equals ``ejecta_mass_msun`` and the
    velocity scale so the kinetic energy equals ``explosion_energy_erg`` -- both
    exactly, independent of resolution and taper. On contact with the ambient the
    ejecta drives a forward shock and (for a steep envelope) a reverse shock.

    Args:
        helper_data: The simulation :class:`HelperData` (for cell centres).
        code_units: The :class:`CodeUnits` in use.
        box_size: Cubic box side (code length).
        num_cells: Cells per axis.
        explosion_energy_erg: Target ejecta kinetic energy (erg).
        ejecta_mass_msun: Target ejecta mass (solar masses).
        ejecta_radius: Outer ejecta radius at t = 0 (code length).
        rho_ambient: Ambient density field (code density), shape (N, N, N).
        p_ambient: Ambient pressure field (code pressure), shape (N, N, N).
        core_fraction: Flat-core radius as a fraction of ``ejecta_radius``.
        envelope_slope: Power-law index n of the outer envelope (rho ∝ r^-n).
        taper_cells: tanh taper width of the ejecta edge, in cells.
        ejecta_temperature: (Cold) ejecta temperature, sets its tiny pressure.
        mass_per_nucleus: Mean gas mass per H nucleus, in m_p (~1.4).
        clump_field: Optional fractional density perturbation (mean ~0), same
            shape as the grid, multiplying the ejecta density as (1 + clump_field)
            to break spherical symmetry and seed Rayleigh-Taylor fingers at the
            reverse shock. The total ejecta mass is still renormalised to the
            target, so clumping redistributes mass without changing M_ej.
        edge_radius_field: Optional angle-dependent outer ejecta radius (same
            shape as the grid) overriding ``ejecta_radius`` in the edge taper --
            e.g. elongated along a cone to model a jet/counter-jet piston (the
            homologous v ∝ r then automatically makes the elongated material
            the fastest). The flat-core radius stays ``core_fraction *
            ejecta_radius``.

    Returns:
        A tuple ``(fields, info)`` where ``fields`` is the primitive-field dict
        for :func:`construct_primitive_state` and ``info`` holds diagnostics
        (achieved mass/energy, max velocity in km/s, dx, ...).
    """
    r, X, Y, Z = centered_radius(helper_data, box_size, num_cells)
    dx = box_size / num_cells
    cell_vol = dx ** 3
    r_safe = jnp.maximum(r, 0.5 * dx)

    E = float((explosion_energy_erg * u.erg).to(code_units.code_energy).value)
    M_ej = float((ejecta_mass_msun * u.Msun).to(code_units.code_mass).value)
    r_core = core_fraction * ejecta_radius

    # dimensionless ejecta density shape: flat core + power-law envelope, tapered
    edge_radius = ejecta_radius if edge_radius_field is None else edge_radius_field
    shape = ejecta_radial_shape(r, r_core, edge_radius, dx,
                                envelope_slope=envelope_slope,
                                taper_cells=taper_cells)
    shape_smooth = shape
    if clump_field is not None:
        # multiplicative clumping; total mass is renormalised below so this only
        # redistributes ejecta mass into denser knots / thinner voids.
        shape = shape * jnp.clip(1.0 + clump_field, 0.0, None)

    # renormalise so injected ejecta mass == M_ej
    d_rho = M_ej / (jnp.sum(shape) * cell_vol)
    m_ej = d_rho * shape                                    # moving ejecta mass density
    rho = rho_ambient + m_ej

    # homologous velocity v_r = s * r on the ejecta mass; renormalise s to KE == E
    # KE = 1/2 s^2 * sum( m_ej^2 r^2 / rho ) dV
    I = jnp.sum(m_ej ** 2 * r ** 2 / rho) * cell_vol
    s = jnp.sqrt(E / (0.5 * I))
    mom_r = m_ej * s * r
    vx = mom_r * X / r_safe / rho
    vy = mom_r * Y / r_safe / rho
    vz = mom_r * Z / r_safe / rho

    # Pressure: cold ejecta blended into the ambient. ISOBARIC perturbations:
    # the pressure follows the SMOOTH (unperturbed) ejecta profile, not the
    # clumped/bubbled density -- so dense knots are cold, rarefied (Ni-heated)
    # bubbles are hot, and small-scale structure sits in pressure equilibrium
    # (what it would relax to anyway). The previous p ∝ rho_perturbed gave
    # bubbles a pressure deficit and walls a surplus (unphysical cold
    # pressure-inflated structure). With no clump_field this is identical to
    # the old behavior.
    rho_per_n = float((mass_per_nucleus * const.m_p / u.cm ** 3).to(code_units.code_density).value)
    p_per_n_ej = float((const.k_B * ejecta_temperature / u.cm ** 3).to(code_units.code_pressure).value)
    rho_smooth = rho_ambient + d_rho * shape_smooth
    p_cold = (rho_smooth / rho_per_n) * p_per_n_ej
    p = p_ambient * (1.0 - shape_smooth) + p_cold * shape_smooth

    fields = dict(density=rho, velocity_x=vx, velocity_y=vy, velocity_z=vz,
                  gas_pressure=p)

    vsq = vx ** 2 + vy ** 2 + vz ** 2
    info = dict(
        dx=dx,
        E_target=E,
        KE_achieved=float(jnp.sum(0.5 * rho * vsq) * cell_vol),
        M_ej_target=M_ej,
        M_ej_achieved=float(jnp.sum(m_ej) * cell_vol),
        v_max_kms=float((float(jnp.sqrt(jnp.max(vsq))) * code_units.code_velocity).to(u.km / u.s).value),
        rho_max=float(jnp.max(rho)),
        r_core=float(r_core),
        cells_across_ejecta=float(ejecta_radius / dx),
    )
    return fields, info


MU_PARTICLE = 0.61              # mean molecular weight per particle (ionized)
MASS_PER_NUCLEUS = 1.4          # mean gas mass per H nucleus, in m_p (cosmic)


def temperature_K(rho, p, code_units):
    """Gas temperature (K) from code-unit density and pressure (ionized, mu=0.61)."""
    p_cgs = (np.asarray(p) * code_units.code_pressure).to(u.erg / u.cm ** 3).value
    rho_cgs = (np.asarray(rho) * code_units.code_density).to(u.g / u.cm ** 3).value
    # NB: use CGS m_p (g); const.m_p.value is SI (kg) and would make T 1000x too low.
    return (MU_PARTICLE * const.m_p.cgs.value / const.k_B.cgs.value) * p_cgs / rho_cgs


def schure_cooling_setup(code_units, floor_temperature_K=1e4,
                         hydrogen_mass_fraction=0.7, metal_mass_fraction=0.02,
                         resolution_limiter_alpha=4.0, explicit=False,
                         max_cooling_fraction=0.0, clamp_to_floor=False):
    """(CoolingConfig, CoolingParams) for the Schure et al. (2009) ISM cooling curve.

    Radiative cooling Lambda(T) applied with the unconditionally-stable implicit
    method. ``floor_temperature_K`` is a physical temperature floor (K) below
    which cooling is switched off (a stand-in for the ISM heating balance).
    ``resolution_limiter_alpha`` suppresses cooling where the cooling length is
    below that many grid cells (see ``CoolingParams.resolution_limiter_alpha``)
    -- the guard against the unresolved-radiative-shock crush runaway.

    ``explicit=True`` switches to the forward update, which brings the cooling
    time into the CFL (see the ``EXPLICIT_COOLING`` branch of the FD timestep
    estimator). That is far slower -- the constraint bites hardest exactly in
    the cells that are crushing -- but it removes the possibility of a single
    backward-Euler step taking a cell from its post-shock temperature to the
    floor, so it is the clean DIAGNOSTIC for whether the crush is an
    operator-splitting artefact or real unresolved physics.
    """
    config = CoolingConfig(
        cooling=True,
        cooling_method=EXPLICIT_COOLING if explicit else IMPLICIT_COOLING,
        cooling_curve_config=CoolingCurveConfig(cooling_curve_type=PIECEWISE_POWER_LAW),
    )
    # The cooling kernel works in the RESCALED temperature T~ = p * mu / rho
    # (code units), not Kelvin: the floor must be converted or the kernel's
    # "stay above the floor" gate (T~_new > floor) rejects every update and
    # silently disables cooling (T~ is O(1e-8) per Kelvin here). T~ / T_K is an
    # exact constant for fixed composition, so calibrate it off any (rho, p).
    from astronomix._modules._cooling._cooling import get_temperature_from_pressure
    tilde_per_kelvin = float(
        get_temperature_from_pressure(
            1.0, 1.0, hydrogen_mass_fraction, metal_mass_fraction)
        / temperature_K(1.0, 1.0, code_units)
    )
    params = CoolingParams(
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        metal_mass_fraction=metal_mass_fraction,
        floor_temperature=float(floor_temperature_K) * tilde_per_kelvin,
        resolution_limiter_alpha=float(resolution_limiter_alpha),
        max_cooling_fraction=float(max_cooling_fraction),
        clamp_to_floor=bool(clamp_to_floor),
        cooling_curve_params=schure_cooling(code_units),
    )
    return config, params


def ism_ti_cooling_setup(code_units, hrate_cgs=5.0e-26, mu_athena=0.618,
                         floor_temperature_K=10.0,
                         hydrogen_mass_fraction=0.7, metal_mass_fraction=0.02,
                         resolution_limiter_alpha=0.0, explicit=False):
    """(CoolingConfig, CoolingParams) matching mainline AthenaK's ISM physics.

    Cooling: AthenaK's exact curve (Koyama-Inutsuka 2002 below log T = 4.2,
    Schure SPEX table to 8.15, CGOLS tail), applied as de/dt = -n_p^2 Lambda
    with the TOTAL particle density n_p = rho / (mu_athena m_u). Heating: the
    constant per-particle rate de/dt = +n_p * hrate (the ISM photoelectric
    stand-in; Guo-Kim-Stone use hrate = 5e-26 erg/s). Together they carry the
    classic two-phase thermal instability, so ``floor_temperature_K`` sits far
    below the ~180 K cold branch instead of standing in for the heating.
    """
    from astronomix._modules._cooling._cooling_tables import athenak_ism_cooling
    from astronomix._modules._cooling._cooling import (
        get_effective_molecular_weights,
        get_temperature_from_pressure,
    )
    # ``explicit=True`` replicates AthenaK exactly: a single forward-Euler
    # source evaluation per stage, with the time step limited to the local
    # thermal time min(T/|dT/dt|) (see the estimator). Far cheaper than the
    # implicit fixed point (~33 curve evaluations per call) and, because the
    # net rate vanishes at the two-phase equilibrium, essentially free in dt
    # for this problem.
    config = CoolingConfig(
        cooling=True,
        cooling_method=EXPLICIT_COOLING if explicit else IMPLICIT_COOLING,
        cooling_curve_config=CoolingCurveConfig(cooling_curve_type=PIECEWISE_POWER_LAW),
    )
    curve = athenak_ism_cooling(
        code_units, hydrogen_mass_fraction, metal_mass_fraction, mu_athena)
    mu_cool, _, _ = get_effective_molecular_weights(
        hydrogen_mass_fraction, metal_mass_fraction)
    # effective T~ heating constant: dT~/dt = (gamma-1) * heating_rate with
    # heating_rate = mu_cool * [n_p Gamma for rho_code = 1] in code units
    heating_eff = mu_cool * float((
        (1.0 * code_units.code_density / (mu_athena * const.u))
        * (hrate_cgs * u.erg / u.s)
    ).to(code_units.code_pressure / code_units.code_time).value)
    # Kelvin -> kernel T~ with the SAME map the athenak_ism_cooling table
    # uses (T~ = T_K * k_B * mu_cool / (mu_athena * m_u) in code p/rho units)
    tilde_per_kelvin = float((
        1.0 * u.K * const.k_B * mu_cool / (mu_athena * const.u)
    ).to(code_units.code_pressure / code_units.code_density).value)
    params = CoolingParams(
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        metal_mass_fraction=metal_mass_fraction,
        floor_temperature=float(floor_temperature_K) * tilde_per_kelvin,
        resolution_limiter_alpha=float(resolution_limiter_alpha),
        heating_rate=heating_eff,
        cooling_curve_params=curve,
    )
    return config, params


def turbulent_field(num_cells, key, kmin=4, kmax=16, slope=-1.0):
    """A zero-mean, unit-standard-deviation band-limited random field (N, N, N).

    Thin wrapper over ``create_turb_field`` used to seed turbulent velocity or
    (fractional / log-normal) density perturbations in an initial condition.
    Power is carried in wavenumbers ``kmin..kmax`` with amplitude ~ k^slope. The
    caller scales it to the desired rms / fluctuation amplitude.
    """
    f = create_turb_field(num_cells, A0=1.0, slope=slope, kmin=kmin, kmax=kmax, key=key)
    f = f - jnp.mean(f)
    return f / (jnp.std(f) + 1e-30)


def nickel_bubble_field(X, Y, Z, key, ejecta_radius, n_bubbles=5,
                        bubble_radius_frac=(0.15, 0.35), center_max_frac=0.6,
                        depth=0.5, wall_boost=0.5, wall_width_frac=0.3):
    """Multiplicative ejecta-density field for radioactive Ni-bubble structure.

    In real core-collapse ejecta the heating from freshly synthesised ⁵⁶Ni
    inflates low-density bubbles whose compressed walls later light up as the
    RING-shaped ejecta emission seen across Cas A's interior (Milisavljevic &
    Fesen 2013; Cas A's pristine-debris cavities in the 2024 Webb analysis).
    This is a parameterised stand-in: ``n_bubbles`` spheres at random centres
    within ``center_max_frac * ejecta_radius``, each evacuated by ``depth``
    inside and wrapped in a compressed wall (Gaussian shell of relative
    amplitude ``wall_boost`` and width ``wall_width_frac`` of the bubble
    radius). The caller multiplies this onto the ejecta density shape; the
    ejecta mass renormalisation makes it a pure redistribution.

    Defaults are deliberately GENTLE: with sharp walls (depth 0.7, boost 1.2,
    width 0.18) the 512^3 run blew up at t=0.032 when the reverse shock met
    the first fully-resolved dense cold wall (radiative-crush runaway at
    r~1.07 pc), while 256^3 -- where the same walls are grid-smeared -- was
    stable. Softer, wider walls keep the ring morphology without handing the
    reverse shock a cell-sharp cold sheet.
    """
    keys = jax.random.split(key, 3)
    lo, hi = bubble_radius_frac
    centers = (jax.random.ball(keys[0], d=3, shape=(n_bubbles,))
               * center_max_frac * ejecta_radius)
    radii = (jax.random.uniform(keys[1], (n_bubbles,)) * (hi - lo) + lo) * ejecta_radius

    m = jnp.ones_like(X)
    for i in range(n_bubbles):
        d = jnp.sqrt((X - centers[i, 0]) ** 2 + (Y - centers[i, 1]) ** 2
                     + (Z - centers[i, 2]) ** 2)
        w = wall_width_frac * radii[i]
        inside = 0.5 * (1.0 - jnp.tanh((d - radii[i]) / w))
        wall = jnp.exp(-0.5 * ((d - radii[i]) / w) ** 2)
        m = m * (1.0 - depth * inside) * (1.0 + wall_boost * wall)
    return m


def dense_csm_shell(r, X, Y, Z, *, shell_radius, shell_thickness, peak_number_density,
                    rho_per_n, asymmetry=0.0):
    """Additive dense circumstellar shell (Gaussian in radius), optionally lopsided.

    Models the dense circumstellar shell around Cas A (Orlando et al. 2025's
    "Green Monster": n ~ 180 cm^-3 at ~1.5-1.9 pc, from a pre-SN eruptive
    mass-loss event). Returns the code-density to ADD to the ambient.

    Args:
        r, X, Y, Z: radius and centered coordinate fields (code length).
        shell_radius: shell centre radius (code length).
        shell_thickness: Gaussian sigma of the shell (code length).
        peak_number_density: peak shell number density (cm^-3).
        rho_per_n: code density per cm^-3 (from the ambient composition).
        asymmetry: 0 = spherical; up to 1 = fully one-sided. The shell amplitude
            is modulated by (1 + asymmetry * x/r), denser toward +x (the Cas A
            shell is markedly asymmetric; +x keeps the lopsidedness visible in the
            x-y slices / z-projection).

    Returns:
        The shell density field to add to the ambient (code density).
    """
    radial = jnp.exp(-0.5 * ((r - shell_radius) / shell_thickness) ** 2)
    lopsided = 1.0 + asymmetry * (X / jnp.maximum(r, 1e-6))
    lopsided = jnp.clip(lopsided, 0.0, None)
    return peak_number_density * rho_per_n * radial * lopsided


# -----------------------------------------------------------------------------
# Orlando et al. Route B: a calibrated 1D profile mapped into 3D, with the
# multi-D structure imposed at the mapping time.
# -----------------------------------------------------------------------------
def map_1d_profile(r3d, profile_npz, *, floor_density=None):
    """Interpolate a 1D spherical radial profile onto the 3D radius field.

    This is the mapping step of Orlando et al. (2016)'s Route B: the expensive,
    high-dynamic-range early evolution (free expansion, reverse-shock formation,
    the first ~150 yr of deceleration against the progenitor wind) is done once
    in 1D by ``casa_calibrate_1d.py``, calibrated against the observed shock
    radii and speeds, and the resulting radial profile is then laid down on the
    3D grid -- where the multi-D structure is imposed and the rest of the
    evolution is computed.

    Density and pressure are interpolated in the log (they span many decades
    across the shocks), the radial velocity linearly. Outside the 1D domain the
    outermost value is held; the caller normally overwrites the pre-shock region
    with a full 3D circumstellar model anyway.

    Args:
        r3d: The 3D radius field (code length).
        profile_npz: Path to (or an open ``NpzFile`` of) a
            ``casa_calibrate_1d.py --save-profile`` output.
        floor_density: Optional lower clip on the interpolated density.

    Returns:
        ``(rho, v_r, p, meta)`` -- three fields with the shape of ``r3d`` (code
        units) plus the profile's metadata dict (age, shock radii, the
        calibrated explosion/wind parameters).
    """
    data = profile_npz if hasattr(profile_npz, "files") else np.load(profile_npz)
    r1 = np.asarray(data["r"], dtype=np.float64)
    rho1 = np.asarray(data["rho"], dtype=np.float64)
    v1 = np.asarray(data["v"], dtype=np.float64)
    p1 = np.asarray(data["press"], dtype=np.float64)

    rr = jnp.clip(r3d, r1[0], r1[-1])
    tiny = 1e-300
    rho = jnp.exp(jnp.interp(rr, r1, jnp.asarray(np.log(np.maximum(rho1, tiny)))))
    p = jnp.exp(jnp.interp(rr, r1, jnp.asarray(np.log(np.maximum(p1, tiny)))))
    v_r = jnp.interp(rr, r1, jnp.asarray(v1))
    if floor_density is not None:
        rho = jnp.maximum(rho, floor_density)

    meta = {k: float(data[k]) for k in data.files
            if data[k].ndim == 0 and k not in ("r", "rho", "v", "press")}
    return rho, v_r, p, meta


#: Chemical stratification of a Type IIb ejecta, inside out, as masses in
#: M_sun for a 3 M_sun ejecta. NOTE the smooth Fe layer is deliberately small:
#: Orlando et al.'s Table-4 knots are drawn FROM the Fe core, not added on top
#: of it, and they carry 0.2015 M_sun between them, so the smooth layer is the
#: remainder. Putting a full Fe layer underneath them would double-count. Cas A's light echoes make it a Type IIb (Krause
#: et al. 2008), so the hydrogen envelope is nearly stripped; the oxygen layer
#: carries most of the mass, which is why Cas A's X-ray ejecta emission is
#: O/Si/S/Fe-dominated. Composition is assigned by the enclosed-ejecta-mass
#: coordinate rather than by radius, which is the Lagrangian label a 1D
#: spherical flow preserves exactly.
IIB_LAYERS = (
    # species, mass [Msun] for M_ej = 3.0
    ("Fe", 0.02),      # see below: most of the Fe is in the Table-4 knots
    ("Si", 0.24),      # Si + S; Hwang & Laming 2012 find only 0.08 SHOCKED
    ("O", 1.71),       # O/Ne/Mg -- the bulk of the ejecta mass
    ("He", 0.93),
    ("H", 0.10),       # the stripped IIb envelope
)

#: Circumstellar (progenitor wind) composition: cosmic abundances.
CSM_COMPOSITION = {"H": 0.70, "He": 0.28, "O": 0.01, "Si": 0.005, "Fe": 0.005}


def enclosed_mass_profile(profile_npz):
    """Cumulative mass ``M(<r)`` of a 1D spherical profile, in code mass units."""
    data = profile_npz if hasattr(profile_npz, "files") else np.load(profile_npz)
    r = np.asarray(data["r"], dtype=np.float64)
    rho = np.asarray(data["rho"], dtype=np.float64)
    dr = np.gradient(r)
    shell_mass = 4.0 * np.pi * r ** 2 * rho * dr
    return r, np.cumsum(shell_mass)


def ejecta_mass_coordinate(r3d, profile_npz, ejecta_mass, interior_wind_mass=0.0):
    """Enclosed-ejecta-mass coordinate ``m`` in [0, 1] and the ejecta fraction.

    In a 1D spherical flow the Lagrangian ordering is preserved exactly, so the
    enclosed mass ``M(<r)`` *is* a material label: a parcel that started at
    ejecta mass fraction ``m`` is still at ``m`` at the mapping time, however
    much the profile has been reshaped by the reverse shock. That makes the
    chemical stratification assignable without any composition tracer in the 1D
    stage — and it locates the contact discontinuity exactly, as the radius
    enclosing the whole ejecta mass.

    Args:
        r3d: The 3D radius field (code length).
        profile_npz: The 1D profile (see :func:`map_1d_profile`).
        ejecta_mass: The ejecta mass in code mass units.
        interior_wind_mass: Circumstellar mass that was already inside the
            initial ejecta radius, and is therefore mixed through the ejecta in
            the enclosed-mass ordering rather than sitting below it (a few per
            cent; it only shifts the contact discontinuity).

    Returns:
        ``(m, ejecta_fraction, r_cd)`` -- the mass coordinate (clipped to [0, 1]
        outside the ejecta), a smooth 1/0 ejecta-versus-circumstellar indicator,
        and the contact-discontinuity radius in code length.
    """
    r1, m_enc = enclosed_mass_profile(profile_npz)
    m_cd = float(ejecta_mass + interior_wind_mass)
    # the contact discontinuity: the radius enclosing all the ejecta
    idx = int(np.searchsorted(m_enc, m_cd))
    r_cd = float(r1[min(idx, len(r1) - 1)])

    m_of_r = np.clip(m_enc / m_cd, 0.0, 1.0)
    rr = jnp.clip(r3d, r1[0], r1[-1])
    m = jnp.interp(rr, jnp.asarray(r1), jnp.asarray(m_of_r))

    dr = float(r1[1] - r1[0])
    ejecta_fraction = 0.5 * (1.0 - jnp.tanh((r3d - r_cd) / (2.0 * dr)))
    return m, ejecta_fraction, r_cd


def layered_composition(m, ejecta_fraction, layers=IIB_LAYERS,
                        csm=CSM_COMPOSITION, smoothing=0.15):
    """Per-species mass fractions from the ejecta mass coordinate.

    The layers are laid down inside out in cumulative mass fraction and blended
    rather than stacked as sharp shells: real supernova ejecta layers overlap,
    and a sharp jump in a passive scalar is a discontinuity the advection scheme
    then has to carry for the whole run for no physical reason.

    ``smoothing`` is a fraction of EACH LAYER'S OWN WIDTH, not an absolute width
    in mass coordinate. That distinction is not cosmetic: with a constant
    absolute blend the thin inner layers are smeared far beyond themselves, and
    a 0.04 blend against an Fe layer only 0.10/3.0 = 0.033 wide put **8x too
    much Fe** into the initial condition (Si 2x, while O — 16x wider — came out
    right at 1.02x). The measured shocked-Fe mass is the diagnostic this feeds,
    so the error went straight into the comparison with Hwang & Laming.

    Args:
        m: Enclosed-ejecta-mass coordinate in [0, 1].
        ejecta_fraction: 1 in the ejecta, 0 in the circumstellar medium.
        layers: ``(species, mass)`` inside out.
        csm: Circumstellar mass fractions per species.
        smoothing: Blend width as a fraction of each layer's own width.

    Returns:
        ``dict`` of species -> mass-fraction field, normalised to sum to 1.
    """
    total = sum(mass for _, mass in layers)
    edges, acc = [], 0.0
    for _, mass in layers:
        acc += mass / total
        edges.append(acc)

    out = {}
    lo = 0.0
    for (species, _), hi in zip(layers, edges):
        # a smooth top-hat in mass coordinate, blended over a fraction of THIS
        # layer's width so a thin inner layer is not smeared beyond itself
        w = max(smoothing * (hi - lo), 1e-4)
        window = (0.5 * (1.0 + jnp.tanh((m - lo) / w))
                  * 0.5 * (1.0 - jnp.tanh((m - hi) / w)))
        ej = window
        out[species] = ejecta_fraction * ej + (1.0 - ejecta_fraction) * csm.get(species, 0.0)
        lo = hi

    # renormalise so the mass fractions sum to one everywhere
    norm = sum(out.values())
    norm = jnp.where(norm > 1e-12, norm, 1.0)
    return {k: v / norm for k, v in out.items()}


def prior_shock_history(r3d, profile_npz, density, *, map_age, code_units,
                        mass_coordinate=None, ejecta_fraction=None):
    """Seed the shock history for material already shocked at the mapping time.

    The 3D run starts at ~150 yr, by which point the reverse shock has already
    swept most of the ejecta and the forward shock a comparable amount of wind.
    If the shock-history scalars start at zero there, the ionization age
    ``n_e t`` restarts from the mapping time and is under-counted by the whole
    pre-mapping history — up to ~40 % for the earliest-shocked material, which
    is exactly the material Hwang & Laming find at the most advanced ionization
    age. So the history is seeded from the 1D run's own shock trajectories.

    A parcel is taken to have been shocked when the relevant shock passed its
    *current* radius, inverting the saved ``r_FS(t)`` / ``r_RS(t)`` tables. That
    is essentially exact for the shocked circumstellar gas, which barely moves
    before the blast reaches it; it is an approximation for the shocked ejecta,
    which does move, and there it errs by assuming the parcel has stayed with
    the shock. ``density_time`` is then estimated as ``rho * Delta t`` at the
    present density rather than as the true integral, which is fair because the
    post-shock density is roughly constant behind a strong shock.

    Args:
        r3d: The 3D radius field (code length).
        profile_npz: The 1D profile, which must carry ``history_*`` arrays.
        density: The mapped 3D density field (code units).
        map_age: The age (yr) at which the profile is being mapped.
        code_units: The :class:`CodeUnits` in use.
        mass_coordinate: The enclosed-ejecta-mass coordinate in [0, 1] (from
            :func:`ejecta_mass_coordinate`), used to invert the reverse shock's
            Lagrangian position for the shocked ejecta.
        ejecta_fraction: 1 in the ejecta, 0 in the circumstellar medium, which
            selects which shock's trajectory to invert.

    Returns:
        ``(shocked_fraction, time_since_shock, density_time)`` in code units, or
        ``(None, None, None)`` if the profile predates the history being saved.
    """
    data = profile_npz if hasattr(profile_npz, "files") else np.load(profile_npz)
    if "history_age_yr" not in data:
        return None, None, None

    age = np.asarray(data["history_age_yr"], dtype=np.float64)
    r_fs = np.asarray(data["history_r_fs"], dtype=np.float64)
    r_rs = np.asarray(data["history_r_rs"], dtype=np.float64)
    m_rs = (np.asarray(data["history_m_rs"], dtype=np.float64)
            if "history_m_rs" in data else np.full_like(age, np.nan))
    ok = np.isfinite(r_fs) & np.isfinite(r_rs) & (age <= map_age + 1e-9)
    age, r_fs, r_rs, m_rs = age[ok], r_fs[ok], r_rs[ok], m_rs[ok]
    if age.size < 2:
        return None, None, None
    r_fs_now, r_rs_now = r_fs[-1], r_rs[-1]

    # Circumstellar gas barely moves before the blast reaches it, so inverting
    # the forward shock's RADIUS at the parcel's present radius is essentially
    # exact for it.
    t_cross_fs = jnp.interp(r3d, jnp.asarray(r_fs), jnp.asarray(age),
                            left=np.nan, right=np.nan)

    # Shocked ejecta is a different matter: it has moved a long way since the
    # reverse shock swept it, and its present radius says nothing about where it
    # was then. Its Lagrangian label is the enclosed mass, so invert the reverse
    # shock's MASS coordinate instead. (Both m_RS and the parcel's own m are
    # monotone, so this is a well-posed inversion.)
    # ``m_rs`` is an ABSOLUTE code mass while ``mass_coordinate`` is normalised
    # to the whole ejecta, so the parcel's label must be denormalised before the
    # two can be compared.
    m_total = float(data["M_ej"]) + float(data.get("M_wind_inside_r0", 0.0)) \
        if "M_ej" in data else np.nan
    t_cross_rs = jnp.full_like(r3d, jnp.nan)
    shocked_ejecta = jnp.zeros_like(r3d, dtype=bool)
    if (mass_coordinate is not None and np.all(np.isfinite(m_rs))
            and np.isfinite(m_total) and m_total > 0):
        m_target = mass_coordinate * m_total
        # the reverse shock eats INWARD in mass, so m_RS decreases with age;
        # sort into increasing mass for the interpolation
        order = np.argsort(m_rs)
        t_cross_rs = jnp.interp(m_target, jnp.asarray(m_rs[order]),
                                jnp.asarray(age[order]), left=np.nan, right=np.nan)
        # ejecta outside the reverse shock in MASS has been swept
        shocked_ejecta = m_target > float(m_rs[-1])

    is_ejecta = (ejecta_fraction > 0.5 if ejecta_fraction is not None
                 else jnp.zeros_like(r3d, dtype=bool))
    t_shock = jnp.where(is_ejecta, t_cross_rs, t_cross_fs)
    t_shock = jnp.where(jnp.isfinite(t_shock), t_shock, map_age)

    shocked = jnp.where(is_ejecta,
                        shocked_ejecta,
                        (r3d <= r_fs_now) & (r3d >= r_rs_now))
    dt_yr = jnp.where(shocked, jnp.clip(map_age - t_shock, 0.0, map_age), 0.0)
    dt_code = float((1.0 * u.yr).to(code_units.code_time).value) * dt_yr
    # the shocked FRACTION is what latches the parcel as already shocked
    f_shocked = jnp.where(shocked, 1.0, 0.0)
    # rho * Delta t rather than the true integral of rho dt: fair because the
    # post-shock density is roughly constant behind a strong shock
    return f_shocked, dt_code, density * dt_code


def orlando_csm_shell(r, X, Y, Z, *, shell_radius, thickness, peak_number_density,
                      theta_deg, phi_deg, scale_length, rho_per_n,
                      min_thickness=None):
    """The asymmetric circumstellar shell of Orlando et al. (2022), their Eq. 1.

    ``n_sh * exp[-(r - r_sh)^2 / (2 sigma^2)] * exp[(r.D) / H]`` with the
    direction cosine ``r.D = x cos(theta) cos(phi) - y sin(phi)
    + z sin(theta) cos(phi)``, the Earth vantage point on the -y axis, theta
    measured in the plane of the sky counterclockwise from west and phi
    counterclockwise from the plane of the sky. Interpreted as the relic of an
    eruptive mass-loss event 1e4-1e5 yr before collapse; hitting it is what
    drives a reflected shock back into the ejecta and makes Cas A's reverse
    shock move *inward* in the west by 350 yr.

    The favoured parameters of model ``W15-IIb-sh-HD-1eta-az`` are
    ``n_sh = 20 cm^-3``, ``r_sh = 1.5 pc``, ``sigma = 0.02 pc``, ``theta = 30``,
    ``phi = 50``, ``H = 0.7 pc``. Note that sigma = 0.02 pc is thinner than the
    cell of any whole-remnant grid used here (0.0137 pc at 512^3 in a 7 pc box),
    so ``min_thickness`` broadens the shell to a resolvable width **at fixed
    surface density** (``n_sh * sigma`` held constant). That preserves the
    column the blast wave runs into -- which is what sets the deceleration and
    the reflected shock -- while making the shell representable on the grid; the
    unresolved-shell caveat then applies to its fragmentation, not its dynamics.

    Args:
        r, X, Y, Z: radius and centered coordinate fields (code length).
        shell_radius: shell centre radius r_sh (code length).
        thickness: Gaussian sigma of the shell (code length).
        peak_number_density: n_sh (cm^-3), before the angular modulation.
        theta_deg, phi_deg: the two orientation angles (degrees).
        scale_length: H, the density scale length of the gradient (code length).
        rho_per_n: code density per cm^-3.
        min_thickness: if given and larger than ``thickness``, broaden the shell
            to this width and reduce ``n_sh`` by the same factor.

    Returns:
        ``(rho_shell, info)`` -- the code-density field to ADD to the wind, and
        a dict with the effective thickness/amplitude and the resulting peak
        densities along the two extremes of the gradient.
    """
    sigma, n_sh = thickness, peak_number_density
    if min_thickness is not None and min_thickness > sigma:
        n_sh = n_sh * sigma / min_thickness      # hold n_sh * sigma (the column)
        sigma = min_thickness

    th, ph = jnp.deg2rad(theta_deg), jnp.deg2rad(phi_deg)
    r_dot_D = (X * jnp.cos(th) * jnp.cos(ph) - Y * jnp.sin(ph)
               + Z * jnp.sin(th) * jnp.cos(ph))
    radial = jnp.exp(-0.5 * ((r - shell_radius) / sigma) ** 2)
    gradient = jnp.exp(r_dot_D / scale_length)
    rho_shell = n_sh * rho_per_n * radial * gradient

    info = dict(
        thickness=float(sigma), peak_number_density=float(n_sh),
        column_preserved=float(n_sh * sigma),
        n_peak_near=float(n_sh * np.exp(shell_radius / scale_length)),
        n_peak_far=float(n_sh * np.exp(-shell_radius / scale_length)),
    )
    return rho_shell, info


#: The five large-scale ejecta anisotropies ("pistons") of Orlando et al.
#: (2016), their Table 4 -- the configuration that reproduces Cas A's Fe and
#: Si/S morphology. Lengths are in units of the remnant radius at the mapping
#: time; ``chi_n`` and ``chi_v`` are the density and velocity contrasts.
#: Without these, essentially no Fe is shocked by 350 yr: small-scale clumping
#: alone cannot do it. The knot DIRECTIONS are a fitted orientation (Orlando
#: rotates the remnant so the Fe fingers point where Cas A's do), so they are
#: listed explicitly and are meant to be adjusted, not derived.
ORLANDO_PISTONS = (
    # name,        direction (x, y, z),   D_knot, r_knot, chi_n, chi_v, M_knot, species
    ("Fe-rich SE", (0.79, -0.35, -0.50),   0.15,   0.05,  100.0,  4.2,  0.100,  "Fe"),
    ("Fe-rich SW", (-0.66, -0.38, -0.65),  0.15,   0.02,   50.0,  4.2,  0.0015, "Fe"),
    ("Fe-rich NW", (-0.42, 0.30, 0.86),    0.15,   0.06,   50.0,  4.2,  0.100,  "Fe"),
    ("Si-rich NE", (-0.36, 0.36, 0.86),    0.35,   0.10,    5.0,  3.0,  0.040,  "Si"),
    ("Si-rich SW", (0.36, -0.36, -0.86),   0.35,   0.10,    1.2,  3.0,  0.0091, "Si"),
)


def orlando_piston_fields(X, Y, Z, remnant_radius, density, cell_volume,
                          pistons=ORLANDO_PISTONS, smoothing_cells=2.0, dx=None):
    """Density, velocity and composition of the Orlando et al. (2016) pistons.

    Each piston is an overdense sphere in pressure equilibrium with the
    surrounding ejecta, at ``D_knot * remnant_radius`` from the centre with
    radius ``r_knot * remnant_radius``.

    **Parametrised by MASS, not by density contrast.** Orlando's Table 4 quotes
    both, but the density contrast ``chi_n`` refers to the ejecta density at
    THEIR mapping time (~1 day after the explosion). Applying the same contrast
    at 150 yr, when the core has expanded by five orders of magnitude in volume
    and is rarefied, produces a knot with a negligible fraction of the intended
    mass — the five knots together came to 0.067 M_sun against Orlando's 0.25.
    The knot MASS is epoch-independent, so it is what is imposed here and the
    amplitude is solved for.

    **The knots carry their own composition.** This is the point of them: Cas A
    has ~0.14 M_sun of SHOCKED Fe but only ~0.08 M_sun of shocked Si+S, while a
    spherically stratified ejecta puts Fe deepest and therefore shocks it LAST.
    The observed ordering is inverted, and Orlando et al. show it is the
    large-scale anisotropies that do it: knots inside the Fe core with
    ``chi_v > 2.5`` punch Fe out through the Si/S layer to meet the reverse
    shock early. A piston that only multiplies the density inherits whatever
    composition the smooth stratification has at its location, and so cannot
    produce that inversion at all.

    Args:
        X, Y, Z: centered coordinate fields (code length).
        remnant_radius: the radius the ``D_knot``/``r_knot`` fractions refer to
            (the forward-shock radius at the mapping time), code length.
        density: the mapped density field, for solving the knot amplitude.
        cell_volume: cell volume in code units.
        pistons: the piston table (see :data:`ORLANDO_PISTONS`).
        smoothing_cells: tanh edge width of each knot, in cells.
        dx: cell width (code length), for the edge smoothing.

    Returns:
        ``(density_multiplier, velocity_multiplier, species_weight, info)``.
        ``species_weight`` maps each species to the added-mass-weighted field
        that the caller blends into the composition.
    """
    dens = jnp.ones_like(X)
    velo = jnp.ones_like(X)
    species_weight = {}
    info = []
    edge = (smoothing_cells * dx) if dx is not None else (0.05 * remnant_radius)
    for name, direction, d_knot, r_tab, chi_n, chi_v, m_knot, species in pistons:
        d = jnp.asarray(direction, dtype=X.dtype)
        d = d / jnp.sqrt(jnp.sum(d ** 2))
        cx, cy, cz = (d * d_knot * remnant_radius)
        dist = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)

        # Orlando's table fixes M_knot, chi_n AND r_knot together, but only at
        # THEIR epoch (~1 day). By the mapping time the reverse shock has
        # evacuated the interior, so the three are no longer mutually
        # consistent and one has to give. The MASS and the CONTRAST are the
        # physically meaningful pair -- mass is what Hwang & Laming measure,
        # contrast is what sets whether the knot penetrates -- so the RADIUS is
        # solved for. Imposing the tabulated radius instead forces a contrast of
        # ~1e3 rather than 1e2 and the run aborts on a dt collapse at t = 0.032,
        # which is the same cold-dense-clump crush this setup has hit before.
        rho_amb = float(jnp.mean(jnp.where(
            dist < 0.5 * d_knot * remnant_radius, density, 0.0)) * 0.0
            + jnp.sum(jnp.where(dist < 0.3 * remnant_radius, density, 0.0))
            / jnp.maximum(jnp.sum(jnp.where(dist < 0.3 * remnant_radius, 1.0, 0.0)), 1.0))
        r_solve = (3.0 * m_knot / (4.0 * np.pi * max(rho_amb, 1e-30)
                                   * max(chi_n - 1.0, 1e-3))) ** (1.0 / 3.0)
        # NOT capped. Capping the radius at twice the tabulated value was tried
        # -- the low-contrast SW Si knot otherwise solves to 0.33 R_SNR, which
        # is a global asymmetry rather than a knot -- but holding the mass
        # inside a smaller radius forces the CONTRAST up, and the run then
        # aborted on a dt collapse at t = 0.013, EARLIER than the uncapped
        # version which completes. Contrast is what destabilises this setup, so
        # the honest trade is to let the radius grow and say so: these knots are
        # larger and gentler than Orlando's because the interior they are placed
        # into has been evacuated by the reverse shock in a way theirs had not.
        w = 0.5 * (1.0 - jnp.tanh((dist - r_solve) / edge))
        r_knot = r_solve / remnant_radius

        base = float(jnp.sum(w * density) * cell_volume)
        amp = m_knot / max(base, 1e-30)
        dens = dens + amp * w
        velo = velo * (1.0 + (chi_v - 1.0) * w)
        added = amp * w * density
        species_weight[species] = species_weight.get(species, 0.0) + added

        chi_eff = 1.0 + amp * float(jnp.max(w)) if base > 0 else chi_n
        info.append(dict(name=name, center=(float(cx), float(cy), float(cz)),
                         radius=float(r_solve), chi_v=chi_v, chi_n=chi_n,
                         chi_eff=float(amp / max(rho_amb, 1e-30) + 1.0),

                         m_knot=m_knot, species=species, r_frac=float(r_knot),
                         cells_across=float(2 * r_solve / dx) if dx else np.nan))
    return dens, velo, species_weight, info


# =============================================================================
# ============ ↑ Initial-condition builders ↑ =================================
# =============================================================================


# =============================================================================
# ============ ↓ Shared figure (slices + radial profiles) ↓ ===================
# =============================================================================
def snr_figure(rho, T, r, vr_kms, box_size, title, out_path,
               rho_cmap="cividis"):
    """2x2 SNR panel: density & temperature z-slices + radial density & velocity.

    Args:
        rho, T: 3D density (code) and temperature (K) fields.
        r: 3D radius-from-centre field (code length), same shape.
        vr_kms: 3D radial-velocity field (km/s).
        box_size: cubic box side (code length) for the slice extent.
        title: figure suptitle.
        out_path: output PNG path.
        rho_cmap: colormap for the density slice.
    """
    rho = np.asarray(rho); T = np.asarray(T); r = np.asarray(r); vr_kms = np.asarray(vr_kms)
    n = rho.shape[-1]
    zc = n // 2
    ext = [-box_size / 2, box_size / 2, -box_size / 2, box_size / 2]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    im0 = axes[0, 0].imshow(rho[:, :, zc].T, origin="lower", extent=ext,
                            norm=LogNorm(), cmap=rho_cmap)
    axes[0, 0].set_title("density (z-midplane)")
    fig.colorbar(im0, ax=axes[0, 0], label=r"$\rho$ [code]")

    im1 = axes[0, 1].imshow(T[:, :, zc].T, origin="lower", extent=ext,
                            norm=LogNorm(vmin=max(1e2, np.nanmin(T[T > 0]))), cmap="inferno")
    axes[0, 1].set_title("temperature (z-midplane)")
    fig.colorbar(im1, ax=axes[0, 1], label="T [K]")
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlabel("x [pc]")
        ax.set_ylabel("y [pc]")

    rf = r.flatten()
    nbin = 120
    bins = np.linspace(0, rf.max(), nbin + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.searchsorted(bins, rf, "right") - 1, 0, nbin - 1)
    raw_cnt = np.bincount(idx, minlength=nbin)
    cnt = np.maximum(raw_cnt, 1)
    empty = raw_cnt == 0                              # mask empty inner bins (avoid spikes)
    mean_rho = np.bincount(idx, weights=rho.flatten(), minlength=nbin) / cnt
    mean_vr = np.bincount(idx, weights=vr_kms.flatten(), minlength=nbin) / cnt
    mean_rho[empty] = np.nan
    mean_vr[empty] = np.nan
    sub = np.arange(0, rf.size, max(1, rf.size // 40000))

    axes[1, 0].scatter(rf[sub], rho.flatten()[sub], s=1, color="lightgray",
                       alpha=0.5, rasterized=True)
    axes[1, 0].plot(bc, mean_rho, color="C0", lw=1.8)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("r [pc]")
    axes[1, 0].set_ylabel(r"$\rho$ [code]")
    axes[1, 0].set_title("radial density (forward shock / shell / reverse shock / core)")
    axes[1, 0].set_xlim(0, box_size / 2)

    axes[1, 1].scatter(rf[sub], vr_kms.flatten()[sub], s=1, color="lightgray",
                       alpha=0.5, rasterized=True)
    axes[1, 1].plot(bc, mean_vr, color="C3", lw=1.8)
    axes[1, 1].set_xlabel("r [pc]")
    axes[1, 1].set_ylabel(r"$v_r$ [km/s]")
    axes[1, 1].set_title("radial velocity")
    axes[1, 1].set_xlim(0, box_size / 2)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# =============================================================================
# ============ ↓ Chandra-like X-ray synthesis ↓ ===============================
# =============================================================================
# Photon-energy bands, keyed to plasma temperature as a proxy for the emitted
# spectrum (kT ~ the band energy). Roughly the Chandra soft/medium/hard split.
XRAY_BANDS_K = {
    "soft":   (3.0e6, 1.2e7),    # ~0.26-1.0 keV
    "medium": (1.2e7, 4.0e7),    # ~1.0-3.4 keV
    "hard":   (4.0e7, 1.0e10),   # >3.4 keV
}


def xray_band_emissivity(rho, p, code_units, mass_per_nucleus=1.4):
    """Per-cell optically-thin X-ray emissivity in the soft/medium/hard bands.

    Uses a thermal-bremsstrahlung continuum proxy, emissivity ∝ n_e n_H sqrt(T),
    with each cell assigned to the band matching its temperature (kT ~ band).
    This is a visualisation proxy (no spectral model / line emission), enough for
    a Chandra-like 3-colour composite where brightness ∝ emission measure.

    Returns:
        ``(bands, T)`` with ``bands`` a dict of per-cell emissivity fields
        (arbitrary units) and ``T`` the temperature field (K).
    """
    T = temperature_K(rho, p, code_units)
    rho_cgs = (np.asarray(rho) * code_units.code_density).to(u.g / u.cm ** 3).value
    n_H = rho_cgs / (mass_per_nucleus * const.m_p.cgs.value)
    n_e = 1.2 * n_H                                   # fully ionized H+He
    emission_measure = n_e * n_H
    brems = np.sqrt(np.maximum(T, 0.0))
    bands = {}
    for name, (lo, hi) in XRAY_BANDS_K.items():
        bands[name] = emission_measure * brems * ((T >= lo) & (T < hi))
    return bands, T


def synchrotron_emissivity(rho, p, code_units, shock_threshold=0.3):
    """Per-cell nonthermal (synchrotron) X-ray emissivity proxy.

    Cas A's highest-energy X-rays are not hot thermal gas: they are produced by
    relativistic electrons accelerated at strong shocks, lighting up as THIN
    ARCS at the blast wave (and at shock filaments in the interior) — the
    "power-law component" that Vink et al. (2024) also find in the Green
    Monster spectra. The proxy localises emission at shocks with a
    pressure-jump detector (cell-scale ``|grad p| / p`` above
    ``shock_threshold``, which is razor-thin because the WENO scheme captures
    shocks over ~2 cells) and scales it with the post-shock pressure (a stand-in
    for the shock ram pressure that sets the accelerated-electron + amplified-B
    energy density). A visualisation proxy — no electron transport, cutoffs or
    magnetic field model.
    """
    p = np.asarray(p)
    grads = np.gradient(p)
    gmag = np.sqrt(sum(g * g for g in grads))     # |grad p| per cell width
    jump = gmag / np.maximum(p, 1e-30)
    shock = np.clip(jump - shock_threshold, 0.0, None)
    return shock * p


def xray_rgb_projection(rho, p, code_units, dx_cm, axis=2, gamma=0.45):
    """Project X-ray channels along a line of sight into a Chandra-like RGB image.

    Follows the science-colour mapping of the labeled Chandra Cas A image:
    red = low energies (Fe/Mg-dominated shocked-ejecta lines; here the soft
    thermal band), green = intermediate energies (Si lines; the medium thermal
    band), blue = the highest energies, which in Cas A are SYNCHROTRON emission
    from shock-accelerated electrons (thin blast-wave arcs), not hot thermal
    gas. With this mapping the blast-wave rim and the shocked dense CSM shell
    (the "Green Monster" analogue) read blue against the red/green ejecta
    shell, mirroring how Vink et al. (2024) distinguish them spectrally.

    Returns:
        ``(rgb, broadband)``: an (H, W, 3) float image in [0, 1] and the summed
        broadband surface-brightness map (for a grayscale panel).
    """
    bands, _ = xray_band_emissivity(rho, p, code_units)
    sync = synchrotron_emissivity(rho, p, code_units)
    proj = {k: np.asarray(v).sum(axis=axis) * dx_cm for k, v in bands.items()}
    proj["sync"] = np.asarray(sync).sum(axis=axis) * dx_cm

    def stretch(a):
        a = a.T
        hi = np.percentile(a, 99.5)
        hi = hi if hi > 0 else (a.max() or 1.0)
        return np.clip(a / hi, 0.0, 1.0) ** gamma

    rgb = np.stack([stretch(proj["soft"] + 0.5 * proj["hard"]),
                    stretch(proj["medium"] + 0.5 * proj["hard"]),
                    stretch(proj["sync"])], axis=-1)
    broadband = (proj["soft"] + proj["medium"] + proj["hard"]).T
    return rgb, broadband


def _xray_zoom_halfwidth(broadband, box_size, floor=1e-2, pad=1.25):
    """Half-width (pc) that frames the X-ray-emitting region (for a telescope-like crop)."""
    n = broadband.shape[0]
    coord = (np.arange(n) + 0.5) / n * box_size - box_size / 2
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    bright = broadband > floor * broadband.max()
    if not np.any(bright):
        return box_size / 2
    rmax = np.sqrt(np.maximum(xx[bright] ** 2, yy[bright] ** 2)).max()
    return float(min(box_size / 2, pad * rmax))


def ir_dust_emissivity(rho, p, code_units, mass_per_nucleus=1.4, T_sputter=2.0e6):
    """Per-cell dust thermal-IR emissivity proxy (dense, cooler shocked gas).

    Infrared traces warm dust, which is collisionally heated by the gas
    (heating ∝ n_gas sqrt(T)) but destroyed by sputtering in the hottest plasma.
    So ε_IR ∝ n_H^2 sqrt(T) exp(-(T/T_sputter)^1.5): brightest in the DENSE,
    warm shocked shell and ejecta knots (the Cas A "Green Monster" / dust
    filaments), and suppressed in the hot diffuse X-ray plasma -- complementary
    to the X-ray emissivity. A visualisation proxy (no dust model / grain size
    distribution / line emission).
    """
    T = temperature_K(rho, p, code_units)
    rho_cgs = (np.asarray(rho) * code_units.code_density).to(u.g / u.cm ** 3).value
    n_H = rho_cgs / (mass_per_nucleus * const.m_p.cgs.value)
    Tpos = np.maximum(T, 0.0)
    return n_H ** 2 * np.sqrt(Tpos) * np.exp(-(Tpos / T_sputter) ** 1.5)


def pristine_debris_emissivity(rho, p, code_units, mass_per_nucleus=1.4,
                               T_cold=3.0e4):
    """Per-cell emissivity proxy for the cold, UNSHOCKED ("pristine") ejecta.

    Webb sees cool supernova debris that no shock has touched yet -- the
    filamentary web interior to the reverse shock (the material whose fine
    structure the Milisavljevic et al. 2024 analysis ties to the explosion's
    radioactive-bubble mixing). The proxy selects cold (T < ``T_cold``) dense
    gas and weights it by density squared, so the un-shocked homologous ejecta
    interior glows while both the shock-heated plasma and the thin cold CSM are
    excluded. Meaningful only with the dual-energy formalism: without it the
    cold interior's float32 pressure (hence temperature) is cancellation noise.
    """
    T = temperature_K(rho, p, code_units)
    rho_cgs = (np.asarray(rho) * code_units.code_density).to(u.g / u.cm ** 3).value
    n_H = rho_cgs / (mass_per_nucleus * const.m_p.cgs.value)
    cold = np.exp(-np.maximum(T, 0.0) / T_cold)
    return n_H ** 2 * cold


def _stretch_project(field3d, dx_cm, axis=2, gamma=0.5, pct=99.5):
    """Line-of-sight surface-brightness projection, normalised with a gamma stretch."""
    a = (np.asarray(field3d).sum(axis=axis) * dx_cm).T
    pos = a[a > 0]
    hi = np.percentile(pos, pct) if pos.size else 1.0
    hi = hi if hi > 0 else 1.0
    return np.clip(a / hi, 0.0, 1.0) ** gamma


def multiwavelength_figure(rho, p, code_units, box_size, dx_cm, title, out_path,
                           axis=2):
    """Chandra(X-ray)+JWST(IR)-style composite: X-ray, IR, and their overlay.

    Emulates the multiwavelength Cas A composites: hot shock-heated plasma plus
    thin synchrotron blast-wave arcs in X-ray (blue) over TWO infrared
    components -- warm dust embedded in the hot shocked gas (gold), and the
    cold "pristine" un-shocked ejecta debris interior to the reverse shock
    (deep red, the Webb debris web; physically meaningful only with dual
    energy). Three panels: X-ray, IR, and the composite. ``axis`` picks the
    line of sight (0 = along the CSM-shell asymmetry axis, which projects the
    shocked dense shell IN FRONT of the interior like the real near-side
    "Green Monster").
    """
    bands, _ = xray_band_emissivity(rho, p, code_units)
    xray3d = bands["soft"] + bands["medium"] + bands["hard"]     # hot plasma (T>3e6 K)
    sync3d = synchrotron_emissivity(rho, p, code_units)           # blast-wave arcs
    ir3d = ir_dust_emissivity(rho, p, code_units)                 # warm dense dust
    pris3d = pristine_debris_emissivity(rho, p, code_units)       # cold un-shocked ejecta

    X = _stretch_project(xray3d, dx_cm, axis=axis, gamma=0.5)
    S = _stretch_project(sync3d, dx_cm, axis=axis, gamma=0.5)
    I = _stretch_project(ir3d, dx_cm, axis=axis, gamma=0.5)
    P = _stretch_project(pris3d, dx_cm, axis=axis, gamma=0.5)

    # X-ray (+arcs) -> blue/cyan, warm dust -> gold, pristine debris -> deep red
    comp = np.clip(np.stack([
        1.05 * I + 0.90 * P + 0.10 * X + 0.08 * S,     # R
        0.55 * I + 0.22 * P + 0.55 * X + 0.45 * S,     # G
        0.12 * I + 0.10 * P + 1.05 * X + 1.00 * S,     # B
    ], axis=-1), 0.0, 1.0)

    # blue-tinted X-ray-only and gold+red IR-only images for the side panels
    xray_rgb = np.clip(np.stack([0.10 * X + 0.08 * S, 0.55 * X + 0.45 * S,
                                 1.05 * X + 1.00 * S], -1), 0, 1)
    ir_rgb = np.clip(np.stack([1.05 * I + 0.90 * P, 0.55 * I + 0.22 * P,
                               0.12 * I + 0.10 * P], -1), 0, 1)

    ext = [-box_size / 2, box_size / 2, -box_size / 2, box_size / 2]
    bb = np.asarray(xray3d).sum(axis=axis).T + np.asarray(ir3d).sum(axis=axis).T
    hw = _xray_zoom_halfwidth(bb, box_size)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, img, ttl in zip(
        axes,
        [xray_rgb, ir_rgb, comp],
        ["X-ray (hot plasma + synchrotron arcs)",
         "infrared (warm dust + cold pristine debris)",
         "X-ray + IR composite"],
    ):
        ax.imshow(img, origin="lower", extent=ext, interpolation="bilinear")
        ax.set_title(ttl)
        ax.set_xlabel("x [pc]")
        ax.set_ylabel("y [pc]")
        ax.set_facecolor("black")
        ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw); ax.set_aspect("equal")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def xray_figure(rho, p, code_units, box_size, dx_cm, title, out_path, axis=2):
    """Chandra-like X-ray view: 3-colour composite + broadband surface brightness.

    The frame is cropped to the emitting region so the remnant fills the panel,
    as in a real telescope image (for side-by-side comparison).
    """
    rgb, broadband = xray_rgb_projection(rho, p, code_units, dx_cm, axis=axis)
    ext = [-box_size / 2, box_size / 2, -box_size / 2, box_size / 2]
    hw = _xray_zoom_halfwidth(broadband, box_size)
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(13, 6.2))

    axc.imshow(rgb, origin="lower", extent=ext, interpolation="bilinear")
    axc.set_title("X-ray colour composite (R=soft/Fe-Mg, G=medium/Si, B=synchrotron)")

    bb = broadband.copy()
    vmax = np.percentile(bb[bb > 0], 99.7) if np.any(bb > 0) else 1.0
    vmin = vmax * 1e-3
    axb.imshow(np.clip(bb, vmin, vmax), origin="lower", extent=ext,
               norm=LogNorm(vmin=vmin, vmax=vmax), cmap="afmhot", interpolation="bilinear")
    axb.set_title("X-ray broadband surface brightness")
    for ax in (axc, axb):
        ax.set_xlabel("x [pc]")
        ax.set_ylabel("y [pc]")
        ax.set_facecolor("black")
        ax.set_xlim(-hw, hw)
        ax.set_ylim(-hw, hw)
        ax.set_aspect("equal")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# Deep-image colormap: black -> deep blue -> bright blue -> pale blue-white,
# matching the classic deep Chandra Cas A press image. The low end ramps up
# quickly so the faint filamentary interior reads as medium blue, not black.
CHANDRA_BLUE = LinearSegmentedColormap.from_list("chandra_blue", [
    (0.00, (0.00, 0.00, 0.00)),
    (0.12, (0.02, 0.06, 0.28)),
    (0.35, (0.08, 0.22, 0.65)),
    (0.62, (0.15, 0.45, 0.95)),
    (0.85, (0.55, 0.78, 1.00)),
    (1.00, (0.97, 0.99, 1.00)),
])


def chandra_deep_figure(rho, p, code_units, box_size, dx_cm, out_path,
                        axis=2, knot_weight=3.0,
                        unsharp=((0.006, 8.0), (0.025, 3.0)),
                        asinh_softening=0.01, pct=99.9, pad=1.12,
                        sync_weight=0.6, observe=False,
                        nh_transmission=(0.35, 0.70, 0.92),
                        psf_sigma_pc=0.017, peak_counts=350.0,
                        observe_seed=17):
    """Deep single-band Chandra-style press image (blue on black, full-bleed).

    Emulates the look of the deep Chandra Cas A exposure, which is dominated by
    thin tangled filaments and knots of reverse-shocked ejecta inside a faint,
    sharp forward-shock rim. Three imaging steps on top of the raw emissivity:

    1. **knot weighting**: the real blue image is mostly *line* emission from
       dense shocked ejecta knots, which scales more steeply with density than
       the bremsstrahlung continuum. Blend in a density-boosted component
       (``knot_weight`` times an extra ``n_H``-weighting) to let the clumps
       dominate over the smooth shell.
    2. **multi-scale unsharp mask**: for each ``(sigma_frac, gain)`` in
       ``unsharp``, add back ``gain`` times the positive residual against a
       Gaussian-smoothed image (sigma = ``sigma_frac`` of the frame) -- the
       standard trick that turns limb-brightened shells into the thin wispy
       filaments of the deep X-ray look (a fine and a medium scale together
       give both crisp wisps and larger filament complexes).
    3. **asinh stretch** (the astro-imaging standard): linear in the faint
       outer rim, logarithmic in the bright shell, so both are visible at once.

    A ``sync_weight`` fraction of a shock-localised nonthermal component
    (:func:`synchrotron_emissivity`) is blended in so the thin forward-shock
    arcs -- which in the real deep Chandra image are synchrotron, invisible to
    a thermal proxy at the low-density blast wave -- appear at the periphery.

    ``observe=True`` adds an observational forward model, which supplies the
    grainy, speckled texture of the real deep exposure that no hydro
    improvement can:

    * **Galactic absorption** -- per-band transmission factors
      (``nh_transmission`` for soft/medium/hard; defaults roughly matching
      N_H ~ 1.2e22 cm^-2 toward Cas A, which strongly suppresses the soft
      band);
    * **PSF blur** -- Gaussian of ``psf_sigma_pc`` (default 0.017 pc ~ 1
      arcsec at 3.4 kpc);
    * **photon noise** -- the projected image is scaled so its bright shell
      reaches ``peak_counts`` counts/pixel and Poisson-sampled (deep-exposure
      statistics: smooth where bright, speckled where faint).

    Without ``observe`` this remains a clean visualisation proxy (no spectral
    model, no instrument response).
    """
    from scipy.ndimage import gaussian_filter

    bands, T = xray_band_emissivity(rho, p, code_units)
    if observe:
        # Galactic absorption: apply the per-band transmission before summing,
        # so the (heavily absorbed) soft band contributes like it does through
        # N_H ~ 1.2e22 cm^-2
        xray3d = (nh_transmission[0] * bands["soft"]
                  + nh_transmission[1] * bands["medium"]
                  + nh_transmission[2] * bands["hard"])
    else:
        xray3d = bands["soft"] + bands["medium"] + bands["hard"]
    if knot_weight > 0:
        # extra density weighting for the (line-emitting) dense shocked knots,
        # normalised by the mean emitting density so units stay comparable
        rho_cgs = (np.asarray(rho) * code_units.code_density).to(u.g / u.cm ** 3).value
        n_H = rho_cgs / (1.4 * const.m_p.cgs.value)
        emitting = xray3d > 0
        n_ref = n_H[emitting].mean() if np.any(emitting) else 1.0
        xray3d = xray3d * (1.0 + knot_weight * (n_H / n_ref))

    sb = np.asarray(xray3d).sum(axis=axis).T * dx_cm

    if sync_weight > 0:
        # blend the (separately-normalised) synchrotron arcs into the thermal
        # surface brightness at sync_weight of its bright end
        sb_sync = np.asarray(
            synchrotron_emissivity(rho, p, code_units)).sum(axis=axis).T * dx_cm
        s_pos = sb_sync[sb_sync > 0]
        t_pos = sb[sb > 0]
        if s_pos.size and t_pos.size:
            scale = np.percentile(t_pos, pct) / max(np.percentile(s_pos, pct), 1e-30)
            sb = sb + sync_weight * scale * sb_sync

    if observe:
        # instrument PSF + deep-exposure photon statistics
        from scipy.ndimage import gaussian_filter as _gauss
        dx_pc = box_size / sb.shape[0]
        sb = _gauss(sb, max(psf_sigma_pc / dx_pc, 0.3))
        pos = sb[sb > 0]
        hi = np.percentile(pos, pct) if pos.size else 1.0
        rng = np.random.default_rng(observe_seed)
        counts = rng.poisson(np.clip(sb / max(hi, 1e-30), 0.0, 4.0) * peak_counts)
        sb = counts.astype(np.float64)

    enhanced = sb.copy()
    for sigma_frac, gain in unsharp:
        sigma = max(1.0, sigma_frac * sb.shape[0])
        detail = np.clip(sb - gaussian_filter(sb, sigma), 0.0, None)
        enhanced = enhanced + gain * detail

    pos = enhanced[enhanced > 0]
    hi = np.percentile(pos, pct) if pos.size else 1.0
    x = np.clip(enhanced / max(hi, 1e-30), 0.0, 1.0)
    img = np.arcsinh(x / asinh_softening) / np.arcsinh(1.0 / asinh_softening)

    ext = [-box_size / 2, box_size / 2, -box_size / 2, box_size / 2]
    hw = _xray_zoom_halfwidth(sb, box_size, pad=pad)
    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, origin="lower", extent=ext, cmap=CHANDRA_BLUE,
              vmin=0.0, vmax=1.0, interpolation="bilinear")
    ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path, dpi=200, facecolor="black")
    plt.close(fig)
    return out_path


def radial_velocity_field(fs, registered_variables, helper_data, box_size, num_cells, code_units):
    """Radial velocity field in km/s (and the radius field), from a final state."""
    r, X, Y, Z = centered_radius(helper_data, box_size, num_cells)
    r = np.asarray(r); X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)
    vx = np.asarray(fs[registered_variables.velocity_index.x])
    vy = np.asarray(fs[registered_variables.velocity_index.y])
    vz = np.asarray(fs[registered_variables.velocity_index.z])
    vr = (vx * X + vy * Y + vz * Z) / np.maximum(r, 1e-6)
    vr_kms = (vr * code_units.code_velocity).to(u.km / u.s).value
    return r, vr_kms


def realistic_figure(rho, T, r, box_size, title, out_path):
    """Density & temperature slices + projected column density + radial density."""
    rho = np.asarray(rho); T = np.asarray(T); r = np.asarray(r)
    n = rho.shape[-1]
    zc = n // 2
    ext = [-box_size / 2, box_size / 2, -box_size / 2, box_size / 2]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    im0 = axes[0, 0].imshow(rho[:, :, zc].T, origin="lower", extent=ext,
                            norm=LogNorm(), cmap="cividis")
    axes[0, 0].set_title("density (z-midplane slice)")
    fig.colorbar(im0, ax=axes[0, 0], label=r"$\rho$ [code]")

    im1 = axes[0, 1].imshow(T[:, :, zc].T, origin="lower", extent=ext,
                            norm=LogNorm(vmin=max(1e2, np.nanmin(T[T > 0]))), cmap="inferno")
    axes[0, 1].set_title("temperature (z-midplane slice)")
    fig.colorbar(im1, ax=axes[0, 1], label="T [K]")

    # projected column density (sum along the line of sight) -- the "observed" view
    col = rho.sum(axis=2).T * (box_size / n)
    im2 = axes[1, 0].imshow(col, origin="lower", extent=ext, norm=LogNorm(), cmap="bone")
    axes[1, 0].set_title("projected column density (filaments / clumps)")
    fig.colorbar(im2, ax=axes[1, 0], label=r"$\Sigma$ [code]")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("x [pc]")
        ax.set_ylabel("y [pc]")

    rf = r.flatten()
    nbin = 140
    bins = np.linspace(0, rf.max(), nbin + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.searchsorted(bins, rf, "right") - 1, 0, nbin - 1)
    raw_cnt = np.bincount(idx, minlength=nbin)
    cnt = np.maximum(raw_cnt, 1)
    mean_rho = np.bincount(idx, weights=rho.flatten(), minlength=nbin) / cnt
    mean_rho[raw_cnt == 0] = np.nan
    sub = np.arange(0, rf.size, max(1, rf.size // 50000))
    axes[1, 1].scatter(rf[sub], rho.flatten()[sub], s=1, color="lightgray",
                       alpha=0.5, rasterized=True)
    axes[1, 1].plot(bc, mean_rho, color="C0", lw=1.6)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("r [pc]")
    axes[1, 1].set_ylabel(r"$\rho$ [code]")
    axes[1, 1].set_title("radial density (scatter = clumping spread)")
    axes[1, 1].set_xlim(0, box_size / 2)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
# =============================================================================
# ============ ↑ Shared figure (slices + radial profiles) ↑ ===================
# =============================================================================
