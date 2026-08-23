"""
Heating and cooling for the chemistry module (thermochemistry).

LICENSE: GPL-3.0 (NOT astronomix's MIT). This file is a derivative work of KROME
(Grassi et al. 2014, GPL-3.0); see ``LICENSE.md`` in this directory.

Provides the net gas heating minus cooling rate and the corresponding
temperature derivative used to evolve the gas temperature alongside the chemical
abundances. Everything works in CGS with the temperature in Kelvin, matching the
units the per-cell reaction network is integrated in. Species are located by
index rather than a hard-coded layout: any referenced species that is absent
from the network (index ``-1``) drops out of its term.

Provenance of the physics:

  * heating (cosmic-ray, grain photoelectric, H2-formation) is ported from the
    carbox ``latent_tgas`` example;
  * H2 line cooling is the **Glover & Abel (2008)** multi-collider form (H, H+,
    H2, He, e), ported verbatim from KROME's ``krome_cooling.f90::cooling_H2``
    (a substantial upgrade over the single-collider example fit);
  * fine-structure metal cooling — [C II] 158 um and [O I] 63 um — are the
    dominant coolants of the cold neutral gas that KROME includes but the carbox
    example omits. They are implemented here as standard two-level atoms. The
    line data (Einstein A, level spacing) are exact; the collisional
    de-excitation rate coefficients are representative literature fits
    (Draine 2011; Wolfire et al. 2003; Barinovs et al. 2005), not KROME's
    tabulated collision data — good enough to capture the thermal balance, to be
    refined against KROME later.
"""

# jax
import jax.numpy as jnp

# Boltzmann constant in CGS (erg / K).
BOLTZMANN_CONSTANT_CGS = 1.38064852e-16

# H2 binding energy released per molecule formed (4.48 eV) in erg.
ELECTRON_VOLT_ERG = 1.602176634e-12
HYDROGEN_MOLECULE_BINDING_ERG = 4.48 * ELECTRON_VOLT_ERG


def _species_density(abundances, species_index):
    """Return the number density of one species, or zero if it is absent.

    Args:
        abundances: The per-cell absolute abundances [cm^-3].
        species_index: The static index of the species, or -1 when the network
            does not contain it.

    Returns:
        The species number density, or ``0.0`` for a missing species so its
        contribution vanishes.
    """
    # ``species_index`` is a static (configuration) integer, so this branch is
    # resolved at trace time and adds no runtime cost.
    if species_index < 0:
        return 0.0
    return abundances[species_index]


# =============================================================
# ======================= ↓ Heating ↓ =========================
# =============================================================


def _grain_photoelectric_heating(
    abundances,
    temperature_kelvin,
    fuv_field,
    dust_to_gas_ratio,
    hydrogen_index,
    electron_index,
):
    """Net grain photoelectric heating minus grain-recombination cooling.

    Ported from the carbox example ``chem_heating.py:get_photoelectric_heating``
    (Bakes & Tielens / Wolfire form). The heating scales with the total number
    density and the FUV field; the recombination-cooling correction depends on
    the grain charging parameter ``psi``.

    Args:
        abundances: The per-cell abundances [cm^-3].
        temperature_kelvin: The gas temperature [K].
        fuv_field: The FUV radiation field (Draine units).
        dust_to_gas_ratio: The dust-to-gas mass ratio.
        hydrogen_index: Index of atomic H (or -1).
        electron_index: Index of the electron (or -1).

    Returns:
        The net photoelectric heating rate.
    """
    total_number_density = jnp.sum(abundances)
    electron_density = _species_density(abundances, electron_index)
    hydrogen_density = _species_density(abundances, hydrogen_index)

    # Charging parameter of the grains; the small floor avoids a divide-by-zero
    # where the electron abundance underflows.
    charging_parameter = (
        fuv_field * jnp.sqrt(temperature_kelvin) / (electron_density + 1e-20)
    )

    recombination_exponent = 0.735 * temperature_kelvin ** (-0.068)
    grain_recombination_cooling = (
        4.65e-30
        * temperature_kelvin**0.94
        * charging_parameter**recombination_exponent
        * electron_density
        * hydrogen_density
    )

    heating_efficiency = 4.9e-2 / (
        1.0 + 4e-3 * charging_parameter**0.73
    ) + 3.7e-2 * (temperature_kelvin * 1e-4) ** 0.7 / (
        1.0 + 2e-4 * charging_parameter
    )

    return (
        1.3e-24 * heating_efficiency * fuv_field * total_number_density
        - grain_recombination_cooling
    ) * dust_to_gas_ratio


def _hydrogen_molecule_formation_heating(
    abundances,
    temperature_kelvin,
    formation_rate_coefficient,
    hydrogen_index,
    molecular_hydrogen_index,
):
    """H2 grain-formation heating with KROME's critical-density partition.

    Each H2 formed on a grain releases 4.48 eV; the fraction that thermalises
    (rather than escaping as internal excitation / radiation) is set by the
    critical-density factor ``h2heatfac`` ported verbatim from KROME's
    ``heatingChem`` (``krome_heating.f90``). The formation rate is tied to the
    network's H + H -> H2 reaction, ``formation_rate_coefficient * n_H^2``.

    NB the carbox example instead multiplied the *FUV photodissociation* rate by
    n(H2), an unshielded photo term mislabelled as formation heating; a 0-D
    benchmark against KROME showed it overestimated the heating by ~5 dex. This
    replaces it. (True FUV photodissociation heating, with self-shielding,
    belongs in a separate photo module.)

    Args:
        abundances: The per-cell abundances [cm^-3].
        temperature_kelvin: The gas temperature [K].
        formation_rate_coefficient: The H + H -> H2 grain formation rate
            coefficient [cm^3 s^-1].
        hydrogen_index: Index of atomic H (or -1).
        molecular_hydrogen_index: Index of H2 (or -1).

    Returns:
        The H2 formation heating rate [erg cm^-3 s^-1].
    """
    if hydrogen_index < 0:
        return 0.0

    hydrogen_density = abundances[hydrogen_index]
    molecular_hydrogen_density = _species_density(
        abundances, molecular_hydrogen_index
    )
    # Total hydrogen nuclei density (KROME's ``get_Hnuclei``, dominant terms).
    hydrogen_nuclei_density = hydrogen_density + 2.0 * molecular_hydrogen_density

    # Critical-density partition (KROME heatingChem): the fraction of the 4.48 eV
    # that thermalises rises toward unity above the critical density ``ncr``.
    critical_numerator = 1.0e6 * temperature_kelvin ** (-0.5)
    critical_denominator_hydrogen = 1.6 * jnp.exp(-((4.0e2 / temperature_kelvin) ** 2))
    critical_denominator_molecular = 1.4 * jnp.exp(
        -1.2e4 / (temperature_kelvin + 1.2e3)
    )
    hydrogen_fraction = hydrogen_density / hydrogen_nuclei_density
    molecular_fraction = molecular_hydrogen_density / hydrogen_nuclei_density
    critical_density = critical_numerator / (
        critical_denominator_hydrogen * hydrogen_fraction
        + critical_denominator_molecular * molecular_fraction
    )
    thermalised_fraction = 1.0 / (1.0 + critical_density / hydrogen_nuclei_density)

    formation_rate = formation_rate_coefficient * hydrogen_density * hydrogen_density
    return (
        HYDROGEN_MOLECULE_BINDING_ERG * thermalised_fraction * formation_rate
    )


def heating_rate(
    abundances,
    temperature_kelvin,
    cosmic_ray_rate,
    fuv_field,
    dust_to_gas_ratio,
    hydrogen_molecule_formation_rate_coefficient,
    hydrogen_index,
    molecular_hydrogen_index,
    electron_index,
):
    """Total gas heating rate [erg cm^-3 s^-1].

    Sums cosmic-ray heating, grain photoelectric heating and H2 grain-formation
    heating (the last with KROME's critical-density partition).

    Args:
        abundances: The per-cell abundances [cm^-3].
        temperature_kelvin: The gas temperature [K].
        cosmic_ray_rate: The cosmic-ray ionization rate [s^-1].
        fuv_field: The FUV radiation field (Draine units).
        dust_to_gas_ratio: The dust-to-gas mass ratio.
        hydrogen_molecule_formation_rate_coefficient: The H + H -> H2 grain
            formation rate coefficient [cm^3 s^-1].
        hydrogen_index: Index of atomic H (or -1).
        molecular_hydrogen_index: Index of H2 (or -1).
        electron_index: Index of the electron (or -1).

    Returns:
        The total heating rate.
    """
    hydrogen_density = _species_density(abundances, hydrogen_index)
    molecular_hydrogen_density = _species_density(
        abundances, molecular_hydrogen_index
    )

    cosmic_ray_heating = cosmic_ray_rate * (
        5.5e-12 * hydrogen_density + 2.5e-11 * molecular_hydrogen_density
    )

    photoelectric_heating = _grain_photoelectric_heating(
        abundances,
        temperature_kelvin,
        fuv_field,
        dust_to_gas_ratio,
        hydrogen_index,
        electron_index,
    )

    h2_formation_heating = _hydrogen_molecule_formation_heating(
        abundances,
        temperature_kelvin,
        hydrogen_molecule_formation_rate_coefficient,
        hydrogen_index,
        molecular_hydrogen_index,
    )

    return cosmic_ray_heating + photoelectric_heating + h2_formation_heating


# =============================================================
# ======================= ↓ Cooling ↓ =========================
# =============================================================


def _sigmoid(argument, midpoint, steepness):
    """KROME's ``sigmoid`` helper: ``10 / (10 + exp(-s (x - x0)))``."""
    return 10.0 / (10.0 + jnp.exp(-steepness * (argument - midpoint)))


def _cooling_window(log_temperature, log_temperature_low, log_temperature_high):
    """KROME's ``wCool`` smoothing window in [0, 1].

    Ported from ``krome_cooling.f90::wCool``. Blends the piecewise H2 cooling
    fits smoothly to zero outside their fitted temperature range.

    Args:
        log_temperature: ``log10(T)``.
        log_temperature_low: Lower edge of the window in ``log10(T)``.
        log_temperature_high: Upper edge of the window in ``log10(T)``.

    Returns:
        The window value in [0, 1].
    """
    scaled = (log_temperature - log_temperature_low) / (
        log_temperature_high - log_temperature_low
    )
    window = 10.0 ** (
        200.0 * (_sigmoid(scaled, -0.2, 50.0) * _sigmoid(-scaled, -1.2, 50.0) - 1.0)
    )
    return jnp.where(window < 1e-199, 0.0, window)


def _ten_to_the(exponent):
    """``10 ** exponent`` with the exponent capped to avoid overflow.

    The Glover & Abel fits are only valid in their own temperature window; the
    branches evaluated outside it (masked by ``jnp.where``) can otherwise produce
    a non-finite value that would poison gradients. Capping keeps every branch
    finite; the in-window values (log-rates around -16 to -25) are untouched.
    """
    return 10.0 ** jnp.minimum(exponent, 30.0)


def _glover_abel_h2_cooling(
    abundances,
    temperature_kelvin,
    hydrogen_index,
    molecular_hydrogen_index,
    ionized_hydrogen_index,
    helium_index,
    electron_index,
):
    """Glover & Abel (2008) multi-collider H2 line cooling.

    Ported from ``krome_cooling.f90::cooling_H2``. The low-density limit is the
    sum of per-collider excitation rates (H, H+, H2, e, He), each a piecewise
    polynomial in ``log10(T/1000)``; the high-density (LTE) limit follows
    Hollenbach/Glover. The effective rate is the harmonic combination
    ``n(H2) / (1/HDL + 1/LDL)``.

    Args:
        abundances: The per-cell abundances [cm^-3].
        temperature_kelvin: The gas temperature [K].
        hydrogen_index: Index of atomic H (or -1).
        molecular_hydrogen_index: Index of H2 (or -1).
        ionized_hydrogen_index: Index of H+ (or -1).
        helium_index: Index of He (or -1).
        electron_index: Index of the electron (or -1).

    Returns:
        The H2 cooling rate, or ``0.0`` if the network has no H2.
    """
    if molecular_hydrogen_index < 0:
        return 0.0

    molecular_hydrogen_density = abundances[molecular_hydrogen_index]
    hydrogen_density = _species_density(abundances, hydrogen_index)
    ionized_hydrogen_density = _species_density(abundances, ionized_hydrogen_index)
    helium_density = _species_density(abundances, helium_index)
    electron_density = _species_density(abundances, electron_index)

    temperature = temperature_kelvin
    scaled_temperature = temperature * 1e-3
    log_t3 = jnp.log10(scaled_temperature)
    log_temperature = jnp.log10(temperature)

    def polynomial(coefficients):
        return sum(
            coefficient * log_t3**power
            for power, coefficient in enumerate(coefficients)
        )

    window_1_4 = _cooling_window(log_temperature, 1.0, 4.0)
    window_2_4 = _cooling_window(log_temperature, 2.0, 4.0)

    # --- H2-H (Glover & Abel), piecewise in T ---
    h2_h = hydrogen_density * jnp.where(
        temperature <= 1e2,
        _ten_to_the(
            polynomial([-16.818342, 37.383713, 58.145166, 48.656103, 20.159831, 3.8479610])
        ),
        jnp.where(
            temperature <= 1e3,
            _ten_to_the(
                polynomial([-24.311209, 3.5692468, -11.332860, -27.850082, -21.328264, -4.2519023])
            ),
            jnp.where(
                temperature <= 6e3,
                _ten_to_the(
                    polynomial([-24.311209, 4.6450521, -3.7209846, 5.9369081, -5.5108049, 1.5538288])
                ),
                1.862314467912518e-22
                * _cooling_window(log_temperature, 1.0, jnp.log10(6e3)),
            ),
        ),
    )

    # --- H2-H+ ---
    h2_ion = ionized_hydrogen_density * jnp.where(
        (temperature > 1e1) & (temperature <= 1e4),
        _ten_to_the(
            polynomial([-22.089523, 1.5714711, 0.015391166, -0.23619985, -0.51002221, 0.32168730])
        ),
        1.182509139382060e-21 * window_1_4,
    )

    # --- H2-H2 ---
    h2_h2 = molecular_hydrogen_density * window_2_4 * _ten_to_the(
        polynomial([-23.962112, 2.09433740, -0.77151436, 0.43693353, -0.14913216, -0.033638326])
    )

    # --- H2-e ---
    electron_rate = jnp.where(
        temperature <= 5e2,
        _ten_to_the(
            polynomial(
                [-21.928796, 16.815730, 96.743155, 343.19180, 734.71651, 983.67576, 802.01247, 364.14446, 70.609154]
            )
        ),
        _ten_to_the(
            polynomial(
                [-22.921189, 1.6802758, 0.93310622, 4.0406627, -4.7274036, -8.8077017, 8.9167183, 6.4380698, -6.3701156]
            )
        ),
    )
    h2_electron = electron_density * electron_rate * window_2_4

    # --- H2-He ---
    h2_helium = helium_density * jnp.where(
        (temperature > 1e1) & (temperature <= 1e4),
        _ten_to_the(
            polynomial([-23.689237, 2.1892372, -0.81520438, 0.29036281, -0.16596184, 0.19191375])
        ),
        1.002560385050777e-22 * window_1_4,
    )

    low_density_limit = h2_h + h2_ion + h2_h2 + h2_electron + h2_helium

    # High-density (LTE) limit: Hollenbach below 2000 K, Glover fit to 1e4 K, a
    # smooth cut-off above.
    high_density_rotational = (
        (9.5e-22 * scaled_temperature**3.76)
        / (1.0 + 0.12 * scaled_temperature**2.1)
        * jnp.exp(-((0.13 / scaled_temperature) ** 3))
        + 3.0e-24 * jnp.exp(-0.51 / scaled_temperature)
    )
    high_density_vibrational = 6.7e-19 * jnp.exp(-5.86 / scaled_temperature) + 1.6e-18 * jnp.exp(
        -11.7 / scaled_temperature
    )
    high_density_low_temperature = high_density_rotational + high_density_vibrational
    high_density_mid_temperature = _ten_to_the(
        polynomial(
            [-20.584225, 5.0194035, -1.5738805, -4.7155769, 2.4714161, 5.4710750, -3.9467356, -2.2148338, 1.8161874]
        )
    )
    cutoff = 1.0 / (1.0 + jnp.exp(jnp.minimum((temperature - 3e4) * 2e-4, 3e2)))
    high_density_limit = jnp.where(
        temperature < 2e3,
        high_density_low_temperature,
        jnp.where(
            temperature <= 1e4,
            high_density_mid_temperature,
            5.531333679406485e-19 * cutoff,
        ),
    )

    # Harmonic combination; guard the degenerate limits.
    return jnp.where(
        low_density_limit <= 0.0,
        0.0,
        molecular_hydrogen_density
        / (1.0 / (high_density_limit + 1e-100) + 1.0 / (low_density_limit + 1e-100)),
    )


def _two_level_line_cooling(
    species_density,
    temperature_kelvin,
    level_spacing_kelvin,
    einstein_coefficient,
    upper_over_lower_degeneracy,
    collisional_deexcitation_rate,
):
    """Cooling from a single two-level (fine-structure) line.

    The upper-level fraction follows from statistical balance between collisional
    excitation/de-excitation and spontaneous radiative decay (optically thin):
    ``f_upper = C_lu / (C_lu + C_ul + A)`` with
    ``C_lu = C_ul (g_u/g_l) exp(-dE/kT)``. The emitted power per ion is
    ``A · dE · f_upper``.

    Args:
        species_density: Number density of the emitting species [cm^-3].
        temperature_kelvin: The gas temperature [K].
        level_spacing_kelvin: Upper-lower level spacing ``dE/k`` [K].
        einstein_coefficient: Spontaneous decay rate ``A_ul`` [s^-1].
        upper_over_lower_degeneracy: ``g_upper / g_lower``.
        collisional_deexcitation_rate: ``C_ul = sum_c n_c gamma_c(T)`` [s^-1].

    Returns:
        The line cooling rate [erg cm^-3 s^-1].
    """
    boltzmann_factor = jnp.exp(-level_spacing_kelvin / temperature_kelvin)
    collisional_excitation = (
        collisional_deexcitation_rate * upper_over_lower_degeneracy * boltzmann_factor
    )
    upper_fraction = collisional_excitation / (
        collisional_excitation + collisional_deexcitation_rate + einstein_coefficient
    )
    line_energy_erg = level_spacing_kelvin * BOLTZMANN_CONSTANT_CGS
    return species_density * einstein_coefficient * line_energy_erg * upper_fraction


def _ionized_carbon_cooling(
    abundances,
    temperature_kelvin,
    ionized_carbon_index,
    electron_index,
    hydrogen_index,
    molecular_hydrogen_index,
):
    """[C II] 158 um fine-structure cooling (2P3/2 - 2P1/2).

    The dominant coolant of the cold/warm neutral medium. Line data are exact;
    collision rate coefficients (e, H, H2) are representative literature fits.

    Returns:
        The [C II] cooling rate, or ``0.0`` if the network has no C+.
    """
    if ionized_carbon_index < 0:
        return 0.0

    carbon_ion_density = abundances[ionized_carbon_index]
    electron_density = _species_density(abundances, electron_index)
    hydrogen_density = _species_density(abundances, hydrogen_index)
    molecular_hydrogen_density = _species_density(abundances, molecular_hydrogen_index)
    scaled_temperature = temperature_kelvin / 100.0

    # Collisional de-excitation rates [cm^3 s^-1]: electrons via the collision
    # strength (Omega ~ 2, g_upper = 4); H and H2 via power-law fits.
    electron_rate = 4.3e-6 / jnp.sqrt(temperature_kelvin)
    hydrogen_rate = 8.0e-10 * scaled_temperature**0.07
    molecular_hydrogen_rate = 3.8e-10 * scaled_temperature**0.14
    collisional_deexcitation_rate = (
        electron_density * electron_rate
        + hydrogen_density * hydrogen_rate
        + molecular_hydrogen_density * molecular_hydrogen_rate
    )

    return _two_level_line_cooling(
        carbon_ion_density,
        temperature_kelvin,
        level_spacing_kelvin=91.21,
        einstein_coefficient=2.29e-6,
        upper_over_lower_degeneracy=2.0,
        collisional_deexcitation_rate=collisional_deexcitation_rate,
    )


def _atomic_oxygen_cooling(
    abundances,
    temperature_kelvin,
    atomic_oxygen_index,
    hydrogen_index,
    molecular_hydrogen_index,
):
    """[O I] 63 um fine-structure cooling (3P1 - 3P2).

    A major coolant of dense neutral gas. Line data exact; H and H2 collisional
    de-excitation rates are representative literature fits (electron collisions
    with neutral O are weak and neglected).

    Returns:
        The [O I] cooling rate, or ``0.0`` if the network has no atomic O.
    """
    if atomic_oxygen_index < 0:
        return 0.0

    oxygen_density = abundances[atomic_oxygen_index]
    hydrogen_density = _species_density(abundances, hydrogen_index)
    molecular_hydrogen_density = _species_density(abundances, molecular_hydrogen_index)
    scaled_temperature = temperature_kelvin / 100.0

    hydrogen_rate = 9.2e-11 * scaled_temperature**0.67
    molecular_hydrogen_rate = 3.0e-11 * scaled_temperature**0.1
    collisional_deexcitation_rate = (
        hydrogen_density * hydrogen_rate
        + molecular_hydrogen_density * molecular_hydrogen_rate
    )

    return _two_level_line_cooling(
        oxygen_density,
        temperature_kelvin,
        level_spacing_kelvin=227.7,
        einstein_coefficient=8.91e-5,
        upper_over_lower_degeneracy=0.6,
        collisional_deexcitation_rate=collisional_deexcitation_rate,
    )


# CO Jeans column-density coefficient (KROME num2col default):
# N_CO = CO_COLUMN_COEFFICIENT * (n_CO * 1e-3) ** (2/3)  [cm^-2].
CO_COLUMN_COEFFICIENT = 1.87e21


def _trilinear_table_lookup(table, bounds, coordinate_1, coordinate_2, coordinate_3):
    """Trilinearly interpolate a value on a uniform 3D log grid.

    The grid is uniform in each axis between the bounds; coordinates outside the
    upper edge are pulled just inside it (as KROME does), and a flag marks
    coordinates below the lower edge so the caller can zero the contribution.

    Args:
        table: Grid values, shape ``(n1, n2, n3)``.
        bounds: ``[x1_min, x1_max, x2_min, x2_max, x3_min, x3_max]``.
        coordinate_1: Query position along axis 1.
        coordinate_2: Query position along axis 2.
        coordinate_3: Query position along axis 3.

    Returns:
        A tuple ``(interpolated_value, below_grid)`` where ``below_grid`` is True
        when any coordinate fell below the grid's lower edge.
    """
    grid_size_1, grid_size_2, grid_size_3 = table.shape
    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max = (
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
    )
    edge_tolerance = 1e-5

    coordinate_1 = jnp.where(
        coordinate_1 >= x1_max, x1_max * (1.0 - edge_tolerance), coordinate_1
    )
    coordinate_2 = jnp.where(
        coordinate_2 >= x2_max, x2_max * (1.0 - edge_tolerance), coordinate_2
    )
    coordinate_3 = jnp.where(
        coordinate_3 >= x3_max, x3_max * (1.0 - edge_tolerance), coordinate_3
    )
    below_grid = (
        (coordinate_1 < x1_min) | (coordinate_2 < x2_min) | (coordinate_3 < x3_min)
    )

    def axis_position(coordinate, minimum, maximum, grid_size):
        fractional = (coordinate - minimum) / (maximum - minimum) * (grid_size - 1)
        lower = jnp.clip(jnp.floor(fractional).astype(jnp.int32), 0, grid_size - 2)
        weight = fractional - lower
        return lower, weight

    lower_1, weight_1 = axis_position(coordinate_1, x1_min, x1_max, grid_size_1)
    lower_2, weight_2 = axis_position(coordinate_2, x2_min, x2_max, grid_size_2)
    lower_3, weight_3 = axis_position(coordinate_3, x3_min, x3_max, grid_size_3)

    def corner(offset_1, offset_2, offset_3):
        return table[lower_1 + offset_1, lower_2 + offset_2, lower_3 + offset_3]

    # Interpolate along axis 1, then 2, then 3.
    face_low_3 = (
        (1 - weight_2)
        * ((1 - weight_1) * corner(0, 0, 0) + weight_1 * corner(1, 0, 0))
        + weight_2
        * ((1 - weight_1) * corner(0, 1, 0) + weight_1 * corner(1, 1, 0))
    )
    face_high_3 = (
        (1 - weight_2)
        * ((1 - weight_1) * corner(0, 0, 1) + weight_1 * corner(1, 0, 1))
        + weight_2
        * ((1 - weight_1) * corner(0, 1, 1) + weight_1 * corner(1, 1, 1))
    )
    interpolated_value = (1 - weight_3) * face_low_3 + weight_3 * face_high_3
    return interpolated_value, below_grid


def _carbon_monoxide_cooling(
    abundances,
    temperature_kelvin,
    carbon_monoxide_index,
    hydrogen_index,
    molecular_hydrogen_index,
    cooling_table,
    cooling_bounds,
):
    """Tabulated CO rotational-line cooling (Neufeld & Kaufman 1993).

    Ported from KROME's ``cooling_CO``: interpolate the cooling coefficient on a
    uniform 3D log grid of (temperature, H+H2 density, CO column density), then
    scale by the collider density and the CO abundance. The CO column density
    uses KROME's default local Jeans estimate ``N_CO = 1.87e21 (n_CO/1e3)^(2/3)``.

    Args:
        abundances: The per-cell abundances [cm^-3].
        temperature_kelvin: The gas temperature [K].
        carbon_monoxide_index: Index of CO (or -1).
        hydrogen_index: Index of atomic H (or -1).
        molecular_hydrogen_index: Index of H2 (or -1).
        cooling_table: The 3D log-cooling table, shape (n_T, n_n, n_NCO).
        cooling_bounds: The six uniform-log-grid limits.

    Returns:
        The CO cooling rate [erg cm^-3 s^-1], or ``0.0`` if CO is absent.
    """
    if carbon_monoxide_index < 0:
        return 0.0

    carbon_monoxide_density = abundances[carbon_monoxide_index]
    collider_density = _species_density(abundances, hydrogen_index) + _species_density(
        abundances, molecular_hydrogen_index
    )

    # Local Jeans column density of CO (KROME num2col default).
    column_density = CO_COLUMN_COEFFICIENT * jnp.maximum(
        carbon_monoxide_density * 1e-3, 1e-40
    ) ** (2.0 / 3.0)

    log_temperature = jnp.log10(temperature_kelvin)
    log_collider_density = jnp.log10(jnp.maximum(collider_density, 1e-40))
    log_column_density = jnp.log10(column_density)

    log_cooling_coefficient, below_grid = _trilinear_table_lookup(
        cooling_table,
        cooling_bounds,
        log_temperature,
        log_collider_density,
        log_column_density,
    )

    cooling = (
        10.0**log_cooling_coefficient * collider_density * carbon_monoxide_density
    )
    # Outside the fitted grid (too cold / too diffuse) KROME returns zero.
    return jnp.where(below_grid, 0.0, cooling)


def cooling_rate(
    abundances,
    temperature_kelvin,
    hydrogen_index,
    molecular_hydrogen_index,
    electron_index,
    atomic_oxygen_index,
    ionized_hydrogen_index,
    helium_index,
    ionized_carbon_index,
    co_cooling,
    carbon_monoxide_index,
    co_cooling_table,
    co_cooling_bounds,
):
    """Total gas cooling rate [erg cm^-3 s^-1].

    Sums Lyman-alpha (high-T atomic), [C II] 158 um and [O I] 63 um
    fine-structure lines (the dominant cold-neutral coolants), Glover & Abel
    multi-collider H2 line cooling, and — when enabled — tabulated CO rotational
    cooling.

    Args:
        abundances: The per-cell abundances [cm^-3].
        temperature_kelvin: The gas temperature [K].
        hydrogen_index: Index of atomic H (or -1).
        molecular_hydrogen_index: Index of H2 (or -1).
        electron_index: Index of the electron (or -1).
        atomic_oxygen_index: Index of atomic O (or -1).
        ionized_hydrogen_index: Index of H+ (or -1).
        helium_index: Index of He (or -1).
        ionized_carbon_index: Index of C+ (or -1).
        co_cooling: Whether to add tabulated CO rotational cooling.
        carbon_monoxide_index: Index of CO (or -1).
        co_cooling_table: The 3D CO cooling table (unused when co_cooling False).
        co_cooling_bounds: The CO table grid limits.

    Returns:
        The total cooling rate.
    """
    hydrogen_density = _species_density(abundances, hydrogen_index)
    electron_density = _species_density(abundances, electron_index)

    lyman_alpha_cooling = (
        7.3e-19
        * hydrogen_density
        * electron_density
        * jnp.exp(-118400.0 / temperature_kelvin)
    )
    carbon_ion_cooling = _ionized_carbon_cooling(
        abundances,
        temperature_kelvin,
        ionized_carbon_index,
        electron_index,
        hydrogen_index,
        molecular_hydrogen_index,
    )
    oxygen_cooling = _atomic_oxygen_cooling(
        abundances,
        temperature_kelvin,
        atomic_oxygen_index,
        hydrogen_index,
        molecular_hydrogen_index,
    )
    h2_cooling = _glover_abel_h2_cooling(
        abundances,
        temperature_kelvin,
        hydrogen_index,
        molecular_hydrogen_index,
        ionized_hydrogen_index,
        helium_index,
        electron_index,
    )

    total = lyman_alpha_cooling + carbon_ion_cooling + oxygen_cooling + h2_cooling

    # The CO table is only present (and only makes sense) when requested; the
    # ``co_cooling`` flag is static, so this branch is resolved at trace time.
    if co_cooling:
        total = total + _carbon_monoxide_cooling(
            abundances,
            temperature_kelvin,
            carbon_monoxide_index,
            hydrogen_index,
            molecular_hydrogen_index,
            co_cooling_table,
            co_cooling_bounds,
        )

    return total


def temperature_derivative(
    abundances,
    temperature_kelvin,
    adiabatic_index,
    cosmic_ray_rate,
    fuv_field,
    dust_to_gas_ratio,
    hydrogen_molecule_formation_rate_coefficient,
    hydrogen_index,
    molecular_hydrogen_index,
    electron_index,
    atomic_oxygen_index,
    ionized_hydrogen_index,
    helium_index,
    ionized_carbon_index,
    co_cooling,
    carbon_monoxide_index,
    co_cooling_table,
    co_cooling_bounds,
):
    """Gas temperature derivative dT/dt [K s^-1] from net heating minus cooling.

    ``dT/dt = (gamma - 1) * (heating - cooling) / (k_B * n_total)`` with
    ``n_total`` the total particle number density (sum of abundances).

    Args:
        abundances: The per-cell abundances [cm^-3].
        temperature_kelvin: The gas temperature [K].
        adiabatic_index: The adiabatic index gamma.
        cosmic_ray_rate: The cosmic-ray ionization rate [s^-1].
        fuv_field: The FUV radiation field (Draine units).
        dust_to_gas_ratio: The dust-to-gas mass ratio.
        hydrogen_molecule_formation_rate_coefficient: The H + H -> H2 grain
            formation rate coefficient [cm^3 s^-1].
        hydrogen_index: Index of atomic H (or -1).
        molecular_hydrogen_index: Index of H2 (or -1).
        electron_index: Index of the electron (or -1).
        atomic_oxygen_index: Index of atomic O (or -1).
        ionized_hydrogen_index: Index of H+ (or -1).
        helium_index: Index of He (or -1).
        ionized_carbon_index: Index of C+ (or -1).
        co_cooling: Whether to add tabulated CO rotational cooling.
        carbon_monoxide_index: Index of CO (or -1).
        co_cooling_table: The 3D CO cooling table.
        co_cooling_bounds: The CO table grid limits.

    Returns:
        The temperature derivative dT/dt.
    """
    net_heating = heating_rate(
        abundances,
        temperature_kelvin,
        cosmic_ray_rate,
        fuv_field,
        dust_to_gas_ratio,
        hydrogen_molecule_formation_rate_coefficient,
        hydrogen_index,
        molecular_hydrogen_index,
        electron_index,
    ) - cooling_rate(
        abundances,
        temperature_kelvin,
        hydrogen_index,
        molecular_hydrogen_index,
        electron_index,
        atomic_oxygen_index,
        ionized_hydrogen_index,
        helium_index,
        ionized_carbon_index,
        co_cooling,
        carbon_monoxide_index,
        co_cooling_table,
        co_cooling_bounds,
    )

    total_number_density = jnp.sum(abundances)
    return (
        (adiabatic_index - 1.0)
        * net_heating
        / (BOLTZMANN_CONSTANT_CGS * total_number_density)
    )
