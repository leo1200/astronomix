"""
Configuration and parameter containers for the chemistry module.

Defines the integer tags selecting the stiff ODE solver and the NamedTuples that
carry the reaction network (split into a static structure and dynamic array
leaves, following the neural-network cooling pattern), the unit-conversion
factors between astronomix code units and the CGS units carbox expects, and the
per-network physical parameters.
"""

# typing
from typing import NamedTuple, Tuple, Union
from types import NoneType
from jaxtyping import PyTree

# jax
import jax.numpy as jnp

# Stiff-solver tags (select which Diffrax solver integrates the network). Only
# Kvaerno5 is wired up for now; the others leave room to switch without changing
# the container layout.
KVAERNO5 = 1
DOPRI5 = 2
TSIT5 = 3


class ChemistryConfig(NamedTuple):
    """Top-level chemistry configuration.

    Every field must be hashable because the simulation configuration is passed
    as a static argument to the jitted update. The number of species and their
    order are needed at registry-build time to allocate the contiguous species
    block in the state array. The reaction network itself is deliberately *not*
    stored here: a carbox ``JNetwork`` holds a Python list of reactions and is
    therefore unhashable, so it lives in ``ChemistryParams`` (a dynamic argument)
    instead.

    Attributes:
        chemistry: Master switch for the chemistry module.
        number_of_chemical_species: Length of the species block in the state.
        species_names: Species ordering, matching both the state block and the
            carbox network's ``species`` list.
        number_of_reactions: Number of reactions in the network (length of the
            rate-modifier vectors).
        solver: Stiff-solver tag (see the module-level constants).
        max_steps: Maximum internal Diffrax steps per cell per hydro step.
        thermochemistry: When True, the temperature is evolved together with the
            abundances (heating/cooling feedback) and written back into the
            pressure field. When False the species react at a fixed temperature
            and the energy field is left untouched.
        hydrogen_index: Position of atomic H in the species block, or -1 if the
            network has no such species. Resolved once from ``species_names``;
            the thermochemistry terms that reference a missing species contribute
            zero.
        molecular_hydrogen_index: Position of H2, or -1.
        electron_index: Position of the electron (E), or -1.
        atomic_oxygen_index: Position of atomic O, or -1.
        ionized_hydrogen_index: Position of H+, or -1 (H2-H+ cooling collider).
        helium_index: Position of He, or -1 (H2-He cooling collider).
        ionized_carbon_index: Position of C+, or -1 ([C II] 158 um cooling).
        co_cooling: When True, add tabulated CO rotational-line cooling
            (Neufeld & Kaufman 1993). Requires the table in ``ChemistryParams``.
        carbon_monoxide_index: Position of CO, or -1 (CO cooling).
    """

    chemistry: bool = False
    number_of_chemical_species: int = 0
    species_names: Tuple[str, ...] = ()
    number_of_reactions: int = 0
    solver: int = KVAERNO5
    max_steps: int = 4096

    # thermochemistry (chemistry-driven heating/cooling of the energy field)
    thermochemistry: bool = False
    hydrogen_index: int = -1
    molecular_hydrogen_index: int = -1
    electron_index: int = -1
    atomic_oxygen_index: int = -1
    ionized_hydrogen_index: int = -1
    helium_index: int = -1
    ionized_carbon_index: int = -1
    co_cooling: bool = False
    carbon_monoxide_index: int = -1


class ChemistryParams(NamedTuple):
    """Runtime chemistry parameters (differentiable leaves).

    Bundles the network's array leaves, the code-unit <-> CGS conversion factors,
    the physical parameters the rate expressions read, and the per-reaction rate
    modifiers.

    Attributes:
        network: The carbox reaction network (``JNetwork``). It is an Equinox
            module — a JAX pytree of arrays plus structure — so it rides here as
            a dynamic argument and stays differentiable in its rate constants.
        hydrogen_mass_fraction: Hydrogen mass fraction X, used for the mean
            molecular weight in the pressure-to-temperature conversion.
        metal_mass_fraction: Metal mass fraction Z, used likewise.
        number_density_unit_cgs: Number density [cm^-3] per code number-density
            unit; converts species and hydrogen densities into CGS.
        temperature_unit_kelvin: Kelvin per code (rescaled) temperature unit.
        time_unit_seconds: Seconds per code time unit.
        cosmic_ray_rate: Cosmic-ray ionization rate [s^-1].
        fuv_field: FUV radiation field strength (Draine units).
        visual_extinction: Visual extinction Av [mag].
        dust_to_gas_ratio: Dust-to-gas mass ratio, used by the grain
            photoelectric heating term.
        floor_temperature: Lower bound [K] on the post-reaction temperature
            before it is written back to the pressure field.
        hydrogen_molecule_formation_rate_coefficient: The H + H -> H2 grain
            formation rate coefficient [cm^3 s^-1], used by the H2 formation
            heating term.
        atol: Absolute tolerance of the stiff solver.
        rtol: Relative tolerance of the stiff solver.
        rate_modifier_a: Per-reaction multiplicative rate modifier (ones = no-op).
        rate_modifier_b: Per-reaction additive rate modifier (zeros = no-op).
        co_cooling_table: The 3D Neufeld & Kaufman CO cooling table, shape
            ``(n_logT, n_logn, n_logNCO)`` of ``log10`` cooling coefficients.
            Empty when CO cooling is inactive.
        co_cooling_bounds: The six uniform-log-grid limits
            ``[logT_min, logT_max, logn_min, logn_max, logNCO_min, logNCO_max]``.
    """

    network: Union[PyTree, NoneType] = None

    # composition used to convert pressure to temperature (mean molecular weight)
    hydrogen_mass_fraction: float = 0.76
    metal_mass_fraction: float = 0.02

    # code-unit <-> CGS conversion factors (the chemistry boundary)
    number_density_unit_cgs: float = 1.0
    temperature_unit_kelvin: float = 1.0
    time_unit_seconds: float = 1.0

    # physical parameters entering the rate expressions
    cosmic_ray_rate: float = 1e-17
    fuv_field: float = 1.0
    visual_extinction: float = 2.0

    # thermochemistry parameters (heating/cooling)
    dust_to_gas_ratio: float = 1e-2
    floor_temperature: float = 1e1
    # H + H -> H2 grain formation rate coefficient [cm^3 s^-1] (formation heating)
    hydrogen_molecule_formation_rate_coefficient: float = 3e-17

    # stiff-solver tolerances
    atol: float = 1e-18
    rtol: float = 1e-10

    # per-reaction rate modifiers, rate -> rate * a + b
    rate_modifier_a: jnp.ndarray = jnp.array([])
    rate_modifier_b: jnp.ndarray = jnp.array([])

    # tabulated CO rotational cooling (Neufeld & Kaufman 1993)
    co_cooling_table: jnp.ndarray = jnp.array([])
    co_cooling_bounds: jnp.ndarray = jnp.array([])
