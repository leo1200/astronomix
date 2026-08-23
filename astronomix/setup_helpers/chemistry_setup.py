"""
Build the chemistry configuration from a carbox reaction network file.

Turns a carbox network CSV into the pair of containers the chemistry module
consumes: a static ``ChemistryConfig`` (species count and ordering, solver, and
the network's structural half) and a dynamic ``ChemistryParams`` (the network's
array half, unit-conversion factors, physical parameters and rate modifiers).

The reaction network (a carbox ``JNetwork``) is an Equinox module, i.e. a JAX
pytree of arrays plus structure. Because it holds a Python list of reactions it
is not hashable, so it is carried in the dynamic ``ChemistryParams`` rather than
the static ``ChemistryConfig``; it stays differentiable in its rate constants.

WARNING: This helper imports carbox. It is an optional dependency; the import is
kept local so that ``import astronomix`` does not require carbox unless this
helper is actually called.
"""

# typing
from typing import Tuple

# numerics
import numpy as np

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix._modules._chemistry.chemistry_options import KVAERNO5

# astronomix containers
from astronomix._modules._chemistry.chemistry_options import (
    ChemistryConfig,
    ChemistryParams,
)

# The Neufeld & Kaufman CO cooling table is KROME/Omukai data (GPL) and is NOT
# bundled here; the caller must supply the path to their own ``coolCO.dat`` when
# CO cooling is requested. There is deliberately no default path.
DEFAULT_CO_COOLING_TABLE_PATH = None


def _load_co_cooling_table(table_path):
    """Load KROME's ``coolCO.dat`` into a dense 3D table and its grid bounds.

    The file lists, per row, the integer grid indices (i, j, k) followed by
    ``log10(T)``, ``log10(n_H + n_H2)``, ``log10(N_CO)`` and ``log10`` of the
    cooling coefficient. The grid is uniform in each log axis.

    Args:
        table_path: Path to ``coolCO.dat``.

    Returns:
        A tuple ``(table, bounds)`` with the table shaped
        ``(n_logT, n_logn, n_logNCO)`` and the six grid limits
        ``[logT_min, logT_max, logn_min, logn_max, logNCO_min, logNCO_max]``.
    """
    rows = np.loadtxt(table_path, comments="#")
    index_temperature = rows[:, 0].astype(int) - 1
    index_density = rows[:, 1].astype(int) - 1
    index_column = rows[:, 2].astype(int) - 1
    log_temperature = rows[:, 3]
    log_density = rows[:, 4]
    log_column = rows[:, 5]
    log_cooling = rows[:, 6]

    table = np.zeros(
        (
            index_temperature.max() + 1,
            index_density.max() + 1,
            index_column.max() + 1,
        )
    )
    table[index_temperature, index_density, index_column] = log_cooling
    bounds = np.array(
        [
            log_temperature.min(),
            log_temperature.max(),
            log_density.min(),
            log_density.max(),
            log_column.min(),
            log_column.max(),
        ]
    )
    return jnp.asarray(table), jnp.asarray(bounds)


def build_chemistry_from_network_file(
    network_csv_path: str,
    network_format: str,
    number_density_unit_cgs: float,
    temperature_unit_kelvin: float,
    time_unit_seconds: float,
    cosmic_ray_rate: float = 1e-17,
    fuv_field: float = 1.0,
    visual_extinction: float = 2.0,
    absolute_tolerance: float = 1e-18,
    relative_tolerance: float = 1e-10,
    solver: int = KVAERNO5,
    max_steps: int = 4096,
    thermochemistry: bool = False,
    dust_to_gas_ratio: float = 1e-2,
    floor_temperature: float = 1e1,
    hydrogen_molecule_formation_rate_coefficient: float = 3e-17,
    co_cooling: bool = False,
    co_cooling_table_path=DEFAULT_CO_COOLING_TABLE_PATH,
) -> Tuple[ChemistryConfig, ChemistryParams, Tuple[str, ...]]:
    """Assemble the chemistry containers from a carbox network file.

    Args:
        network_csv_path: Path to the carbox reaction-network CSV.
        network_format: carbox parser format, e.g. ``"latent_tgas"``.
        number_density_unit_cgs: Number density [cm^-3] per code number-density
            unit (converts species densities into CGS for the network).
        temperature_unit_kelvin: Kelvin per code (rescaled) temperature unit.
        time_unit_seconds: Seconds per code time unit.
        cosmic_ray_rate: Cosmic-ray ionization rate [s^-1].
        fuv_field: FUV radiation field strength (Draine units).
        visual_extinction: Visual extinction Av [mag].
        absolute_tolerance: Absolute tolerance of the stiff solver.
        relative_tolerance: Relative tolerance of the stiff solver.
        solver: Stiff-solver tag (see ``chemistry_options``).
        max_steps: Maximum internal Diffrax steps per cell per hydro step.
        thermochemistry: When True, evolve the temperature with the abundances
            (heating/cooling) and write the result back into the pressure field.
        dust_to_gas_ratio: Dust-to-gas mass ratio (grain photoelectric heating).
        floor_temperature: Lower bound [K] on the post-reaction temperature.
        hydrogen_molecule_formation_rate_coefficient: H + H -> H2 grain formation
            rate coefficient [cm^3 s^-1] (H2 formation heating).
        co_cooling: When True, add tabulated CO rotational cooling (loads the
            Neufeld & Kaufman table from ``co_cooling_table_path``).
        co_cooling_table_path: Path to KROME's ``coolCO.dat``.

    Returns:
        A tuple ``(chemistry_config, chemistry_params, species_names)``. The
        species ordering is returned so the caller can place initial abundances
        into the matching state slots.
    """

    # carbox is an optional dependency; import locally so the package imports
    # without it when chemistry is unused.
    from carbox.network import Network
    from carbox.parsers import parse_chemical_network

    # A dense stoichiometry matrix keeps the per-cell right-hand side a plain
    # matmul, which vmaps cleanly across the grid (the default sparse BCOO does
    # not).
    parsed_network = parse_chemical_network(network_csv_path, network_format)
    dense_network = Network(
        parsed_network.species,
        parsed_network.reactions,
        use_sparse=False,
    )
    reaction_network = dense_network.get_ode()

    species_names = tuple(species.name for species in dense_network.species)
    number_of_species = len(species_names)
    number_of_reactions = reaction_network.reactions_number

    # Resolve the species the thermochemistry terms reference. A species absent
    # from the network maps to -1, and its heating/cooling contribution drops out.
    def species_index(name):
        return species_names.index(name) if name in species_names else -1

    # Load the CO cooling table only when requested.
    co_cooling_table = jnp.array([])
    co_cooling_bounds = jnp.array([])
    if co_cooling:
        if co_cooling_table_path is None:
            raise ValueError(
                "co_cooling=True requires co_cooling_table_path pointing to a "
                "Neufeld & Kaufman CO cooling table (e.g. KROME's coolCO.dat); "
                "no table is bundled."
            )
        co_cooling_table, co_cooling_bounds = _load_co_cooling_table(
            co_cooling_table_path
        )

    chemistry_config = ChemistryConfig(
        chemistry=True,
        number_of_chemical_species=number_of_species,
        species_names=species_names,
        number_of_reactions=number_of_reactions,
        solver=solver,
        max_steps=max_steps,
        thermochemistry=thermochemistry,
        hydrogen_index=species_index("H"),
        molecular_hydrogen_index=species_index("H2"),
        electron_index=species_index("E"),
        atomic_oxygen_index=species_index("O"),
        ionized_hydrogen_index=species_index("H+"),
        helium_index=species_index("He"),
        ionized_carbon_index=species_index("C+"),
        co_cooling=co_cooling,
        carbon_monoxide_index=species_index("CO"),
    )

    chemistry_params = ChemistryParams(
        network=reaction_network,
        number_density_unit_cgs=number_density_unit_cgs,
        temperature_unit_kelvin=temperature_unit_kelvin,
        time_unit_seconds=time_unit_seconds,
        cosmic_ray_rate=cosmic_ray_rate,
        fuv_field=fuv_field,
        visual_extinction=visual_extinction,
        dust_to_gas_ratio=dust_to_gas_ratio,
        floor_temperature=floor_temperature,
        hydrogen_molecule_formation_rate_coefficient=(
            hydrogen_molecule_formation_rate_coefficient
        ),
        atol=absolute_tolerance,
        rtol=relative_tolerance,
        rate_modifier_a=jnp.ones(number_of_reactions),
        rate_modifier_b=jnp.zeros(number_of_reactions),
        co_cooling_table=co_cooling_table,
        co_cooling_bounds=co_cooling_bounds,
    )

    return chemistry_config, chemistry_params, species_names
