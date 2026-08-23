"""
Configuration and parameter containers for advected chemical species.

Carries the number and ordering of the species tracked in the state array and an
optional reaction ``source_term`` — a user-supplied operator-split update applied
once per hydro step. astronomix does not implement any chemistry itself; it only
registers the species (so they advect) and calls the source term if one is given.
This keeps the core free of any specific chemistry engine or its dependencies.
"""

# typing
from typing import Callable, NamedTuple, Tuple, Union
from types import NoneType
from jaxtyping import PyTree


class ChemistryConfig(NamedTuple):
    """Static configuration for advected chemical species.

    Every field is hashable, since the configuration is passed as a static
    argument to the jitted update; the ``source_term`` is a plain callable.

    Attributes:
        chemistry: Master switch for carrying/advecting species and applying the
            source term.
        number_of_chemical_species: Number of species carried in the state array
            (a contiguous, advected scalar block).
        species_names: Species ordering — labels for the state block.
        source_term: Optional operator-split update applied once per hydro step,
            with signature
            ``source_term(primitive_state, registered_variables, chemistry_config,
            chemistry_params, dt) -> primitive_state``.
            ``None`` leaves the species advected but chemically inert. The update
            is entirely user-defined (e.g. a reaction network plus heating /
            cooling); astronomix only invokes it.
    """

    chemistry: bool = False
    number_of_chemical_species: int = 0
    species_names: Tuple[str, ...] = ()
    source_term: Union[Callable, NoneType] = None


class ChemistryParams(NamedTuple):
    """Runtime parameters forwarded, uninterpreted, to the reaction source term.

    Attributes:
        source_term_params: An opaque pytree the source term needs (e.g. rate
            data, unit-conversion factors, a network object). astronomix does not
            read it — it is threaded straight through to ``source_term`` so the
            update stays differentiable in whatever it contains.
    """

    source_term_params: Union[PyTree, NoneType] = None
