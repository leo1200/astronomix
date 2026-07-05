"""
Dispatch wrapper selecting the finite-volume Riemann solver.

Routes to the configured solver (HLL, HLLC / HLLC-LM, AM-HLLC / hybrid HLLC, or
Lax-Friedrichs) and returns the conservative interface fluxes. Also carries a
back-compatibility shim for the FV-Pallas support predicate.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    AM_HLLC,
    HLL,
    HLLC,
    HLLC_LM,
    HYBRID_HLLC,
    LAX_FRIEDRICHS,
    PALLAS,
    STATE_TYPE,
    STATE_TYPE_ALTERED,
)

# astronomix containers
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from astronomix._finite_volume._riemann_solver._lax_friedrichs import _lax_friedrichs_solver
from astronomix._finite_volume._riemann_solver.hll import _am_hllc_solver, _hll_solver, _hllc_solver
from astronomix._fluid_equations._equations import conserved_state_from_primitive, speed_of_sound
from astronomix._fluid_equations._fluxes import _euler_flux


def _fv_pallas_supported(state, config: SimulationConfig) -> bool:
    """Back-compat shim.  The real FV-Pallas predicate now lives in
    ``astronomix._finite_volume._state_evolution._pallas_evolve`` as
    ``_fv_pallas_evolve_supported``; this function just re-exports it so
    older external callers still resolve.
    """
    from astronomix._finite_volume._state_evolution._pallas_evolve import (
        _fv_pallas_evolve_supported,
    )
    return _fv_pallas_evolve_supported(state, config)


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["config", "registered_variables", "flux_direction_index"]
)
def _riemann_solver(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    primitive_state: STATE_TYPE_ALTERED,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int,
) -> STATE_TYPE:
    """Wrapper function for the Riemann solver."""
    if config.riemann_solver == HLL:
        return _hll_solver(
            primitives_left,
            primitives_right,
            gamma,
            config,
            registered_variables,
            flux_direction_index,
        )
    elif config.riemann_solver == HLLC or config.riemann_solver == HLLC_LM:
        return _hllc_solver(
            primitives_left,
            primitives_right,
            gamma,
            config,
            registered_variables,
            flux_direction_index,
        )
    elif config.riemann_solver == AM_HLLC or config.riemann_solver == HYBRID_HLLC:
        return _am_hllc_solver(
            primitives_left,
            primitives_right,
            primitive_state,
            gamma,
            config,
            registered_variables,
            flux_direction_index,
        )
    elif config.riemann_solver == LAX_FRIEDRICHS:
        return _lax_friedrichs_solver(
            primitives_left,
            primitives_right,
            primitive_state,
            gamma,
            config,
            registered_variables,
            flux_direction_index,
        )

    else:
        raise ValueError("Riemann solver not supported.")
