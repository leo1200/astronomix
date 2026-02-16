# general imports
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# general astronomix
from astronomix._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from astronomix._finite_volume._riemann_solver._lax_friedrichs import _lax_friedrichs_solver
from astronomix._finite_volume._riemann_solver.hll import _am_hllc_solver, _hll_solver, _hllc_solver
from astronomix.variable_registry.registered_variables import RegisteredVariables

# fluid stuff
from astronomix._fluid_equations._equations import conserved_state_from_primitive, speed_of_sound
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix.option_classes.simulation_config import (
    AM_HLLC,
    HLL,
    HLLC,
    HLLC_LM,
    HYBRID_HLLC,
    LAX_FRIEDRICHS,
    STATE_TYPE,
    STATE_TYPE_ALTERED,
    SimulationConfig,
)


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
