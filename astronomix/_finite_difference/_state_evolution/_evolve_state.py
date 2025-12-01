# general imports
import jax.numpy as jnp
import jax
from functools import partial


# type checking imports
from jaxtyping import Array, Float
from beartype import beartype as typechecker
from typing import Union

# general astronomix imports
from astronomix._finite_difference._fluid_equations._equations import conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from astronomix._finite_difference._time_integrators._ssprk import _ssprk4_with_ct
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    STATE_TYPE,
    SimulationConfig,
)

from astronomix.option_classes.simulation_params import SimulationParams

@partial(jax.jit, static_argnames=["config", "registered_variables"], donate_argnames=["primitive_state"])
def _evolve_state_fd(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    
    # NOTE: here we assume the magnetic field at interfaces
    # is stored in the last three indices of the state array
    conserved_state = conserved_state_from_primitive_mhd(
        primitive_state[:-3], gamma, registered_variables
    )

    # extract interface magnetic fields
    bxb = primitive_state[registered_variables.interface_magnetic_field_index.x]
    byb = primitive_state[registered_variables.interface_magnetic_field_index.y]
    bzb = primitive_state[registered_variables.interface_magnetic_field_index.z]

    # update conserved state and interface magnetic fields
    conserved_state, bxb, byb, bzb = _ssprk4_with_ct(
        conserved_state,
        bxb,
        byb,
        bzb,
        gamma,
        config.grid_spacing,
        dt,
        params,
        helper_data,
        config,
        registered_variables,
    )

    # back to primitive state
    primitive_state = primitive_state_from_conserved_mhd(
        conserved_state, params.minimum_density, params.minimum_pressure, gamma, config, registered_variables
    )

    # append updated interface magnetic fields
    # NOTE: same assumption as above
    primitive_state = jnp.concatenate(
        [primitive_state, bxb[None, :], byb[None, :], bzb[None, :]], axis=0
    )

    return primitive_state