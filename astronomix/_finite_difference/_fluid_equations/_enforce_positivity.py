"""
Here we protect the density and pressure from going negative.

In my view this is a bit of a shady practice, hiding unphysical
updates under the rug. However, it is common practice.
"""

import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float

from typing import Union

from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    STATE_TYPE,
)

@partial(
    jax.jit, static_argnames=["registered_variables"]
)
def _enforce_positivity(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    

    rho = conserved_state[registered_variables.density_index]

    # enforce minimum density
    rho = jnp.maximum(rho, minimum_density)

    v_x = conserved_state[registered_variables.momentum_index.x] / rho
    v_y = conserved_state[registered_variables.momentum_index.y] / rho
    v_z = conserved_state[registered_variables.momentum_index.z] / rho
    energy = conserved_state[registered_variables.energy_index]
    B_x = conserved_state[registered_variables.magnetic_index.x]
    B_y = conserved_state[registered_variables.magnetic_index.y]
    B_z = conserved_state[registered_variables.magnetic_index.z]

    b2 = B_x**2 + B_y**2 + B_z**2
    v2 = v_x**2 + v_y**2 + v_z**2

    # calculate pressure
    pressure = (gamma - 1.0) * (energy - 0.5 * rho * v2 - 0.5 * b2)
    pressure = jnp.maximum(pressure, minimum_pressure)

    # redefine energy with new pressure
    energy = pressure / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * b2

    # reconstruct conserved state
    conserved_state = conserved_state.at[registered_variables.density_index].set(rho)
    conserved_state = conserved_state.at[registered_variables.energy_index].set(energy)

    return conserved_state