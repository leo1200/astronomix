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

from astronomix._pallas_helpers import diffable_pallas_call_n
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    STATE_TYPE,
    SimulationConfig,
)


def _enforce_positivity_native(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    return _enforce_positivity_native_impl(
        conserved_state, config, gamma,
        minimum_density, minimum_pressure, registered_variables,
    )


@partial(
    jax.jit, static_argnames=["registered_variables", "config"]
)
def _enforce_positivity(
    conserved_state: STATE_TYPE,
    config: SimulationConfig,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    if _enforce_positivity_pallas_supported(conserved_state, config):
        pallas = lambda s, g, mr, mp: _enforce_positivity_pallas(  # noqa: E731
            s, config, g, mr, mp, registered_variables,
        )
        native = lambda s, g, mr, mp: _enforce_positivity_native(  # noqa: E731
            s, g, mr, mp, config, registered_variables,
        )
        return diffable_pallas_call_n(
            (conserved_state, gamma, minimum_density, minimum_pressure),
            pallas_branch=pallas, native_branch=native,
            ad_mode=config.pallas_ad_mode,
        )
    return _enforce_positivity_native(
        conserved_state, gamma, minimum_density, minimum_pressure,
        config, registered_variables,
    )


def _enforce_positivity_native_impl(
    conserved_state: STATE_TYPE,
    config: SimulationConfig,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    rho = conserved_state[registered_variables.density_index]

    # enforce minimum density
    rho = jnp.maximum(rho, minimum_density)

    # the energy only needs to be updated in the ideal gas case
    if config.equation_of_state == IDEAL_GAS:

        if config.dimensionality == 1:
            v_x = conserved_state[registered_variables.momentum_index] / rho
        else:
            v_x = conserved_state[registered_variables.momentum_index.x] / rho

        if config.dimensionality == 2:
            v_y = conserved_state[registered_variables.momentum_index.y] / rho
            v_z = 0.0
        elif config.dimensionality == 3:
            v_y = conserved_state[registered_variables.momentum_index.y] / rho
            v_z = conserved_state[registered_variables.momentum_index.z] / rho

        energy = conserved_state[registered_variables.energy_index]

        if config.mhd:
            B_x = conserved_state[registered_variables.magnetic_index.x]
            B_y = conserved_state[registered_variables.magnetic_index.y]
            B_z = conserved_state[registered_variables.magnetic_index.z]

            b2 = B_x**2 + B_y**2 + B_z**2
        
        if config.dimensionality == 1:
            v2 = v_x**2
        elif config.dimensionality == 2:
            v2 = v_x**2 + v_y**2
        elif config.dimensionality == 3:
            v2 = v_x**2 + v_y**2 + v_z**2

        # calculate pressure
        if config.mhd:
            pressure = (gamma - 1.0) * (energy - 0.5 * rho * v2 - 0.5 * b2)
        else:
            pressure = (gamma - 1.0) * (energy - 0.5 * rho * v2)
        
        pressure = jnp.maximum(pressure, minimum_pressure)

        # redefine energy with new pressure
        if config.mhd:
            energy = pressure / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * b2
        else:
            energy = pressure / (gamma - 1.0) + 0.5 * rho * v2

        # reconstruct conserved state
        conserved_state = conserved_state.at[registered_variables.energy_index].set(energy)
    
    # for both the ideal gas and isothermal case, we need to update the density
    conserved_state = conserved_state.at[registered_variables.density_index].set(rho)

    return conserved_state

# Bottom-of-file Pallas import (avoids circular import — see guide §2.4).
from astronomix._finite_difference._fluid_equations._enforce_positivity_pallas import (  # noqa: E402
    _enforce_positivity_pallas,
    _enforce_positivity_pallas_supported,
)
