"""
Geometric source terms for curvilinear finite-volume updates.

Provides the pressure-nozzling source term that accounts for the changing cell
cross-section in spherical geometry (Crittenden & Balachandar, 2018).
"""

# general
from functools import partial

# typing
from typing import Union
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE_ALTERED

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _pressure_nozzling_source(
    pressure,
    pressure_gradient,
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE_ALTERED:
    """Pressure nozzling source term as of the geometry of the domain.

    Args:
        pressure: The gas pressure of the cells (given by the caller, so that the
            source serves a state without pressure slot as well).
        pressure_gradient: The limited radial pressure gradient of the cells
            (the scheme's own, so that the source matches its reconstruction).
        config: The simulation configuration.
        helper_data: The helper data.
        registered_variables: The registered variables.

    Returns:
        The pressure nozzling source
    """

    p = pressure

    dp_dr = pressure_gradient

    # calculate the pressure nozzling term, following
    # eq. 14 in Crittendend and Balachandar, 2018
    # https://doi.org/10.1007/s00193-017-0784-y
    pressure_nozzling = (
        helper_data.geometric_centers ** (config.geometry - 1) * p
        + (
            helper_data.r_hat_alpha
            - helper_data.volumetric_centers
            * helper_data.geometric_centers ** (config.geometry - 1)
        )
        * dp_dr
    )
    nozzling = jnp.zeros((registered_variables.num_vars, p.shape[0]))
    nozzling = nozzling.at[registered_variables.velocity_index].set(
        config.geometry * pressure_nozzling
    )

    return nozzling
