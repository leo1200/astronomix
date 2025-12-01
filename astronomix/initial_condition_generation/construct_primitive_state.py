from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import FIELD_TYPE, FINITE_DIFFERENCE, STATE_TYPE, SimulationConfig

from typing import Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped


from functools import partial
from types import NoneType


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["registered_variables", "config", "sharding"])
def construct_primitive_state(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    density: FIELD_TYPE,
    velocity_x: Union[FIELD_TYPE, NoneType] = None,
    velocity_y: Union[FIELD_TYPE, NoneType] = None,
    velocity_z: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_x: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_y: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_z: Union[FIELD_TYPE, NoneType] = None,
    interface_magnetic_field_x: Union[FIELD_TYPE, NoneType] = None,
    interface_magnetic_field_y: Union[FIELD_TYPE, NoneType] = None,
    interface_magnetic_field_z: Union[FIELD_TYPE, NoneType] = None,
    gas_pressure: Union[FIELD_TYPE, NoneType] = None,
    cosmic_ray_pressure: Union[FIELD_TYPE, NoneType] = None,
    sharding=None,
) -> STATE_TYPE:
    """Stack the primitive variables into the state array.

    IN 1D SET ONLY THE XCOMPONENTS, in 2D SET X AND Y COMPONENTS,
    in 3D SET X, Y AND Z COMPONENTS

    Args:
        config: The simulation configuration.
        registered_variables: The indices of the variables in the state array.
        density: The density of the fluid.
        velocity_x: The x-component of the velocity of the fluid.
        velocity_y: The y-component of the velocity of the fluid.
        velocity_z: The z-component of the velocity of the fluid.
        magnetic_field_x: The x-component of the magnetic field in B / sqrt(\mu_0).
        magnetic_field_y: The y-component of the magnetic field in B / sqrt(\mu_0).
        magnetic_field_z: The z-component of the magnetic field in B / sqrt(\mu_0).
        gas_pressure: The thermal pressure of the fluid.
        cosmic_ray_pressure: The cosmic ray pressure of the fluid.

    Returns:
        The state array.
    """
    if sharding is not None:
        state = jax.lax.with_sharding_constraint(
            jnp.zeros((registered_variables.num_vars, *density.shape)), sharding
        )
    else:
        state = jnp.zeros((registered_variables.num_vars, *density.shape))

    state = state.at[registered_variables.density_index].set(density)

    if config.dimensionality == 1:
        state = state.at[registered_variables.velocity_index].set(velocity_x)
    elif config.dimensionality == 2:
        state = state.at[registered_variables.velocity_index.x].set(velocity_x)
        state = state.at[registered_variables.velocity_index.y].set(velocity_y)
    elif config.dimensionality == 3:
        state = state.at[registered_variables.velocity_index.x].set(velocity_x)
        state = state.at[registered_variables.velocity_index.y].set(velocity_y)
        state = state.at[registered_variables.velocity_index.z].set(velocity_z)

    if config.mhd:
        if config.dimensionality >= 2:
            if magnetic_field_x is not None:
                state = state.at[registered_variables.magnetic_index.x].set(
                    magnetic_field_x
                )
            if magnetic_field_y is not None:
                state = state.at[registered_variables.magnetic_index.y].set(
                    magnetic_field_y
                )
            if magnetic_field_z is not None:
                state = state.at[registered_variables.magnetic_index.z].set(
                    magnetic_field_z
            )

        if config.solver_mode == FINITE_DIFFERENCE:
            if interface_magnetic_field_x is not None:
                state = state.at[
                    registered_variables.interface_magnetic_field_index.x
                ].set(interface_magnetic_field_x)
            if interface_magnetic_field_y is not None:
                state = state.at[
                    registered_variables.interface_magnetic_field_index.y
                ].set(interface_magnetic_field_y)
            if interface_magnetic_field_z is not None:
                state = state.at[
                    registered_variables.interface_magnetic_field_index.z
                ].set(interface_magnetic_field_z)
    
    state = state.at[registered_variables.pressure_index].set(gas_pressure)

    if registered_variables.cosmic_ray_n_active:
        # TODO: get from params
        gamma_cr = 4 / 3

        state = state.at[registered_variables.pressure_index].set(
            gas_pressure + cosmic_ray_pressure
        )
        state = state.at[registered_variables.cosmic_ray_n_index].set(
            cosmic_ray_pressure ** (1 / gamma_cr)
        )

    return state