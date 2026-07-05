"""
Assemble the primitive state array from individual primitive fields.

Given the per-variable fields (density, velocities, magnetic field components,
pressures, ...) this stacks them into the single state array used throughout
the solver, placing each field at the index dictated by ``registered_variables``
for the active configuration (dimensionality, MHD, solver mode, equation of
state, cosmic rays).
"""

# general
from functools import partial

# typing
from typing import Union
from types import NoneType
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables


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

    In 1D set only the x-components, in 2D set the x- and y-components, and in
    3D set the x-, y- and z-components.

    Args:
        config: The simulation configuration.
        registered_variables: The indices of the variables in the state array.
        density: The density of the fluid.
        velocity_x: The x-component of the velocity of the fluid.
        velocity_y: The y-component of the velocity of the fluid.
        velocity_z: The z-component of the velocity of the fluid.
        magnetic_field_x: The x-component of the magnetic field in B / sqrt(mu_0).
        magnetic_field_y: The y-component of the magnetic field in B / sqrt(mu_0).
        magnetic_field_z: The z-component of the magnetic field in B / sqrt(mu_0).
        interface_magnetic_field_x: The x-component of the face-centered
            (interface) magnetic field, used by the finite-difference solver.
        interface_magnetic_field_y: The y-component of the face-centered
            (interface) magnetic field, used by the finite-difference solver.
        interface_magnetic_field_z: The z-component of the face-centered
            (interface) magnetic field, used by the finite-difference solver.
        gas_pressure: The thermal pressure of the fluid.
        cosmic_ray_pressure: The cosmic ray pressure of the fluid.
        sharding: An optional sharding to apply to the allocated state array.

    Returns:
        The state array.
    """
    # Allocate the (optionally sharded) empty state array; the per-variable
    # fields are written into their registered slots below.
    if sharding is not None:
        state = jax.lax.with_sharding_constraint(
            jnp.zeros((registered_variables.num_vars, *density.shape)), sharding
        )
    else:
        state = jnp.zeros((registered_variables.num_vars, *density.shape))

    state = state.at[registered_variables.density_index].set(density)

    # The velocity index is a scalar in 1D and a per-axis vector otherwise.
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
            # The finite-difference MHD state always carries all three velocity
            # components; any not supplied stay zero by default.
            if velocity_y is not None:
                state = state.at[registered_variables.velocity_index.y].set(velocity_y)
            if velocity_z is not None:
                state = state.at[registered_variables.velocity_index.z].set(velocity_z)

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

    # For an ideal gas the pressure is an independent variable; in the
    # isothermal case it instead follows from p = c_s^2 * rho and is not stored.
    if config.equation_of_state == IDEAL_GAS:
        state = state.at[registered_variables.pressure_index].set(gas_pressure)

    if registered_variables.cosmic_ray_n_active:
        # TODO: take the cosmic-ray adiabatic index from params instead of
        # hard-coding the relativistic value 4/3.
        gamma_cr = 4 / 3

        # The stored pressure is the combined gas + cosmic-ray pressure, while
        # the cosmic-ray number variable encodes the CR pressure to the power
        # 1/gamma_cr.
        state = state.at[registered_variables.pressure_index].set(
            gas_pressure + cosmic_ray_pressure
        )
        state = state.at[registered_variables.cosmic_ray_n_index].set(
            cosmic_ray_pressure ** (1 / gamma_cr)
        )

    return state
