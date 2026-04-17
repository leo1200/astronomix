"""
The registered_variables module tells the code where 
in the state array which field is stored. This is important 
for modularity and readability of the code.

When you want to add new variables to the state array, e.g.
densities for chemical species, you have to register them here.

NOTE: For finite volume MHD simulation, the magnetic field
is assumed to be stored in the last three indices of the state array
and for finite difference MHD the magnetic field at interfaces
is assumed to be stored in the three indices.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jaxtyping import Array, Float, Int

from typing import Union

from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, FINITE_VOLUME, IDEAL_GAS, ISOTHERMAL, XAXIS, YAXIS, ZAXIS, SimulationConfig, StaticIntVector
)


# =============================================================

"""
We perform simulations in multiple spatial dimensions. To an axis,
e.g. the x-axis, corresponds the axis in the array storing the state
(along which e.g. x varies) but also fields in the state related
to that axis, e.g. the x-velocity or the x-magnetic field component.

A common pattern will be a loop over the spatial dimensions where
we exactly need to know this information.
"""

class AxisInfo(NamedTuple):
    # corresponding axis in the array
    axis_in_array: int

    # corresponding field indices
    velocity_index: int
    magnetic_index: int

# =============================================================


class RegisteredVariables(NamedTuple):
    """
    The registered variables are the variables that are
    stored in the state array. The order of the variables
    in the state array is important and should be consistent
    throughout the code.
    """

    #: Number of variables
    num_vars: int = 3

    # Baseline variables

    #: Density index
    density_index: int = 0

    #: Velocity index
    velocity_index: Union[int, StaticIntVector] = 1
    # in e.g. 3D, we have three velocity components, each with its own index

    #: Momentum density index, same as velocity index
    #: introduced for readability when dealing with
    #: the conserved state
    momentum_index: Union[int, StaticIntVector] = 1

    #: Magnetic field index
    magnetic_index: Union[int, StaticIntVector] = -1

    #: Magnetic field at interfaces index
    #: used in finite difference MHD constrained transport
    interface_magnetic_field_index: Union[int, StaticIntVector] = -1

    #: Pressure index
    pressure_index: int = 2

    #: Energy index, same as pressure index
    #: introduced for readability when dealing with
    #: the conserved state.
    energy_index: int = 2

    # Additional variables, these
    # have to be registered

    #: stellar wind density index
    wind_density_index: int = -1
    wind_density_active: bool = False

    #: simplified cosmic rays
    # in the simplest CR model witout CR diffusion,
    # streaming and no explicitly modeled magnetic field
    # n_CR = P_CR^(1/gamma_CR) is a conserved quantity.
    # This is the cosmic_ray_n, the index below points to.
    cosmic_ray_n_index: int = -1
    cosmic_ray_n_active: bool = False

    # here you can add more variables

# TODO: update
def get_registered_variables(config: SimulationConfig) -> RegisteredVariables:
    """Get the registered variables for the simulation.

    Args:
        config: The simulation configuration.

    Returns:
        The registered variables.
    """

    registered_variables = RegisteredVariables()
    
    if config.solver_mode == FINITE_VOLUME:

        if config.dimensionality == 2:
            # we have two velocity components
            registered_variables = registered_variables._replace(
                num_vars=registered_variables.num_vars + 1
            )

            # update the velocity index
            registered_variables = registered_variables._replace(
                velocity_index=StaticIntVector(1, 2, -1)
            )

            # TODO: unified MHD approach in 1D/2D/3D
            # magnetic field index
            if config.mhd:
                # TODO: better indexing
                registered_variables = registered_variables._replace(pressure_index=3)
                registered_variables = registered_variables._replace(
                    magnetic_index=StaticIntVector(4, 5, 6)
                )
                registered_variables = registered_variables._replace(
                    num_vars=registered_variables.num_vars + 3
                )
            else:
                # update the pressure index
                registered_variables = registered_variables._replace(
                    pressure_index=registered_variables.num_vars - 1
                )

        if config.dimensionality == 3:
            # we have three velocity components
            registered_variables = registered_variables._replace(
                num_vars=registered_variables.num_vars + 2
            )

            # update the velocity index to be an array
            registered_variables = registered_variables._replace(
                velocity_index=StaticIntVector(1, 2, 3)
            )

            # update the pressure index
            registered_variables = registered_variables._replace(
                pressure_index=registered_variables.num_vars - 1
            )

            # update the magnetic field index
            if config.mhd:
                registered_variables = registered_variables._replace(
                    magnetic_index=StaticIntVector(5, 6, 7)
                )
                registered_variables = registered_variables._replace(
                    num_vars=registered_variables.num_vars + 3
                )
        
        # NOTE: CURRENTLY ONLY IMPLEMENTED FOR FINITE VOLUME MODE
        if config.wind_config.trace_wind_density:
            registered_variables = registered_variables._replace(
                wind_density_index=registered_variables.num_vars
            )
            registered_variables = registered_variables._replace(
                num_vars=registered_variables.num_vars + 1
            )
            registered_variables = registered_variables._replace(wind_density_active=True)

        # NOTE: CURRENTLY ONLY IMPLEMENTED FOR FINITE VOLUME MODE
        if config.cosmic_ray_config.cosmic_rays:
            registered_variables = registered_variables._replace(
                cosmic_ray_n_index=registered_variables.num_vars
            )
            registered_variables = registered_variables._replace(
                num_vars=registered_variables.num_vars + 1
            )
            registered_variables = registered_variables._replace(cosmic_ray_n_active=True)


    if config.solver_mode == FINITE_DIFFERENCE:

        if config.mhd:
            
            # we always assume all velocity fields present, 
            # even in 1D and 2D, for the magnetic update

            # TEMPORARY: set the registered variables manually
            if config.equation_of_state == IDEAL_GAS:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=StaticIntVector(1, 2, 3),
                    pressure_index=4,
                    magnetic_index=StaticIntVector(5, 6, 7),
                    interface_magnetic_field_index=StaticIntVector(8, 9, 10),
                    num_vars=11,
                )
            elif config.equation_of_state == ISOTHERMAL:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=StaticIntVector(1, 2, 3),
                    pressure_index=-1,
                    magnetic_index=StaticIntVector(4, 5, 6),
                    interface_magnetic_field_index=StaticIntVector(7, 8, 9),
                    num_vars=10,
                )
        else:
            # redundant with the FINITE_VOLUME case
            # included for readability
            if config.dimensionality == 1:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=1,
                    pressure_index=2,
                    num_vars=3,
                )
            elif config.dimensionality == 2:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=StaticIntVector(1, 2),
                    pressure_index=3,
                    num_vars=4,
                )
            elif config.dimensionality == 3:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=StaticIntVector(1, 2, 3),
                    pressure_index=4,
                    num_vars=5,
                )
            
            if config.equation_of_state == ISOTHERMAL:
                registered_variables = registered_variables._replace(
                    pressure_index=-1
                )
                registered_variables = registered_variables._replace(
                    num_vars=registered_variables.num_vars - 1
                )

    # shorthands
    registered_variables = registered_variables._replace(
        momentum_index=registered_variables.velocity_index
    )
    registered_variables = registered_variables._replace(
        energy_index=registered_variables.pressure_index
    )

    # here you can register more variables

    return registered_variables
