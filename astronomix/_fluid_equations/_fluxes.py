from functools import partial
import jax.numpy as jnp
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

from astronomix._physics_modules._cosmic_rays.cr_fluid_equations import (
    total_energy_from_primitives_with_crs,
)
from astronomix._fluid_equations._equations import (
    get_absolute_velocity,
    total_energy_from_primitives,
    # total_pressure_from_conserved_mhd,
)
from astronomix.variable_registry.registered_variables import AxisInfo, RegisteredVariables
from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["config", "registered_variables", "flux_direction_index"]
)
def _euler_flux(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int,
) -> STATE_TYPE:
    """Compute the Euler fluxes for the given primitive states.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        gamma: The adiabatic index of the fluid.
        registered_variables: The registered variables.
        flux_direction_index: The index of the velocity component in the flux direction of interest.

    Returns:
        The Euler fluxes for the given primitive states.
    """
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]

    # start with a copy of the primitive states
    flux_vector = (
        primitive_state # .copy()
    )  # copy should not be necessary (no side effects)

    # calculate the total energy
    utotal = get_absolute_velocity(primitive_state, config, registered_variables)

    if registered_variables.cosmic_ray_n_active:
        E = total_energy_from_primitives_with_crs(primitive_state, registered_variables)
    else:
        E = total_energy_from_primitives(rho, utotal, p, gamma)

    # add the total energy to the pressure_index of the flux vector
    flux_vector = flux_vector.at[registered_variables.pressure_index].add(E)

    # scale the velocity components with the density
    if config.dimensionality == 1:
        flux_vector = flux_vector.at[registered_variables.velocity_index].set(
            primitive_state[registered_variables.velocity_index] * rho
        )
    elif config.dimensionality == 2:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(
            primitive_state[registered_variables.velocity_index.x] * rho
        )
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(
            primitive_state[registered_variables.velocity_index.y] * rho
        )
    elif config.dimensionality == 3:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(
            primitive_state[registered_variables.velocity_index.x] * rho
        )
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(
            primitive_state[registered_variables.velocity_index.y] * rho
        )
        flux_vector = flux_vector.at[registered_variables.velocity_index.z].set(
            primitive_state[registered_variables.velocity_index.z] * rho
        )

    # multiply the whole vector with the velocity component in the flux direction
    # primitive_state[flux_direction_index] only works because of how the registry is
    # currently ordered: TODO: generalize
    flux_vector = primitive_state[flux_direction_index] * flux_vector

    # add the pressure to the velocity component in the flux direction
    flux_vector = flux_vector.at[flux_direction_index].add(p)

    return flux_vector

# # TODO: improve readability
# @partial(
#     jax.jit, static_argnames=["config", "registered_variables", "flux_axis"]
# )
# def _mhd_flux(
#     conserved_state: STATE_TYPE,
#     gamma: Union[float, Float[Array, ""]],
#     config: SimulationConfig,
#     registered_variables: RegisteredVariables,
#     flux_axis: AxisInfo,
# ) -> STATE_TYPE:
#     """
#     Compute the MHD fluxes, for now only in 3D.

#     NOTE: We work on the conserved state here, not the primitive state!
#     (Because I'm currently writing this for the finite difference MHD solver, 
#     where we handle conserved variables.)

#     Conserved state:
#         [ ρ,

#           ρ v_x,
#           ρ v_y,
#           ρ v_z,

#           E,

#           B_x,
#           B_y,
#           B_z,

#         ]

#     The flux in x-direction is given by:

#         F = [
#             ρ v_x,                         // density_index

#             ρ v_x^2 + P_tot - B_x^2,       // velocity_index.x
#             ρ v_x v_y - B_x B_y,           // velocity_index.y
#             ρ v_x v_z - B_x B_z,           // velocity_index.z

#             (E + P_tot) v_x - (B · v) B_x  // energy_index

#             0,                             // magnetic_index.x
#             B_y v_x - B_x v_y,             // magnetic_index.y
#             B_z v_x - B_x v_z,             // magnetic_index.z

#         ]

#     The flux in y-direction is given by:
#         F = [
#             ρ v_y,                         // density_index

#             ρ v_y v_x - B_y B_x,           // velocity_index.x
#             ρ v_y^2 + P_tot - B_y^2,       // velocity_index.y
#             ρ v_y v_z - B_y B_z,           // velocity_index.z

#             (E + P_tot) v_y - (B · v) B_y  // energy_index

#             B_x v_y - B_y v_x,             // magnetic_index.x
#             0,                             // magnetic_index.y
#             B_z v_y - B_y v_z,             // magnetic_index.z

#         ]

#     The flux in z-direction is given by:
#         F = [
#             ρ v_z,                         // density_index

#             ρ v_z v_x - B_z B_x,           // velocity_index.x
#             ρ v_z v_y - B_z B_y,           // velocity_index.y
#             ρ v_z^2 + P_tot - B_z^2,       // velocity_index.z

#             (E + P_tot) v_z - (B · v) B_z  // energy_index

#             B_x v_z - B_z v_x,             // magnetic_index.x
#             B_y v_z - B_z v_y,             // magnetic_index.y
#             0,                             // magnetic_index.z
#         ]

#     where P_tot = p + 0.5 * |B|^2 is the total pressure.

#     """

#     total_pressure = total_pressure_from_conserved_mhd(
#         conserved_state, gamma, registered_variables
#     )

#     # we can get from the conserved state to the flux by
#     # - add the total pressure to the pressure_index
#     # - multipy everything with conserved_state[axis_info.velocity_index] / density
#     # - add the pressure term to the axis_info.velocity_index
#     # - subtract the the magnetic terms * axis_info.magnetic_index from the velocity components
#     # - subtract the velocity terms * axis_info.magnetic_index from the magnetic components
#     #  - subtract the (B · v) B_axis from the energy component
#     # - set the field at axis_info.magnetic_index to zero

#     flux_vector = conserved_state.at[registered_variables.pressure_index].add(
#         total_pressure
#     )

#     v_axis = conserved_state[flux_axis.velocity_index] / conserved_state[
#         registered_variables.density_index
#     ]

#     flux_vector = v_axis * flux_vector

#     flux_vector = flux_vector.at[flux_axis.velocity_index].add(
#         total_pressure
#     )

#     flux_vector = flux_vector.at[
#         registered_variables.velocity_index.x:registered_variables.velocity_index.z + 1
#     ].add(
#         -conserved_state[registered_variables.magnetic_index.x:registered_variables.magnetic_index.z + 1]
#         * conserved_state[flux_axis.magnetic_index]
#     )

#     flux_vector = flux_vector.at[
#         registered_variables.magnetic_index.x:registered_variables.magnetic_index.z + 1
#     ].add(
#         -conserved_state[
#             registered_variables.velocity_index.x:registered_variables.velocity_index.z + 1
#         ] / conserved_state[registered_variables.density_index]
#         * conserved_state[flux_axis.magnetic_index]
#     )

#     v_dot_B = (
#         conserved_state[registered_variables.magnetic_index.x]
#         * conserved_state[registered_variables.velocity_index.x]
#         + conserved_state[registered_variables.magnetic_index.y]
#         * conserved_state[registered_variables.velocity_index.y]
#         + conserved_state[registered_variables.magnetic_index.z]
#         * conserved_state[registered_variables.velocity_index.z]
#     ) / conserved_state[registered_variables.density_index]
#     flux_vector = flux_vector.at[registered_variables.pressure_index].add(
#         -v_dot_B * conserved_state[flux_axis.magnetic_index]
#     )

#     flux_vector = flux_vector.at[flux_axis.magnetic_index].set(0.0)

#     return flux_vector