# general imports
import jax.numpy as jnp
import jax
from functools import partial


# type checking imports
from jaxtyping import Array, Float
from beartype import beartype as typechecker
from typing import Union

# general astronomix imports
from astronomix._finite_difference._fluid_equations._eigen_hydro import _eigen_all_lambdas_hydro
from astronomix._finite_difference._fluid_equations._eigen_hydro_iso import _eigen_all_lambdas_hydro_iso
from astronomix._finite_difference._fluid_equations._eigen_mhd import _eigen_all_lambdas
from astronomix._finite_difference._fluid_equations._eigen_mhd_iso import _eigen_all_lambdas_iso
from astronomix._finite_difference._fluid_equations._equations import conserved_state_from_primitive_isothermal, conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from astronomix._fluid_equations._equations import conserved_state_from_primitive
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    DYNAMIC_VISCOSITY,
    IDEAL_GAS,
    ISOTHERMAL,
    KINEMATIC_VISCOSITY,
    STATE_TYPE,
    SimulationConfig,
)

from astronomix.option_classes.simulation_params import SimulationParams


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step_fd(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt_max: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    C_CFL: Union[float, Float[Array, ""]] = 0.8,
) -> Float[Array, ""]:
    
    if config.equation_of_state == IDEAL_GAS:
        conserved_state = conserved_state_from_primitive_mhd(
            primitive_state, gamma, registered_variables
        )
    elif config.equation_of_state == ISOTHERMAL:
        conserved_state = conserved_state_from_primitive_isothermal(
            primitive_state, config, registered_variables
        )

    if config.equation_of_state == IDEAL_GAS:
        lambda_x = _eigen_all_lambdas(
            conserved_state, params.minimum_density, params.minimum_pressure, gamma, registered_variables
        )
    elif config.equation_of_state == ISOTHERMAL:
        lambda_x = _eigen_all_lambdas_iso(
            conserved_state, params.minimum_density, params.isothermal_sound_speed, registered_variables
        )
    
    lambda_x = jnp.max(jnp.abs(lambda_x))

    if config.dimensionality >= 2:
        if config.dimensionality == 2:
            qy = jnp.transpose(conserved_state, (0, 2, 1))
        else:
            qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
        
        momentum_x = qy[registered_variables.momentum_index.x]
        momentum_y = qy[registered_variables.momentum_index.y]
        B_x = qy[registered_variables.magnetic_index.x]
        B_y = qy[registered_variables.magnetic_index.y]
        qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
        qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)
        qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
        qy = qy.at[registered_variables.magnetic_index.y].set(B_x)

        if config.equation_of_state == IDEAL_GAS:
            lambda_y = _eigen_all_lambdas(
                qy, params.minimum_density, params.minimum_pressure, gamma, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            lambda_y = _eigen_all_lambdas_iso(
                qy, params.minimum_density, params.isothermal_sound_speed, registered_variables
            )
        lambda_y = jnp.max(jnp.abs(lambda_y))
    else:
        lambda_y = 0.0

    if config.dimensionality == 3:
        qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
        
        momentum_x = qz[registered_variables.momentum_index.x]
        momentum_z = qz[registered_variables.momentum_index.z]
        B_x = qz[registered_variables.magnetic_index.x]
        B_z = qz[registered_variables.magnetic_index.z]
        qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
        qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)
        qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
        qz = qz.at[registered_variables.magnetic_index.z].set(B_x)

        if config.equation_of_state == IDEAL_GAS:
            lambda_z = _eigen_all_lambdas(
                qz, params.minimum_density, params.minimum_pressure, gamma, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            lambda_z = _eigen_all_lambdas_iso(
                qz, params.minimum_density, params.isothermal_sound_speed, registered_variables
            )
        lambda_z = jnp.max(jnp.abs(lambda_z))
    else:
        lambda_z = 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)

    # viscous time step constraint
    if config.diffusion:
        if config.enforce_positivity:
            rho_min = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min = jnp.min(primitive_state[registered_variables.density_index])
       
        if config.viscosity_type == DYNAMIC_VISCOSITY:
            nu_max = params.viscosity / rho_min
        elif config.viscosity_type == KINEMATIC_VISCOSITY:
            nu_max = params.viscosity
        
        dt_visc = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * nu_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_visc)

    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    return dt_cfl


# @partial(jax.jit, static_argnames=["config", "registered_variables"])
# def _cfl_time_step_fd(
#     primitive_state: STATE_TYPE,
#     grid_spacing: Union[float, Float[Array, ""]],
#     dt_max: Union[float, Float[Array, ""]],
#     gamma: Union[float, Float[Array, ""]],
#     config: SimulationConfig,
#     params: SimulationParams,
#     registered_variables: RegisteredVariables,
#     C_CFL: Union[float, Float[Array, ""]] = 0.8,
# ) -> Float[Array, ""]:
    
#     # TODO: use specific lambda function

#     conserved_state = conserved_state_from_primitive_mhd(
#         primitive_state, gamma, registered_variables
#     )

#     lambda_x = _eigen_all_lambdas(
#         conserved_state, params.minimum_density, params.minimum_pressure, gamma, registered_variables
#     )

#     lambda_x = jnp.max(jnp.abs(lambda_x))

#     if config.dimensionality == 1:
#         qy = conserved_state
#     elif config.dimensionality == 2:
#         qy = jnp.transpose(conserved_state, (0, 2, 1))
#     elif config.dimensionality == 3:
#         qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
    
#     momentum_x = qy[registered_variables.momentum_index.x]
#     momentum_y = qy[registered_variables.momentum_index.y]
#     B_x = qy[registered_variables.magnetic_index.x]
#     B_y = qy[registered_variables.magnetic_index.y]
#     qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
#     qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)
#     qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
#     qy = qy.at[registered_variables.magnetic_index.y].set(B_x)

#     # lambda_y, _, _ = _eigen_x(
#     #     qy, gamma, registered_variables
#     # )

#     lambda_y = _eigen_all_lambdas(
#         qy, params.minimum_density, params.minimum_pressure, gamma, registered_variables
#     )

#     lambda_y = jnp.max(jnp.abs(lambda_y))

#     if config.dimensionality < 3:
#         qz = conserved_state
#     else:
#         qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
    
#     momentum_x = qz[registered_variables.momentum_index.x]
#     momentum_z = qz[registered_variables.momentum_index.z]
#     B_x = qz[registered_variables.magnetic_index.x]
#     B_z = qz[registered_variables.magnetic_index.z]
#     qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
#     qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)
#     qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
#     qz = qz.at[registered_variables.magnetic_index.z].set(B_x)

#     # lambda_z, _, _ = _eigen_x(
#     #     qz, gamma, registered_variables
#     # )

#     lambda_z = _eigen_all_lambdas(
#         qz, params.minimum_density, params.minimum_pressure, gamma, registered_variables
#     )

#     lambda_z = jnp.max(jnp.abs(lambda_z))

#     dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)

#     # viscous time step constraint
#     if config.diffusion:
#         if config.enforce_positivity:
#             rho_min = jnp.maximum(
#                 jnp.min(primitive_state[registered_variables.density_index]),
#                 params.minimum_density,
#             )
#         else:
#             rho_min = jnp.min(primitive_state[registered_variables.density_index])
#         nu_max = params.viscosity / rho_min
#         dt_visc = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * nu_max)
#         dt_cfl = jnp.minimum(dt_cfl, dt_visc)


#     dt_cfl = jnp.minimum(dt_cfl, dt_max)

#     return dt_cfl


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step_fd_hydro(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt_max: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    C_CFL: Union[float, Float[Array, ""]] = 0.8,
) -> Float[Array, ""]:
    
    # TODO: use specific lambda function

    if config.equation_of_state == IDEAL_GAS:
        conserved_state = conserved_state_from_primitive(
            primitive_state, gamma, config, registered_variables
        )
    elif config.equation_of_state == ISOTHERMAL:
        conserved_state = conserved_state_from_primitive_isothermal(
            primitive_state, config, registered_variables
        )

    if config.equation_of_state == IDEAL_GAS:
        lambda_x = _eigen_all_lambdas_hydro(
            conserved_state, params.minimum_density, params.minimum_pressure, gamma, config, registered_variables
        )
    elif config.equation_of_state == ISOTHERMAL:
        lambda_x = _eigen_all_lambdas_hydro_iso(
            conserved_state, params.minimum_density, params.isothermal_sound_speed, config, registered_variables
        )
    lambda_x = jnp.max(jnp.abs(lambda_x))

    if config.dimensionality >= 2:

        if config.dimensionality == 2:
            qy = jnp.transpose(conserved_state, (0, 2, 1))
        else:
            qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
        momentum_x = qy[registered_variables.momentum_index.x]
        momentum_y = qy[registered_variables.momentum_index.y]
        qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
        qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)

        # lambda_y, _, _ = _eigen_x(
        #     qy, gamma, registered_variables
        # )

        if config.equation_of_state == IDEAL_GAS:
            lambda_y = _eigen_all_lambdas_hydro(
                qy, params.minimum_density, params.minimum_pressure, gamma, config, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            lambda_y = _eigen_all_lambdas_hydro_iso(
                qy, params.minimum_density, params.isothermal_sound_speed, config, registered_variables
            )

        lambda_y = jnp.max(jnp.abs(lambda_y))
    else:
        lambda_y = 0.0

    if config.dimensionality == 3:
        qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
        momentum_x = qz[registered_variables.momentum_index.x]
        momentum_z = qz[registered_variables.momentum_index.z]
        qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
        qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)

        # lambda_z, _, _ = _eigen_x(
        #     qz, gamma, registered_variables
        # )

        if config.equation_of_state == IDEAL_GAS:
            lambda_z = _eigen_all_lambdas_hydro(
                qz, params.minimum_density, params.minimum_pressure, gamma, config, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            lambda_z = _eigen_all_lambdas_hydro_iso(
                qz, params.minimum_density, params.isothermal_sound_speed, config, registered_variables
            )

        lambda_z = jnp.max(jnp.abs(lambda_z))
    else:
        lambda_z = 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)
    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    # viscous time step constraint
    if config.diffusion:
        
        if config.enforce_positivity:
            rho_min = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min = jnp.min(primitive_state[registered_variables.density_index])
        
        if config.viscosity_type == DYNAMIC_VISCOSITY:
            nu_max = params.viscosity / rho_min
        elif config.viscosity_type == KINEMATIC_VISCOSITY:
            nu_max = params.viscosity
        
        dt_visc = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * nu_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_visc)

    return dt_cfl