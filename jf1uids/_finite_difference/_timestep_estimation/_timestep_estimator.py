# general imports
import jax.numpy as jnp
import jax
from functools import partial


# type checking imports
from jaxtyping import Array, Float
from beartype import beartype as typechecker
from typing import Union

# general jf1uids imports
from jf1uids._finite_difference._fluid_equations._eigen import _eigen_all_lambdas, _eigen_x
from jf1uids._finite_difference._fluid_equations._equations import conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.variable_registry.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import (
    STATE_TYPE,
    SimulationConfig,
)

from jf1uids.option_classes.simulation_params import SimulationParams


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
    
    # TODO: use specific lambda function

    conserved_state = conserved_state_from_primitive_mhd(
        primitive_state, gamma, registered_variables
    )

    lambda_x = _eigen_all_lambdas(
        conserved_state, params.minimum_density, params.minimum_pressure, gamma, registered_variables
    )

    lambda_x = jnp.max(jnp.abs(lambda_x))

    qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
    momentum_x = qy[registered_variables.momentum_index.x]
    momentum_y = qy[registered_variables.momentum_index.y]
    B_x = qy[registered_variables.magnetic_index.x]
    B_y = qy[registered_variables.magnetic_index.y]
    qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
    qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)
    qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
    qy = qy.at[registered_variables.magnetic_index.y].set(B_x)

    # lambda_y, _, _ = _eigen_x(
    #     qy, gamma, registered_variables
    # )

    lambda_y = _eigen_all_lambdas(
        qy, params.minimum_density, params.minimum_pressure, gamma, registered_variables
    )

    lambda_y = jnp.max(jnp.abs(lambda_y))

    qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
    momentum_x = qz[registered_variables.momentum_index.x]
    momentum_z = qz[registered_variables.momentum_index.z]
    B_x = qz[registered_variables.magnetic_index.x]
    B_z = qz[registered_variables.magnetic_index.z]
    qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
    qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)
    qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
    qz = qz.at[registered_variables.magnetic_index.z].set(B_x)

    # lambda_z, _, _ = _eigen_x(
    #     qz, gamma, registered_variables
    # )

    lambda_z = _eigen_all_lambdas(
        qz, params.minimum_density, params.minimum_pressure, gamma, registered_variables
    )

    lambda_z = jnp.max(jnp.abs(lambda_z))
    
    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)
    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    return dt_cfl