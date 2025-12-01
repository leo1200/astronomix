import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float

from typing import Union

from astronomix._finite_difference._fluid_equations._equations import primitive_state_from_conserved_mhd, total_energy_from_primitives_mhd, total_pressure_from_conserved_mhd
from astronomix.variable_registry.registered_variables import AxisInfo, RegisteredVariables
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
    SimulationConfig,
)

# We only define the flux in x-direction here,
# since the other directions can be obtained
# by permuting the arrays accordingly.
@partial(
    jax.jit, static_argnames=["registered_variables", "config"]
)
def _mhd_flux_x(
    conserved_state: STATE_TYPE,
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:

    primitive_state = primitive_state_from_conserved_mhd(
        conserved_state, minimum_density, minimum_pressure, gamma, config, registered_variables
    )
    
    # retrieve necessary quantities
    rho = primitive_state[registered_variables.density_index]
    v_x = primitive_state[registered_variables.velocity_index.x]
    v_y = primitive_state[registered_variables.velocity_index.y]
    v_z = primitive_state[registered_variables.velocity_index.z]
    B_x = primitive_state[registered_variables.magnetic_index.x]
    B_y = primitive_state[registered_variables.magnetic_index.y]
    B_z = primitive_state[registered_variables.magnetic_index.z]
    p_gas = primitive_state[registered_variables.pressure_index]

    # compute derived quantities
    b_squared = B_x**2 + B_y**2 + B_z**2
    v_squared = v_x**2 + v_y**2 + v_z**2
    total_pressure = p_gas + 0.5 * b_squared
    v_dot_B = v_x * B_x + v_y * B_y + v_z * B_z
    E = total_energy_from_primitives_mhd(
        rho,
        v_squared,
        p_gas,
        b_squared,
        gamma,
    )

    # compute fluxes
    flux = jnp.zeros_like(primitive_state)
    flux = flux.at[registered_variables.density_index].set(rho * v_x)
    flux = flux.at[registered_variables.velocity_index.x].set(rho * v_x**2 + total_pressure - B_x**2)
    flux = flux.at[registered_variables.velocity_index.y].set(rho * v_x * v_y - B_x * B_y)
    flux = flux.at[registered_variables.velocity_index.z].set(rho * v_x * v_z - B_x * B_z)
    flux = flux.at[registered_variables.pressure_index].set((E + total_pressure) * v_x - v_dot_B * B_x)
    flux = flux.at[registered_variables.magnetic_index.x].set(0.0)
    flux = flux.at[registered_variables.magnetic_index.y].set(B_y * v_x - B_x * v_y)
    flux = flux.at[registered_variables.magnetic_index.z].set(B_z * v_x - B_x * v_z)

    return flux