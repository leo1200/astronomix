"""
Computations of the eigenvalues and eigenvectors for the Euler (hydrodynamics) equations.

Problems for differentiation largely follow from square roots and divisions:
The derivative of sqrt(x) is 1/(2*sqrt(x)) and of 1/x is -1/x^2, where both expressions
are problematic for small x, especially when multiplying gradients in the backward pass,
-> exploding gradients.
"""

# general
from functools import partial

# typing
from typing import Union

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _shift


def diff_safe_sqrt(x):
    """Square root with a small floor, so its derivative stays finite at x = 0.

    The derivative of sqrt(x) is 1 / (2 sqrt(x)), which blows up as x -> 0;
    clamping the argument to a tiny epsilon keeps the backward pass well-behaved.
    """
    epsilon = 1e-12
    x_safe = jnp.maximum(x, epsilon)
    return jnp.sqrt(x_safe)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigenvalue_building_blocks(
    conserved_state,
    gamma,
    rhomin,
    pgmin,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    # unpack the conserved variables
    density = conserved_state[registered_variables.density_index]

    if config.dimensionality == 1:
        momentum_x = conserved_state[registered_variables.momentum_index]
    else:
        momentum_x = conserved_state[registered_variables.momentum_index.x]

    if config.dimensionality == 1:
        momentum_y = 0.0
        momentum_z = 0.0
    elif config.dimensionality == 2:
        momentum_y = conserved_state[registered_variables.momentum_index.y]
        momentum_z = 0.0
    elif config.dimensionality == 3:
        momentum_y = conserved_state[registered_variables.momentum_index.y]
        momentum_z = conserved_state[registered_variables.momentum_index.z]

    energy = conserved_state[registered_variables.energy_index]

    # compute primitives
    rho = density
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    velocity_z = momentum_z / rho
    velocity_squared = (
        velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z
    )
    
    gas_pressure = (gamma - 1.0) * (
        energy - 0.5 * rho * velocity_squared
    )

    # redefine the density and pressure, and energy based on floors
    rho = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin), jnp.maximum(rho, rhomin), rho
    )
    gas_pressure = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin),
        jnp.maximum(gas_pressure, pgmin),
        gas_pressure,
    )
    energy = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin),
        gas_pressure / (gamma - 1.0) + 0.5 * rho * velocity_squared,
        energy,
    )

    # compute derived quantities
    sound_speed = diff_safe_sqrt(jnp.maximum(0.0, gamma * jnp.abs(gas_pressure / rho)))

    return (
        velocity_x,
        sound_speed,
    )

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigenvector_building_blocks(
    conserved_state,
    gamma,
    rhomin,
    pgmin,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    # unpack conserved variables
    rho = conserved_state[registered_variables.density_index]  

    if config.dimensionality == 1:
        momentum_x = conserved_state[registered_variables.momentum_index]
    else:
        momentum_x = conserved_state[registered_variables.momentum_index.x]


    if config.dimensionality == 1:
        momentum_y = 0.0
        momentum_z = 0.0
    elif config.dimensionality == 2:
        momentum_y = conserved_state[registered_variables.momentum_index.y]
        momentum_z = 0.0
    elif config.dimensionality == 3:
        momentum_y = conserved_state[registered_variables.momentum_index.y]
        momentum_z = conserved_state[registered_variables.momentum_index.z]
        
    energy = conserved_state[registered_variables.energy_index]

    # compute primitives
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    velocity_z = momentum_z / rho
    velocity_sq = (
        velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z
    )
    
    gas_pressure = (gamma - 1.0) * (energy - 0.5 * rho * velocity_sq)

    # redefine the density and pressure, and energy based on floors
    rho = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin), jnp.maximum(rho, rhomin), rho
    )
    gas_pressure = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin),
        jnp.maximum(gas_pressure, pgmin),
        gas_pressure,
    )
    energy = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin),
        gas_pressure / (gamma - 1.0) + 0.5 * rho * velocity_sq,
        energy,
    )

    specific_enthalpy = (energy + gas_pressure) / rho

    # Periodic average from cell centres to interfaces.
    def avg_x(arr):
        return 0.5 * (arr + _shift(arr, shift=-1, axis=0))

    # Average the momenta (rather than the velocities) to the interface and divide
    # by the interface density; this keeps the interface velocity consistent with
    # the averaged conserved quantities.
    rho_interface = avg_x(jnp.maximum(rho, rhomin))
    rho_interface = jnp.maximum(rho_interface, rhomin)
    velocity_x_interface = avg_x(momentum_x) / rho_interface

    if config.dimensionality == 1:
        velocity_y_interface = 0.0
        velocity_z_interface = 0.0
    elif config.dimensionality == 2:
        velocity_y_interface = avg_x(momentum_y) / rho_interface
        velocity_z_interface = 0.0
    elif config.dimensionality == 3:
        velocity_y_interface = avg_x(momentum_y) / rho_interface
        velocity_z_interface = avg_x(momentum_z) / rho_interface

    specific_enthalpy_interface = avg_x(specific_enthalpy)

    # interface derived quantities
    velocity_sq_interface = (
        velocity_x_interface * velocity_x_interface
        + velocity_y_interface * velocity_y_interface
        + velocity_z_interface * velocity_z_interface
    )

    # enthalpy based sound speed at interfaces
    sound_speed_sq_interface = (gamma - 1.0) * (
        specific_enthalpy_interface - 0.5 * velocity_sq_interface
    )
    sound_speed_interface = diff_safe_sqrt(jnp.maximum(0.0, sound_speed_sq_interface))

    sound_speed_sq_inverse = jnp.where(
        sound_speed_sq_interface > 0.0, 1.0 / sound_speed_sq_interface, 0.0
    )

    gamma_minus_one = gamma - 1.0

    return (
        rho_interface,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        velocity_sq_interface,
        specific_enthalpy_interface,
        sound_speed_interface,
        sound_speed_sq_interface,
        sound_speed_sq_inverse,
        gamma_minus_one,
    )


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigen_R_col_hydro(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    col: int,
):
    (
        rho_interface,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        velocity_sq_interface,
        specific_enthalpy_interface,
        sound_speed_interface,
        sound_speed_sq_interface,
        sound_speed_sq_inverse,
        gamma_minus_one,
    ) = _eigenvector_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        config,
        registered_variables,
    )

    # shorter names for registry indices
    density_index = registered_variables.density_index


    if config.dimensionality == 1:
        momentum_index_x = registered_variables.momentum_index
    else:
        momentum_index_x = registered_variables.momentum_index.x
    
    if config.dimensionality >= 2:
        momentum_index_y = registered_variables.momentum_index.y

    if config.dimensionality == 3:
        momentum_index_z = registered_variables.momentum_index.z

    
    energy_index = registered_variables.energy_index

    def col_0():
        # Column 1 (Left Acoustic / u - c)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(1.0)
        R = R.at[momentum_index_x].set(velocity_x_interface - sound_speed_interface)

        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(velocity_y_interface)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(velocity_z_interface)
        
        R = R.at[energy_index].set(specific_enthalpy_interface - velocity_x_interface * sound_speed_interface)
        return R

    def col_1():
        # Column 2 (Entropy / u)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(1.0)

        R = R.at[momentum_index_x].set(velocity_x_interface)

        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(velocity_y_interface)

        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(velocity_z_interface)

        R = R.at[energy_index].set(0.5 * velocity_sq_interface)
        return R

    def col_2():
        # Column 3 (Shear Y / u)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(1.0)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(0.0)
        R = R.at[energy_index].set(velocity_y_interface)
        return R

    def col_3():
        # Column 4 (Shear Z / u)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(0.0)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(1.0)
        R = R.at[energy_index].set(velocity_z_interface)
        return R

    def col_4():
        # Column 5 (Right Acoustic / u + c)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(1.0)
        R = R.at[momentum_index_x].set(velocity_x_interface + sound_speed_interface)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(velocity_y_interface)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(velocity_z_interface)
        R = R.at[energy_index].set(specific_enthalpy_interface + velocity_x_interface * sound_speed_interface)
        return R

    if config.dimensionality == 1:
        R = jax.lax.switch(col, [col_0, col_1, col_4])
    if config.dimensionality == 2:
        R = jax.lax.switch(col, [col_0, col_1, col_2, col_4])
    if config.dimensionality == 3:
        R = jax.lax.switch(col, [col_0, col_1, col_2, col_3, col_4])

    return R


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigen_L_row_hydro(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    row: int,
):
    (
        rho_interface,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        velocity_sq_interface,
        specific_enthalpy_interface,
        sound_speed_interface,
        sound_speed_sq_interface,
        sound_speed_sq_inverse,
        gamma_minus_one,
    ) = _eigenvector_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        config,
        registered_variables,
    )

    # shorter names for registry indices
    density_index = registered_variables.density_index

    if config.dimensionality == 1:
        momentum_index_x = registered_variables.momentum_index
    else:
        momentum_index_x = registered_variables.momentum_index.x

    if config.dimensionality >= 2:
        momentum_index_y = registered_variables.momentum_index.y
    if config.dimensionality == 3:
        momentum_index_z = registered_variables.momentum_index.z
    energy_index = registered_variables.energy_index

    def row_0():
        # Row 1 (Left Acoustic / u - c)
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(0.5 * gamma_minus_one * velocity_sq_interface + velocity_x_interface * sound_speed_interface)
        L = L.at[momentum_index_x].set(-(gamma_minus_one * velocity_x_interface + sound_speed_interface))
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(-gamma_minus_one * velocity_y_interface)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(-gamma_minus_one * velocity_z_interface)
        L = L.at[energy_index].set(gamma_minus_one)
        L = 0.5 * sound_speed_sq_inverse * L
        return L

    def row_1():
        # Row 2 (Entropy / u)
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(sound_speed_sq_interface - 0.5 * gamma_minus_one * velocity_sq_interface)
        L = L.at[momentum_index_x].set(gamma_minus_one * velocity_x_interface)
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(gamma_minus_one * velocity_y_interface)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(gamma_minus_one * velocity_z_interface)
        L = L.at[energy_index].set(-gamma_minus_one)
        L = sound_speed_sq_inverse * L
        return L

    def row_2():
        # Row 3 (Shear Y / u)
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(-velocity_y_interface)
        L = L.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(1.0)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(0.0)
        L = L.at[energy_index].set(0.0)
        return L

    def row_3():
        # Row 4 (Shear Z / u)
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(-velocity_z_interface)
        L = L.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(0.0)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(1.0)
        L = L.at[energy_index].set(0.0)
        return L

    def row_4():
        # Row 5 (Right Acoustic / u + c)
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(0.5 * gamma_minus_one * velocity_sq_interface - velocity_x_interface * sound_speed_interface)
        L = L.at[momentum_index_x].set(-(gamma_minus_one * velocity_x_interface - sound_speed_interface))
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(-gamma_minus_one * velocity_y_interface)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(-gamma_minus_one * velocity_z_interface)
        L = L.at[energy_index].set(gamma_minus_one)
        L = 0.5 * sound_speed_sq_inverse * L
        return L

    if config.dimensionality == 1:
        L = jax.lax.switch(row, [row_0, row_1, row_4])
    if config.dimensionality == 2:
        L = jax.lax.switch(row, [row_0, row_1, row_2, row_4])
    if config.dimensionality == 3:
        L = jax.lax.switch(row, [row_0, row_1, row_2, row_3, row_4])

    return L


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigen_all_lambdas_hydro(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    (
        velocity_x,
        sound_speed,
    ) = _eigenvalue_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        config,
        registered_variables,
    )

    if config.dimensionality == 1:
        return jnp.stack(
            [
                velocity_x - sound_speed,
                velocity_x,
                velocity_x + sound_speed,
            ],
            axis=0,
        )
    
    if config.dimensionality == 2:
        return jnp.stack(
            [
                velocity_x - sound_speed,
                velocity_x,
                velocity_x,
                velocity_x + sound_speed,
            ],
            axis=0,
        )
    
    if config.dimensionality == 3:
        return jnp.stack(
            [
                velocity_x - sound_speed,
                velocity_x,
                velocity_x,
                velocity_x,
                velocity_x + sound_speed,
            ],
            axis=0,
        )


def _eigen_lambdas_hydro(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    mode: int,
):
    (
        velocity_x,
        sound_speed,
    ) = _eigenvalue_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        config,
        registered_variables,
    )

    def mode_0():
        return velocity_x - sound_speed

    def mode_1():
        return velocity_x

    def mode_2():
        return velocity_x

    def mode_3():
        return velocity_x

    def mode_4():
        return velocity_x + sound_speed

    if config.dimensionality == 1:
        return jax.lax.switch(mode, [mode_0, mode_1, mode_4])
    if config.dimensionality == 2:
        return jax.lax.switch(mode, [mode_0, mode_1, mode_2, mode_4])
    if config.dimensionality == 3:
        return jax.lax.switch(
            mode, [mode_0, mode_1, mode_2, mode_3, mode_4]
        )