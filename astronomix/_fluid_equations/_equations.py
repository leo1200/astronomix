"""
Hydrodynamic fluid equations: conversions between primitive and conserved
states and the basic thermodynamic relations (pressure, internal energy, total
energy and sound speed) for the (ideal-gas) Euler equations.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    total_energy_from_primitives_with_crs,
    total_pressure_from_conserved_with_crs,
)

# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def primitive_state_from_conserved(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Convert the conserved state to the primitive state.

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The primitive state.
    """
    # The conserved variables share the same registry indices as the primitive
    # ones, so velocity and momentum density (and pressure and energy) occupy the
    # same slot; we read the conserved values out and overwrite them in place.
    rho = conserved_state[registered_variables.density_index]
    E = conserved_state[registered_variables.pressure_index]

    if config.dimensionality == 1:
        u = conserved_state[registered_variables.velocity_index] / rho
    elif config.dimensionality == 2:
        ux = conserved_state[registered_variables.velocity_index.x] / rho
        uy = conserved_state[registered_variables.velocity_index.y] / rho
        # The 1e-20 offset keeps the gradient of sqrt finite at u = 0, where
        # d/dx sqrt(0) would otherwise be infinite. TODO: find a cleaner way.
        u = jnp.sqrt(ux**2 + uy**2 + 1e-20)
    elif config.dimensionality == 3:
        ux = conserved_state[registered_variables.velocity_index.x] / rho
        uy = conserved_state[registered_variables.velocity_index.y] / rho
        uz = conserved_state[registered_variables.velocity_index.z] / rho
        u = jnp.sqrt(ux**2 + uy**2 + uz**2 + 1e-20)

    p = pressure_from_energy(E, rho, u, gamma)

    if registered_variables.cosmic_ray_n_active:
        p = total_pressure_from_conserved_with_crs(
            conserved_state, registered_variables
        )
    else:
        p = pressure_from_energy(E, rho, u, gamma)

    # Write the recovered pressure and velocities into the primitive state.
    primitive_state = conserved_state.at[registered_variables.pressure_index].set(p)

    if config.dimensionality == 1:
        primitive_state = primitive_state.at[registered_variables.velocity_index].set(u)
    elif config.dimensionality == 2:
        primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(
            ux
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(
            uy
        )
    elif config.dimensionality == 3:
        primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(
            ux
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(
            uy
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(
            uz
        )

    # All other variables (e.g. the mass density) coincide between the primitive
    # and conserved representations, so they are left untouched.
    return primitive_state


# -------------------------------------------------------------
# =============== ↓ Create the conserved state ↓ ==============
# -------------------------------------------------------------


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def conserved_state_from_primitive(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Convert the primitive state to the conserved state.

    Args:
        primitive_state: The primitive state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The conserved state.
    """

    rho = primitive_state[registered_variables.density_index]

    u = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]

    if registered_variables.cosmic_ray_n_active:
        E = total_energy_from_primitives_with_crs(primitive_state, registered_variables)
    else:
        E = total_energy_from_primitives(rho, u, p, gamma)

    conserved_state = primitive_state.at[registered_variables.pressure_index].set(E)

    if config.dimensionality == 1:
        conserved_state = conserved_state.at[registered_variables.velocity_index].set(
            rho * primitive_state[registered_variables.velocity_index]
        )
    elif config.dimensionality == 2:
        conserved_state = conserved_state.at[registered_variables.velocity_index.x].set(
            rho * primitive_state[registered_variables.velocity_index.x]
        )
        conserved_state = conserved_state.at[registered_variables.velocity_index.y].set(
            rho * primitive_state[registered_variables.velocity_index.y]
        )
    elif config.dimensionality == 3:
        conserved_state = conserved_state.at[registered_variables.velocity_index.x].set(
            rho * primitive_state[registered_variables.velocity_index.x]
        )
        conserved_state = conserved_state.at[registered_variables.velocity_index.y].set(
            rho * primitive_state[registered_variables.velocity_index.y]
        )
        conserved_state = conserved_state.at[registered_variables.velocity_index.z].set(
            rho * primitive_state[registered_variables.velocity_index.z]
        )
    else:
        raise ValueError("Invalid dimension.")

    return conserved_state

# -------------------------------------------------------------
# =============== ↑ Create the conserved state ↑ ==============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===================== ↓ Fluid physics ↓ =====================
# -------------------------------------------------------------


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def get_absolute_velocity(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> Union[
    Float[Array, "num_cells"],
    Float[Array, "num_cells_x num_cells_y"],
    Float[Array, "num_cells_x num_cells_y num_cells_z"],
]:
    """Get the absolute velocity of the fluid.

    Args:
        primitive_state: The primitive state of the fluid.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The absolute velocity.
    """
    if config.dimensionality == 1:
        return jnp.abs(primitive_state[registered_variables.velocity_index])
    elif config.dimensionality == 2:
        return jnp.sqrt(
            primitive_state[registered_variables.velocity_index.x] ** 2
            + primitive_state[registered_variables.velocity_index.y] ** 2
            + 1e-20
        )
    elif config.dimensionality == 3:
        return jnp.sqrt(
            primitive_state[registered_variables.velocity_index.x] ** 2
            + primitive_state[registered_variables.velocity_index.y] ** 2
            + primitive_state[registered_variables.velocity_index.z] ** 2
            + 1e-20
        )
    else:
        raise ValueError("Invalid dimension.")


@jax.jit
def pressure_from_internal_energy(e, rho, gamma):
    """
    Calculate the pressure from the internal energy.

    Args:
        e: The internal energy.
        rho: The density.
        gamma: The adiabatic index.

    Returns:
        The pressure.
    """
    return (gamma - 1) * rho * e


@jax.jit
def internal_energy_from_energy(E, rho, u):
    """Calculate the internal energy from the total energy.

    Args:
        E: The total energy.
        rho: The density.
        u: The velocity.

    Returns:
        The internal energy.
    """
    return E / rho - 0.5 * u**2


@jax.jit
def pressure_from_energy(E, rho, u, gamma):
    """Calculate the pressure from the total energy.

    Args:
        E: The total energy.
        rho: The density.
        u: The velocity.
        gamma: The adiabatic index.

    Returns:
        The pressure.
    """

    e = internal_energy_from_energy(E, rho, u)
    return pressure_from_internal_energy(e, rho, gamma)



@jax.jit
def total_energy_from_primitives(rho, u, p, gamma):
    """Calculate the total energy from the primitive variables.

    Args:
        rho: The density.
        u: The velocity.
        p: The pressure.
        gamma: The adiabatic index.

    Returns:
        The total energy.
    """

    return p / (gamma - 1) + 0.5 * rho * u**2

@jax.jit
def speed_of_sound(rho, p, gamma):
    """Calculate the speed of sound.

    Args:
        rho: The density.
        p: The pressure.
        gamma: The adiabatic index.

    Returns:
        The speed of sound.
    """
    return jnp.sqrt(gamma * p / rho)


# -------------------------------------------------------------
# ===================== ↑ Fluid physics ↑ =====================
# -------------------------------------------------------------
