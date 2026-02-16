import jax.numpy as jnp
import jax
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    pressure_from_energy,
    primitive_state_from_conserved,
)

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from functools import partial

from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

from astronomix._physics_modules._stellar_wind.stellar_wind_options import WindParams
from astronomix._physics_modules._stellar_wind.stellar_wind_options import (
    WindConfig,
    MEO,
    MEI,
    EI,
)

from typing import Union


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _wind_injection(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Inject stellar wind into the simulation.

    Args:
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    if config.dimensionality == 1:
        if config.wind_config.wind_injection_scheme == MEO:
            primitive_state = _wind_meo(
                params.wind_params,
                primitive_state,
                dt,
                helper_data,
                config.num_ghost_cells,
                config.wind_config.num_injection_cells,
                params.gamma,
            )
        elif config.wind_config.wind_injection_scheme == MEI:
            primitive_state = _wind_mei(
                params.wind_params,
                primitive_state,
                dt,
                config,
                helper_data,
                config.num_ghost_cells,
                config.wind_config.num_injection_cells,
                params.gamma,
                registered_variables,
            )
        elif config.wind_config.wind_injection_scheme == EI:
            primitive_state = _wind_ei(
                params.wind_params,
                primitive_state,
                dt,
                helper_data,
                config.num_ghost_cells,
                config.wind_config.num_injection_cells,
                params.gamma,
                registered_variables,
            )
        else:
            raise ValueError("Invalid wind injection scheme")
    elif config.dimensionality == 3:
        if config.wind_config.wind_injection_scheme == EI:
            primitive_state = _wind_ei3D(
                params.wind_params,
                primitive_state,
                dt,
                config,
                helper_data,
                config.num_ghost_cells,
                config.wind_config.num_injection_cells,
                params.gamma,
                registered_variables,
            )
        else:
            raise ValueError("Invalid wind injection scheme")
    else:
        raise ValueError("Invalid dimensionality")

    return primitive_state


# ================= Wind injection schemes =================

# here we implement all the injection schemes from
# https://arxiv.org/abs/2107.14673


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["num_ghost_cells", "num_injection_cells"])
def _wind_meo(
    wind_params: WindParams,
    primitive_state: Float[Array, "num_vars num_cells"],
    dt: Float[Array, ""],
    helper_data: HelperData,
    num_ghost_cells: int,
    num_injection_cells: int,
    gamma: Union[float, Float[Array, ""]],
) -> Float[Array, "num_vars num_cells"]:
    """Inject stellar wind into the simulation by a mass-and-energy-overwrite scheme (MEO).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    # set density
    density_overwrite = (
        wind_params.wind_mass_loss_rate
        / helper_data.cell_volumes[
            num_ghost_cells : num_injection_cells + num_ghost_cells
        ]
        / wind_params.wind_final_velocity
        * (
            helper_data.outer_cell_boundaries[
                num_ghost_cells : num_injection_cells + num_ghost_cells
            ]
            - helper_data.inner_cell_boundaries[
                num_ghost_cells : num_injection_cells + num_ghost_cells
            ]
        )
    )
    primitive_state = primitive_state.at[
        0, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].set(density_overwrite)

    # set velocity
    primitive_state = primitive_state.at[
        1, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].set(wind_params.wind_final_velocity)

    # set pressure to the floor value
    primitive_state = primitive_state.at[
        2, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].set(wind_params.pressure_floor)

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit,
    static_argnames=[
        "config",
        "num_ghost_cells",
        "num_injection_cells",
        "registered_variables",
    ],
)
def _wind_mei(
    wind_params: WindParams,
    primitive_state: Float[Array, "num_vars num_cells"],
    dt: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    num_ghost_cells: int,
    num_injection_cells: int,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_vars num_cells"]:
    """Inject stellar wind into the simulation by a momentum-and-energy-injection scheme (MEI).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    conservative_state = conserved_state_from_primitive(
        primitive_state, gamma, config, registered_variables
    )

    V_inj = (
        4
        / 3
        * jnp.pi
        * helper_data.outer_cell_boundaries[num_injection_cells + num_ghost_cells] ** 3
    )

    drho = wind_params.wind_mass_loss_rate * dt / V_inj
    dmomentum = wind_params.wind_final_velocity * drho
    denergy = 0.5 * wind_params.wind_final_velocity**2 * drho

    conservative_state = conservative_state.at[
        0, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].add(drho)
    conservative_state = conservative_state.at[
        1, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].add(dmomentum)
    conservative_state = conservative_state.at[
        2, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].add(denergy)

    primitive_state = primitive_state_from_conserved(
        conservative_state, gamma, config, registered_variables
    )

    return primitive_state


# not really ei
# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit,
    static_argnames=["num_ghost_cells", "num_injection_cells", "registered_variables"],
)
def _wind_ei(
    wind_params: WindParams,
    primitive_state: Float[Array, "num_vars num_cells"],
    dt: Float[Array, ""],
    helper_data: HelperData,
    num_ghost_cells: int,
    num_injection_cells: int,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_vars num_cells"]:
    """Inject stellar wind into the simulation by an thermal-energy-injection scheme (EI).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    source_term = jnp.zeros_like(primitive_state)

    # r = helper_data.volumetric_centers
    # r = helper_data.outer_cell_boundaries
    # r_inj = r[num_injection_cells + 2]
    # V = 4/3 * jnp.pi * r_inj**3

    V = jnp.sum(
        helper_data.cell_volumes[
            num_ghost_cells : num_injection_cells + num_ghost_cells
        ]
    )

    # mass injection
    drho_dt = wind_params.wind_mass_loss_rate / V
    source_term = source_term.at[
        0, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].set(drho_dt)
    updated_density = (
        primitive_state[0, num_ghost_cells : num_injection_cells + num_ghost_cells]
        + drho_dt * dt
    )

    if registered_variables.wind_density_active:
        source_term = source_term.at[
            registered_variables.wind_density_index,
            num_ghost_cells : num_injection_cells + num_ghost_cells,
        ].set(drho_dt)

    # energy injection
    dE_dt = (
        0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate / V
    )

    dp_dt = pressure_from_energy(
        dE_dt,
        updated_density,
        primitive_state[1, num_ghost_cells : num_injection_cells + num_ghost_cells],
        gamma,
    )

    source_term = source_term.at[
        2, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].set(dp_dt)

    primitive_state = primitive_state + source_term * dt

    return primitive_state


# not really ei
# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit,
    static_argnames=["num_ghost_cells", "num_injection_cells", "registered_variables"],
)
def dummy_multi_star_wind(
    wind_params: WindParams,
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    num_ghost_cells: int,
    num_injection_cells: int,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    star_positions = [
        jnp.array([0.2, 0.3, 0.5]),
        jnp.array([0.5, 0.7, 0.5]),
        jnp.array([0.7, 0.4, 0.5]),
        jnp.array([0.3, 0.6, 0.5]),
    ]

    for star_position in star_positions:
        r = jnp.linalg.norm(helper_data.geometric_centers - star_position, axis=-1)

        source_term = jnp.zeros_like(primitive_state)

        r_inj = num_injection_cells * config.grid_spacing
        V = 4 / 3 * jnp.pi * r_inj**3

        # for now only allow injection at the box center
        injection_mask = r <= r_inj - config.grid_spacing / 2

        # mass injection
        drho_dt = wind_params.wind_mass_loss_rate / V
        # source_term = source_term.at[registered_variables.density_index].set(jnp.where(injection_mask, drho_dt, source_term[registered_variables.density_index]))
        source_term = source_term.at[registered_variables.density_index].set(
            drho_dt * injection_mask
        )

        updated_density = primitive_state[registered_variables.density_index]
        updated_density = jnp.where(
            injection_mask > 0,
            updated_density + drho_dt * dt * injection_mask,
            updated_density,
        )

        # energy injection
        dE_dt = (
            0.5
            * wind_params.wind_final_velocity**2
            * wind_params.wind_mass_loss_rate
            / V
        )
        u = jnp.sqrt(
            primitive_state[registered_variables.velocity_index.x] ** 2
            + primitive_state[registered_variables.velocity_index.y] ** 2
            + primitive_state[registered_variables.velocity_index.z] ** 2
        )
        dp_dt = pressure_from_energy(dE_dt, updated_density, u, gamma)

        source_term = source_term.at[registered_variables.pressure_index].set(
            dp_dt * injection_mask
        )

        primitive_state = primitive_state + source_term * dt

    return primitive_state


# not really ei
# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit,
    static_argnames=["num_ghost_cells", "num_injection_cells", "registered_variables"],
)
def _wind_ei3D(
    wind_params: WindParams,
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    num_ghost_cells: int,
    num_injection_cells: int,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Inject stellar wind into the simulation by an thermal-energy-injection scheme (EI).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    source_term = jnp.zeros_like(primitive_state)

    r_inj = num_injection_cells * config.grid_spacing
    V = 4 / 3 * jnp.pi * r_inj**3

    # for now only allow injection at the box center
    injection_mask = helper_data.r <= r_inj - config.grid_spacing / 2
    # overlap_weights = (r_inj + config.grid_spacing / 2 - helper_data.r) / config.grid_spacing
    # overlap_mask = (helper_data.r > r_inj - config.grid_spacing / 2) & (helper_data.r < r_inj + config.grid_spacing / 2)
    # overlap_weights = overlap_weights * overlap_mask
    # injection_mask = injection_mask | overlap_mask
    # injection_mask = injection_mask / jnp.sum(injection_mask * config.grid_spacing**3) * V

    # mass injection
    drho_dt = wind_params.wind_mass_loss_rate / V
    # source_term = source_term.at[registered_variables.density_index].set(jnp.where(injection_mask, drho_dt, source_term[registered_variables.density_index]))
    source_term = source_term.at[registered_variables.density_index].set(
        drho_dt * injection_mask
    )

    updated_density = primitive_state[registered_variables.density_index]
    updated_density = jnp.where(
        injection_mask > 0,
        updated_density + drho_dt * dt * injection_mask,
        updated_density,
    )

    # scale down the velocity in the primitive state to conserve momentum
    # density_ratio = updated_density / primitive_state[registered_variables.density_index]
    # primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(jnp.where(injection_mask, primitive_state[registered_variables.velocity_index.x] * density_ratio, primitive_state[registered_variables.velocity_index.x]))
    # primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(jnp.where(injection_mask, primitive_state[registered_variables.velocity_index.y] * density_ratio, primitive_state[registered_variables.velocity_index.y]))
    # primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(jnp.where(injection_mask, primitive_state[registered_variables.velocity_index.z] * density_ratio, primitive_state[registered_variables.velocity_index.z]))

    # energy injection
    dE_dt = (
        0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate / V
    )
    u = jnp.sqrt(
        primitive_state[registered_variables.velocity_index.x] ** 2
        + primitive_state[registered_variables.velocity_index.y] ** 2
        + primitive_state[registered_variables.velocity_index.z] ** 2
        + 1e-20
    )
    dp_dt = pressure_from_energy(dE_dt, updated_density, u, gamma)

    source_term = source_term.at[registered_variables.pressure_index].set(
        dp_dt * injection_mask
    )

    primitive_state = primitive_state + source_term * dt

    return primitive_state

@partial(
    jax.jit,
    static_argnames=["num_injection_cells", "registered_variables", "config"],
)
def _wind_ei3D_source(
    wind_params: WindParams,
    conserved_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    num_injection_cells: int,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:

    source_term = jnp.zeros_like(conserved_state)

    r_inj = num_injection_cells * config.grid_spacing
    r_1 = 1.3 * r_inj

    # taper down injection to avoid sharp cut-off
    # injection_mask = helper_data.r <= r_inj
    injection_mask = jnp.where(
        helper_data.r <= r_inj,
        1.0,
        jnp.where(
            (helper_data.r > r_inj) & (helper_data.r <= r_1),
            (r_1 - helper_data.r) / (r_1 - r_inj),
            0.0,
        ),
    )

    V = jnp.sum(injection_mask) * config.grid_spacing**3

    # mass injection
    # V = sum(injection_mask) * config.grid_spacing**3
    drho_dt = wind_params.wind_mass_loss_rate / V

    source_term = source_term.at[registered_variables.density_index].set(
        drho_dt * injection_mask * dt
    )

    # energy injection
    dE = (
        0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate / V * dt
    )

    # we do not want to inject kinetic energy, only thermal, the kinetic energy
    # is 1/2 * rho * v^2 = 1/2 * m^2 / rho, momentum m, as we change the density,
    # and until now keep the momentum constant, we would change the kinetic energy
    # to keep the kinetic energy constant
    # 1/2 m_old^2 / rho_old = 1/2 m_new^2 / rho_new ->
    # m_new = m_old * sqrt(rho_new / rho_old) ->
    # dm = m_old * (sqrt(rho_new / rho_old) - 1)
    #    = m_old * (sqrt((rho_old + drho * dt) / rho_old) - 1)
    #    = m_old * (sqrt(1 + drho * dt / rho_old) - 1)
    momentum_source_factor = jnp.sqrt(
        1 + drho_dt * dt * injection_mask
        / (conserved_state[registered_variables.density_index])
    ) - 1.0
    # ensure we only inject momentum within r_1, while
    # the momentum source factor should be zero outside r_1 anyway
    # (sqrt(1 + 0) - 1) = 0 there might be small numerical errors
    momentum_source_factor = jnp.where(
        helper_data.r <= r_1, momentum_source_factor, 0.0
    )

    source_term = source_term.at[registered_variables.momentum_index.x].set(
        conserved_state[registered_variables.momentum_index.x] * momentum_source_factor
    )
    source_term = source_term.at[registered_variables.momentum_index.y].set(
        conserved_state[registered_variables.momentum_index.y] * momentum_source_factor
    )
    source_term = source_term.at[registered_variables.momentum_index.z].set(
        conserved_state[registered_variables.momentum_index.z] * momentum_source_factor
    )

    source_term = source_term.at[registered_variables.energy_index].set(
        dE * injection_mask
    )

    return source_term


# # @jaxtyped(typechecker=typechecker)
# @partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells', 'registered_variables'])
# def _wind_ei3D_superres(wind_params: WindParams, primitive_state: STATE_TYPE, dt: Float[Array, ""], config: SimulationConfig, helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables) -> STATE_TYPE:
#     """Inject stellar wind into the simulation by an thermal-energy-injection scheme (EI).

#     Args:
#         wind_params: The wind parameters.
#         primitive_state: The primitive state array.
#         dt: The time step.
#         helper_data: The helper data.
#         num_ghost_cells: The number of ghost cells.
#         num_injection_cells: The number of injection cells.
#         gamma: The adiabatic index.

#     Returns:
#         The primitive state array with the stellar wind injected.
#     """

#     source_term = jnp.zeros_like(primitive_state)
#     r_inj = num_injection_cells * config.grid_spacing

#     total_mass_change = wind_params.wind_mass_loss_rate * dt
#     total_energy_change = 0.5 * wind_params.wind_final_velocity**2 * total_mass_change

#     superres_factor = 8
#     superres_grid_size = superres_factor * num_injection_cells * 2
#     superres_grid_spacing = config.grid_spacing / superres_factor

#     half_width = superres_grid_size * superres_grid_spacing / 2

#     x = jnp.linspace(-half_width, half_width, superres_grid_size)
#     y = jnp.linspace(-half_width, half_width, superres_grid_size)
#     z = jnp.linspace(-half_width, half_width, superres_grid_size)
#     X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
#     R = jnp.sqrt(X**2 + Y**2 + Z**2)
#     superres_injection_weights = R <= r_inj
#     superres_injection_weights = superres_injection_weights / jnp.sum(superres_injection_weights)


#     # sum pool down the mask to get to a mask of size (num_injection_cells * 2)^3
#     superres_injection_weights = superres_injection_weights.reshape((num_injection_cells * 2, superres_factor,
#                                    num_injection_cells * 2, superres_factor,
#                                    num_injection_cells * 2, superres_factor)).sum(axis=(1, 3, 5))


#     injection_weights = jnp.zeros_like(primitive_state[0])
#     half_index = primitive_state[0].shape[0] // 2
#     injection_weights = injection_weights.at[half_index - num_injection_cells:half_index + num_injection_cells, half_index - num_injection_cells:half_index + num_injection_cells, half_index - num_injection_cells:half_index + num_injection_cells].set(superres_injection_weights)

#     source_term = source_term.at[registered_variables.density_index].set(total_mass_change * injection_weights / (config.grid_spacing**3))
#     gamma = 4/3
#     source_term = source_term.at[registered_variables.pressure_index].set(total_energy_change * (gamma - 1) * injection_weights / (config.grid_spacing**3))

#     primitive_state = primitive_state + source_term

#     return primitive_state

# ======================================================
