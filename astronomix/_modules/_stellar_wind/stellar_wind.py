"""
Stellar-wind injection into the fluid state.

Implements the wind injection schemes of https://arxiv.org/abs/2107.14673 for
the finite-volume solver (mass-and-energy overwrite, momentum-and-energy
injection, thermal-energy injection) in 1D and 3D, plus the source-term variant
used by the finite-difference solver. ``_wind_injection`` dispatches to the
appropriate scheme based on the configuration.
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
from astronomix.option_classes.simulation_config import STATE_TYPE
from astronomix._modules._stellar_wind.stellar_wind_options import (
    MEO,
    MEI,
    EI,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix._modules._stellar_wind.stellar_wind_options import (
    WindConfig,
    WindParams,
)

# astronomix functions
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    pressure_from_energy,
    primitive_state_from_conserved,
)


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

    Dispatches to the configured injection scheme for the active dimensionality.

    Args:
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.
        registered_variables: The registered variables.

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


# -------------------------------------------------------------
# =============== ↓ Wind injection schemes ↓ ==================
# -------------------------------------------------------------
#
# All injection schemes here follow https://arxiv.org/abs/2107.14673.


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
    """Inject stellar wind by a momentum-and-energy-overwrite scheme (MEO).

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

    # Overwrite the density in the injection cells with the steady free-wind
    # density rho = M_dot * (r_out - r_in) / (v_inf * V_cell).
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

    # Overwrite the velocity with the wind terminal velocity.
    primitive_state = primitive_state.at[
        1, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].set(wind_params.wind_final_velocity)

    # Overwrite the pressure with the configured floor value.
    primitive_state = primitive_state.at[
        2, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].set(wind_params.pressure_floor)

    return primitive_state


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
    """Inject stellar wind by a momentum-and-energy-injection scheme (MEI).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.
        registered_variables: The registered variables.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    conservative_state = conserved_state_from_primitive(
        primitive_state, gamma, config, registered_variables
    )

    # Spherical injection volume out to the outer boundary of the last
    # injection cell.
    injection_volume = (
        4
        / 3
        * jnp.pi
        * helper_data.outer_cell_boundaries[num_injection_cells + num_ghost_cells] ** 3
    )

    # Distribute the per-step wind mass, momentum and energy over the injection
    # volume and add them to the conserved state.
    delta_density = wind_params.wind_mass_loss_rate * dt / injection_volume
    delta_momentum = wind_params.wind_final_velocity * delta_density
    delta_energy = 0.5 * wind_params.wind_final_velocity**2 * delta_density

    conservative_state = conservative_state.at[
        0, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].add(delta_density)
    conservative_state = conservative_state.at[
        1, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].add(delta_momentum)
    conservative_state = conservative_state.at[
        2, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].add(delta_energy)

    primitive_state = primitive_state_from_conserved(
        conservative_state, gamma, config, registered_variables
    )

    return primitive_state


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
    """Inject stellar wind by a thermal-energy-injection scheme (EI).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.
        registered_variables: The registered variables.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    source_term = jnp.zeros_like(primitive_state)

    # Total volume of the injection cells, over which the wind mass and energy
    # rates are distributed.
    injection_volume = jnp.sum(
        helper_data.cell_volumes[
            num_ghost_cells : num_injection_cells + num_ghost_cells
        ]
    )

    # Mass injection: a uniform density source rate over the injection cells.
    density_rate = wind_params.wind_mass_loss_rate / injection_volume
    source_term = source_term.at[
        0, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].set(density_rate)
    updated_density = (
        primitive_state[0, num_ghost_cells : num_injection_cells + num_ghost_cells]
        + density_rate * dt
    )

    # When a wind-density tracer is active, tag the injected mass with the same
    # source rate so the tracer follows the wind material.
    if registered_variables.wind_density_active:
        source_term = source_term.at[
            registered_variables.wind_density_index,
            num_ghost_cells : num_injection_cells + num_ghost_cells,
        ].set(density_rate)

    # Energy injection: convert the kinetic luminosity of the wind into a
    # pressure source rate at the freshly injected density.
    energy_rate = (
        0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate
        / injection_volume
    )

    pressure_rate = pressure_from_energy(
        energy_rate,
        updated_density,
        primitive_state[1, num_ghost_cells : num_injection_cells + num_ghost_cells],
        gamma,
    )

    source_term = source_term.at[
        2, num_ghost_cells : num_injection_cells + num_ghost_cells
    ].set(pressure_rate)

    primitive_state = primitive_state + source_term * dt

    return primitive_state


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
    """Inject identical winds from several hard-coded star positions (3D).

    A placeholder multi-source variant of the thermal-energy-injection scheme:
    it loops over a fixed list of star positions and adds a spherical mass and
    energy source around each.

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.
        registered_variables: The registered variables.

    Returns:
        The primitive state array with the stellar winds injected.
    """
    star_positions = [
        jnp.array([0.2, 0.3, 0.5]),
        jnp.array([0.5, 0.7, 0.5]),
        jnp.array([0.7, 0.4, 0.5]),
        jnp.array([0.3, 0.6, 0.5]),
    ]

    for star_position in star_positions:
        # Distance of every cell from this star.
        radius = jnp.linalg.norm(
            helper_data.geometric_centers - star_position, axis=-1
        )

        source_term = jnp.zeros_like(primitive_state)

        injection_radius = num_injection_cells * config.grid_spacing
        injection_volume = 4 / 3 * jnp.pi * injection_radius**3

        # Inject only inside the spherical injection region around the star.
        injection_mask = radius <= injection_radius - config.grid_spacing / 2

        # Mass injection: a uniform density source rate inside the mask.
        density_rate = wind_params.wind_mass_loss_rate / injection_volume
        source_term = source_term.at[registered_variables.density_index].set(
            density_rate * injection_mask
        )

        updated_density = primitive_state[registered_variables.density_index]
        updated_density = jnp.where(
            injection_mask > 0,
            updated_density + density_rate * dt * injection_mask,
            updated_density,
        )

        # Energy injection: the kinetic luminosity converted to a pressure
        # source rate at the freshly injected density.
        energy_rate = (
            0.5
            * wind_params.wind_final_velocity**2
            * wind_params.wind_mass_loss_rate
            / injection_volume
        )
        speed = jnp.sqrt(
            primitive_state[registered_variables.velocity_index.x] ** 2
            + primitive_state[registered_variables.velocity_index.y] ** 2
            + primitive_state[registered_variables.velocity_index.z] ** 2
        )
        pressure_rate = pressure_from_energy(
            energy_rate, updated_density, speed, gamma
        )

        source_term = source_term.at[registered_variables.pressure_index].set(
            pressure_rate * injection_mask
        )

        primitive_state = primitive_state + source_term * dt

    return primitive_state


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
    """Inject stellar wind by a thermal-energy-injection scheme in 3D (EI).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.
        registered_variables: The registered variables.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    source_term = jnp.zeros_like(primitive_state)

    injection_radius = num_injection_cells * config.grid_spacing
    injection_volume = 4 / 3 * jnp.pi * injection_radius**3

    # Inject only inside the spherical injection region at the box centre.
    injection_mask = helper_data.r <= injection_radius - config.grid_spacing / 2

    # Mass injection: a uniform density source rate inside the mask.
    density_rate = wind_params.wind_mass_loss_rate / injection_volume
    source_term = source_term.at[registered_variables.density_index].set(
        density_rate * injection_mask
    )

    updated_density = primitive_state[registered_variables.density_index]
    updated_density = jnp.where(
        injection_mask > 0,
        updated_density + density_rate * dt * injection_mask,
        updated_density,
    )

    # Energy injection: the kinetic luminosity converted to a pressure source
    # rate at the freshly injected density. A small floor on the speed avoids a
    # division by zero in the pressure conversion at rest.
    energy_rate = (
        0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate
        / injection_volume
    )
    speed = jnp.sqrt(
        primitive_state[registered_variables.velocity_index.x] ** 2
        + primitive_state[registered_variables.velocity_index.y] ** 2
        + primitive_state[registered_variables.velocity_index.z] ** 2
        + 1e-20
    )
    pressure_rate = pressure_from_energy(energy_rate, updated_density, speed, gamma)

    source_term = source_term.at[registered_variables.pressure_index].set(
        pressure_rate * injection_mask
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
    """Build the 3D stellar-wind source term for the conserved state (FD path).

    Returns the conserved-state increment for one time step: a tapered spherical
    mass and thermal-energy injection, plus a momentum correction that keeps the
    kinetic energy unchanged as the density grows.

    Args:
        wind_params: The wind parameters.
        conserved_state: The conserved state array.
        dt: The time step.
        config: The simulation configuration.
        helper_data: The helper data.
        num_injection_cells: The number of injection cells.
        registered_variables: The registered variables.

    Returns:
        The conserved-state source-term increment for this time step.
    """

    source_term = jnp.zeros_like(conserved_state)

    injection_radius = num_injection_cells * config.grid_spacing
    taper_radius = 1.3 * injection_radius

    # Taper the injection linearly between the injection radius and the taper
    # radius to avoid a sharp cut-off at the injection boundary.
    injection_mask = jnp.where(
        helper_data.r <= injection_radius,
        1.0,
        jnp.where(
            (helper_data.r > injection_radius) & (helper_data.r <= taper_radius),
            (taper_radius - helper_data.r) / (taper_radius - injection_radius),
            0.0,
        ),
    )

    injection_volume = jnp.sum(injection_mask) * config.grid_spacing**3

    # Mass injection.
    density_rate = wind_params.wind_mass_loss_rate / injection_volume

    source_term = source_term.at[registered_variables.density_index].set(
        density_rate * injection_mask * dt
    )

    # Energy injection.
    delta_energy = (
        0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate
        / injection_volume * dt
    )

    # We only want to inject thermal, not kinetic, energy. The kinetic energy is
    # 1/2 rho v^2 = 1/2 m^2 / rho (momentum m). Adding mass while holding the
    # momentum fixed would change the kinetic energy, so we rescale the momentum
    # to keep the kinetic energy constant:
    #   1/2 m_old^2 / rho_old = 1/2 m_new^2 / rho_new
    #   -> m_new = m_old sqrt(rho_new / rho_old)
    #   -> dm = m_old (sqrt(1 + drho * dt / rho_old) - 1).
    momentum_source_factor = jnp.sqrt(
        1 + density_rate * dt * injection_mask
        / (conserved_state[registered_variables.density_index])
    ) - 1.0
    # Restrict the momentum correction to within the taper radius. Outside it the
    # factor is already (sqrt(1 + 0) - 1) = 0 up to small numerical error, so
    # this just zeroes that residual.
    momentum_source_factor = jnp.where(
        helper_data.r <= taper_radius, momentum_source_factor, 0.0
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
        delta_energy * injection_mask
    )

    return source_term

# -------------------------------------------------------------
# =============== ↑ Wind injection schemes ↑ ==================
# -------------------------------------------------------------
