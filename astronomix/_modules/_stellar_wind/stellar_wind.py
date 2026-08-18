"""
Stellar-wind injection into the fluid state.

Implements the wind injection schemes of https://arxiv.org/abs/2107.14673 for
the finite-volume solver (mass-and-energy overwrite, momentum-and-energy
injection, thermal-energy injection) in 1D and 3D, plus the source-term variant
used by the finite-difference solver. ``_wind_injection`` dispatches to the
appropriate scheme based on the configuration.

The 3D thermal-energy-injection scheme (``_wind_ei3D`` / ``_wind_ei3D_source``)
supports multiple simultaneous wind sources: their injection positions are
taken from the current N-body state every step when
``config.nbody_config.nbody`` is enabled (so the wind sources track the N-body
orbits), else from ``params.wind_params.wind_injection_positions``; and their
mass-loss rate / terminal velocity are taken from tabulated, time-dependent
stellar-evolution tracks (``params.wind_params.real_params``) when
``config.wind_config.real_wind_params`` is enabled, else from the static
``wind_mass_loss_rates`` / ``wind_final_velocities`` arrays. See
``_wind_source_params``.
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
from astronomix._modules._stellar_wind.stellar_wind_functions import (
    get_current_wind_params,
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
                params,
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


def _wind_source_params(
    wind_params: WindParams,
    config: SimulationConfig,
    params: SimulationParams,
):
    """Resolve the per-source injection positions, mass-loss rates and
    terminal velocities for the 3D multi-source wind schemes below.

    Positions come from the current N-body state (``config.nbody_config.nbody``)
    when active — so the wind sources track the N-body orbits every step,
    computed jointly with the hydro update (see
    ``astronomix.time_stepping.time_integration``) — else from
    ``wind_params.wind_injection_positions``. Both are in the same
    box-centered coordinates as the N-body state (box center at the origin;
    see ``astronomix._modules._nbody._nbody``).

    Rates come from the tabulated stellar-evolution tracks
    (``wind_params.real_params``) interpolated to the current time when
    ``config.wind_config.real_wind_params``, else from the static per-source
    ``wind_mass_loss_rates`` / ``wind_final_velocities`` arrays.

    Args:
        wind_params: The wind parameters (``params.wind_params``).
        config: The simulation configuration.
        params: The simulation parameters (provides the N-body state).

    Returns:
        ``(source_positions, mass_rates, vel_scales)`` of shape
        (n_sources, 3) / (n_sources,) / (n_sources,).
    """
    if config.nbody_config.nbody:
        if config.nbody_config.central_object_only:
            source_positions = jnp.zeros((1, 3), dtype=wind_params.wind_injection_positions.dtype)
        else:
            n_bodies = params.nbody_params.masses.size
            source_positions = params.nbody_params.nbody_state.reshape((n_bodies, 7))[:, 1:4]
    else:
        source_positions = wind_params.wind_injection_positions

    if config.wind_config.real_wind_params:
        time_value, mass_rates_value, vel_scales_value = wind_params.real_params
        mass_rates, vel_scales = get_current_wind_params(
            mass_rates_value, vel_scales_value, wind_params.current_time, time_value,
        )
    else:
        mass_rates = jnp.asarray(wind_params.wind_mass_loss_rates)
        vel_scales = jnp.asarray(wind_params.wind_final_velocities)

    return source_positions, mass_rates, vel_scales


def _wind_source_distances(config: SimulationConfig, helper_data: HelperData, source_positions):
    """Distance of every grid cell from every wind source.

    Args:
        config: The simulation configuration.
        helper_data: The helper data (provides ``geometric_centers``).
        source_positions: Per-source positions, shape (n_sources, 3), in the
            same box-centered coordinates as ``geometric_centers`` once
            re-centered below.

    Returns:
        Distances, shape (n_sources, nx, ny, nz).
    """
    box_center = jnp.array(
        [config.box_size.x / 2, config.box_size.y / 2, config.box_size.z / 2]
    )
    centered = helper_data.geometric_centers - box_center
    delta = centered[None, ...] - source_positions[:, None, None, None, :]
    return jnp.linalg.norm(delta, axis=-1)


@partial(
    jax.jit,
    static_argnames=["num_ghost_cells", "num_injection_cells", "registered_variables", "config"],
)
def _wind_ei3D(
    params: SimulationParams,
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    num_ghost_cells: int,
    num_injection_cells: int,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Inject stellar wind by a thermal-energy-injection scheme in 3D (EI),
    from one or more sources (see ``_wind_source_params``).

    Args:
        params: The simulation parameters (provides the wind and, when
            active, N-body parameters).
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
    wind_params = params.wind_params
    source_positions, mass_rates, vel_scales = _wind_source_params(
        wind_params, config, params
    )

    source_term = jnp.zeros_like(primitive_state)

    injection_radius = num_injection_cells * config.grid_spacing
    injection_volume = 4 / 3 * jnp.pi * injection_radius**3

    # Inject only inside the spherical injection region around each source.
    dist = _wind_source_distances(config, helper_data, source_positions)
    per_source_mask = (dist <= injection_radius - config.grid_spacing / 2).astype(
        primitive_state.dtype
    )

    # Mass injection: a uniform density source rate inside each source's
    # mask, summed over sources.
    density_rate_sources = (mass_rates / injection_volume)[:, None, None, None] * per_source_mask
    density_rate = jnp.sum(density_rate_sources, axis=0)
    source_term = source_term.at[registered_variables.density_index].set(density_rate)

    updated_density = primitive_state[registered_variables.density_index]
    updated_density = jnp.where(
        density_rate > 0,
        updated_density + density_rate * dt,
        updated_density,
    )

    # Energy injection: the per-source kinetic luminosity converted to a
    # pressure source rate at the freshly injected density. A small floor on
    # the speed avoids a division by zero in the pressure conversion at rest.
    energy_rate_sources = (
        (0.5 * vel_scales**2 * mass_rates / injection_volume)[:, None, None, None]
        * per_source_mask
    )
    energy_rate = jnp.sum(energy_rate_sources, axis=0)
    speed = jnp.sqrt(
        primitive_state[registered_variables.velocity_index.x] ** 2
        + primitive_state[registered_variables.velocity_index.y] ** 2
        + primitive_state[registered_variables.velocity_index.z] ** 2
        + 1e-20
    )
    pressure_rate = pressure_from_energy(energy_rate, updated_density, speed, gamma)

    source_term = source_term.at[registered_variables.pressure_index].set(pressure_rate)

    primitive_state = primitive_state + source_term * dt

    return primitive_state


@partial(
    jax.jit,
    static_argnames=["num_injection_cells", "registered_variables", "config"],
)
def _wind_ei3D_source(
    params: SimulationParams,
    conserved_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    num_injection_cells: int,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Build the 3D stellar-wind source term for the conserved state (FD path),
    from one or more sources (see ``_wind_source_params``).

    Returns the conserved-state increment for one time step: a tapered spherical
    mass and thermal-energy injection per source, plus a momentum correction
    that keeps the kinetic energy unchanged as the density grows.

    Args:
        params: The simulation parameters (provides the wind and, when
            active, N-body parameters).
        conserved_state: The conserved state array.
        dt: The time step.
        config: The simulation configuration.
        helper_data: The helper data.
        num_injection_cells: The number of injection cells.
        registered_variables: The registered variables.

    Returns:
        The conserved-state source-term increment for this time step.
    """
    wind_params = params.wind_params
    source_positions, mass_rates, vel_scales = _wind_source_params(
        wind_params, config, params
    )

    source_term = jnp.zeros_like(conserved_state)

    injection_radius = num_injection_cells * config.grid_spacing
    taper_radius = 1.3 * injection_radius

    # Taper the injection linearly between the injection radius and the taper
    # radius to avoid a sharp cut-off at the injection boundary, per source.
    dist = _wind_source_distances(config, helper_data, source_positions)
    injection_mask = jnp.where(
        dist <= injection_radius,
        1.0,
        jnp.where(
            (dist > injection_radius) & (dist <= taper_radius),
            (taper_radius - dist) / (taper_radius - injection_radius),
            0.0,
        ),
    )  # (n_sources, nx, ny, nz)

    injection_volume = jnp.sum(injection_mask, axis=(1, 2, 3)) * config.grid_spacing**3
    injection_volume_safe = jnp.where(injection_volume > 0, injection_volume, 1.0)

    # Mass injection, summed over sources.
    density_rate_sources = (mass_rates / injection_volume_safe)[:, None, None, None] * injection_mask
    density_rate = jnp.sum(density_rate_sources, axis=0)
    added_density = density_rate * dt

    source_term = source_term.at[registered_variables.density_index].set(added_density)

    # Energy injection, summed over sources.
    energy_rate_sources = (
        (0.5 * vel_scales**2 * mass_rates / injection_volume_safe)[:, None, None, None]
        * injection_mask
    )
    delta_energy = jnp.sum(energy_rate_sources, axis=0) * dt

    # We only want to inject thermal, not kinetic, energy. The kinetic energy is
    # 1/2 rho v^2 = 1/2 m^2 / rho (momentum m). Adding mass while holding the
    # momentum fixed would change the kinetic energy, so we rescale the momentum
    # to keep the kinetic energy constant:
    #   1/2 m_old^2 / rho_old = 1/2 m_new^2 / rho_new
    #   -> m_new = m_old sqrt(rho_new / rho_old)
    #   -> dm = m_old (sqrt(1 + drho / rho_old) - 1).
    momentum_source_factor = jnp.sqrt(
        1 + added_density / conserved_state[registered_variables.density_index]
    ) - 1.0
    # Restrict the momentum correction to cells touched by at least one
    # source's taper region. Untouched cells already give factor
    # (sqrt(1 + 0) - 1) = 0 up to small numerical error, so this just zeroes
    # that residual.
    touched = jnp.any(dist <= taper_radius, axis=0)
    momentum_source_factor = jnp.where(touched, momentum_source_factor, 0.0)

    source_term = source_term.at[registered_variables.momentum_index.x].set(
        conserved_state[registered_variables.momentum_index.x] * momentum_source_factor
    )
    source_term = source_term.at[registered_variables.momentum_index.y].set(
        conserved_state[registered_variables.momentum_index.y] * momentum_source_factor
    )
    source_term = source_term.at[registered_variables.momentum_index.z].set(
        conserved_state[registered_variables.momentum_index.z] * momentum_source_factor
    )

    source_term = source_term.at[registered_variables.energy_index].set(delta_energy)

    return source_term

# -------------------------------------------------------------
# =============== ↑ Wind injection schemes ↑ ==================
# -------------------------------------------------------------
