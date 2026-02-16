"""
Fourier-based Poisson solver and simple source term handling
of self gravity. To be improved to an energy-conserving scheme.
"""

# general
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Union

# astronomix data classes
from astronomix._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from astronomix._physics_modules._self_gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._finite_volume._riemann_solver._riemann_solver import _riemann_solver
from astronomix._stencil_operations._stencil_operations import _stencil_add
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    DONOR_ACCOUNTING,
    GHOST_CELLS,
    HALF_SPLIT,
    HLLC_LM,
    LAX_FRIEDRICHS,
    RIEMANN_SPLIT,
    RIEMANN_SPLIT_UNSTABLE,
    SIMPLE_SOURCE_TERM,
    SPLIT,
    STATE_TYPE_ALTERED,
    SimulationConfig,
)

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    HLL,
    HLLC,
    OPEN_BOUNDARY,
    STATE_TYPE,
)

# astronomix functions
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._finite_volume._riemann_solver.hll import _hll_solver, _hllc_solver
from astronomix._finite_volume._state_evolution.reconstruction import (
    _reconstruct_at_interface_split,
    _reconstruct_at_interface_unsplit,
)
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    primitive_state_from_conserved,
    speed_of_sound,
)
from astronomix.option_classes.simulation_params import SimulationParams


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["axis", "grid_spacing", "registered_variables", "config"]
)
def _gravitational_source_term_along_axis(
    gravitational_potential: FIELD_TYPE,
    primitive_state: STATE_TYPE,
    grid_spacing: float,
    registered_variables: RegisteredVariables,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    axis: int,
) -> STATE_TYPE:
    """
    Compute the source term for the self-gravity solver along a single axis.
    Currently, simply density * gravitational_acceleration for the momentum
    and density * velocity * gravitational_acceleration for the energy.

    Args:
        gravitational_potential: The gravitational potential.
        primitive_state: The primitive state.
        grid_spacing: The grid spacing.
        registered_variables: The registered variables.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.
        axis: The axis along which to compute the source term.

    Returns:
        The source term.

    """

    rho = primitive_state[registered_variables.density_index]
    v_axis = primitive_state[axis]

    # a_i = - (phi_{i+1} - phi_{i-1}) / (2 * dx)
    acceleration = -_stencil_add(
        gravitational_potential, indices=(1, -1), factors=(1.0, -1.0), axis=axis - 1
    ) / (2 * grid_spacing)
    # it is axis - 1 because the axis is 1-indexed as usually the zeroth axis are the different
    # fields in the state vector not the spatial dimensions, but here we only have the spatial dimensions

    source_term = jnp.zeros_like(primitive_state)

    # set momentum source
    source_term = source_term.at[axis].set(rho * acceleration)

    if config.self_gravity_version == SIMPLE_SOURCE_TERM:
        # set energy source
        source_term = source_term.at[registered_variables.pressure_index].set(
            rho * v_axis * acceleration
        )

    else:
        # the other schemes are based on the reconstructed states
        if config.first_order_fallback:
            primitive_state_left = jnp.roll(primitive_state, shift=1, axis=axis)
            primitive_state_right = primitive_state
        else:
            if config.split == SPLIT:
                primitive_state_left, primitive_state_right = (
                    _reconstruct_at_interface_split(
                        primitive_state,
                        dt,
                        gamma,
                        config,
                        helper_data,
                        registered_variables,
                        axis,
                    )
                )
            else:
                # TODO: improve efficiency
                # this is currently suboptimal, the reconstruction is done for all axes
                # but we only need it for the current axis
                primitives_left_interface, primitives_right_interface = (
                    _reconstruct_at_interface_unsplit(
                        primitive_state,
                        dt,
                        gamma,
                        config,
                        params,
                        helper_data,
                        registered_variables,
                    )
                )
                primitive_state_left = primitives_left_interface[axis - 1]
                primitive_state_right = primitives_right_interface[axis - 1]

        if (
            config.self_gravity_version == RIEMANN_SPLIT
            or config.self_gravity_version == RIEMANN_SPLIT_UNSTABLE
        ):
            # improve code reuse, instead of this copied
            # Riemann solver

            if not (config.riemann_solver == HLLC or config.riemann_solver == HLLC_LM):
                raise NotImplementedError(
                    "The RIEMANN_SPLIT gravity scheme is currently only implemented for HLLC and HLLC_LM."
                )

            rho_L = primitive_state_left[registered_variables.density_index]
            u_L = primitive_state_left[axis]

            rho_R = primitive_state_right[registered_variables.density_index]
            u_R = primitive_state_right[axis]

            p_L = primitive_state_left[registered_variables.pressure_index]
            p_R = primitive_state_right[registered_variables.pressure_index]

            # calculate the sound speeds
            c_L = speed_of_sound(rho_L, p_L, gamma)
            c_R = speed_of_sound(rho_R, p_R, gamma)

            # get the left and right states and fluxes
            F_L = _euler_flux(
                primitive_state_left, gamma, config, registered_variables, axis
            )
            F_R = _euler_flux(
                primitive_state_right, gamma, config, registered_variables, axis
            )

            # Roe average of the velocity
            u_hat = (jnp.sqrt(rho_L) * u_L + jnp.sqrt(rho_R) * u_R) / (
                jnp.sqrt(rho_L) + jnp.sqrt(rho_R)
            )

            # Roe average of the sound speed
            c_hat_squared = (c_L**2 * jnp.sqrt(rho_L) + c_R**2 * jnp.sqrt(rho_R)) / (
                jnp.sqrt(rho_L) + jnp.sqrt(rho_R)
            ) + 0.5 * (
                jnp.sqrt(rho_L)
                * jnp.sqrt(rho_R)
                / (jnp.sqrt(rho_L) + jnp.sqrt(rho_R)) ** 2
            ) * (u_R - u_L) ** 2
            c_hat = jnp.sqrt(c_hat_squared)

            # Einfeldt estimates of maximum left and right signal speeds
            S_L = jnp.minimum(u_L - c_L, u_hat - c_hat)
            S_R = jnp.maximum(u_R + c_R, u_hat + c_hat)

            # contact wave signal speed
            S_star = (
                p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
            ) / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R))

            # intermediate states
            U_L = conserved_state_from_primitive(
                primitive_state_left, gamma, config, registered_variables
            )
            U_R = conserved_state_from_primitive(
                primitive_state_right, gamma, config, registered_variables
            )

            U_star_L = U_L.at[axis].set(rho_L * S_star)
            U_star_L = U_star_L.at[registered_variables.pressure_index].add(
                (S_star - u_L) * (rho_L * S_star + p_L / (S_L - u_L))
            )
            U_star_L = U_star_L * (S_L - u_L) / (S_L - S_star)

            U_star_R = U_R.at[axis].set(rho_R * S_star)
            U_star_R = U_star_R.at[registered_variables.pressure_index].add(
                (S_star - u_R) * (rho_R * S_star + p_R / (S_R - u_R))
            )
            U_star_R = U_star_R * (S_R - u_R) / (S_R - S_star)

            # HLLC-LM adaptation
            # following
            # https://doi.org/10.1016/j.jcp.2020.109762
            if config.riemann_solver == HLLC_LM:
                Ma_limit = 0.1
                Ma_local = jnp.maximum(jnp.abs(u_L / c_L), jnp.abs(u_R / c_R))
                phi = jnp.sin(jnp.minimum(1, Ma_local / Ma_limit) * jnp.pi / 2)
                S_Llm = S_L * phi
                S_Rlm = S_R * phi

            if config.riemann_solver == HLLC_LM:
                S_Lstar = S_Llm
                S_Rstar = S_Rlm
            else:
                S_Lstar = S_L
                S_Rstar = S_R

            bulk_flux_star = 0.5 * (F_L + F_R)
            dissipation_term_star = 0.5 * (
                S_Lstar * (U_star_L - U_L)
                + jnp.abs(S_star) * (U_star_L - U_star_R)
                + S_Rstar * (U_star_R - U_R)
            )

            F_star = bulk_flux_star + dissipation_term_star

            fluxes = jnp.where(S_L >= 0, F_L, F_star)
            fluxes = jnp.where(S_R <= 0, F_R, fluxes)

            # what cell i accounts for regarding the flux between i-1 and i
            fluxes_i_to_im1 = 0.5 * F_R + jnp.minimum(dissipation_term_star, 0)

            # what cell i-1 accounts for regarding the flux between i-1 and i
            fluxes_im1 = 0.5 * F_L + jnp.maximum(dissipation_term_star, 0)

            if config.self_gravity_version == RIEMANN_SPLIT_UNSTABLE:
                # stable but big spread in specific entropy
                # fluxes_i_to_im1 = jnp.where(S_R <= 0, F_R, fluxes_i_to_im1)
                # fluxes_i_to_im1 = jnp.where(S_L >= 0, 0, fluxes_i_to_im1)
                # fluxes_im1 = jnp.where(S_L >= 0, F_L, fluxes_im1)
                # fluxes_im1 = jnp.where(S_R <= 0, 0, fluxes_im1)

                # less stable but reduced spread

                fluxes_i_to_im1 = jnp.where(S_R <= 0, F_R / 2, fluxes_i_to_im1)
                fluxes_i_to_im1 = jnp.where(S_L >= 0, F_L / 2, fluxes_i_to_im1)
                fluxes_im1 = jnp.where(S_L >= 0, F_L / 2, fluxes_im1)
                fluxes_im1 = jnp.where(S_R <= 0, F_R / 2, fluxes_im1)

            fluxes_i_to_ip1 = jnp.roll(fluxes_im1, shift=-1, axis=axis)

        elif config.self_gravity_version == DONOR_ACCOUNTING:
            # at index i, the fluxes array contains the flux from i-1 to i
            fluxes = _riemann_solver(
                primitive_state_left,
                primitive_state_right,
                primitive_state,
                gamma,
                config,
                registered_variables,
                axis,
            )
            fluxes_i_to_ip1 = jnp.maximum(jnp.roll(fluxes, shift=-1, axis=axis), 0)
            fluxes_i_to_im1 = jnp.minimum(fluxes, 0)

        elif config.self_gravity_version == HALF_SPLIT:
            # at index i, the fluxes array contains the flux from i-1 to i
            fluxes = _riemann_solver(
                primitive_state_left,
                primitive_state_right,
                primitive_state,
                gamma,
                config,
                registered_variables,
                axis,
            )
            fluxes_i_to_ip1 = 0.5 * jnp.roll(fluxes, shift=-1, axis=axis)
            fluxes_i_to_im1 = 0.5 * fluxes

        acc_backward = (
            -_stencil_add(
                gravitational_potential,
                indices=(0, -1),
                factors=(1.0, -1.0),
                axis=axis - 1,
            )
            / grid_spacing
        )
        acc_forward = (
            -_stencil_add(
                gravitational_potential,
                indices=(1, 0),
                factors=(1.0, -1.0),
                axis=axis - 1,
            )
            / grid_spacing
        )

        fluxes_acc = fluxes_i_to_im1 * acc_backward + fluxes_i_to_ip1 * acc_forward

        source_term = source_term.at[registered_variables.pressure_index].set(
            fluxes_acc[0]
        )

    return source_term


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _apply_self_gravity(
    primitive_state: STATE_TYPE,
    old_primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    dt: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    rho = old_primitive_state[registered_variables.density_index]

    potential = _compute_gravitational_potential(
        rho, config.grid_spacing, config, gravitational_constant
    )

    source_term = jnp.zeros_like(primitive_state)

    for i in range(config.dimensionality):
        source_term = source_term + _gravitational_source_term_along_axis(
            potential,
            old_primitive_state,
            config.grid_spacing,
            registered_variables,
            dt,
            gamma,
            config,
            params,
            helper_data,
            i + 1,
        )

    conserved_state = conserved_state_from_primitive(
        primitive_state, gamma, config, registered_variables
    )

    conserved_state = conserved_state + dt * source_term

    primitive_state = primitive_state_from_conserved(
        conserved_state, gamma, config, registered_variables
    )


    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(primitive_state, config)

    return primitive_state