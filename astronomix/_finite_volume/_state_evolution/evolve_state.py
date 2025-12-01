# general imports
import jax.numpy as jnp
import jax
from functools import partial

# runtime debugging
from jax.experimental import checkify

# type checking imports
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# general astronomix imports
from astronomix._finite_volume._riemann_solver._riemann_solver import _riemann_solver
from astronomix._finite_volume._magnetic_update._magnetic_field_update import magnetic_update
from astronomix._physics_modules._self_gravity._self_gravity import _apply_self_gravity
from astronomix._stencil_operations._stencil_operations import _stencil_add
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    GHOST_CELLS,
    RK2_SSP,
    SPHERICAL,
    STATE_TYPE,
    UNSPLIT,
    VAN_ALBADA_PP,
    SimulationConfig,
)

from astronomix._geometry.geometric_terms import _pressure_nozzling_source
from astronomix._finite_volume._state_evolution.reconstruction import (
    _reconstruct_at_interface_split,
    _reconstruct_at_interface_unsplit,
    _reconstruct_at_interface_unsplit_single,
)
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._fluid_equations._equations import (
    primitive_state_from_conserved,
    conserved_state_from_primitive,
)
from astronomix._finite_volume._riemann_solver._lax_friedrichs import _lax_friedrichs_solver
from astronomix.option_classes.simulation_params import SimulationParams

# -------------------------------------------------------------
# ====================== ↓ Split Scheme ↓ =====================
# -------------------------------------------------------------


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables", "axis"])
def _evolve_state_along_axis(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    axis: int,
) -> STATE_TYPE:
    
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(primitive_state, config)

    # get conserved variables
    conservative_states = conserved_state_from_primitive(
        primitive_state, gamma, config, registered_variables
    )

    if config.first_order_fallback:
        primitive_state_left = jnp.roll(primitive_state, shift=1, axis=axis)
        primitive_state_right = primitive_state
    else:
        primitive_state_left, primitive_state_right = _reconstruct_at_interface_split(
            primitive_state, dt, gamma, config, helper_data, registered_variables, axis
        )

    fluxes = _riemann_solver(
        primitive_state_left,
        primitive_state_right,
        primitive_state,
        gamma,
        config,
        registered_variables,
        axis,
    )

    # ================ update the conserved variables =================

    # usual cartesian case
    if config.geometry == CARTESIAN:
        conserved_change = (
            1
            / grid_spacing
            * _stencil_add(fluxes, indices=(0, 1), factors=(1.0, -1.0), axis=axis)
            * dt
        )

    # in spherical geometry, we have to take special care
    elif config.geometry == SPHERICAL and config.dimensionality == 1 and axis == 1:
        r = helper_data.geometric_centers
        r_hat_alpha = helper_data.r_hat_alpha

        alpha = config.geometry

        r_plus_half = r + grid_spacing / 2
        r_minus_half = r - grid_spacing / 2

        # calculate the source terms
        nozzling_source = _pressure_nozzling_source(
            primitive_state, config, helper_data, registered_variables
        )

        # update the conserved variables using the fluxes and source terms
        conserved_change = (
            1
            / r_hat_alpha
            * (
                +(
                    r_minus_half**alpha * fluxes
                    - r_plus_half**alpha * jnp.roll(fluxes, shift=-1, axis=axis)
                )
                / grid_spacing
                + nozzling_source
            )
            * dt
        )

    # misconfiguration
    else:
        raise ValueError("Geometry and dimensionality combination not supported.")

    # =================================================================

    conservative_states = conservative_states + conserved_change

    primitive_state = primitive_state_from_conserved(
        conservative_states, gamma, config, registered_variables
    )
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(primitive_state, config)

    # check if the pressure is still positive
    p = primitive_state[registered_variables.pressure_index]
    rho = primitive_state[registered_variables.density_index]

    if config.runtime_debugging:
        checkify.check(
            jnp.all(p >= 0),
            "pressure needs to be non-negative, minimum pressure {pmin} at index {index}",
            pmin=jnp.min(p),
            index=jnp.unravel_index(jnp.argmin(p), p.shape),
        )
        checkify.check(
            jnp.all(rho >= 0),
            "density needs to be non-negative, minimum density {rhomin} at index {index}",
            rhomin=jnp.min(rho),
            index=jnp.unravel_index(jnp.argmin(rho), rho.shape),
        )

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _evolve_gas_state_split(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    if config.dimensionality == 1:
        if config.self_gravity:
            old_primitive_state = primitive_state

        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt,
            gamma,
            config,
            helper_data,
            registered_variables,
            1,
        )

        if config.self_gravity:
            primitive_state = _apply_self_gravity(
                primitive_state,
                old_primitive_state,
                config,
                params,
                registered_variables,
                helper_data,
                gamma,
                gravitational_constant,
                dt,
            )

    elif config.dimensionality == 2:
        if config.self_gravity:
            old_primitive_state = primitive_state

        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt / 2,
            gamma,
            config,
            helper_data,
            registered_variables,
            1,
        )
        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt,
            gamma,
            config,
            helper_data,
            registered_variables,
            2,
        )
        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt / 2,
            gamma,
            config,
            helper_data,
            registered_variables,
            1,
        )

        if config.self_gravity:
            primitive_state = _apply_self_gravity(
                primitive_state,
                old_primitive_state,
                config,
                params,
                registered_variables,
                helper_data,
                gamma,
                gravitational_constant,
                dt,
            )

    elif config.dimensionality == 3:
        if config.self_gravity:
            old_primitive_state = primitive_state

        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt / 2,
            gamma,
            config,
            helper_data,
            registered_variables,
            1,
        )
        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt / 2,
            gamma,
            config,
            helper_data,
            registered_variables,
            2,
        )
        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt,
            gamma,
            config,
            helper_data,
            registered_variables,
            3,
        )
        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt / 2,
            gamma,
            config,
            helper_data,
            registered_variables,
            2,
        )
        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt / 2,
            gamma,
            config,
            helper_data,
            registered_variables,
            1,
        )

        if config.self_gravity:
            primitive_state = _apply_self_gravity(
                primitive_state,
                old_primitive_state,
                config,
                params,
                registered_variables,
                helper_data,
                gamma,
                gravitational_constant,
                dt,
            )

    else:
        raise ValueError("Dimensionality not supported.")

    return primitive_state


# -------------------------------------------------------------
# ====================== ↑ Split Scheme ↑ =====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ Unsplit Scheme ↓ ====================
# -------------------------------------------------------------


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _evolve_gas_state_unsplit_inner(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(primitive_state, config)
    conservative_states = conserved_state_from_primitive(
        primitive_state, gamma, config, registered_variables
    )

    # in case of the van albada pp limiter, the limited
    # gradients along all dimensions are needed at once for
    # the proper multidimensional limiting
    if config.limiter == VAN_ALBADA_PP:
        # get left and right states along all dimensions
        pls, prs = _reconstruct_at_interface_unsplit(
            primitive_state,
            dt,
            gamma,
            config,
            params,
            helper_data,
            registered_variables
        )

    for axis in range(1, config.dimensionality + 1):

        if config.boundary_handling == GHOST_CELLS:
            primitive_state = _boundary_handler(primitive_state, config)

        if config.limiter == VAN_ALBADA_PP:
            primitives_left_interface = pls[axis - 1]
            primitives_right_interface = prs[axis - 1]
        else:
            primitives_left_interface, primitives_right_interface = (
                _reconstruct_at_interface_unsplit_single(
                    primitive_state, config, helper_data, axis
                )
            )

        # get the fluxes at the interfaces
        fluxes = _riemann_solver(
            primitives_left_interface,
            primitives_right_interface,
            primitive_state,
            gamma,
            config,
            registered_variables,
            axis,
        )

        # update the conserved variables
        conserved_change = (
            1
            / config.grid_spacing
            * _stencil_add(fluxes, indices=(0, 1), factors=(1.0, -1.0), axis=axis)
            * dt
        )
        conservative_states += conserved_change

    # update the primitive state
    primitive_state = primitive_state_from_conserved(
        conservative_states, gamma, config, registered_variables
    )

    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(primitive_state, config)

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _evolve_gas_state_unsplit(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:

    if config.self_gravity:
        old_primitive_state = primitive_state

    if config.time_integrator == RK2_SSP:
        primitive_state_1 = _evolve_gas_state_unsplit_inner(
            primitive_state,
            dt,
            gamma,
            gravitational_constant,
            config,
            params,
            helper_data,
            registered_variables,
        )

        primitive_state_2 = _evolve_gas_state_unsplit_inner(
            primitive_state_1,
            dt,
            gamma,
            gravitational_constant,
            config,
            params,
            helper_data,
            registered_variables,
        )

        conserved_state = conserved_state_from_primitive(
            primitive_state, gamma, config, registered_variables
        )

        conserved_state_2 = conserved_state_from_primitive(
            primitive_state_2, gamma, config, registered_variables
        )

        # RK2
        conserved_state = 0.5 * (conserved_state + conserved_state_2)

        primitive_state = primitive_state_from_conserved(
            conserved_state, gamma, config, registered_variables
        )
    else:
        raise ValueError(
            "Only the RK2 SSP time integrator is currently supported for the unsplit scheme."
        )

    if config.self_gravity:
        primitive_state = _apply_self_gravity(
            primitive_state,
            old_primitive_state,
            config,
            params,
            registered_variables,
            helper_data,
            gamma,
            gravitational_constant,
            dt,
        )

    return primitive_state

# -------------------------------------------------------------
# ===================== ↑ Unsplit Scheme ↑ ====================
# -------------------------------------------------------------


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _evolve_state_fv(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    if config.mhd:
        if config.dimensionality > 1:

            # HERE WE EXPLICITLY ASSUME THAT THE LAST 3 INDICES 
            # ARE THE MAGNETIC FIELD COMPONENTS
            registered_variables_gas = registered_variables._replace(
                num_vars=registered_variables.num_vars - 3
            )

            gas_state = primitive_state[:-3, ...]
            magnetic_field = primitive_state[-3:, ...]

            if config.split == UNSPLIT:
                evolved_gas = _evolve_gas_state_unsplit(
                    gas_state,
                    dt / 2,
                    gamma,
                    gravitational_constant,
                    config,
                    params,
                    helper_data,
                    registered_variables_gas,
                )
            else:
                evolved_gas = _evolve_gas_state_split(
                    gas_state,
                    dt / 2,
                    gamma,
                    gravitational_constant,
                    config,
                    params,
                    helper_data,
                    registered_variables_gas,
                )

            magnetic_field, evolved_gas = magnetic_update(
                magnetic_field,
                evolved_gas,
                config.grid_spacing,
                dt,
                registered_variables,
                config,
            )

            if config.split == UNSPLIT:
                evolved_gas = _evolve_gas_state_unsplit(
                    evolved_gas,
                    dt / 2,
                    gamma,
                    gravitational_constant,
                    config,
                    params,
                    helper_data,
                    registered_variables_gas,
                )
            else:
                evolved_gas = _evolve_gas_state_split(
                    evolved_gas,
                    dt / 2,
                    gamma,
                    gravitational_constant,
                    config,
                    params,
                    helper_data,
                    registered_variables_gas,
                )

            return jnp.concatenate((evolved_gas, magnetic_field), axis=0)
        else:
            raise ValueError("MHD currently not supported in 1D.")

    else:
        if config.split == UNSPLIT:
            return _evolve_gas_state_unsplit(
                primitive_state,
                dt,
                gamma,
                gravitational_constant,
                config,
                params,
                helper_data,
                registered_variables,
            )
        else:
            return _evolve_gas_state_split(
                primitive_state,
                dt,
                gamma,
                gravitational_constant,
                config,
                params,
                helper_data,
                registered_variables,
            )