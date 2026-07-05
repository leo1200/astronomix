"""
Finite-volume state evolution.

Drives one finite-volume hydro step: the dimensionally split (Strang) and the
unsplit (SSP-RK2, optionally Pallas-fused) gas updates, the operator-split
self-gravity source, and the top-level :func:`_evolve_state_fv` that couples
the gas update to the magnetic-field update for MHD runs.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp
from jax.experimental import checkify

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    GHOST_CELLS,
    RK2_SSP,
    SPHERICAL,
    STATE_TYPE,
    UNSPLIT,
    VAN_ALBADA_PP,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_volume._riemann_solver._riemann_solver import _riemann_solver
from astronomix._finite_volume._magnetic_update._magnetic_field_update import magnetic_update
from astronomix._integrators._explicit_rk import rk2_ssp
from astronomix._modules._time_integrator_sources import _time_integrator_sources
from astronomix._stencil_operations._stencil_operations import _stencil_add
from astronomix._geometry.geometric_terms import _pressure_nozzling_source
from astronomix._finite_volume._state_evolution.reconstruction import (
    _reconstruct_at_interface_split,
    _reconstruct_at_interface_unsplit,
    _reconstruct_at_interface_unsplit_single,
)
from astronomix._finite_volume._state_evolution._pallas_evolve import (
    _evolve_gas_state_unsplit_pallas,
    _fv_pallas_evolve_supported,
)
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._fluid_equations._equations import (
    primitive_state_from_conserved,
    conserved_state_from_primitive,
)

# -------------------------------------------------------------
# ====================== ↓ Self-gravity ↓ =====================
# -------------------------------------------------------------


def _gravity_source_presolve(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Build the operator-split self-gravity source from the pre-hydro state.

    The source depends only on the pre-hydro state, so evaluating it before
    the hydro update and adding it afterwards reproduces the former
    ``_apply_self_gravity`` scheme exactly.
    """
    return _time_integrator_sources(
        conserved_state_from_primitive(
            primitive_state, gamma, config, registered_variables
        ),
        None,
        None,
        dt,
        gamma,
        config,
        params,
        helper_data,
        registered_variables,
    )


def _apply_gravity_source(
    primitive_state: STATE_TYPE,
    gravity_source: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Add the pre-computed self-gravity source to the post-hydro state."""
    conserved_state = (
        conserved_state_from_primitive(
            primitive_state, gamma, config, registered_variables
        )
        + gravity_source
    )
    primitive_state = primitive_state_from_conserved(
        conserved_state, gamma, config, registered_variables
    )
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(
            primitive_state, config, registered_variables, params
        )
    return primitive_state


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
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    axis: int,
) -> STATE_TYPE:
    
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(primitive_state, config, registered_variables, params)

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
        primitive_state = _boundary_handler(primitive_state, config, registered_variables, params)

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
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    if config.dimensionality == 1:
        if config.gravity_config.gravity:
            gravity_source = _gravity_source_presolve(
                primitive_state, dt, gamma, config, params, helper_data, registered_variables
            )

        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
            1,
        )

        if config.gravity_config.gravity:
            primitive_state = _apply_gravity_source(
                primitive_state, gravity_source, gamma, config, params, registered_variables
            )

    elif config.dimensionality == 2:
        if config.gravity_config.gravity:
            gravity_source = _gravity_source_presolve(
                primitive_state, dt, gamma, config, params, helper_data, registered_variables
            )

        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt / 2,
            gamma,
            config,
            params,
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
            params,
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
            params,
            helper_data,
            registered_variables,
            1,
        )

        if config.gravity_config.gravity:
            primitive_state = _apply_gravity_source(
                primitive_state, gravity_source, gamma, config, params, registered_variables
            )

    elif config.dimensionality == 3:
        if config.gravity_config.gravity:
            gravity_source = _gravity_source_presolve(
                primitive_state, dt, gamma, config, params, helper_data, registered_variables
            )

        primitive_state = _evolve_state_along_axis(
            primitive_state,
            config.grid_spacing,
            dt / 2,
            gamma,
            config,
            params,
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
            params,
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
            params,
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
            params,
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
            params,
            helper_data,
            registered_variables,
            1,
        )

        if config.gravity_config.gravity:
            primitive_state = _apply_gravity_source(
                primitive_state, gravity_source, gamma, config, params, registered_variables
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
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(primitive_state, config, registered_variables, params)

    conservative_states = conserved_state_from_primitive(
        primitive_state, gamma, config, registered_variables
    )

    # Pallas-fused per-axis recon+Riemann+divergence: each axis writes
    # ``conservative_states += -(dt/dx) * (F[i+1/2] - F[i-1/2])`` directly
    # into the conservative buffer via ``input_output_aliases``.  No
    # full-state q_L, q_R, fluxes are materialised.  Falls back to the
    # original native chain when ``_fv_pallas_evolve_supported`` says no.
    if _fv_pallas_evolve_supported(primitive_state, config):
        return _evolve_gas_state_unsplit_pallas(
            primitive_state,
            conservative_states,
            dt,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
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
            primitive_state = _boundary_handler(primitive_state, config, registered_variables, params)

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
        primitive_state = _boundary_handler(primitive_state, config, registered_variables, params)

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _evolve_gas_state_unsplit(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:

    if config.gravity_config.gravity:
        gravity_source = _gravity_source_presolve(
            primitive_state, dt, gamma, config, params, helper_data, registered_variables
        )

    if config.time_integrator == RK2_SSP:
        # Generic SSP-RK2 (Heun) over the conserved state.  The stage
        # increment is one forward-Euler hydro step expressed in conserved
        # variables: dt * L(u) = conserved(inner(primitive(u))) - u.
        def rhs(u, dt_step):
            p = primitive_state_from_conserved(
                u, gamma, config, registered_variables
            )
            p_stepped = _evolve_gas_state_unsplit_inner(
                p,
                dt_step,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
            return (
                conserved_state_from_primitive(
                    p_stepped, gamma, config, registered_variables
                )
                - u
            )

        u0 = conserved_state_from_primitive(
            primitive_state, gamma, config, registered_variables
        )
        u_final = rk2_ssp(u0, dt, rhs=rhs)
        primitive_state = primitive_state_from_conserved(
            u_final, gamma, config, registered_variables
        )
    else:
        raise ValueError(
            "Only the RK2 SSP time integrator is currently supported for the unsplit scheme."
        )

    if config.gravity_config.gravity and config.time_integrator:
        primitive_state = _apply_gravity_source(
            primitive_state, gravity_source, gamma, config, params, registered_variables
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
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    if config.mhd:
        if config.dimensionality > 1:

            # WARNING: this relies on the last three state indices being the
            # magnetic-field components, so that stripping them off yields the
            # pure gas sub-state and a matching gas variable registry.
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
                params
            )

            if config.split == UNSPLIT:
                evolved_gas = _evolve_gas_state_unsplit(
                    evolved_gas,
                    dt / 2,
                    gamma,
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
                config,
                params,
                helper_data,
                registered_variables,
            )