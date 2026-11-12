"""
Finite-volume state evolution.

Drives one finite-volume step: the dimensionally split (Strang) gas update,
the operator-split self-gravity source, and the top-level
:func:`_evolve_state_fv` that dispatches the time integrators (the unsplit
schemes ``vl2_step`` and ``rk2_ssp_step`` of ``_integrators.py`` and the
split scheme).
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
    DONOR_CELL,
    GHOST_CELLS,
    RK2_SSP,
    SPHERICAL,
    STATE_TYPE,
    VL2,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_volume._riemann_solver._riemann_solver import _riemann_solver
from astronomix._modules._time_integrator_sources import _time_integrator_sources
from astronomix._stencil_operations._stencil_operations import _stencil_add
from astronomix._geometry.geometric_terms import _pressure_nozzling_source
from astronomix._finite_volume._state_evolution.reconstruction import (
    _reconstruct_at_interface_split,
)
from astronomix._finite_volume._integrators import rk2_ssp_step, vl2_step
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

    if config.solver_config.reconstruction == DONOR_CELL:
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
    # The AthenaPK-equivalent VL2 scheme (hydro, GLM-MHD and MHD with
    # constrained transport) replaces the whole finite-volume step with its own
    # Cartesian update, including Athena++'s self-gravity coupling. It has no
    # geometric source terms (applied by the split scheme below);
    # finalize_config rejects that combination.
    if config.solver_config.time_integrator == VL2:
        return vl2_step(
            primitive_state,
            dt,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
        )

    # The split and unsplit schemes are hydrodynamic: finite-volume MHD is the
    # VL2 scheme above (finalize_config rejects other combinations).
    if config.solver_config.time_integrator == RK2_SSP:
        # Gravity couples as in the split scheme: the source is built from
        # the state before the step and added after it.
        if config.gravity_config.gravity:
            gravity_source = _gravity_source_presolve(
                primitive_state,
                dt,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
        primitive_state = rk2_ssp_step(
            primitive_state,
            dt,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
        )
        if config.gravity_config.gravity:
            primitive_state = _apply_gravity_source(
                primitive_state,
                gravity_source,
                gamma,
                config,
                params,
                registered_variables,
            )
        return primitive_state
    return _evolve_gas_state_split(
        primitive_state,
        dt,
        gamma,
        config,
        params,
        helper_data,
        registered_variables,
    )
