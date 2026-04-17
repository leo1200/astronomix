"""
Strong Stability Preserving Runge-Kutta (SSPRK) time integrator.

See _magnetic_update/_constrained_transport.py for more details on the
Constrained Transport (CT) implementation following (Seo & Ryu 2023,
https://arxiv.org/abs/2304.04360).
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Union, Tuple

from astronomix._finite_difference._fluid_equations._enforce_positivity import (
    _enforce_positivity,
)
from astronomix._finite_difference._fluid_equations._equations import conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)

from astronomix._finite_difference._magnetic_update._constrained_transport import (
    constrained_transport_rhs,
    update_cell_center_fields,
)
from astronomix._physics_modules.run_physics_modules import _physics_sources
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

@partial(jax.jit, static_argnames=["registered_variables", "config"], donate_argnames=["conserved_state", "bx_interface", "by_interface", "bz_interface"])
def _ssprk4_with_ct(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Integrates the MHD equations for one time step using a 5-stage, 4th-order
    Strong Stability Preserving Runge-Kutta (SSPRK) method
    with Constrained Transport (CT).
    """

    # for procceses with similar or smaller time scales as the hydrodynamics,
    # they should be included as source terms in the RK stages, otherwise
    # they could be handled outside

    def compute_rhs(current_q, bx, by, bz, k2_coeff):
        """
        Computes the right-hand side (RHS) of the MHD equations for a given stage.
        The `k2_coeff` scales the timestep `dt` for the current RK stage.
        """

        current_q = update_cell_center_fields(
            current_q, bx, by, bz, config, registered_variables
        )

        dt_tilde = k2_coeff * dt

        # in the future we might support
        # different grid spacings in each direction
        dtdx = dt_tilde / grid_spacing
        dtdy = dt_tilde / grid_spacing
        dtdz = dt_tilde / grid_spacing

        # Calculate fluxes based on the state of the current stage
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)

        if config.dimensionality == 1:
            dF_y = 0.0
            dF_z = 0.0

        if config.dimensionality == 2:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            dF_z = 0.0

        if config.dimensionality == 3:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)

        # Calculate RHS for interface magnetic fields using Constrained Transport
        rhs_bx, rhs_by, rhs_bz = constrained_transport_rhs(
            current_q, dF_x, dF_y, dF_z, dtdx, dtdy, dtdz, config, registered_variables
        )

        # Calculate RHS for conserved fluid variables
        if config.dimensionality == 1:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
            )
        elif config.dimensionality == 2:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
                + (dF_y - _shift(dF_y, 1, axis=2))
            )
        elif config.dimensionality == 3:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
                + (dF_y - _shift(dF_y, 1, axis=2))
                + (dF_z - _shift(dF_z, 1, axis=3))
            )

        if config.dimensionality == 1:
            density_fluxes = (dF_x[registered_variables.density_index],)
        elif config.dimensionality == 2:
            density_fluxes = (dF_x[registered_variables.density_index], dF_y[registered_variables.density_index])
        elif config.dimensionality == 3:
            density_fluxes = (dF_x[registered_variables.density_index], dF_y[registered_variables.density_index], dF_z[registered_variables.density_index])


        # Add physics source terms
        rhs_q += _physics_sources(
            current_q,
            density_fluxes,
            rhs_q[registered_variables.density_index], # drho
            dt_tilde,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
        )

        return rhs_q, rhs_bx, rhs_by, rhs_bz

    # define the SSPRK4 coefficients

    k1_1 = 1.0
    k2_1 = 0.39175222700392
    k3_1 = 0.0

    k1_2 = 0.44437049406734
    k2_2 = 0.36841059262959
    k3_2 = 0.55562950593266

    k1_3 = 0.62010185138540
    k2_3 = 0.25189177424738
    k3_3 = 0.37989814861460
    
    k1_4 = 0.17807995410773
    k2_4 = 0.54497475021237
    k3_4 = 0.82192004589227

    k1_5 = -2.081261929715610e-02
    k2_5 = 0.22600748319395
    k3_5 = 5.03580947213895e-01
    k4_5 = 0.51723167208978
    k5_5 = -6.518979800418380e-12

    final_factors = jnp.array([k1_5, 0.0, k4_5, k5_5, k3_5])
    k_rhs_s = jnp.array([k2_1, k2_2, k2_3, k2_4, k2_5])
    k_0_s = jnp.array([k1_1, k1_2, k1_3, k1_4, k1_5])
    k_curr_s = jnp.array([k3_1, k3_2, k3_3, k3_4, k3_5])

    # Store the initial state (t = n)
    q0 = conserved_state
    bx0, by0, bz0 = bx_interface, by_interface, bz_interface

    def ssprk_stage(stage_idx, carry):

        # unpack carry
        current_state, final_state = carry
        q_curr, bx_curr, by_curr, bz_curr = current_state
        q_final, bx_final, by_final, bz_final = final_state

        if config.enforce_positivity:
            q_curr = _enforce_positivity(
                q_curr,
                config,
                gamma,
                params.minimum_density,
                params.minimum_pressure,
                registered_variables,
            )

        k_rhs = k_rhs_s[stage_idx]
        k_0 = k_0_s[stage_idx]
        k_curr = k_curr_s[stage_idx]

        # update the current state
        rhs_q, rhs_bx, rhs_by, rhs_bz = compute_rhs(q_curr, bx_curr, by_curr, bz_curr, k_rhs)
        q_curr = k_0 * q0 + k_curr * q_curr + rhs_q
        bx_curr = k_0 * bx0 + k_curr * bx_curr + rhs_bx
        by_curr = k_0 * by0 + k_curr * by_curr + rhs_by
        bz_curr = k_0 * bz0 + k_curr * bz_curr + rhs_bz

        # update the final state
        final_factor = final_factors[stage_idx + 1]
        q_final += q_curr * final_factor
        bx_final += bx_curr * final_factor
        by_final += by_curr * final_factor
        bz_final += bz_curr * final_factor

        return (
            (q_curr, bx_curr, by_curr, bz_curr), 
            (q_final, bx_final, by_final, bz_final)
        )

    (
        (q4, bx4, by4, bz4),
        (q_final, bx_final, by_final, bz_final)
    ) = jax.lax.fori_loop(0, 4, ssprk_stage, 
        (
            (q0, bx0, by0, bz0), 
            (final_factors[0] * q0, final_factors[0] * bx0, final_factors[0] * by0, final_factors[0] * bz0)
        )
    )

    # Final Stage (Stage 5)
    rhs_q4, rhs_bx4, rhs_by4, rhs_bz4 = compute_rhs(q4, bx4, by4, bz4, k2_5)
    q_final = q_final + rhs_q4
    bx_final = bx_final + rhs_bx4
    by_final = by_final + rhs_by4
    bz_final = bz_final + rhs_bz4

    # Update the cell-centered magnetic fields in the conserved state array
    # from the final interface magnetic fields.
    q_final = update_cell_center_fields(
        q_final, bx_final, by_final, bz_final, config, registered_variables
    )

    if config.enforce_positivity:
        q_final = _enforce_positivity(
            q_final,
            config,
            gamma,
            params.minimum_density,
            params.minimum_pressure,
            registered_variables,
        )
    
    return q_final, bx_final, by_final, bz_final


@partial(jax.jit, static_argnames=["registered_variables", "config"], donate_argnames=["conserved_state"])
def _ssprk4_hydro(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params, # Assuming SimulationParams type
    helper_data, # Assuming HelperData type
    config, # Assuming SimulationConfig type
    registered_variables: RegisteredVariables,
):
    """
    Integrates the Euler (hydrodynamics) equations for one time step using a 
    5-stage, 4th-order Strong Stability Preserving Runge-Kutta (SSPRK) method.
    """

    # for procceses with similar or smaller time scales as the hydrodynamics,
    # they should be included as source terms in the RK stages, otherwise
    # they could be handled outside

    def compute_rhs(current_q, k2_coeff):
        """
        Computes the right-hand side (RHS) of the hydro equations for a given stage.
        The `k2_coeff` scales the timestep `dt` for the current RK stage.
        """

        dt_tilde = k2_coeff * dt

        # in the future we might support
        # different grid spacings in each direction
        dtdx = dt_tilde / grid_spacing
        dtdy = dt_tilde / grid_spacing
        dtdz = dt_tilde / grid_spacing

        # Calculate fluxes based on the state of the current stage
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)

        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)

        if config.dimensionality == 3:
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)

        # Calculate RHS for conserved fluid variables
        if config.dimensionality == 1:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
            )
        elif config.dimensionality == 2:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
                + (dF_y - _shift(dF_y, 1, axis=2))
            )
        elif config.dimensionality == 3:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
                + (dF_y - _shift(dF_y, 1, axis=2))
                + (dF_z - _shift(dF_z, 1, axis=3))
            )

        if config.dimensionality == 1:
            density_fluxes = (dF_x[registered_variables.density_index],)
        elif config.dimensionality == 2:
            density_fluxes = (dF_x[registered_variables.density_index], dF_y[registered_variables.density_index])
        elif config.dimensionality == 3:
            density_fluxes = (dF_x[registered_variables.density_index], dF_y[registered_variables.density_index], dF_z[registered_variables.density_index])

        # Add physics source terms
        rhs_q += _physics_sources(
            current_q,
            density_fluxes,
            rhs_q[registered_variables.density_index], # drho
            dt_tilde,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
        )

        return rhs_q

    # define the SSPRK4 coefficients

    k1_1 = 1.0
    k2_1 = 0.39175222700392
    k3_1 = 0.0

    k1_2 = 0.44437049406734
    k2_2 = 0.36841059262959
    k3_2 = 0.55562950593266

    k1_3 = 0.62010185138540
    k2_3 = 0.25189177424738
    k3_3 = 0.37989814861460
    
    k1_4 = 0.17807995410773
    k2_4 = 0.54497475021237
    k3_4 = 0.82192004589227

    k1_5 = -2.081261929715610e-02
    k2_5 = 0.22600748319395
    k3_5 = 5.03580947213895e-01
    k4_5 = 0.51723167208978
    k5_5 = -6.518979800418380e-12

    final_factors = jnp.array([k1_5, 0.0, k4_5, k5_5, k3_5])
    k_rhs_s = jnp.array([k2_1, k2_2, k2_3, k2_4, k2_5])
    k_0_s = jnp.array([k1_1, k1_2, k1_3, k1_4, k1_5])
    k_curr_s = jnp.array([k3_1, k3_2, k3_3, k3_4, k3_5])

    # Store the initial state (t = n)
    q0 = conserved_state

    def ssprk_stage(stage_idx, carry):

        # unpack carry
        q_curr, q_final = carry

        if config.enforce_positivity:
            q_curr = _enforce_positivity(
                q_curr,
                config,
                gamma,
                params.minimum_density,
                params.minimum_pressure,
                registered_variables,
            )

        k_rhs = k_rhs_s[stage_idx]
        k_0 = k_0_s[stage_idx]
        k_curr = k_curr_s[stage_idx]

        # update the current state
        rhs_q = compute_rhs(q_curr, k_rhs)
        q_curr = k_0 * q0 + k_curr * q_curr + rhs_q

        # update the final state
        final_factor = final_factors[stage_idx + 1]
        q_final += q_curr * final_factor

        return (q_curr, q_final)

    q4, q_final = jax.lax.fori_loop(
        0, 4, ssprk_stage, (q0, final_factors[0] * q0)
    )

    # Final Stage (Stage 5)
    rhs_q4 = compute_rhs(q4, k2_5)
    q_final = q_final + rhs_q4

    if config.enforce_positivity:
        q_final = _enforce_positivity(
            q_final,
            config,
            gamma,
            params.minimum_density,
            params.minimum_pressure,
            registered_variables,
        )
    
    return q_final