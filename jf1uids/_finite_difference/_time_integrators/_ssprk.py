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

from jf1uids._finite_difference._fluid_equations._enforce_positivity import (
    _enforce_positivity,
)
from jf1uids._finite_difference._interface_fluxes._weno import (
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)

from jf1uids._finite_difference._magnetic_update._constrained_transport import (
    constrained_transport_rhs,
    update_cell_center_fields,
)
from jf1uids._finite_difference._maths._differencing import finite_difference_int6
from jf1uids._physics_modules._stellar_wind.stellar_wind import _wind_ei3D_source
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids.variable_registry.registered_variables import RegisteredVariables


@partial(jax.jit, static_argnames=["registered_variables", "config"])
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

    def compute_rhs(current_q, bx, by, bz, k2_coeff):
        """
        Computes the right-hand side (RHS) of the MHD equations for a given stage.
        The `k2_coeff` scales the timestep `dt` for the current RK stage.
        """

        current_q = update_cell_center_fields(
            current_q, bx, by, bz, registered_variables
        )

        # in the future we might support
        # different grid spacings in each direction
        dtdx = k2_coeff * dt / grid_spacing
        dtdy = k2_coeff * dt / grid_spacing
        dtdz = k2_coeff * dt / grid_spacing

        # Calculate fluxes based on the state of the current stage
        dF_x = _weno_flux_x(
            current_q,
            params.minimum_density,
            params.minimum_pressure,
            gamma,
            registered_variables,
        )
        dF_y = _weno_flux_y(
            current_q,
            params.minimum_density,
            params.minimum_pressure,
            gamma,
            registered_variables,
        )
        dF_z = _weno_flux_z(
            current_q,
            params.minimum_density,
            params.minimum_pressure,
            gamma,
            registered_variables,
        )

        # Calculate RHS for interface magnetic fields using Constrained Transport
        rhs_bx, rhs_by, rhs_bz = constrained_transport_rhs(
            current_q, dF_x, dF_y, dF_z, dtdx, dtdy, dtdz, registered_variables
        )

        # Calculate RHS for conserved fluid variables
        rhs_q = -dtdx * (
            (dF_x - jnp.roll(dF_x, 1, axis=1))
            + (dF_y - jnp.roll(dF_y, 1, axis=2))
            + (dF_z - jnp.roll(dF_z, 1, axis=3))
        )

        if config.wind_config.stellar_wind:
            rhs_q += (
                _wind_ei3D_source(
                    params.wind_params,
                    current_q,
                    config,
                    helper_data,
                    config.wind_config.num_injection_cells,
                    registered_variables,
                )
                * k2_coeff
                * dt
            )

        return rhs_q, rhs_bx, rhs_by, rhs_bz

    # Store the initial state (t = n)
    q0 = conserved_state
    bx0, by0, bz0 = bx_interface, by_interface, bz_interface

    # Stage 1
    k1_1 = 1.0
    k2_1 = 0.39175222700392
    # k3_1 = 0.0, so it's omitted

    rhs_q0, rhs_bx0, rhs_by0, rhs_bz0 = compute_rhs(q0, bx0, by0, bz0, k2_1)

    q1 = q0 + rhs_q0
    bx1, by1, bz1 = bx0 + rhs_bx0, by0 + rhs_by0, bz0 + rhs_bz0

    # Stage 2
    k1_2 = 0.44437049406734
    k2_2 = 0.36841059262959
    k3_2 = 0.55562950593266

    rhs_q1, rhs_bx1, rhs_by1, rhs_bz1 = compute_rhs(q1, bx1, by1, bz1, k2_2)

    q2 = k1_2 * q0 + k3_2 * q1 + rhs_q1
    bx2 = k1_2 * bx0 + k3_2 * bx1 + rhs_bx1
    by2 = k1_2 * by0 + k3_2 * by1 + rhs_by1
    bz2 = k1_2 * bz0 + k3_2 * bz1 + rhs_bz1

    # Stage 3
    k1_3 = 0.62010185138540
    k2_3 = 0.25189177424738
    k3_3 = 0.37989814861460

    rhs_q2, rhs_bx2, rhs_by2, rhs_bz2 = compute_rhs(q2, bx2, by2, bz2, k2_3)

    q3 = k1_3 * q0 + k3_3 * q2 + rhs_q2
    bx3 = k1_3 * bx0 + k3_3 * bx2 + rhs_bx2
    by3 = k1_3 * by0 + k3_3 * by2 + rhs_by2
    bz3 = k1_3 * bz0 + k3_3 * bz2 + rhs_bz2

    # Stage 4
    k1_4 = 0.17807995410773
    k2_4 = 0.54497475021237
    k3_4 = 0.82192004589227

    rhs_q3, rhs_bx3, rhs_by3, rhs_bz3 = compute_rhs(q3, bx3, by3, bz3, k2_4)

    q4 = k1_4 * q0 + k3_4 * q3 + rhs_q3
    bx4 = k1_4 * bx0 + k3_4 * bx3 + rhs_bx3
    by4 = k1_4 * by0 + k3_4 * by3 + rhs_by3
    bz4 = k1_4 * bz0 + k3_4 * bz3 + rhs_bz3

    # Stage 5 (Final Stage)
    k1_5 = -2.081261929715610e-02
    k2_5 = 0.22600748319395
    k3_5 = 5.03580947213895e-01  # This corresponds to k3 in the Fortran code
    k4_5 = 0.51723167208978  # This corresponds to k4
    k5_5 = -6.518979800418380e-12  # This corresponds to k5

    rhs_q4, rhs_bx4, rhs_by4, rhs_bz4 = compute_rhs(q4, bx4, by4, bz4, k2_5)

    q5 = (k1_5 * q0) + (k4_5 * q2) + (k5_5 * q3) + (k3_5 * q4) + rhs_q4
    bx_final = (k1_5 * bx0) + (k4_5 * bx2) + (k5_5 * bx3) + (k3_5 * bx4) + rhs_bx4
    by_final = (k1_5 * by0) + (k4_5 * by2) + (k5_5 * by3) + (k3_5 * by4) + rhs_by4
    bz_final = (k1_5 * bz0) + (k4_5 * bz2) + (k5_5 * bz3) + (k3_5 * bz4) + rhs_bz4

    # Update the cell-centered magnetic fields in the conserved state array
    # from the final interface magnetic fields.
    q_final = update_cell_center_fields(
        q5, bx_final, by_final, bz_final, registered_variables
    )

    return q_final, bx_final, by_final, bz_final
