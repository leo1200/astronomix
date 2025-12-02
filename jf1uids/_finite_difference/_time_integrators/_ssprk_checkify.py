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

from functools import partial
import jax
import jax.numpy as jnp
import jax.experimental.checkify as jax_checkify
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


# @partial(jax.jit, static_argnames=["registered_variables", "config"])
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
    SSPRK4 with Constrained Transport instrumented with checkify checks to
    detect forward-pass issues that can produce NaN/Inf gradients.
    """

    # -------------------------
    # Basic input sanity checks
    # -------------------------
    jax_checkify.check(jnp.all(jnp.isfinite(conserved_state)), "Input conserved_state contains NaN/Inf")
    jax_checkify.check(jnp.max(jnp.abs(conserved_state)) < 1e10, "Input conserved_state magnitude extremely large")

    jax_checkify.check(jnp.isfinite(dt), "dt is NaN/Inf")
    jax_checkify.check(dt > 0.0, "dt must be positive")
    jax_checkify.check(dt < 1e4, "dt suspiciously large")

    jax_checkify.check(jnp.isfinite(grid_spacing), "grid_spacing is NaN/Inf")
    jax_checkify.check(jnp.all(grid_spacing > 0), "grid_spacing must be positive")

    jax_checkify.check(jnp.all(jnp.isfinite(bx_interface)), "bx_interface contains NaN/Inf")
    jax_checkify.check(jnp.all(jnp.isfinite(by_interface)), "by_interface contains NaN/Inf")
    jax_checkify.check(jnp.all(jnp.isfinite(bz_interface)), "bz_interface contains NaN/Inf")

    # optional quick physics check: ensure densities in conserved_state not too small
    # (assumes density is one of the conserved variables; adapt idx if needed)
    try:
        density_idx = registered_variables.density_index_conserved  # if your registry has this
    except Exception:
        # fallback: assume conserved_state[0] is density in many schemes
        density_idx = 0

    rho = conserved_state[density_idx]
    jax_checkify.check(jnp.all(jnp.isfinite(rho)), "Input density contains NaN/Inf")
    jax_checkify.check(jnp.all(rho > params.minimum_density * 0.01), "Input density extremely small (can cause huge gradients)")

    # small helper for bounding checks
    def _check_array(name, arr, mag_thresh=1e8):
        jax_checkify.check(jnp.all(jnp.isfinite(arr)), f"{name} contains NaN/Inf")
        jax_checkify.check(jnp.max(jnp.abs(arr)) < mag_thresh, f"{name} magnitude exceeds {mag_thresh} → gradient risk")

    # -------------------------
    # RHS builder with checks
    # -------------------------
    def compute_rhs(current_q, bx, by, bz, k2_coeff):
        current_q = update_cell_center_fields(
            current_q, bx, by, bz, registered_variables
        )

        _check_array("current_q (after update_cell_center_fields)", current_q, mag_thresh=1e9)

        # stage-scaled dt/spacing
        dtdx = k2_coeff * dt / grid_spacing
        dtdy = dtdx
        dtdz = dtdx

        jax_checkify.check(jnp.isfinite(dtdx), "dtdx is NaN/Inf")
        jax_checkify.check(jnp.isfinite(dtdy), "dtdy is NaN/Inf")
        jax_checkify.check(jnp.isfinite(dtdz), "dtdz is NaN/Inf")

        # Compute WENO fluxes (these are common culprits)
        dF_x = _weno_flux_x(current_q, params.minimum_density, params.minimum_pressure, gamma, registered_variables)
        dF_y = _weno_flux_y(current_q, params.minimum_density, params.minimum_pressure, gamma, registered_variables)
        dF_z = _weno_flux_z(current_q, params.minimum_density, params.minimum_pressure, gamma, registered_variables)

        _check_array("dF_x (WENO flux)", dF_x, mag_thresh=1e7)
        _check_array("dF_y (WENO flux)", dF_y, mag_thresh=1e7)
        _check_array("dF_z (WENO flux)", dF_z, mag_thresh=1e7)

        # Constrained transport RHS for interface magnetic fields
        rhs_bx, rhs_by, rhs_bz = constrained_transport_rhs(
            current_q, dF_x, dF_y, dF_z, dtdx, dtdy, dtdz, registered_variables
        )

        _check_array("rhs_bx (CT)", rhs_bx, mag_thresh=1e7)
        _check_array("rhs_by (CT)", rhs_by, mag_thresh=1e7)
        _check_array("rhs_bz (CT)", rhs_bz, mag_thresh=1e7)

        # Compute fluid RHS (finite-differencing of flux divergences)
        rhs_q = -dtdx * (
            (dF_x - jnp.roll(dF_x, 1, axis=1))
            + (dF_y - jnp.roll(dF_y, 1, axis=2))
            + (dF_z - jnp.roll(dF_z, 1, axis=3))
        )

        _check_array("rhs_q (fluid RHS)", rhs_q, mag_thresh=1e8)

        if config.wind_config.stellar_wind:
            wind_src = _wind_ei3D_source(
                params.wind_params,
                current_q,
                config,
                helper_data,
                config.wind_config.num_injection_cells,
                registered_variables,
            )
            _check_array("wind_src", wind_src, mag_thresh=1e6)

            rhs_q = rhs_q + (wind_src * k2_coeff * dt)
            _check_array("rhs_q (after wind)", rhs_q, mag_thresh=1e8)

        return rhs_q, rhs_bx, rhs_by, rhs_bz

    # -------------------------
    # Start RK stages with checks
    # -------------------------
    q0 = conserved_state
    bx0, by0, bz0 = bx_interface, by_interface, bz_interface

    _check_array("q0", q0)
    _check_array("bx0", bx0)
    _check_array("by0", by0)
    _check_array("bz0", bz0)

    # Stage 1
    k1_1 = 1.0
    k2_1 = 0.39175222700392

    rhs_q0, rhs_bx0, rhs_by0, rhs_bz0 = compute_rhs(q0, bx0, by0, bz0, k2_1)

    q1 = q0 + rhs_q0
    bx1, by1, bz1 = bx0 + rhs_bx0, by0 + rhs_by0, bz0 + rhs_bz0

    _check_array("q1", q1)
    _check_array("bx1", bx1)
    _check_array("by1", by1)
    _check_array("bz1", bz1)

    # Stage 2
    k1_2 = 0.44437049406734
    k2_2 = 0.36841059262959
    k3_2 = 0.55562950593266

    rhs_q1, rhs_bx1, rhs_by1, rhs_bz1 = compute_rhs(q1, bx1, by1, bz1, k2_2)

    q2 = k1_2 * q0 + k3_2 * q1 + rhs_q1
    bx2 = k1_2 * bx0 + k3_2 * bx1 + rhs_bx1
    by2 = k1_2 * by0 + k3_2 * by1 + rhs_by1
    bz2 = k1_2 * bz0 + k3_2 * bz1 + rhs_bz1

    _check_array("q2", q2)
    _check_array("bx2", bx2)
    _check_array("by2", by2)
    _check_array("bz2", bz2)

    # Stage 3
    k1_3 = 0.62010185138540
    k2_3 = 0.25189177424738
    k3_3 = 0.37989814861460

    rhs_q2, rhs_bx2, rhs_by2, rhs_bz2 = compute_rhs(q2, bx2, by2, bz2, k2_3)

    q3 = k1_3 * q0 + k3_3 * q2 + rhs_q2
    bx3 = k1_3 * bx0 + k3_3 * bx2 + rhs_bx2
    by3 = k1_3 * by0 + k3_3 * by2 + rhs_by2
    bz3 = k1_3 * bz0 + k3_3 * bz2 + rhs_bz2

    _check_array("q3", q3)
    _check_array("bx3", bx3)
    _check_array("by3", by3)
    _check_array("bz3", bz3)

    # Stage 4
    k1_4 = 0.17807995410773
    k2_4 = 0.54497475021237
    k3_4 = 0.82192004589227

    rhs_q3, rhs_bx3, rhs_by3, rhs_bz3 = compute_rhs(q3, bx3, by3, bz3, k2_4)

    q4 = k1_4 * q0 + k3_4 * q3 + rhs_q3
    bx4 = k1_4 * bx0 + k3_4 * bx3 + rhs_bx3
    by4 = k1_4 * by0 + k3_4 * by3 + rhs_by3
    bz4 = k1_4 * bz0 + k3_4 * bz3 + rhs_bz3

    _check_array("q4", q4)
    _check_array("bx4", bx4)
    _check_array("by4", by4)
    _check_array("bz4", bz4)

    # Stage 5 (Final Stage)
    k1_5 = -2.081261929715610e-02
    k2_5 = 0.22600748319395
    k3_5 = 5.03580947213895e-01
    k4_5 = 0.51723167208978
    k5_5 = -6.518979800418380e-12

    rhs_q4, rhs_bx4, rhs_by4, rhs_bz4 = compute_rhs(q4, bx4, by4, bz4, k2_5)

    q5 = (k1_5 * q0) + (k4_5 * q2) + (k5_5 * q3) + (k3_5 * q4) + rhs_q4
    bx_final = (k1_5 * bx0) + (k4_5 * bx2) + (k5_5 * bx3) + (k3_5 * bx4) + rhs_bx4
    by_final = (k1_5 * by0) + (k4_5 * by2) + (k5_5 * by3) + (k3_5 * by4) + rhs_by4
    bz_final = (k1_5 * bz0) + (k4_5 * bz2) + (k5_5 * bz3) + (k3_5 * bz4) + rhs_bz4

    _check_array("q5", q5)
    _check_array("bx_final", bx_final)
    _check_array("by_final", by_final)
    _check_array("bz_final", bz_final)

    # Update the cell-centered magnetic fields in the conserved state array
    q_final = update_cell_center_fields(
        q5, bx_final, by_final, bz_final, registered_variables
    )

    _check_array("q_final (after update_cell_center_fields)", q_final, mag_thresh=1e9)

    # Final sanity: ensure densities/pressures remain > minima (if accessible)
    try:
        # If you have conserved->primitive utilities here, you can check physical pressure
        # primitive = conserved_to_primitive(q_final, gamma, registered_variables)
        # jax_checkify.check(jnp.all(primitive[pressure_idx] > params.minimum_pressure * 0.01), "Pressure too small after RK")
        pass
    except Exception:
        pass

    return q_final, bx_final, by_final, bz_final
