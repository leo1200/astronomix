"""
Frame tracking for the cooling-layer test.

Continuously re-centres the simulation frame on the peak-cooling interface by
boosting the vertical (z) velocity. A proportional-derivative controller damps
the interface velocity and restores its position to the centre of the z-domain,
keeping the mixing layer inside the box. The scheme is preliminary and currently
hard-codes the parameters of a specific 3D test.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._geometry.boundaries import _boundary_handler


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _frame_tracking(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data_pad: HelperData,
) -> STATE_TYPE:
    """
    Re-centre the frame on the cooling interface by boosting the z-velocity.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        helper_data_pad: The padded helper data.

    Returns:
        The primitive state with the frame-tracking velocity boost applied.
    """

    # WARNING: these reference values are hard-coded for one specific cooling
    # test and are not yet derived from the configuration.
    rho_hot = 1.0
    P0 = 1.0
    L_box = config.box_size.x
    box_size_z = config.box_size.z

    # cell_centers_z is a 1D array along the z axis; jnp
    # broadcasts it against the 3D pk_mask / Z-reduction below.
    Z = helper_data_pad.cell_centers_z
    density_contrast = params.cooling_params.cooling_curve_params.density_contrast
    mach_number = params.cooling_params.cooling_curve_params.mach_number
    rho_cold = density_contrast * rho_hot
    T_hot = P0 / rho_hot
    T_cold = P0 / rho_cold
    c_hot = (params.gamma * P0 / rho_hot) ** 0.5
    v_rel = mach_number * c_hot
    # Geometric-mean temperature where the cooling curve peaks.
    T_pk = (T_cold**2 * T_hot) ** (1 / 3)

    rho = primitive_state[registered_variables.density_index]
    v_z = primitive_state[registered_variables.velocity_index.z]
    P = primitive_state[registered_variables.pressure_index]
    T = P / rho

    # -------------------------------------------------------------
    # ============== ↓ Peak-cooling measurements ↓ ===============
    # -------------------------------------------------------------

    # Select the cells within a narrow temperature band (a factor f_w either
    # side) around the peak-cooling temperature, and average the z-velocity and
    # z-position over them to locate and track the cooling layer.
    f_w = 10.0**0.1
    T_ratio = T / T_pk
    pk_mask = (T_ratio > 1.0 / f_w) & (T_ratio < f_w)
    n_pk_safe = jnp.maximum(jnp.sum(pk_mask), 1)
    v_z_pk = jnp.sum(jnp.where(pk_mask, v_z, 0.0)) / n_pk_safe
    z_layer = jnp.sum(jnp.where(pk_mask, Z, 0.0)) / n_pk_safe

    # -------------------------------------------------------------
    # ====== ↓ PD control: velocity damping + restoring ↓ ========
    # -------------------------------------------------------------

    # Target the centre of the z-domain (where the interface was initialised).
    # k_p sets the correction timescale: k_p = v_rel / L_box closes a
    # displacement in roughly one shear time, and pairing it with the
    # instantaneous velocity term below gives critical damping.
    z_target = box_size_z / 2.0
    dz = z_layer - z_target
    k_p = v_rel / L_box

    boost = -v_z_pk - k_p * dz

    # Cap the boost to keep the frame shift deeply subsonic and avoid shocks.
    # The limit v_rel / 10 corresponds to Mach ~0.05 in the hot gas. This is
    # intentionally looser than the paper's v_rel / 100, because that tighter
    # cap was the mechanism that was failing.
    boost = jnp.clip(boost, -v_rel / 10.0, v_rel / 10.0)

    primitive_state = primitive_state.at[
        registered_variables.velocity_index.z
    ].add(boost)

    # Re-apply the boundary conditions for safety after modifying the interior.
    primitive_state = _boundary_handler(
        primitive_state,
        config,
        registered_variables,
        params,
    )
    return primitive_state
