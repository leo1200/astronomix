if __name__ == "__main__":
    from autocvd import autocvd

    autocvd(num_gpus=1)
    from astronomix import time_integration


import jax.numpy as jnp
from astronomix.data_classes.simulation_helper_data import HelperData
from typing import Tuple

from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    initialize_interface_fields,
)

from astronomix.data_classes.simulation_snapshot_data import SnapshotData
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    FORWARDS,
    OPEN_BOUNDARY,
    STATE_TYPE,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix._finite_difference._maths._interpolate import interp_face_to_center
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    XAXIS,
    YAXIS,
    ZAXIS,
)

from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    FINITE_DIFFERENCE,
)
from arena.arena_tests.solver_in_the_loop.plot_states_comparison import (
    plot_states,
    plot_and_animate_states,
    plot_and_animate_states_with_diagonal,
)
import logging

logger = logging.getLogger(__name__)


def _periodic_diff(f, axis: int, grid_spacing: float, order: int = 2):
    """Compute the periodic finite difference derivative along a given axis.

    Args:
        f: The field to differentiate.
        axis: The axis along which to differentiate.
        grid_spacing: The width of the cells.
        order: The order of accuracy (2, 4, or 6).

    Returns:
        The derivative of the field along the given axis.
    """
    if order == 2:
        return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (
            2 * grid_spacing
        )
    elif order == 4:
        return (
            -jnp.roll(f, -2, axis=axis)
            + 8 * jnp.roll(f, -1, axis=axis)
            - 8 * jnp.roll(f, 1, axis=axis)
            + jnp.roll(f, 2, axis=axis)
        ) / (12 * grid_spacing)
    elif order == 6:
        return (
            jnp.roll(f, -3, axis=axis)
            - 9 * jnp.roll(f, -2, axis=axis)
            + 45 * jnp.roll(f, -1, axis=axis)
            - 45 * jnp.roll(f, 1, axis=axis)
            + 9 * jnp.roll(f, 2, axis=axis)
            - jnp.roll(f, 3, axis=axis)
        ) / (60 * grid_spacing)
    else:
        raise ValueError(f"Unsupported order {order}, must be 2, 4, or 6.")


def curl3D(field, grid_spacing: float, order: int = 2):
    """Calculate the curl of a 3d field on a 3d grid using periodic boundaries.

    Args:
        field: The field to calculate the curl of.
        grid_spacing: The width of the cells.
        order: The order of accuracy for finite differences (2, 4, or 6).

    Returns:
        The curl of the field.
    """
    # curl_x = dF_z/dy - dF_y/dz
    curl_x = _periodic_diff(field[2], 1, grid_spacing, order) - _periodic_diff(
        field[1], 2, grid_spacing, order
    )
    # curl_y = dF_x/dz - dF_z/dx
    curl_y = _periodic_diff(field[0], 2, grid_spacing, order) - _periodic_diff(
        field[2], 0, grid_spacing, order
    )
    # curl_z = dF_y/dx - dF_x/dy
    curl_z = _periodic_diff(field[1], 0, grid_spacing, order) - _periodic_diff(
        field[0], 1, grid_spacing, order
    )

    return jnp.stack([curl_x, curl_y, curl_z])


def initial_state_mhd_jet(
    config: SimulationConfig,
    params: SimulationParams,
) -> Tuple[
    STATE_TYPE, SimulationConfig, SimulationParams, HelperData, RegisteredVariables
]:
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    r = helper_data.r

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r)

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    # create x and y for 0,0 to be in the center
    x = jnp.linspace(
        -config.box_size / 2,
        config.box_size / 2,
        # we do num_cells cause we are using periodic boundary the last interface is the same as the last one
        config.num_cells,
        endpoint=True,
    )
    y = jnp.linspace(
        -config.box_size / 2,
        config.box_size / 2,
        config.num_cells,
        endpoint=True,
    )
    z = jnp.linspace(
        -config.box_size / 2,
        config.box_size / 2,
        config.num_cells,
        endpoint=True,
    )
    interfaces = jnp.array(jnp.meshgrid(x, y, z))
    r_interfaces = jnp.linalg.norm(interfaces, axis=0)

    A_0 = 20
    alpha = 1.0
    r_cut = 0.5 * config.box_size
    A_x = alpha * -jnp.exp(-(r_interfaces**2)) * interfaces[0]
    A_y = alpha * jnp.exp(-(r_interfaces**2)) * interfaces[1]
    A_z = alpha * 0.5 * A_0 * jnp.exp(-(r_interfaces**2))
    A_x = jnp.where(r_interfaces <= r_cut, A_x, 0.0)
    A_y = jnp.where(r_interfaces <= r_cut, A_y, 0.0)
    A_z = jnp.where(r_interfaces <= r_cut, A_z, 0.0)

    magnetic_potential = jnp.stack([A_x, A_y, A_z])
    B_interface = curl3D(
        field=magnetic_potential, grid_spacing=config.grid_spacing, order=6
    )

    Bx_center = interp_face_to_center(B_interface[0], XAXIS)
    By_center = interp_face_to_center(B_interface[1], YAXIS)
    Bz_center = interp_face_to_center(B_interface[2], ZAXIS)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=V_x,
        velocity_y=V_y,
        velocity_z=V_z,
        magnetic_field_x=Bx_center,
        magnetic_field_y=By_center,
        magnetic_field_z=Bz_center,
        interface_magnetic_field_x=B_interface[0],
        interface_magnetic_field_y=B_interface[1],
        interface_magnetic_field_z=B_interface[2],
        gas_pressure=P,
    )

    config = finalize_config(config, initial_state.shape)
    return (initial_state, config, params, helper_data, registered_variables)


if __name__ == "__main__":
    num_cells = 128
    t_end = 0.05
    c_cfl = 0.4
    snapshot_timepoints = jnp.linspace(0.0, t_end, 30, True)
    simulation_params = SimulationParams(
        C_cfl=c_cfl, t_end=t_end, snapshot_timepoints=snapshot_timepoints
    )

    simulation_config = SimulationConfig(
        num_cells=num_cells,
        box_size=24.0,
        solver_mode=FINITE_DIFFERENCE,
        differentiation_mode=FORWARDS,
        mhd=True,
        dimensionality=3,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ),
        use_specific_snapshot_timepoints=True,
        return_snapshots=True,
        num_snapshots=len(snapshot_timepoints),
        progress_bar=False,
        runtime_debugging=False,
        limiter=4,
    )
    #
    # simulation_config = SimulationConfig(
    #     num_cells=num_cells,
    #     box_size=24.0,
    #     solver_mode=FINITE_VOLUME,
    #     differentiation_mode=FORWARDS,
    #     mhd=True,
    #     dimensionality=3,
    #     boundary_settings=BoundarySettings(
    #         BoundarySettings1D(
    #             left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
    #         ),
    #         BoundarySettings1D(
    #             left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
    #         ),
    #         BoundarySettings1D(
    #             left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
    #         ),
    #     ),
    #     use_specific_snapshot_timepoints=True,
    #     return_snapshots=True,
    #     num_snapshots=len(snapshot_timepoints),
    #     progress_bar=False,
    #     runtime_debugging=False,
    #     limiter=4,
    # )
    #
    initial_state, config, params, helper_data, registered_variables = (
        initial_state_mhd_jet(
            config=simulation_config,
            params=simulation_params,
        )
    )

    plot_states(
        states_list=[initial_state],
        z_levels=[num_cells // 2],
        model_name="test_jet",
    )

    snapshot_data = time_integration(
        primitive_state=initial_state,
        config=config,
        params=params,
        registered_variables=registered_variables,
    )
    assert isinstance(snapshot_data, SnapshotData)

    last_true_state = -1
    for state in snapshot_data.states:
        if jnp.any(state != 0.0):
            last_true_state += 1
    plot_states(
        states_list=[initial_state, snapshot_data.states[last_true_state]],
        z_levels=[num_cells // 2, num_cells // 2],
        model_name="test_jet",
        fig_name="initial_final_state",
    )
    plot_and_animate_states_with_diagonal(
        states=snapshot_data.states,
        timepoints=snapshot_data.time_points,
        z_level=num_cells // 2,
        slice_axis="x",
        save_path="results/mhd_jet",
    )
