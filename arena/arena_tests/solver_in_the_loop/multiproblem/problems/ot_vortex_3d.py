"""
Comes from https://www.sciencedirect.com/science/article/abs/pii/S0045793018300410
"""

if __name__ == "__main__":
    from autocvd import autocvd

    autocvd(num_gpus=1)
    from astronomix import time_integration

# import jax

# jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from matplotlib import figure
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

from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    FINITE_DIFFERENCE,
)
from arena.arena_tests.solver_in_the_loop.plot_states_comparison import (
    plot_states,
    plot_and_animate_states_with_diagonal,
    animate_several_sims_states_with_diagonal,
)
import logging
from arena.arena_tests.solver_in_the_loop.multiproblem.problems.baseproblem import (
    BaseProblem,
)
from jaxtyping import Array

logger = logging.getLogger(__name__)


class OtVortex(BaseProblem):
    def __init__(
        self, vortex_axis: str = "z", parity: bool = False, epsilon_p: float = 0.2
    ) -> None:
        self.vortex_axis = vortex_axis
        self.parity = parity
        self.epsilon_p = epsilon_p

    @property
    def name(self) -> str:
        return "ot_vortex"

    @property
    def t_end(self) -> float:
        return jnp.pi

    @property
    def dimensionalty(self) -> float:
        return 3

    def get_hyperparams(self) -> dict:
        return {
            "problem_name": self.name,
            "params": {
                "vortex_axis": self.vortex_axis,
                "parity": self.parity,
                "epsilon_p": self.epsilon_p,
            },
        }

    def generate_initial_state(
        self, config: SimulationConfig, params: SimulationParams
    ) -> Tuple[STATE_TYPE, SimulationConfig, SimulationParams, RegisteredVariables]:
        config = config._replace(**{"box_size": 2 * jnp.pi})
        params = params._replace(
            **{
                "t_end": jnp.pi,
                "dt_max": 0.1,
                "minimum_density": 1e-2,
                "minimum_pressure": 1e-2,
                "C_cfl": 0.7,
            }
        )
        helper_data = get_helper_data(config)
        registered_variables = get_registered_variables(config)
        grid_spacing = config.box_size / config.num_cells
        if self.vortex_axis == "z":
            perpendicular_axis = 2
            plane_axis = [0, 1]
        elif self.vortex_axis == "y":
            perpendicular_axis = 1
            plane_axis = [0, 2]
        elif self.vortex_axis == "x":
            perpendicular_axis = 0
            plane_axis = [1, 2]
        else:
            raise ValueError("vortex_axis given is not x,y or z")
        plane_axis = jnp.array(plane_axis)

        x = jnp.linspace(
            grid_spacing / 2,
            config.box_size + grid_spacing / 2,
            config.num_cells,
            endpoint=False,
        )
        coordinates = jnp.array(jnp.meshgrid(x, x, x, indexing="ij"))

        r = helper_data.r

        rho = jnp.ones_like(r) * (params.gamma) ** 2
        P = jnp.ones_like(r) * (params.gamma)

        v_s = {}
        B_s = {}

        plane_axis_asymmetry = [1, 0]
        for i, axis in enumerate(plane_axis):
            axis = int(axis)

            v_s[axis] = (
                (-1) ** (i + self.parity)
                * (1 + self.epsilon_p * jnp.sin(coordinates[perpendicular_axis]))
                * jnp.sin(coordinates[plane_axis[plane_axis_asymmetry[i]]])
            )
            B_s[axis] = (-1) ** (i + self.parity) * jnp.sin(
                2 ** (i) * coordinates[plane_axis[plane_axis_asymmetry[i]]]
            )

        v_s[perpendicular_axis] = self.epsilon_p * jnp.sin(
            coordinates[perpendicular_axis]
        )
        B_s[perpendicular_axis] = jnp.zeros_like(r)

        V_x = v_s[0]
        V_y = v_s[1]
        V_z = v_s[2]
        B_x = B_s[0]
        B_y = B_s[1]
        B_z = B_s[2]

        bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)
        initial_state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=rho,
            velocity_x=V_x,
            velocity_y=V_y,
            velocity_z=V_z,
            magnetic_field_x=B_x,
            magnetic_field_y=B_y,
            magnetic_field_z=B_z,
            interface_magnetic_field_x=bxb,
            interface_magnetic_field_y=byb,
            interface_magnetic_field_z=bzb,
            gas_pressure=P,
        )

        config = finalize_config(config, initial_state.shape)
        return (initial_state, config, params, registered_variables)


def initial_state_ot_vortex(
    config: SimulationConfig,
    params: SimulationParams,
    vortex_axis: str = "z",
    parity: bool = False,
) -> Tuple[
    STATE_TYPE, SimulationConfig, SimulationParams, HelperData, RegisteredVariables
]:
    """from paper https://doi.org/10.1016/j.compfluid.2018.01.032"""
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    grid_spacing = config.box_size / config.num_cells
    if vortex_axis == "z":
        perpendicular_axis = 2
        plane_axis = [0, 1]
    elif vortex_axis == "y":
        perpendicular_axis = 1
        plane_axis = [0, 2]
    elif vortex_axis == "x":
        perpendicular_axis = 0
        plane_axis = [1, 2]
    else:
        raise ValueError("vortex_axis given is not x,y or z")
    plane_axis = jnp.array(plane_axis)

    x = jnp.linspace(
        grid_spacing / 2,
        config.box_size + grid_spacing / 2,
        config.num_cells,
        endpoint=False,
    )
    coordinates = jnp.array(jnp.meshgrid(x, x, x, indexing="ij"))

    r = helper_data.r

    rho = jnp.ones_like(r) * (params.gamma) ** 2
    P = jnp.ones_like(r) * (params.gamma)

    epsilon_p = 0.2

    v_s = {}
    B_s = {}

    plane_axis_asymmetry = [1, 0]
    for i, axis in enumerate(plane_axis):
        axis = int(axis)

        v_s[axis] = (
            (-1) ** (i + parity)
            * (1 + epsilon_p * jnp.sin(coordinates[perpendicular_axis]))
            * jnp.sin(coordinates[plane_axis[plane_axis_asymmetry[i]]])
        )
        B_s[axis] = (-1) ** (i + parity) * jnp.sin(
            2 ** (i) * coordinates[plane_axis[plane_axis_asymmetry[i]]]
        )

    v_s[perpendicular_axis] = epsilon_p * jnp.sin(coordinates[perpendicular_axis])
    B_s[perpendicular_axis] = jnp.zeros_like(r)

    V_x = v_s[0]
    V_y = v_s[1]
    V_z = v_s[2]
    B_x = B_s[0]
    B_y = B_s[1]
    B_z = B_s[2]

    bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)
    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=V_x,
        velocity_y=V_y,
        velocity_z=V_z,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
        gas_pressure=P,
    )

    config = finalize_config(config, initial_state.shape)
    return (initial_state, config, params, helper_data, registered_variables)


if __name__ == "__main__":
    """Creates an animation comparing high res and low res states into the folder"""
    axis = "x"
    parity = False
    folder = f"results/ot_vortex_3d/axis_{axis}_{parity}"
    num_cells_hr = 64
    num_cells_lr = 32
    num_snapshots = 50
    t_end = jnp.pi
    c_cfl = 0.7
    snapshot_timepoints = jnp.linspace(0.0, t_end, num_snapshots, True)
    simulation_params = SimulationParams(
        C_cfl=c_cfl,
        t_end=t_end,
        snapshot_timepoints=snapshot_timepoints,
        dt_max=0.1,
        minimum_density=1e-2,
        minimum_pressure=1e-2,
    )

    hr_simulation_config = SimulationConfig(
        num_cells=num_cells_hr,
        box_size=2 * jnp.pi,
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
        # limiter=4,
        enforce_positivity=True,
    )

    lr_simulation_config = SimulationConfig(
        num_cells=num_cells_lr,
        box_size=2 * jnp.pi,
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
        # limiter=4,
        enforce_positivity=True,
    )

    initial_state_hr, config_hr, params, helper_data_hr, registered_variables_hr = (
        initial_state_ot_vortex(
            config=hr_simulation_config,
            params=simulation_params,
            vortex_axis=axis,
            parity=parity,
        )
    )

    initial_state_lr, config_lr, params, helper_data_lr, registered_variables_lr = (
        initial_state_ot_vortex(
            config=lr_simulation_config,
            params=simulation_params,
            vortex_axis=axis,
            parity=parity,
        )
    )
    plot_states(
        states_list=[initial_state_hr, initial_state_lr],
        z_levels=[num_cells_hr // 2, num_cells_lr // 2],
        folder=folder,
        fig_name=f"initial_state_{num_cells_hr}_{num_cells_lr}",
        slice_axis=axis,
    )

    snapshot_data_hr = time_integration(
        primitive_state=initial_state_hr,
        config=config_hr,
        params=params,
        registered_variables=registered_variables_hr,
    )
    snapshot_data_lr = time_integration(
        primitive_state=initial_state_lr,
        config=config_lr,
        params=params,
        registered_variables=registered_variables_lr,
    )
    assert isinstance(snapshot_data_hr, SnapshotData)
    assert isinstance(snapshot_data_lr, SnapshotData)

    last_true_state_lr = -1
    for state in snapshot_data_lr.states:
        if jnp.any(state != 0.0):
            last_true_state_lr += 1

    last_true_state_hr = -1
    for state in snapshot_data_hr.states:
        if jnp.any(state != 0.0):
            last_true_state_hr += 1

    plot_states(
        states_list=[initial_state_hr, snapshot_data_hr.states[last_true_state_hr]],
        z_levels=[num_cells_hr // 2, num_cells_hr // 2],
        folder=folder,
        fig_name=f"initial_final_state_{num_cells_hr}",
        slice_axis=axis,
    )
    plot_states(
        states_list=[initial_state_lr, snapshot_data_lr.states[last_true_state_lr]],
        z_levels=[num_cells_lr // 2, num_cells_lr // 2],
        folder=folder,
        fig_name=f"initial_final_state_{num_cells_lr}",
        slice_axis=axis,
    )

    animate_several_sims_states_with_diagonal(
        sim_states=[snapshot_data_hr.states, snapshot_data_lr.states],
        sim_legend=[f"{num_cells_hr}^3", f"{num_cells_lr}^3"],
        z_levels=[num_cells_hr // 2, num_cells_lr // 2],
        timepoints=snapshot_data_hr.time_points,
        slice_axis=axis,
        save_path=folder,
        file_name=f"animation_{num_cells_hr}_{num_cells_lr}",
        fps=5,
    )
