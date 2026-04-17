if __name__ == "__main__":
    from autocvd import autocvd

    #
    autocvd(num_gpus=1)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["jax_enable_x64"] = "True"

    from astronomix import time_integration
import os
from pathlib import Path
import jax.numpy as jnp
from jaxtyping import Array
from astronomix.data_classes.simulation_helper_data import HelperData
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
from astronomix.option_classes.simulation_config import STATE_TYPE
from astronomix.variable_registry.registered_variables import RegisteredVariables
from arena.arena_tests.solver_in_the_loop.multiproblem.problems.baseproblem import (
    BaseProblem,
)

import logging

logger = logging.getLogger(__name__)


class MhdBlast(BaseProblem):
    def __init__(
        self,
        B0: float = 10.0,
        r0: float = 0.125,
        B_direction: Array = jnp.array([1, 1, 0]),
    ) -> None:
        self.B0 = B0
        self.r0 = r0
        B_direction = jnp.asarray(B_direction)
        assert B_direction.shape == (3,), ValueError(
            f"B_direction is a 3D vector, shape given {B_direction.shape}"
        )
        self.B_direction = B_direction

    @property
    def name(self) -> str:
        return "mhd_blast"

    @property
    def t_end(self) -> float:
        return 0.2

    def get_hyperparams(self) -> dict:
        return {
            "problem_name": self.name,
            "params": {"B0": self.B0, "r0": self.r0, "B_direction": self.B_direction},
        }

    def generate_initial_state(
        self, config: SimulationConfig, params: SimulationParams
    ) -> Tuple[STATE_TYPE, SimulationConfig, SimulationParams, RegisteredVariables]:
        # PROBLEM OVERRIDING
        config = config._replace(box_size=1.0)
        params = params._replace(t_end=0.2)

        helper_data = get_helper_data(config)
        registered_variables = get_registered_variables(config)

        r = helper_data.r
        r0 = self.r0
        r1 = 1.1 * self.r0

        rho = jnp.ones_like(r)
        P = jnp.ones_like(r) * 1.0
        P = jnp.where(r <= r0, 100.0, P)
        P = jnp.where((r > r0) & (r <= r1), 1.0 + 99.0 * (r1 - r) / (r1 - r0), P)

        V_x = jnp.zeros_like(r)
        V_y = jnp.zeros_like(r)
        V_z = jnp.zeros_like(r)

        B_direction_normalized = self.B_direction / jnp.linalg.norm(self.B_direction)
        B_x = self.B0 * B_direction_normalized[0]
        B_y = self.B0 * B_direction_normalized[1]
        B_z = self.B0 * B_direction_normalized[2]

        B_x = jnp.ones_like(r) * B_x
        B_y = jnp.ones_like(r) * B_y
        B_z = jnp.ones_like(r) * B_z

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


def initial_state_mhd_blast(
    config: SimulationConfig, params: SimulationParams
) -> Tuple[
    STATE_TYPE, SimulationConfig, SimulationParams, HelperData, RegisteredVariables
]:
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    B0 = 10.0
    r = helper_data.r
    r0 = 0.125
    r1 = 1.1 * r0

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r) * 1.0
    P = jnp.where(r <= r0, 100.0, P)
    P = jnp.where((r > r0) & (r <= r1), 1.0 + 99.0 * (r1 - r) / (r1 - r0), P)

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    B_x = B0 / jnp.sqrt(2)
    B_y = B0 / jnp.sqrt(2)
    B_z = 0.0

    B_x = jnp.ones_like(r) * B_x
    B_y = jnp.ones_like(r) * B_y
    B_z = jnp.ones_like(r) * B_z

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


def _plot_states_with_left_theta_labels(
    *,
    states_list: list[Array],
    z_levels: list[int],
    thetas: list[float],
    folder: str,
    fig_name: str,
) -> None:
    if not (len(states_list) == len(z_levels) == len(thetas)):
        raise ValueError("states_list, z_levels, and thetas must have the same length.")

    n_states = len(states_list)
    n_fields = 4  # density, velocity magnitude, pressure, magnetic field

    vmax_density = max(jnp.max(states[0]).astype(float) for states in states_list)
    vmin_density = min(jnp.min(states[0]).astype(float) for states in states_list)
    vmax_pressure = max(jnp.max(states[4]).astype(float) for states in states_list)
    vmin_pressure = min(jnp.min(states[4]).astype(float) for states in states_list)
    vmin_speed = min(
        jnp.min(jnp.sqrt(states[1] ** 2 + states[2] ** 2 + states[3] ** 2)).astype(
            float
        )
        for states in states_list
    )
    vmax_speed = max(
        jnp.max(jnp.sqrt(states[1] ** 2 + states[2] ** 2 + states[3] ** 2)).astype(
            float
        )
        for states in states_list
    )
    vmin_mag = min(
        jnp.min(jnp.sqrt(states[5] ** 2 + states[6] ** 2 + states[7] ** 2)).astype(
            float
        )
        for states in states_list
    )
    vmax_mag = max(
        jnp.max(jnp.sqrt(states[5] ** 2 + states[6] ** 2 + states[7] ** 2)).astype(
            float
        )
        for states in states_list
    )

    def _slice_field(field, level):
        return field[:, :, level].T

    fig, axs = plt.subplots(n_states, n_fields, figsize=(5 * n_fields, 5 * n_states))
    if n_states == 1:
        axs = axs[None, :]

    for row, (states, z_level, theta) in enumerate(
        zip(states_list, z_levels, thetas, strict=True)
    ):
        c0 = axs[row][0].imshow(
            _slice_field(states[0], z_level),
            origin="lower",
            norm=Normalize(vmin=vmin_density, vmax=vmax_density),
        )
        fig.colorbar(c0, ax=axs[row][0])
        axs[row][0].set_title("Density")
        axs[row][0].set_xlabel("x")
        axs[row][0].set_ylabel("y")

        c1 = axs[row][1].imshow(
            _slice_field(jnp.sqrt(states[1] ** 2 + states[2] ** 2 + states[3] ** 2), z_level),
            origin="lower",
            norm=Normalize(vmin=vmin_speed, vmax=vmax_speed),
        )
        fig.colorbar(c1, ax=axs[row][1])
        axs[row][1].set_title("Velocity Magnitude")
        axs[row][1].set_xlabel("x")
        axs[row][1].set_ylabel("y")

        c2 = axs[row][2].imshow(
            _slice_field(states[4], z_level),
            origin="lower",
            norm=Normalize(vmin=vmin_pressure, vmax=vmax_pressure),
        )
        fig.colorbar(c2, ax=axs[row][2])
        axs[row][2].set_title("Pressure")
        axs[row][2].set_xlabel("x")
        axs[row][2].set_ylabel("y")

        c3 = axs[row][3].imshow(
            _slice_field(jnp.sqrt(states[5] ** 2 + states[6] ** 2 + states[7] ** 2), z_level),
            origin="lower",
            norm=Normalize(vmin=vmin_mag, vmax=vmax_mag),
        )
        fig.colorbar(c3, ax=axs[row][3])
        axs[row][3].set_title("Magnetic Field")
        axs[row][3].set_xlabel("x")
        axs[row][3].set_ylabel("y")

        axs[row][0].text(
            -0.95,
            0.5,
            f"θ = {theta:.3f}",
            transform=axs[row][0].transAxes,
            ha="right",
            va="center",
            fontsize=12,
            clip_on=False,
        )

    plt.tight_layout(rect=(0.10, 0.0, 1.0, 1.0))
    os.makedirs(folder, exist_ok=True)
    plt.savefig(Path(folder) / f"{fig_name}.png", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    from arena.arena_tests.solver_in_the_loop.multiproblem.problem_manager import (
        _build_hr_config_and_params,
    )
    from arena.arena_tests.solver_in_the_loop.model_manager import TrainingConfig
    import math

    b0 = 10.0
    num_snapshots = 40
    t_end = 0.2

    training_config = TrainingConfig(epochs_per_time=[], snapshot_timepoints_train=[])
    base_config, base_params = _build_hr_config_and_params(
        training_config=training_config
    )

    # Match azimuth_generalized/run_one.sh:
    #   phi = pi/4
    #   theta in {0.8, pi/2, 2.3}
    phi = math.pi / 4.0
    thetas = (0.8, math.pi / 2.0, 2.3)
    b_directions = []
    for theta in thetas:
        b_direction = jnp.array(
            [
                round(math.sin(theta) * math.cos(phi), 8),
                round(math.sin(theta) * math.sin(phi), 8),
                round(math.cos(theta), 8),
            ]
        )
        b_directions.append(b_direction)

    final_states: list[Array] = []
    theta_labels: list[float] = []

    for theta, b_direction in zip(thetas, b_directions, strict=True):
        blast = MhdBlast(B_direction=b_direction, B0=b0)
        config_overrides = blast.get_config_overrides_evaluation(
            snapshot_timepoints=jnp.linspace(
                0.0, t_end, num=num_snapshots, endpoint=True
            )
        )
        initial_state, config, params, registered_variables = (
            blast.generate_initial_state_with_config_overrides(
                config=base_config,
                params=base_params,
                config_overrides=config_overrides,
            )
        )

        snapshot_data = time_integration(
            primitive_state=initial_state,
            config=config,
            params=params,
            registered_variables=registered_variables,
        )
        assert isinstance(snapshot_data, SnapshotData)

        # Keep last valid state in case a crash creates zero-padded tail snapshots.
        final_state = initial_state
        for state in snapshot_data.states:
            if jnp.any(state != 0.0):
                final_state = state

        final_states.append(final_state)
        theta_labels.append(float(theta))

    out_folder = "results/mhd_blast/angle_generalization_no_ml"
    z_level = final_states[0].shape[-1] // 2
    _plot_states_with_left_theta_labels(
        states_list=final_states,
        z_levels=[z_level] * len(final_states),
        thetas=theta_labels,
        folder=out_folder,
        fig_name="final_state_comparison_three_B_directions",
    )
