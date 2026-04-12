import jax.numpy as jnp
from jaxtyping import Array
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
