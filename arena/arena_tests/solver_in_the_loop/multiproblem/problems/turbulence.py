if __name__ == "__main__":
    from autocvd import autocvd

    autocvd(num_gpus=1)
import jax.numpy as jnp
import jax
from jaxtyping import Array
from typing import Tuple, Optional

from arena.arena_tests.solver_in_the_loop.model_manager import TrainingConfig
from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    initialize_interface_fields,
)

from astropy import units as u
from astropy import constants as c
from astronomix.option_classes.simulation_config import STATE_TYPE
from astronomix.variable_registry.registered_variables import RegisteredVariables
from arena.arena_tests.solver_in_the_loop.multiproblem.problems.baseproblem import (
    BaseProblem,
)
from astronomix import CodeUnits
from astronomix.initial_condition_generation.turbulent_ic_generator import (
    create_turb_field,
)
import logging
import time

logger = logging.getLogger(__name__)


class Turbulence(BaseProblem):
    def __init__(
        self,
        B0: float = 0.7071,
        B_direction: Array = jnp.array([1, 1, 0]),
        rng_seed: Optional[int] = None,
    ) -> None:
        self.B0 = B0
        assert B_direction.shape == (3,), ValueError(
            f"B_direction is a 3D vector, shape given {B_direction.shape}"
        )
        self.B_direction = B_direction
        self.rng_seed = rng_seed

    @property
    def name(self) -> str:
        return "turbulence"

    @property
    def t_end(self) -> float:
        return 0.4

    @property
    def dimensionalty(self) -> float:
        return 3

    def get_hyperparams(self) -> dict:
        return {
            "problem_name": self.name,
            "params": {
                "B0": self.B0,
                "B_direction": self.B_direction,
                "rng_seed": self.rng_seed,
            },
        }

    def generate_initial_state(
        self, config: SimulationConfig, params: SimulationParams
    ) -> Tuple[STATE_TYPE, SimulationConfig, SimulationParams, RegisteredVariables]:
        # PROBLEM OVERRIDING
        "Creates a turbulent initial state with mhd"
        adiabatic_index = 5 / 3
        params = params._replace(gamma=adiabatic_index)
        config = config._replace(box_size=1.0)
        params = params._replace(dt_max=0.1)
        params = params._replace(t_end=0.4)
        params = params._replace(C_cfl=0.4)

        helper_data = get_helper_data(config)
        registered_variables = get_registered_variables(config)

        # setup the unit system
        code_length = 3 * u.parsec
        code_mass = 1 * u.M_sun
        code_velocity = 100 * u.km / u.s
        code_units = CodeUnits(code_length, code_mass, code_velocity)

        wanted_rms = 50 * u.km / u.s

        # homogeneous initial state
        rho_0 = 2 * c.m_p / u.cm**3
        p_0 = 3e4 * u.K / u.cm**3 * c.k_B

        density = (
            jnp.ones((config.num_cells, config.num_cells, config.num_cells))
            * rho_0.to(code_units.code_density).value
        )

        # turbulence parameters
        turbulence_slope = -2
        kmin = 2
        kmax = 64

        p = (
            jnp.ones((config.num_cells, config.num_cells, config.num_cells))
            * p_0.to(code_units.code_pressure).value
        )
        if self.rng_seed is None:
            self.rng_seed = int(time.time() * 1e6) % (2**32 - 1)

        key = jax.random.key(self.rng_seed)

        keys = jax.random.split(key, 3)
        u_x = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[0]
        )
        u_y = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[1]
        )
        u_z = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[2]
        )

        # scale the turbulence to the desired rms velocity
        rms_vel = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))

        u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_z = u_z / rms_vel * wanted_rms.to(code_units.code_velocity).value
        # construct primitive state

        grid_spacing = config.box_size / config.num_cells
        x = jnp.linspace(
            grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
        )

        X, *_ = jnp.meshgrid(x, x, x, indexing="ij")

        B_direction_normalized = self.B_direction / jnp.linalg.norm(self.B_direction)
        B_x = self.B0 * B_direction_normalized[0]
        B_y = self.B0 * B_direction_normalized[1]
        B_z = self.B0 * B_direction_normalized[2]

        B_x = jnp.ones_like(X) * B_x
        B_y = jnp.ones_like(X) * B_y
        B_z = jnp.ones_like(X) * B_z

        bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

        initial_state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=density,
            velocity_x=u_x,
            velocity_y=u_y,
            velocity_z=u_z,
            gas_pressure=p,
            magnetic_field_x=B_x,
            magnetic_field_y=B_y,
            magnetic_field_z=B_z,
            interface_magnetic_field_x=bxb,
            interface_magnetic_field_y=byb,
            interface_magnetic_field_z=bzb,
        )

        config = finalize_config(config, initial_state.shape)

        return (initial_state, config, params, registered_variables)


if __name__ == "__main__":
    from arena.arena_tests.solver_in_the_loop.multiproblem.problem_manager import (
        _build_hr_config_and_params,
    )
    from astronomix import time_integration
    from arena.arena_tests.solver_in_the_loop.plot_states_comparison import (
        plot_states,
        plot_and_animate_states_with_diagonal,
    )
    from astronomix.data_classes.simulation_snapshot_data import SnapshotData

    training_config = TrainingConfig(epochs_per_time=[], snapshot_timepoints_train=[])
    config, params = _build_hr_config_and_params(training_config=training_config)

    turb = Turbulence(rng_seed=111)
    config_overrides = turb.get_config_overrides_evaluation(
        snapshot_timepoints=jnp.linspace(0.01, 0.4, num=40, endpoint=True)
    )
    (initial_state, config, params, registered_variables) = (
        turb.generate_initial_state_with_config_overrides(
            config=config, params=params, config_overrides=config_overrides
        )
    )
    plot_states(
        states_list=[initial_state],
        z_levels=[training_config.num_cells_high_res // 2],
        folder="results/turbulence",
        fig_name="initial_state",
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
    print(last_true_state)

    plot_states(
        states_list=[initial_state, snapshot_data.states[last_true_state]],
        z_levels=[
            training_config.num_cells_high_res // 2,
            training_config.num_cells_high_res // 2,
        ],
        folder="results/turbulence",
        model_name="test_jet",
        fig_name="initial_final_state",
    )
    plot_and_animate_states_with_diagonal(
        states=snapshot_data.states,
        timepoints=snapshot_data.time_points,
        z_level=training_config.num_cells_high_res // 2,
        slice_axis="x",
        save_path="results/turbulence",
        file_name="turb_video",
        fps=10,
    )
