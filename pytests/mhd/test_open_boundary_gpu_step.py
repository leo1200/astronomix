"""
The float64 GPU step of ``FD_RIEMANN_ISOTHERMAL`` with open boundaries on (N, 1, 1) cells.

Compiled with the multi-output fusion pass of XLA (jaxlib 0.9.2), the step faults with CUDA_ERROR_ILLEGAL_ADDRESS
from N = 416 on (a loop kernel of the flux sweep reads out of bounds), which ends the process;
``time_integration`` compiles this scheme in float64 on a GPU without the pass. Checks that a few steps of the
isothermal MHD shock tube (Mignone 2007, test 2) at N = 512 run and stay finite.

    python -m pytest pytests/mhd/test_open_boundary_gpu_step.py
"""

# numerics
import numpy as np

# testing
import pytest

# astronomix
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    ISOTHERMAL,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
)
from astronomix.option_classes.presets import FD_RIEMANN_ISOTHERMAL
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.time_stepping.time_integration import time_integration

# jax
import jax.numpy as jnp

# The fault is in the float64 GPU program (the conftest runs the module in float64 and skips it without a GPU).
pytestmark = [pytest.mark.x64, pytest.mark.gpu]

NUM_CELLS = 512
FIELD_SCALE = 1.0 / np.sqrt(4.0 * np.pi)
LEFT_STATE = dict(density=1.08, velocity=(1.2, 0.01, 0.5), field=(2.0, 3.6, 2.0))
RIGHT_STATE = dict(density=1.0, velocity=(0.0, 0.0, 0.0), field=(2.0, 4.0, 2.0))


def test_open_boundary_shock_tube_steps_on_the_gpu():
    open_sides = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)
    periodic_sides = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    cell_size = 1.0 / NUM_CELLS
    config = SimulationConfig(
        mhd=True,
        equation_of_state=ISOTHERMAL,
        solver_mode=FINITE_DIFFERENCE,
        solver_config=FD_RIEMANN_ISOTHERMAL,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        progress_bar=False,
        dimensionality=3,
        num_cells=StaticIntVector(NUM_CELLS, 1, 1),
        box_size=StaticFloatVector(1.0, cell_size, cell_size),
        boundary_settings=BoundarySettings(open_sides, periodic_sides, periodic_sides),
    )
    registered_variables = get_registered_variables(config)
    left_half = (np.arange(NUM_CELLS) + 0.5) * cell_size < 0.5

    def profile(left_value, right_value):
        return jnp.asarray(np.where(left_half, left_value, right_value).reshape(NUM_CELLS, 1, 1))

    velocity = [profile(*values) for values in zip(LEFT_STATE["velocity"], RIGHT_STATE["velocity"])]
    field = [profile(*values) * FIELD_SCALE for values in zip(LEFT_STATE["field"], RIGHT_STATE["field"])]
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=profile(LEFT_STATE["density"], RIGHT_STATE["density"]),
        velocity_x=velocity[0],
        velocity_y=velocity[1],
        velocity_z=velocity[2],
        magnetic_field_x=field[0],
        magnetic_field_y=field[1],
        magnetic_field_z=field[2],
        interface_magnetic_field_x=field[0],
        interface_magnetic_field_y=field[1],
        interface_magnetic_field_z=field[2],
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(C_cfl=1.0, t_end=0.002, isothermal_sound_speed=1.0)
    final_state = np.asarray(time_integration(state, config, params, registered_variables))
    assert np.all(np.isfinite(final_state))
