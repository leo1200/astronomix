"""
Viscosity in every scheme family: a decaying shear wave.

The transverse velocity ``v_y = A sin(k x)`` of a gas at rest and of uniform density and pressure decays as
``A exp(-nu k^2 t)`` under a kinematic viscosity ``nu``; nothing else moves (the HLLC solvers of the
finite-volume families do not diffuse a shear across a contact at rest, and the WENO schemes barely at 32 cells).
After a decay to two thirds every family matches the exponential to 1e-5 (2D, 32 x 8 cells, periodic).

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/modules/test_viscosity_across_schemes.py
"""

# numerics
import numpy as np
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import KINEMATIC_VISCOSITY

# astronomix containers
from astronomix import DiffusionConfig, SimulationParams

# astronomix functions
from astronomix import get_registered_variables

# the scheme families of the module tests
from _schemes import HYDRO_SCHEMES, cell_centres, make_config, mode_amplitude, run

pytestmark = pytest.mark.x64

VISCOSITY = 0.01
AMPLITUDE = 1e-3
WAVENUMBER = 2.0 * np.pi
T_END = 1.0
SHAPE = (32, 8)
TOLERANCE = 1e-5


@pytest.mark.parametrize("name", list(HYDRO_SCHEMES))
def test_decaying_shear_wave(name):
    config = make_config(
        *HYDRO_SCHEMES[name],
        SHAPE,
        diffusion_config=DiffusionConfig(viscosity=True, viscosity_type=KINEMATIC_VISCOSITY),
    )
    registered_variables = get_registered_variables(config)
    position = cell_centres(SHAPE[0], 1.0)
    initial_state = np.zeros((registered_variables.num_vars,) + SHAPE)
    initial_state[registered_variables.density_index] = 1.0
    initial_state[registered_variables.pressure_index] = 1.0
    initial_state[registered_variables.velocity_index.y] = AMPLITUDE * np.sin(WAVENUMBER * position)[:, None]
    params = SimulationParams(t_end=T_END, C_cfl=0.4, gamma=5.0 / 3.0, viscosity=VISCOSITY)
    final_state, _ = run(config, initial_state, params)
    amplitude = mode_amplitude(final_state[registered_variables.velocity_index.y], position, WAVENUMBER, 0.0)
    exact = AMPLITUDE * np.exp(-VISCOSITY * WAVENUMBER**2 * T_END)
    deviation = amplitude / exact - 1.0
    print(f"{name}: amplitude {amplitude / AMPLITUDE:.5f} of the initial one, off the exponential by {deviation:+.2e}")
    assert abs(deviation) < TOLERANCE
    assert np.max(np.abs(final_state[registered_variables.velocity_index.x])) < 1e-3 * AMPLITUDE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
