"""
Thermal conduction in every scheme family: a thermal mode against linear theory.

A gas in pressure balance with a small sinusoidal density perturbation (``rho = rho_0 (1 + eps sin(k x))``,
uniform pressure) carries a temperature perturbation that conduction diffuses. The linearised equations of
continuity, momentum and energy with ``div(kappa grad T)``, ``T = p / rho``, are three ordinary differential
equations for the mode amplitudes, solved exactly by the matrix exponential. After the thermal mode has decayed to
half, the density and pressure amplitudes of every family agree with linear theory to 2e-3 (relative to the
density amplitude and to ``eps``), with the second- and the fourth-order conduction flux (1D, 64 cells).

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/modules/test_conduction_across_schemes.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np
import pytest

# astronomix containers
from astronomix import DiffusionConfig, SimulationParams

# astronomix functions
from astronomix import get_registered_variables

# the scheme families of the module tests
from _schemes import HYDRO_SCHEMES, cell_centres, make_config, mode_amplitude, run

GAMMA = 5.0 / 3.0
CONDUCTIVITY = 0.05
EPSILON = 1e-4
WAVENUMBER = 2.0 * np.pi
T_END = 1.0
CELLS = 64
TOLERANCE = 2e-3


def _linear_theory():
    """The density and pressure amplitudes at the end time: ``d/dt (rho, v, p) = M (rho, v, p)`` for the modes
    ``sin(k x)``, ``cos(k x)``, ``sin(k x)`` (density and pressure 1)."""
    diffusion_rate = (GAMMA - 1.0) * CONDUCTIVITY * WAVENUMBER**2
    matrix = np.array(
        [
            [0.0, WAVENUMBER, 0.0],
            [0.0, 0.0, -WAVENUMBER],
            [diffusion_rate, GAMMA * WAVENUMBER, -diffusion_rate],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    propagator = eigenvectors @ np.diag(np.exp(eigenvalues * T_END)) @ np.linalg.inv(eigenvectors)
    density_amplitude, _, pressure_amplitude = np.real(propagator @ np.array([EPSILON, 0.0, 0.0]))
    return density_amplitude, pressure_amplitude


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("name", list(HYDRO_SCHEMES))
def test_thermal_mode(name, order):
    config = make_config(
        *HYDRO_SCHEMES[name],
        (CELLS,),
        diffusion_config=DiffusionConfig(thermal_conduction=True, conduction_order=order),
    )
    registered_variables = get_registered_variables(config)
    position = cell_centres(CELLS, 1.0)
    initial_state = np.zeros((registered_variables.num_vars, CELLS))
    initial_state[registered_variables.density_index] = 1.0 + EPSILON * np.sin(WAVENUMBER * position)
    initial_state[registered_variables.pressure_index] = 1.0
    params = SimulationParams(t_end=T_END, C_cfl=0.4, gamma=GAMMA, thermal_conductivity=CONDUCTIVITY)
    final_state, _ = run(config, initial_state, params)
    density_amplitude, pressure_amplitude = _linear_theory()
    density_deviation = (
        mode_amplitude(final_state[registered_variables.density_index], position, WAVENUMBER, 0.0)
        / density_amplitude
        - 1.0
    )
    pressure_deviation = (
        mode_amplitude(final_state[registered_variables.pressure_index], position, WAVENUMBER, 0.0)
        - pressure_amplitude
    ) / EPSILON
    print(f"{name}, order {order}: density {density_deviation:+.2e}, pressure {pressure_deviation:+.2e}")
    assert density_amplitude < 0.6 * EPSILON
    assert abs(density_deviation) < TOLERANCE
    assert abs(pressure_deviation) < TOLERANCE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
