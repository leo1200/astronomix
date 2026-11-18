"""
The stellar wind in every scheme family: a wind bubble against Weaver et al. (1977).

A wind of mass-loss rate ``M`` and speed ``v_w`` (luminosity ``L = M v_w^2 / 2``) blows into a uniform, cold
medium of density ``rho_0`` in a periodic 3D box. The source of the stages (``_modules/_explicit_sources.py``)
injects mass and thermal energy at constant rates, so every scheme gains exactly ``M t`` of mass and ``L t`` of
energy (checked to 1e-10). The swept-up shell runs ahead at the outer-shock radius of Weaver et al.,
``R_2 = 0.88 (L t^3 / rho_0)^(1/5)``; the radius where the radially averaged density falls through 1.5 rho_0
matches it to 10 % (24^3 cells, the scheme with the Riemann flux on 16^3 to stay within the time of a test on four
CPU cores).

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/modules/test_stellar_wind_across_schemes.py
"""

# numerics
import numpy as np
import pytest

# astronomix containers
from astronomix import SimulationParams, WindParams
from astronomix.option_classes import WindConfig

# astronomix functions
from astronomix import get_registered_variables
from astronomix._modules._stellar_wind.weaver import Weaver

# the scheme families of the module tests
from _schemes import HYDRO_SCHEMES, cell_centres, make_config, run

pytestmark = pytest.mark.x64

GAMMA = 5.0 / 3.0
AMBIENT_DENSITY = 1.0
AMBIENT_PRESSURE = 0.02
WIND_SPEED = 5.0
LUMINOSITY = 1.0
MASS_LOSS_RATE = 2.0 * LUMINOSITY / WIND_SPEED**2
T_END = 0.21
CELLS = {"FD RIEMANN_FLUX": 16}
DEFAULT_CELLS = 24
RADIUS_TOLERANCE = 0.1
CONSERVATION_TOLERANCE = 1e-10


def _shock_radius(density, cells):
    """The outermost radius at which the radially averaged density falls through 1.5 times the ambient one,
    interpolated between bins of one cell."""
    position = cell_centres(cells, 1.0) - 0.5
    radius = np.sqrt(position[:, None, None] ** 2 + position[None, :, None] ** 2 + position[None, None, :] ** 2)
    edges = np.arange(0.0, 0.5, 1.0 / cells)
    bins = np.digitize(radius.ravel(), edges)
    profile = np.array([density.ravel()[bins == number].mean() for number in range(1, len(edges))])
    centres = 0.5 * (edges[1:] + edges[:-1])
    threshold = 1.5 * AMBIENT_DENSITY
    last = np.nonzero(profile > threshold)[0].max()
    fraction = (profile[last] - threshold) / (profile[last] - profile[last + 1])
    return centres[last] + fraction * (centres[last + 1] - centres[last])


@pytest.mark.slow
@pytest.mark.parametrize("name", list(HYDRO_SCHEMES))
def test_wind_bubble(name):
    cells = CELLS.get(name, DEFAULT_CELLS)
    shape = (cells,) * 3
    config = make_config(*HYDRO_SCHEMES[name], shape, wind_config=WindConfig(stellar_wind=True, num_injection_cells=2))
    registered_variables = get_registered_variables(config)
    initial_state = np.zeros((registered_variables.num_vars,) + shape)
    initial_state[registered_variables.density_index] = AMBIENT_DENSITY
    initial_state[registered_variables.pressure_index] = AMBIENT_PRESSURE
    params = SimulationParams(
        t_end=T_END,
        C_cfl=0.4,
        gamma=GAMMA,
        wind_params=WindParams(wind_mass_loss_rate=MASS_LOSS_RATE, wind_final_velocity=WIND_SPEED),
    )
    final_state, _ = run(config, initial_state, params)
    assert np.all(np.isfinite(final_state))

    cell_volume = (1.0 / cells) ** 3
    density = final_state[registered_variables.density_index]
    velocity = final_state[registered_variables.velocity_index.x : registered_variables.velocity_index.z + 1]
    energy = final_state[registered_variables.pressure_index] / (GAMMA - 1.0) + 0.5 * density * np.sum(velocity**2, 0)
    mass_gain = (density.sum() - AMBIENT_DENSITY * cells**3) * cell_volume
    energy_gain = (energy.sum() - AMBIENT_PRESSURE / (GAMMA - 1.0) * cells**3) * cell_volume
    mass_deviation = mass_gain / (MASS_LOSS_RATE * T_END) - 1.0
    energy_deviation = energy_gain / (LUMINOSITY * T_END) - 1.0

    weaver = Weaver(WIND_SPEED, MASS_LOSS_RATE, AMBIENT_DENSITY, AMBIENT_PRESSURE, gamma=GAMMA)
    outer_shock_radius = weaver.get_outer_shock_radius(T_END)
    radius_deviation = _shock_radius(density, cells) / outer_shock_radius - 1.0
    print(
        f"{name} ({cells}^3): mass {mass_deviation:+.1e}, energy {energy_deviation:+.1e}, "
        f"shock radius off Weaver's by {radius_deviation:+.3f}"
    )
    assert abs(mass_deviation) < CONSERVATION_TOLERANCE
    assert abs(energy_deviation) < CONSERVATION_TOLERANCE
    assert abs(radius_deviation) < RADIUS_TOLERANCE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
