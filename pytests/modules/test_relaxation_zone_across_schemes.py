"""
The relaxation zone in every scheme family: exact relaxation of a uniform box, no effect where the rate is zero,
and absorption of an outgoing sound wave.

* A uniform box has no fluxes, so with a uniform rate every primitive variable follows the exact solution
  ``q_target + (q_0 - q_target) exp(-rate t)`` of the zone's equation, with one rate for all variables or one per
  variable.
* Where the rate is zero the zone changes nothing: a run with the zone switched on and a zero rate is bit for bit the
  run without it.
* A sound pulse that runs into a reflecting wall comes back into the box; with a zone in front of the wall, relaxing
  to the ambient state, it does not.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/modules/test_relaxation_zone_across_schemes.py
"""

# numerics
import numpy as np
import pytest

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import FINITE_VOLUME, REFLECTIVE_BOUNDARY
from astronomix.option_classes.simulation_config import VL2

# astronomix containers
from astronomix import (
    BoundarySettings1D,
    FVSolverConfig,
    RelaxationZoneConfig,
    RelaxationZoneParams,
    SimulationParams,
)

# astronomix functions
from astronomix import finalize_config, get_registered_variables

# the scheme families of the module tests
from _schemes import HYDRO_SCHEMES, cell_centres, make_config, run

pytestmark = pytest.mark.x64

GAMMA = 5.0 / 3.0
ZONE_ON = RelaxationZoneConfig(relaxation_zone=True)


def _uniform_state(registered_variables, num_cells, density, velocity, pressure):
    """A uniform 1D primitive state."""
    state = np.zeros((registered_variables.num_vars, num_cells))
    state[registered_variables.density_index] = density
    state[registered_variables.velocity_index] = velocity
    state[registered_variables.pressure_index] = pressure
    return state


# -------------------------------------------------------------
# ============ ↓ exact relaxation of a uniform box ↓ ============
# -------------------------------------------------------------


@pytest.mark.parametrize("per_variable", [False, True], ids=["one rate", "rate per variable"])
@pytest.mark.parametrize("name", list(HYDRO_SCHEMES))
def test_uniform_box_relaxes_exactly(name, per_variable):
    num_cells = 8
    end_time = 0.5
    config = make_config(*HYDRO_SCHEMES[name], (num_cells,), relaxation_zone_config=ZONE_ON)
    registered_variables = get_registered_variables(config)
    initial_state = _uniform_state(registered_variables, num_cells, density=1.0, velocity=0.3, pressure=1.0)
    target_state = _uniform_state(registered_variables, num_cells, density=2.0, velocity=0.5, pressure=3.0)

    if per_variable:
        rate = np.zeros((registered_variables.num_vars, num_cells))
        rate[registered_variables.density_index] = 2.0
        rate[registered_variables.velocity_index] = 0.0
        rate[registered_variables.pressure_index] = 4.0
    else:
        rate = np.full(num_cells, 2.0)
    params = SimulationParams(
        t_end=end_time,
        C_cfl=0.4,
        gamma=GAMMA,
        relaxation_zone_params=RelaxationZoneParams(target_state=jnp.asarray(target_state), rate=jnp.asarray(rate)),
    )
    final_state, _ = run(config, initial_state, params)

    rate_per_variable = rate if per_variable else np.broadcast_to(rate, initial_state.shape)
    expected = target_state + (initial_state - target_state) * np.exp(-rate_per_variable * end_time)
    print(f"{name}: largest deviation from the exact relaxation {np.abs(final_state - expected).max():.2e}")
    assert np.allclose(final_state, expected, rtol=1e-12, atol=1e-12)


# -------------------------------------------------------------
# ============ ↓ no effect where the rate is zero ↓ ============
# -------------------------------------------------------------


@pytest.mark.parametrize("name", list(HYDRO_SCHEMES))
def test_zero_rate_changes_nothing(name):
    num_cells = 32
    coordinate = cell_centres(num_cells, 1.0)
    config_off = make_config(*HYDRO_SCHEMES[name], (num_cells,))
    config_on = make_config(*HYDRO_SCHEMES[name], (num_cells,), relaxation_zone_config=ZONE_ON)
    registered_variables = get_registered_variables(config_off)
    initial_state = _uniform_state(registered_variables, num_cells, density=1.0, velocity=0.0, pressure=1.0)
    initial_state[registered_variables.density_index] += 0.1 * np.sin(2.0 * np.pi * coordinate)
    params = SimulationParams(t_end=0.2, C_cfl=0.4, gamma=GAMMA)
    zone = RelaxationZoneParams(target_state=jnp.asarray(2.0 * initial_state), rate=jnp.zeros(num_cells))

    final_without_zone, _ = run(config_off, initial_state, params)
    final_with_zone, _ = run(config_on, initial_state, params._replace(relaxation_zone_params=zone))
    assert np.array_equal(final_with_zone, final_without_zone)


# -------------------------------------------------------------
# ============ ↓ absorption of an outgoing sound wave ↓ ============
# -------------------------------------------------------------


def _pulse_energy_left_in_box(with_zone: bool) -> float:
    """
    Run a right-moving sound pulse into a reflecting wall at x = 1 (with or
    without a zone in front of it) and return the squared density perturbation
    left in x < 0.6 once the reflection would have come back, relative to the
    initial one.
    """
    num_cells = 256
    density, pressure = 1.0, 1.0
    sound_speed = np.sqrt(GAMMA * pressure / density)
    coordinate = cell_centres(num_cells, 1.0)

    wall = BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY)
    settings = {"relaxation_zone_config": ZONE_ON} if with_zone else {}
    config = make_config(
        FINITE_VOLUME,
        FVSolverConfig(time_integrator=VL2),
        (num_cells,),
        boundary_settings=wall,
        **settings,
    )
    registered_variables = get_registered_variables(config)

    # A linear sound wave moving in +x: delta v = c delta rho / rho, delta p = c^2 delta rho.
    ambient_state = _uniform_state(registered_variables, num_cells, density, 0.0, pressure)
    perturbation = 1e-3 * np.exp(-0.5 * ((coordinate - 0.3) / 0.03) ** 2)
    initial_state = ambient_state.copy()
    initial_state[registered_variables.density_index] += perturbation
    initial_state[registered_variables.velocity_index] += sound_speed * perturbation / density
    initial_state[registered_variables.pressure_index] += sound_speed**2 * perturbation

    # The zone covers x > 0.6; its rate rises quadratically up to x = 0.8 so
    # that the edge of the zone does not reflect the pulse itself.
    ramp = np.clip((coordinate - 0.6) / 0.2, 0.0, 1.0) ** 2
    zone = RelaxationZoneParams(target_state=jnp.asarray(ambient_state), rate=jnp.asarray(50.0 * ramp))

    # Out to the wall (0.7) and back to x = 0.5.
    end_time = 1.2 / sound_speed
    params = SimulationParams(t_end=end_time, C_cfl=0.4, gamma=GAMMA, relaxation_zone_params=zone)
    final_state, _ = run(config, initial_state, params)

    inside = coordinate < 0.6
    final_perturbation = final_state[registered_variables.density_index] - density
    return float(np.sum(final_perturbation[inside] ** 2) / np.sum(perturbation**2))


def test_zone_absorbs_outgoing_wave():
    reflected_without_zone = _pulse_energy_left_in_box(with_zone=False)
    reflected_with_zone = _pulse_energy_left_in_box(with_zone=True)
    print(f"pulse energy back in the box: {reflected_without_zone:.3f} without, {reflected_with_zone:.2e} with zone")
    assert reflected_without_zone > 0.5
    assert reflected_with_zone < 1e-2 * reflected_without_zone


# -------------------------------------------------------------
# ============ ↓ configuration checks ↓ ============
# -------------------------------------------------------------


def test_zone_is_refused_for_mhd():
    config = make_config(FINITE_VOLUME, FVSolverConfig(time_integrator=VL2), (8,), mhd=True, relaxation_zone_config=ZONE_ON)
    registered_variables = get_registered_variables(config)
    with pytest.raises(ValueError, match="relaxation_zone"):
        finalize_config(config, (registered_variables.num_vars, 8))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
