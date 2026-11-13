"""
Cooling in every scheme family: the cooling curve of a uniform box at rest.

A box at rest and of uniform density cools as the single gas parcel of the ordinary differential equation
``dT/dt = rate(rho, T)`` of the cooling module; its solution at the end time (fourth-order Runge-Kutta with 20000
steps) is the reference. The piecewise power-law curve of ``pytests/hydrodynamics/test_cooling_solver.py`` halves
the temperature over the run. Cooling enters the stages of every scheme (``_modules/_explicit_sources.py``); with
the explicit update the time step follows the cooling time, and every family reaches the reference to 1e-4. The
implicit update is first order in the step and checked to 1e-2.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/modules/test_cooling_across_schemes.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import jax.numpy as jnp
import numpy as np
import pytest

# astronomix constants
from astronomix._modules._cooling.cooling_options import (
    EXPLICIT_COOLING,
    IMPLICIT_COOLING,
    PIECEWISE_POWER_LAW,
)

# astronomix containers
from astronomix import SimulationParams
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    PiecewisePowerLawParams,
)

# astronomix functions
from astronomix import get_registered_variables
from astronomix._modules._cooling._cooling import dtemperature_dt, get_temperature_from_pressure

# the scheme families of the module tests
from _schemes import HYDRO_SCHEMES, make_config, run

GAMMA = 5.0 / 3.0
HYDROGEN_MASS_FRACTION = 0.70
METAL_MASS_FRACTION = 0.02
# density and pressure of the box: the cooling time is about one time unit
DENSITY = 300.0
PRESSURE = 300.0
T_END = 1.0
CELLS = 8
TOLERANCE = {EXPLICIT_COOLING: 1e-4, IMPLICIT_COOLING: 1e-2}


def _curve():
    """The piecewise power-law cooling curve of ``test_cooling_solver.py``."""
    log10_temperature = np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.0, 0.0, 1.0])
    log10_cooling_function = np.array([-3.0, -1.5, -1.0, -1.3, -1.8, -2.3, -2.4, -2.1])
    slopes = np.diff(log10_cooling_function) / np.diff(log10_temperature)
    slopes = np.append(slopes, slopes[-1])
    curve_config = CoolingCurveConfig(cooling_curve_type=PIECEWISE_POWER_LAW)
    curve_params = PiecewisePowerLawParams(
        log10_T_table=jnp.asarray(log10_temperature),
        log10_Lambda_table=jnp.asarray(log10_cooling_function),
        alpha_table=jnp.asarray(slopes),
        Y_table=jnp.zeros(len(log10_temperature)),
        reference_temperature=10.0,
    )
    return curve_config, curve_params


CURVE_CONFIG, CURVE_PARAMS = _curve()
COOLING_PARAMS = CoolingParams(
    hydrogen_mass_fraction=HYDROGEN_MASS_FRACTION,
    metal_mass_fraction=METAL_MASS_FRACTION,
    floor_temperature=1e-4,
    resolution_limiter_alpha=0.0,
    max_cooling_fraction=0.0,
    clamp_to_floor=True,
    cooling_curve_params=CURVE_PARAMS,
)


def _temperature(density, pressure):
    return float(
        get_temperature_from_pressure(
            jnp.asarray(density), jnp.asarray(pressure), HYDROGEN_MASS_FRACTION, METAL_MASS_FRACTION
        )
    )


def _reference_temperature():
    """The temperature of the gas parcel at the end time (fourth-order Runge-Kutta, 20000 steps)."""

    def rate(temperature):
        return float(
            dtemperature_dt(
                jnp.asarray(DENSITY),
                jnp.asarray(temperature),
                HYDROGEN_MASS_FRACTION,
                METAL_MASS_FRACTION,
                GAMMA,
                CURVE_CONFIG,
                CURVE_PARAMS,
                heating_rate=COOLING_PARAMS.heating_rate,
            )
        )

    temperature = _temperature(DENSITY, PRESSURE)
    num_steps = 20000
    step = T_END / num_steps
    for _ in range(num_steps):
        k1 = rate(temperature)
        k2 = rate(temperature + 0.5 * step * k1)
        k3 = rate(temperature + 0.5 * step * k2)
        k4 = rate(temperature + step * k3)
        temperature += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return temperature


@pytest.fixture(scope="module")
def reference_temperature():
    return _reference_temperature()


@pytest.mark.parametrize("method", [EXPLICIT_COOLING, IMPLICIT_COOLING], ids=["explicit", "implicit"])
@pytest.mark.parametrize("name", list(HYDRO_SCHEMES))
def test_cooling_curve(name, method, reference_temperature):
    cooling_config = CoolingConfig(cooling=True, cooling_method=method, cooling_curve_config=CURVE_CONFIG)
    config = make_config(*HYDRO_SCHEMES[name], (CELLS,), cooling_config=cooling_config)
    registered_variables = get_registered_variables(config)
    initial_state = np.zeros((registered_variables.num_vars, CELLS))
    initial_state[registered_variables.density_index] = DENSITY
    initial_state[registered_variables.pressure_index] = PRESSURE
    params = SimulationParams(t_end=T_END, C_cfl=0.4, gamma=GAMMA, cooling_params=COOLING_PARAMS)
    final_state, _ = run(config, initial_state, params)
    pressure = final_state[registered_variables.pressure_index]
    temperature = _temperature(final_state[registered_variables.density_index].mean(), pressure.mean())
    deviation = temperature / reference_temperature - 1.0
    initial_temperature = _temperature(DENSITY, PRESSURE)
    print(f"{name}: T_end / T_0 = {temperature / initial_temperature:.4f}, off the reference by {deviation:+.2e}")
    assert reference_temperature / initial_temperature < 0.6
    assert abs(deviation) < TOLERANCE[method]
    assert np.ptp(pressure) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
