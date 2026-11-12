"""
The polytropic equation of state (``p = K rho^Gamma``, no energy equation) in the finite-volume framework.

* A polytrope of exponent one with ``K = c_s^2`` is the isothermal gas, to rounding level.
* A sound wave with ``Gamma = 5/3`` runs at ``sqrt(Gamma K rho^(Gamma - 1))`` and converges at second order.
* A smooth flow of uniform entropy stays isentropic until a shock forms: there the polytrope and the ideal gas
  with ``gamma = Gamma`` and ``p = K rho^gamma`` solve the same equations, and their difference is truncation
  error that falls at second order.
* A soft polytrope (``Gamma = 0.7``, the dense-gas regime of Federrath and Banerjee 2015) keeps colliding and
  receding streams positive and finite.

Each point is checked for ``VL2``, ``RK2_SSP`` and ``MUSCL_HANCOCK`` with the HLL solver.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_polytropic.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, OPEN_BOUNDARY, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import (
    HLL,
    IDEAL_GAS,
    ISOTHERMAL,
    MUSCL_HANCOCK,
    POLYTROPIC,
    RK2_SSP,
    VAN_LEER,
    VL2,
)

# astronomix containers
from astronomix import (
    BoundarySettings1D,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

INTEGRATORS = (("VL2", VL2), ("RK2_SSP", RK2_SSP), ("MUSCL_HANCOCK", MUSCL_HANCOCK))
POLYTROPIC_CONSTANT = 0.9
POLYTROPIC_EXPONENT = 5.0 / 3.0


def _cell_centres(cells):
    return (jnp.arange(cells) + 0.5) / cells


def _final_state(time_integrator, equation_of_state, boundary, density, velocity, t_end, **params):
    """Final primitive state (density, velocity and, for the ideal gas, pressure) of a 1D run with HLL."""
    config = SimulationConfig(
        dimensionality=1,
        box_size=1.0,
        boundary_settings=BoundarySettings1D(boundary, boundary),
        solver_mode=FINITE_VOLUME,
        equation_of_state=equation_of_state,
        solver_config=FVSolverConfig(time_integrator=time_integrator, riemann_solver=HLL, limiter=VAN_LEER),
        mhd=False,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    fields = dict(density=density, velocity_x=velocity)
    if equation_of_state == IDEAL_GAS:
        fields["gas_pressure"] = params["polytropic_constant"] * density ** params["gamma"]
    initial_state = construct_primitive_state(config, registered_variables, **fields)
    config = finalize_config(config, initial_state.shape)
    result = time_integration(
        initial_state, config, SimulationParams(t_end=t_end, C_cfl=0.4, **params), registered_variables
    )
    return np.asarray(result.final_state)


def test_polytrope_of_exponent_one_is_the_isothermal_gas():
    """A shock tube under ``POLYTROPIC`` with Gamma = 1 and K = c_s^2 against ``ISOTHERMAL``."""
    x = _cell_centres(100)
    density, velocity = jnp.where(x < 0.5, 1.0, 0.25), jnp.zeros(100)
    for name, time_integrator in INTEGRATORS:
        isothermal = _final_state(
            time_integrator, ISOTHERMAL, OPEN_BOUNDARY, density, velocity, 0.15, isothermal_sound_speed=0.8
        )
        polytropic = _final_state(
            time_integrator,
            POLYTROPIC,
            OPEN_BOUNDARY,
            density,
            velocity,
            0.15,
            polytropic_constant=0.8**2,
            polytropic_exponent=1.0,
        )
        difference = float(np.abs(polytropic - isothermal).max())
        print(f"{name}: polytrope of exponent one against the isothermal gas, maximum difference {difference:.2e}")
        assert isothermal[0].max() > 0.4  # the tube has evolved
        assert difference < 1e-13


def _sound_wave_error_and_speed(time_integrator, cells):
    """One period of a right-running sound wave: L1 density error over the amplitude, and the phase speed."""
    amplitude = 1e-6
    sound_speed = np.sqrt(POLYTROPIC_EXPONENT * POLYTROPIC_CONSTANT)  # at the mean density one
    x = _cell_centres(cells)
    wave = amplitude * jnp.sin(2.0 * jnp.pi * x)
    final = _final_state(
        time_integrator,
        POLYTROPIC,
        PERIODIC_BOUNDARY,
        1.0 + wave,
        sound_speed * wave,
        1.0 / sound_speed,
        polytropic_constant=POLYTROPIC_CONSTANT,
        polytropic_exponent=POLYTROPIC_EXPONENT,
    )
    perturbation = (final[0] - 1.0) / amplitude
    error = float(np.abs(perturbation - np.sin(2.0 * np.pi * np.asarray(x))).mean())
    # The wave sin(k (x - c t)) has moved on by one wavelength plus the lag the phase of the first mode shows.
    lag = np.arctan2(
        np.mean(perturbation * np.cos(2.0 * np.pi * np.asarray(x))),
        np.mean(perturbation * np.sin(2.0 * np.pi * np.asarray(x))),
    )
    return error, sound_speed * (1.0 - lag / (2.0 * np.pi)), sound_speed


def test_polytropic_sound_wave_speed_and_second_order():
    """Phase speed ``sqrt(Gamma K rho^(Gamma - 1))`` and second-order convergence, Gamma = 5/3."""
    for name, time_integrator in INTEGRATORS:
        coarse, _, _ = _sound_wave_error_and_speed(time_integrator, 32)
        fine, speed, sound_speed = _sound_wave_error_and_speed(time_integrator, 64)
        order = np.log2(coarse / fine)
        print(
            f"{name}: L1 error / amplitude {coarse:.3e} (32 cells), {fine:.3e} (64 cells), order {order:.2f}; "
            f"phase speed {speed:.5f} against {sound_speed:.5f} (relative {speed / sound_speed - 1.0:+.1e})"
        )
        assert abs(speed / sound_speed - 1.0) < 2e-3
        assert fine < 2e-2
        assert order > 1.7


def _isentropic_difference(time_integrator, cells):
    """L1 density and velocity difference of the polytrope and the ideal gas in a smooth flow of uniform entropy."""
    density = 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * _cell_centres(cells))
    arguments = (PERIODIC_BOUNDARY, density, jnp.zeros(cells), 0.15)
    polytropic = _final_state(
        time_integrator,
        POLYTROPIC,
        *arguments,
        polytropic_constant=POLYTROPIC_CONSTANT,
        polytropic_exponent=POLYTROPIC_EXPONENT,
    )
    ideal = _final_state(
        time_integrator, IDEAL_GAS, *arguments, polytropic_constant=POLYTROPIC_CONSTANT, gamma=POLYTROPIC_EXPONENT
    )
    return float(np.abs(polytropic[0] - ideal[0]).mean()), float(np.abs(polytropic[1] - ideal[1]).mean())


def test_polytrope_follows_the_ideal_gas_in_isentropic_flow():
    """Before a shock forms the two equations of state differ by truncation error: second order."""
    for name, time_integrator in INTEGRATORS:
        coarse = _isentropic_difference(time_integrator, 64)
        fine = _isentropic_difference(time_integrator, 128)
        density_order, velocity_order = np.log2(coarse[0] / fine[0]), np.log2(coarse[1] / fine[1])
        print(
            f"{name}: L1 density difference {coarse[0]:.3e} (64 cells), {fine[0]:.3e} (128 cells), order "
            f"{density_order:.2f}; velocity {coarse[1]:.3e}, {fine[1]:.3e}, order {velocity_order:.2f}"
        )
        assert fine[0] < 1e-4 and fine[1] < 1e-4
        assert density_order > 1.7 and velocity_order > 1.7


def test_soft_polytrope_with_shocks_stays_positive():
    """Gamma = 0.7: colliding streams (two shocks) and receding streams (two rarefactions) stay positive and finite."""
    cells = 200
    x = _cell_centres(cells)
    density = jnp.where(x < 0.5, 1.0, 0.1)
    for name, time_integrator in INTEGRATORS:
        for flow, speed in (("colliding", 2.0), ("receding", -2.0)):
            final = _final_state(
                time_integrator,
                POLYTROPIC,
                OPEN_BOUNDARY,
                density,
                jnp.where(x < 0.5, speed, -speed),
                0.1,
                polytropic_constant=1.0,
                polytropic_exponent=0.7,
            )
            print(f"{name}, {flow} streams, Gamma = 0.7: density in [{final[0].min():.3e}, {final[0].max():.3e}]")
            assert np.all(np.isfinite(final))
            assert final[0].min() > 0.0
            # The streams have met (compression) or parted (rarefaction).
            assert final[0].max() > 1.5 if flow == "colliding" else final[0].min() < 0.05


if __name__ == "__main__":
    test_polytrope_of_exponent_one_is_the_isothermal_gas()
    test_polytropic_sound_wave_speed_and_second_order()
    test_polytrope_follows_the_ideal_gas_in_isentropic_flow()
    test_soft_polytrope_with_shocks_stays_positive()
