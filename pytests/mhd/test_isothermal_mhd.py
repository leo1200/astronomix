"""
Barotropic MHD in the finite-volume framework: the Orszag-Tang vortex without an energy equation.

The state has no pressure slot (rho, v, B and, under constrained transport, the face field b); the pressure follows
from the density, ``p = c_s^2 rho`` (``ISOTHERMAL``) or ``p = K rho^Gamma`` (``POLYTROPIC``). Checked here:

* VL2 with constrained transport and the HLL or the local Lax-Friedrichs solver keeps the face divergence at
  round-off and conserves mass and momentum to round-off;
* a polytrope of exponent one is the isothermal gas, to rounding level;
* the GLM divergence cleaning runs the same case.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/mhd/test_isothermal_mhd.py
"""

# testing
import pytest

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSTRAINED_TRANSPORT,
    DONOR_CELL,
    FINITE_VOLUME,
    GLM,
    HLL,
    ISOTHERMAL,
    LAX_FRIEDRICHS,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    PIECEWISE_LINEAR,
    POLYTROPIC,
    VAN_LEER,
    VL2,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    SimulationConfig,
    StaticFloatVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

pytestmark = pytest.mark.x64

CELLS = 32
DX = 1.0 / CELLS
DENSITY = 25.0 / (36.0 * np.pi)
SOUND_SPEED = np.sqrt(0.6)  # the vortex's usual pressure 5 / (12 pi) over its density
T_END = 0.5


def _orszag_tang(constrained_transport):
    """The vortex without pressure slot: (rho, v, B, b) under constrained transport, else (rho, v, B, psi)."""
    b0 = 1.0 / np.sqrt(4.0 * np.pi)
    centre = (np.arange(CELLS) + 0.5) * DX
    x, y = np.meshgrid(centre, centre, indexing="ij")
    x_corner, y_corner = np.meshgrid(centre + 0.5 * DX, centre + 0.5 * DX, indexing="ij")
    # The face field is the curl of A_z on the cell corners, so its face divergence is zero identically.
    a_z = b0 * (np.cos(4 * np.pi * x_corner) / (4 * np.pi) + np.cos(2 * np.pi * y_corner) / (2 * np.pi))
    zero = np.zeros((CELLS, CELLS))
    faces = [(a_z - np.roll(a_z, 1, axis=1)) / DX, -(a_z - np.roll(a_z, 1, axis=0)) / DX, zero]
    cell_field = [0.5 * (faces[a] + np.roll(faces[a], 1, axis=a)) for a in range(2)] + [zero]
    rows = [np.full((CELLS, CELLS), DENSITY), -np.sin(2 * np.pi * y), np.sin(2 * np.pi * x), zero, *cell_field]
    rows += faces if constrained_transport else [zero]
    return jnp.asarray(np.stack(rows))


def _final_state(
    equation_of_state,
    divergence_control=CONSTRAINED_TRANSPORT,
    riemann_solver=HLL,
    reconstruction=PIECEWISE_LINEAR,
):
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        equation_of_state=equation_of_state,
        solver_config=FVSolverConfig(
            time_integrator=VL2,
            reconstruction=reconstruction,
            limiter=VAN_LEER,
            riemann_solver=riemann_solver,
            divergence_control=divergence_control,
        ),
        mhd=True,
        dimensionality=2,
        box_size=StaticFloatVector(1.0, 1.0, 1.0),
        boundary_settings=BoundarySettings(periodic, periodic, BoundarySettings1D()),
        backend_config=BackendConfig(backend=NATIVE_JAX),
        progress_bar=False,
    )
    state = _orszag_tang(divergence_control == CONSTRAINED_TRANSPORT)
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        C_cfl=0.3,
        t_end=T_END,
        isothermal_sound_speed=SOUND_SPEED,
        polytropic_constant=SOUND_SPEED**2,
        polytropic_exponent=1.0,
    )
    registered_variables = get_registered_variables(config)
    assert registered_variables.pressure_index == -1 and registered_variables.num_vars == state.shape[0]
    return np.asarray(time_integration(state, config, params, registered_variables)), np.asarray(state)


def _relative_face_divergence(state):
    """max |div b| dx / max |B| with the face field in rows 7 and 8."""
    divergence = sum((state[7 + a] - np.roll(state[7 + a], 1, axis=a)) / DX for a in range(2))
    return float(np.abs(divergence).max() * DX / np.abs(state[4:7]).max())


def test_isothermal_orszag_tang_with_constrained_transport():
    """Face divergence at round-off, mass and momentum conserved, for HLL and for donor-cell Lax-Friedrichs."""
    for name, riemann_solver, reconstruction in (
        ("HLL", HLL, PIECEWISE_LINEAR),
        ("LAX_FRIEDRICHS", LAX_FRIEDRICHS, DONOR_CELL),
    ):
        final, initial = _final_state(ISOTHERMAL, riemann_solver=riemann_solver, reconstruction=reconstruction)
        divergence = _relative_face_divergence(final)
        face_average_defect = max(
            float(np.abs(final[4 + a] - 0.5 * (final[7 + a] + np.roll(final[7 + a], 1, axis=a))).max())
            for a in range(2)
        )
        mass_change = abs(final[0].mean() / initial[0].mean() - 1.0)
        momentum = [(state[0] * state[1:4]).mean(axis=(1, 2)) for state in (initial, final)]
        momentum_change = float(np.abs(momentum[1] - momentum[0]).max())
        density_contrast = final[0].max() / final[0].min()
        print(
            f"{name}: relative face divergence {divergence:.2e}, face-average defect {face_average_defect:.2e}, "
            f"mass change {mass_change:.2e}, momentum change {momentum_change:.2e}, density in "
            f"[{final[0].min():.3f}, {final[0].max():.3f}]"
        )
        assert np.isfinite(final).all() and final[0].min() > 0.0
        assert divergence < 1e-12 and face_average_defect < 1e-14
        assert mass_change < 1e-12 and momentum_change < 1e-13
        # The vortex has evolved: shocks have formed in the gas that started uniform.
        assert density_contrast > 2.0


def test_polytrope_of_exponent_one_is_the_isothermal_gas():
    """``POLYTROPIC`` with Gamma = 1 and K = c_s^2 against ``ISOTHERMAL``, constrained transport."""
    isothermal, _ = _final_state(ISOTHERMAL)
    polytropic, _ = _final_state(POLYTROPIC)
    difference = float(np.abs(polytropic - isothermal).max())
    print(f"polytrope of exponent one against the isothermal gas: maximum difference {difference:.2e}")
    assert difference < 1e-12


def test_isothermal_orszag_tang_with_divergence_cleaning():
    """The GLM cleaning carries the layout without pressure slot (rho, v, B, psi)."""
    reference, _ = _final_state(ISOTHERMAL)
    final, initial = _final_state(ISOTHERMAL, divergence_control=GLM)
    mass_change = abs(final[0].mean() / initial[0].mean() - 1.0)
    density_difference = float(np.abs(final[0] - reference[0]).mean() / DENSITY)
    print(
        f"GLM: mass change {mass_change:.2e}, density in [{final[0].min():.3f}, {final[0].max():.3f}], "
        f"mean density difference to the constrained-transport run {density_difference:.2e}"
    )
    assert np.isfinite(final).all() and final[0].min() > 0.0
    assert mass_change < 1e-12
    # Same flow: the two divergence controls differ by truncation error only (3e-2 at 32 cells).
    assert density_difference < 0.1


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_isothermal_orszag_tang_with_constrained_transport()
        test_polytrope_of_exponent_one_is_the_isothermal_gas()
        test_isothermal_orszag_tang_with_divergence_cleaning()
