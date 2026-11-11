"""
Positivity-preserving WENO with the shear waves at their own splitting speed.

The solver splits every field of an ideal gas with the common speed
(``mass_free_modes``), but the interface flux supports fields at their own
speed, as for an isothermal gas. For the shear waves of an ideal gas this
shifts the split states along the shear eigenvector of the interface. In the forward-Euler update of a cell the inflowing face value then
carries the shift jump, the neighbour's shift minus the cell's own at that
interface, and the limiter has to keep that state admissible (see
``_weno_positivity.py``). One interface with a strong jump of the tangential
velocity next to cold gas, whose face value deviates in energy from the split
state, is checked for the two states of the update: the inflow state of the
cell and the mirror its neighbour keeps.

    JAX_PLATFORMS=cpu python -m pytest pytests/hydrodynamics/test_positivity_shear_shift.py
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import get_registered_variables
from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    positivity_preserving_flux_local,
    positivity_preserving_interface_flux,
)

jax.config.update("jax_enable_x64", True)

GAMMA = 5.0 / 3.0


def _conserved(density, velocity_x, velocity_y, pressure):
    """The conserved state (density, momentum_x, momentum_y, energy)."""
    return np.array([
        density,
        density * velocity_x,
        density * velocity_y,
        pressure / (GAMMA - 1.0) + 0.5 * density * (velocity_x**2 + velocity_y**2),
    ])


def _pressure(state):
    """The gas pressure of a conserved state."""
    density, momentum_x, momentum_y, energy = state
    return (GAMMA - 1.0) * (energy - 0.5 * (momentum_x**2 + momentum_y**2) / density)


def _flux_x(state):
    """The physical flux along x."""
    density, momentum_x, momentum_y, energy = state
    velocity_x = momentum_x / density
    pressure = _pressure(state)
    return np.array([
        momentum_x,
        momentum_x * velocity_x + pressure,
        momentum_y * velocity_x,
        (energy + pressure) * velocity_x,
    ])


def _shear_interface():
    """
    One interface: cell i at rest, its right neighbour cold and fast in both
    directions, the minus face value with less energy than its split state.

    Returns:
        A dict with the two cells, their fluxes, the common speed, the shifts,
        the split states and the face values.
    """
    own = _conserved(1.0, 0.0, 0.0, 1.0)
    neighbour = _conserved(1.0, 6.0, 8.0, 0.02)
    sound_speeds = [np.sqrt(GAMMA * _pressure(q) / q[0]) for q in (own, neighbour)]
    common_speed = max(abs(q[1] / q[0]) + c for q, c in zip((own, neighbour), sound_speeds))
    shear_speed = max(abs(q[1] / q[0]) for q in (own, neighbour))

    # The shear field of the interface basis (momentum-weighted velocity) at
    # its own speed shifts the split states by kappa R (L q).
    velocity_y = (own[2] + neighbour[2]) / (own[0] + neighbour[0])
    right_vector = np.array([0.0, 0.0, 1.0, velocity_y])
    left_row = np.array([-velocity_y, 0.0, 1.0, 0.0])
    kappa = shear_speed / common_speed - 1.0
    own_shift = kappa * (left_row @ own) * right_vector
    neighbour_shift = kappa * (left_row @ neighbour) * right_vector

    plus_unshifted = own + _flux_x(own) / common_speed
    minus_unshifted = neighbour - _flux_x(neighbour) / common_speed
    plus_owner = plus_unshifted + own_shift
    minus_owner = minus_unshifted + neighbour_shift
    # The plus face value is first order; the minus face value has less energy.
    deviation = np.array([0.0, 0.0, 0.0, -1.0])
    return dict(
        own=own,
        neighbour=neighbour,
        common_speed=common_speed,
        own_shift=own_shift,
        neighbour_shift=neighbour_shift,
        minus_unshifted=minus_unshifted,
        plus_owner=plus_owner,
        minus_owner=minus_owner,
        deviation=deviation,
        plus_face_flux=0.5 * common_speed * plus_owner,
        minus_face_flux=-0.5 * common_speed * (minus_owner + deviation),
    )


def _array_form_flux(case):
    """The interface flux of the array form (native kernel)."""
    config = SimulationConfig(dimensionality=2)
    registered_variables = get_registered_variables(config)
    slots = [
        registered_variables.density_index,
        registered_variables.momentum_index.x,
        registered_variables.momentum_index.y,
        registered_variables.energy_index,
    ]

    def as_field(*cells):
        """A (variables, Nx = 2, Ny = 1) array of the given cell vectors."""
        field = np.zeros((4, 2, 1))
        for index, vector in enumerate(cells):
            field[slots, index, 0] = vector
        return jnp.asarray(field)

    interface_flux = positivity_preserving_interface_flux(
        as_field(case["own"], case["neighbour"]),
        as_field(_flux_x(case["own"]), _flux_x(case["neighbour"])),
        jnp.full((2, 1), case["common_speed"]),
        as_field(case["plus_face_flux"]),
        as_field(case["minus_face_flux"]),
        as_field(case["own_shift"]),
        as_field(case["neighbour_shift"]),
        SimulationParams(gamma=GAMMA),
        config,
        registered_variables,
        axis=0,
    )
    return np.asarray(interface_flux)[slots, 0, 0]


def test_inflow_state_carries_shift_jump():
    """The limited minus face value keeps the update's inflow state and
    mirror admissible, with the shift jump in the inflow state."""

    case = _shear_interface()
    flux = _array_form_flux(case)
    alpha = case["common_speed"]
    deviation = case["deviation"]

    # F = (alpha/2) w^+ - (alpha/2) (w^- + theta d): recover theta from the energy.
    theta = -(2.0 * flux[3] / alpha - case["plus_owner"][3] + case["minus_owner"][3]) / deviation[3]
    assert 0.0 <= theta <= 1.0

    inflow_state = (
        case["minus_unshifted"] + case["neighbour_shift"] - case["own_shift"] + theta * deviation
    )
    mirror = case["minus_unshifted"] - theta * deviation
    assert inflow_state[0] > 0.0 and _pressure(inflow_state) > 0.0
    assert mirror[0] > 0.0 and _pressure(mirror) > 0.0
    # The face value alone (without the cell's own shift) would admit a much
    # larger theta here; the update needs the small one.
    assert _pressure(case["minus_owner"] + 0.2 * deviation) > 0.0
    assert theta < 0.01


def test_local_form_matches_array_form():
    """The per-cell form of the Pallas kernels gives the array form's flux."""

    case = _shear_interface()

    def as_tuple(vector):
        return tuple(jnp.asarray(component) for component in vector)

    local_flux = positivity_preserving_flux_local(
        as_tuple(case["own"]),
        as_tuple(case["neighbour"]),
        as_tuple(_flux_x(case["own"])),
        as_tuple(_flux_x(case["neighbour"])),
        as_tuple(case["plus_face_flux"]),
        as_tuple(case["minus_face_flux"]),
        as_tuple(case["own_shift"]),
        as_tuple(case["neighbour_shift"]),
        jnp.asarray(case["common_speed"]),
        GAMMA - 1.0,
        SimulationParams().minimum_density,
        SimulationParams().minimum_pressure,
    )
    np.testing.assert_allclose(
        np.array([float(component) for component in local_flux]),
        _array_form_flux(case),
        rtol=1e-12,
        atol=1e-12,
    )
