"""
Finite volume on the spherical 1D grid (the split scheme, MUSCL_HANCOCK).

Geometry enters the scheme at two places: the distances from the centre of volume of a cell to its faces in the
reconstruction, and the area factors with the pressure source in the update. Three checks:

- a uniform gas at rest stays at rest (the pressure source balances the area factors on the pressure part of the
  momentum flux);
- the total mass and energy of a pulse that runs into the centre and back are conserved to rounding level (the
  area-weighted fluxes telescope);
- the wind bubble of ``examples/gallery/stellar_wind/stellar_wind_1d.py`` at 100 cells agrees with a stored profile.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_spherical_1d.py
"""

# general
import os

# testing
import pytest

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, REFLECTIVE_BOUNDARY, SPHERICAL
from astronomix._modules._stellar_wind.stellar_wind_options import EI

# astronomix containers
from astronomix import (
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
    WindParams,
)
from astronomix.option_classes import WindConfig

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

pytestmark = pytest.mark.x64

GAMMA = 5.0 / 3.0
STEPS = 200
SHELL_CELLS = 64
WIND_CELLS = 100
WIND_RECORD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "astronomix", "spherical_wind_1d_100.npz"
)


def _closed_sphere(density, velocity, pressure, time_step):
    """Initial and final state of ``STEPS`` fixed steps between reflective walls, and the cell volumes."""
    config = SimulationConfig(
        geometry=SPHERICAL,
        dimensionality=1,
        box_size=1.0,
        num_cells=SHELL_CELLS,
        boundary_settings=BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
        solver_mode=FINITE_VOLUME,
        mhd=False,
        fixed_timestep=True,
        num_timesteps=STEPS,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    initial_state = construct_primitive_state(
        config, registered_variables, density=density, velocity_x=velocity, gas_pressure=pressure
    )
    config = finalize_config(config, initial_state.shape)
    cell_volumes = np.asarray(get_helper_data(config).cell_volumes)
    params = SimulationParams(t_end=STEPS * time_step, gamma=GAMMA)
    final_state = np.asarray(time_integration(initial_state, config, params, registered_variables).final_state)
    return np.asarray(initial_state), final_state, cell_volumes


def _mass_and_energy(state, cell_volumes):
    density, velocity, pressure = state
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity**2
    return np.sum(density * cell_volumes), np.sum(energy * cell_volumes)


def _wind_bubble():
    """Final state of the wind bubble of the gallery example at ``WIND_CELLS`` cells and a tenth of its end time."""
    config = SimulationConfig(
        geometry=SPHERICAL,
        dimensionality=1,
        box_size=1.0,
        num_cells=WIND_CELLS,
        solver_mode=FINITE_VOLUME,
        mhd=False,
        progress_bar=False,
        wind_config=WindConfig(
            stellar_wind=True, num_injection_cells=20, trace_wind_density=False, wind_injection_scheme=EI
        ),
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    # The example's values in its code units (20 pc, 1 M_sun, 100 km/s): a star cluster of 5000 M_sun losing
    # 2.965e-3 of its mass per Myr at 3230 km/s into a gas of 2 m_p / cm^3 at p / k_B = 3e4 K / cm^3; end time 0.5 Myr.
    t_end = 0.1 * 2.5567804126
    params = SimulationParams(
        C_cfl=0.8,
        dt_max=0.1 * t_end,
        gamma=GAMMA,
        t_end=t_end,
        wind_params=WindParams(wind_mass_loss_rate=2.899153937, wind_final_velocity=32.3),
    )
    initial_state = construct_primitive_state(
        config,
        registered_variables,
        density=jnp.full(WIND_CELLS, 395.42442657),
        velocity_x=jnp.zeros(WIND_CELLS),
        gas_pressure=jnp.full(WIND_CELLS, 4.89598693),
    )
    config = finalize_config(config, initial_state.shape)
    return np.asarray(time_integration(initial_state, config, params, registered_variables).final_state)


def test_uniform_gas_stays_at_rest():
    """The area factors act on the pressure in the momentum flux; the pressure source gives it back."""
    cells = SHELL_CELLS
    _, final_state, _ = _closed_sphere(jnp.ones(cells), jnp.zeros(cells), jnp.full(cells, 0.6), 0.4 / cells)
    density, velocity, pressure = final_state
    deviation = max(np.max(np.abs(density - 1.0)), np.max(np.abs(velocity)), np.max(np.abs(pressure - 0.6)))
    print(f"largest deviation from the uniform state at rest after {STEPS} steps: {deviation:.3e}")
    assert deviation < 1e-12


def test_mass_and_energy_are_conserved():
    """A pulse runs into the centre and back; total mass and energy stay."""
    radius = (np.arange(SHELL_CELLS) + 0.5) / SHELL_CELLS
    # The pulse does not reach the outer wall within the run. A reflective wall of the radial grid is not exactly
    # closed for a sloped state: the centres of volume of the mirrored ghost cells are not mirror images, so the
    # two face states at the wall are not either.
    pulse = np.where(np.abs(radius - 0.2) < 0.1, np.cos(np.pi * (radius - 0.2) / 0.2) ** 2, 0.0)
    initial_state, final_state, cell_volumes = _closed_sphere(
        jnp.asarray(1.0 + 0.5 * pulse), jnp.zeros(SHELL_CELLS), jnp.asarray(0.6 + 0.6 * pulse), 1.25e-3
    )
    mass_start, energy_start = _mass_and_energy(initial_state, cell_volumes)
    mass_end, energy_end = _mass_and_energy(final_state, cell_volumes)
    mass_change, energy_change = mass_end / mass_start - 1.0, energy_end / energy_start - 1.0
    centre_change = np.max(np.abs(final_state[:, :3] - initial_state[:, :3]))
    print(
        f"relative change after {STEPS} steps: mass {mass_change:.3e}, energy {energy_change:.3e}"
        f" (the state at the centre moved by {centre_change:.2f})"
    )
    assert centre_change > 0.05
    assert abs(mass_change) < 1e-12 and abs(energy_change) < 1e-12


def test_wind_bubble_matches_the_record():
    """The wind bubble at low resolution against its stored profile."""
    # The record was written with the classic signal-speed bounds of the HLL solver (the smaller and the larger of
    # u -+ c of the two states). The Einfeldt bounds in use move the profile by up to 6e-3 at the shocks and by
    # 2.4e-4 in the mean; the tolerances sit above that.
    record = np.load(WIND_RECORD)["final_state"]
    final_state = _wind_bubble()
    scale = np.max(np.abs(record), axis=1)
    difference = np.abs(final_state - record)
    largest = np.max(difference.max(axis=1) / scale)
    mean = np.max(difference.mean(axis=1) / scale)
    print(f"wind bubble against the record (unit: largest value per variable): max {largest:.2e}, L1 {mean:.2e}")
    assert largest < 2e-2
    assert mean < 6e-4


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_uniform_gas_stays_at_rest()
        test_mass_and_energy_are_conserved()
        test_wind_bubble_matches_the_record()
