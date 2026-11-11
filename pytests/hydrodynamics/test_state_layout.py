"""
State-layout pytest: the rows the finite-difference solver appends behind the
gas / field variables.

The finite-difference state can carry, behind the gas variables (and for MHD
the cell-centred and interface magnetic fields), the dual-energy internal
energy ``g`` and a passive-scalar block. Checked:

* the registry's ``passive_scalar_index`` matches the layout
  ``construct_primitive_state`` fills, and ``magnetic_psi_active`` marks only
  the VL2 GLM-MHD layout;
* a ``g``-less input state with passive scalars active gets ``g`` inserted in
  front of the scalar block, so it evolves exactly like the full state;
* the initial ghost-cell fill of an FD MHD state with ``g`` and passive scalars
  leaves the interface field to the integrator and fills every trailing row
  (wrapped, mirrored or edge-extended like the gas rows);
* the passive-scalar ghost-cell fill works in 2D;
* the plain-text progress log of a new run starts fresh.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_state_layout.py
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

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    VL2,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)
from astronomix._fluid_equations._passive_scalars import _fill_scalar_ghost_cells
from astronomix._geometry.boundaries import _boundary_handler
from astronomix.time_stepping._progress_bar import (
    _reset_progress_log,
    _show_progress,
)
from astronomix.time_stepping._utils import _pad
from astronomix.time_stepping.time_integration import (
    _prepare_padded_state,
    _seed_internal_energy,
)

GAMMA = 5.0 / 3.0
PERIODIC = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)


def _blast_state(config, num_scalars):
    """A smooth over-pressured blob with ``num_scalars`` user scalars."""
    registered_variables = get_registered_variables(config)
    num_cells = config.num_cells
    x = (jnp.arange(num_cells) + 0.5) / num_cells - 0.5
    if config.dimensionality == 1:
        coordinates = (x,)
    else:
        coordinates = jnp.meshgrid(*([x] * config.dimensionality), indexing="ij")
    radius_squared = sum(coordinate ** 2 for coordinate in coordinates)
    density = 1.0 + 0.5 * jnp.exp(-radius_squared / 0.02)
    pressure = 0.1 + jnp.exp(-radius_squared / 0.01)
    scalars = jnp.stack(
        [jnp.clip(1.0 - radius_squared / 0.04, 0.0, 1.0) ** (k + 1) for k in range(num_scalars)]
    )
    fields = dict(
        density=density,
        velocity_x=0.1 * jnp.sin(2 * jnp.pi * coordinates[0]),
        gas_pressure=pressure,
    )
    if config.mhd:
        fields["velocity_y"] = jnp.zeros_like(density)
        fields["velocity_z"] = jnp.zeros_like(density)
        fields["magnetic_field_x"] = 0.3 + 0.1 * jnp.cos(2 * jnp.pi * coordinates[0])
        fields["magnetic_field_y"] = 0.2 + 0.1 * jnp.sin(2 * jnp.pi * coordinates[1])
        fields["magnetic_field_z"] = 0.1 * jnp.ones_like(density)
    elif config.dimensionality >= 2:
        fields["velocity_y"] = jnp.zeros_like(density)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        passive_scalars=scalars,
        **fields,
    )
    return finalize_config(config, state.shape), registered_variables, state


def test_registry_passive_scalars_and_psi_flags():
    """``passive_scalar_index`` locates the block construct_primitive_state fills,
    the last rows of the state; ``magnetic_psi_active`` marks only the VL2
    GLM-MHD layout."""
    for mhd, dual_energy in ((False, False), (False, True), (True, True)):
        config = SimulationConfig(
            dimensionality=2,
            num_cells=16,
            mhd=mhd,
            dual_energy=dual_energy,
            num_passive_scalars=2,
            boundary_settings=BoundarySettings(PERIODIC, PERIODIC),
        )
        config, registered_variables, state = _blast_state(config, num_scalars=2)
        assert registered_variables.num_passive_scalars == 2
        assert registered_variables.passive_scalar_index + 2 == registered_variables.num_vars
        assert state.shape[0] == registered_variables.num_vars
        # The scalars of ``_blast_state`` are the first and second power of one
        # profile that reaches 1 at the centre of the blob's support.
        scalars = np.asarray(state[registered_variables.passive_scalar_index:])
        assert scalars[0].max() > 0.9
        np.testing.assert_allclose(scalars[1], scalars[0] ** 2, rtol=1e-6)
        assert not registered_variables.magnetic_psi_active

    vl2 = get_registered_variables(
        SimulationConfig(solver_mode=FINITE_VOLUME, mhd=True, time_integrator=VL2)
    )
    assert vl2.magnetic_psi_active and vl2.magnetic_psi_index == 8
    no_scalars = get_registered_variables(SimulationConfig(solver_mode=FINITE_DIFFERENCE))
    assert no_scalars.passive_scalar_index == -1 and not no_scalars.passive_scalars_active


def test_g_less_state_with_passive_scalars():
    """A state without the dual-energy slot gets ``g`` inserted in front of the
    passive scalars and evolves bit for bit like the full state."""
    config = SimulationConfig(
        dimensionality=1,
        num_cells=32,
        dual_energy=True,
        num_passive_scalars=2,
        passive_scalar_bounds=((0.0, 1.0), (0.0, 1.0)),
        boundary_settings=PERIODIC,
        progress_bar=False,
    )
    config, registered_variables, state = _blast_state(config, num_scalars=2)
    internal_energy_index = registered_variables.internal_energy_index
    assert internal_energy_index < registered_variables.passive_scalar_index
    g_less_state = jnp.concatenate(
        [state[:internal_energy_index], state[internal_energy_index + 1:]],
        axis=0,
    )
    params = SimulationParams(gamma=GAMMA, C_cfl=0.3, t_end=0.01)

    seeded_full = np.asarray(_seed_internal_energy(state, params, registered_variables))
    seeded_g_less = np.asarray(_seed_internal_energy(g_less_state, params, registered_variables))
    np.testing.assert_array_equal(seeded_g_less, seeded_full)

    final_full = np.asarray(time_integration(state, config, params, registered_variables))
    final_g_less = np.asarray(time_integration(g_less_state, config, params, registered_variables))
    assert final_g_less.shape == (registered_variables.num_vars, 32)
    # The two inputs compile to different programs (one inserts the g row), so
    # a backend may round them differently; the layout error this guards
    # against overwrote the first user scalar with g.
    np.testing.assert_allclose(final_g_less, final_full, rtol=1e-5, atol=1e-7)


def test_fd_mhd_ghost_fill_with_trailing_rows():
    """FD MHD + ghost cells + dual energy + passive scalars: the gas rows get
    the gas boundary handler, the interface field is left to the integrator,
    and every row behind it (g, scalars) gets the scalar boundary fill."""
    boundaries = BoundarySettings(
        BoundarySettings1D(REFLECTIVE_BOUNDARY, OPEN_BOUNDARY),
        PERIODIC,
    )
    config = SimulationConfig(
        dimensionality=2,
        num_cells=16,
        mhd=True,
        dual_energy=True,
        num_passive_scalars=1,
        boundary_settings=boundaries,
    )
    config, registered_variables, state = _blast_state(config, num_scalars=1)
    params = SimulationParams(gamma=GAMMA)
    state = _seed_internal_energy(state, params, registered_variables)

    padded = np.asarray(_prepare_padded_state(state, config, params, registered_variables))
    edge_padded = _pad(state, config)
    interface_field_start = registered_variables.interface_magnetic_field_index.x
    trailing_rows_start = interface_field_start + 3
    assert trailing_rows_start == registered_variables.internal_energy_index

    expected_gas_rows = _boundary_handler(
        edge_padded[:interface_field_start],
        config,
        registered_variables,
        params,
    )
    expected_trailing_rows = _fill_scalar_ghost_cells(edge_padded[trailing_rows_start:], config)
    np.testing.assert_array_equal(padded[:interface_field_start], np.asarray(expected_gas_rows))
    np.testing.assert_array_equal(
        padded[interface_field_start:trailing_rows_start],
        np.asarray(edge_padded[interface_field_start:trailing_rows_start]),
    )
    np.testing.assert_array_equal(
        padded[trailing_rows_start:],
        np.asarray(expected_trailing_rows),
    )

    # The fill itself: mirrored at the reflective x wall, wrapped along y.
    num_ghost = config.num_ghost_cells
    trailing = padded[trailing_rows_start:]
    np.testing.assert_array_equal(
        trailing[:, :num_ghost],
        trailing[:, num_ghost:2 * num_ghost][:, ::-1],
    )
    np.testing.assert_array_equal(
        trailing[:, :, :num_ghost],
        trailing[:, :, -2 * num_ghost:-num_ghost],
    )
    # g in the ghost cells is the one derived from the filled pressure.
    np.testing.assert_allclose(
        padded[registered_variables.internal_energy_index],
        padded[registered_variables.pressure_index] / (GAMMA - 1.0),
        rtol=1e-6,
    )


def test_passive_scalars_with_ghost_cells_in_2d():
    """2D finite-difference run with ghost-cell boundaries and passive scalars."""
    boundaries = BoundarySettings(
        BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
        BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
    )
    config = SimulationConfig(
        dimensionality=2,
        num_cells=16,
        num_passive_scalars=1,
        passive_scalar_bounds=((0.0, 1.0),),
        boundary_settings=boundaries,
        progress_bar=False,
    )
    config, registered_variables, state = _blast_state(config, num_scalars=1)
    params = SimulationParams(gamma=GAMMA, C_cfl=0.3, t_end=0.005)
    final_state = np.asarray(time_integration(state, config, params, registered_variables))
    assert final_state.shape == state.shape
    assert np.all(np.isfinite(final_state))
    scalar = final_state[registered_variables.passive_scalar_index]
    assert scalar.min() >= 0.0 and scalar.max() <= 1.0


def test_progress_log_starts_fresh_for_a_new_run(capsys):
    """The plain-text log infers dt from consecutive callbacks; a new run must
    not difference its first time against the previous run's last one."""
    _reset_progress_log()
    for current_time in (0.5, 1.0):
        _show_progress(current_time, 1.0)
    first_run = capsys.readouterr().out.splitlines()
    assert "dt" not in first_run[0] and "dt = 5.000e-01" in first_run[1]

    # A new run announced by time_integration ...
    _reset_progress_log()
    _show_progress(0.25, 1.0)
    assert "dt" not in capsys.readouterr().out
    # ... and one detected from the clock going backwards.
    _show_progress(0.5, 1.0)
    _show_progress(0.1, 1.0)
    lines = capsys.readouterr().out.splitlines()
    assert "dt = 2.500e-01" in lines[0] and "dt" not in lines[1]
