"""
The preset ``BEATTIE_REFERENCE`` as a whole (dimensionally split MUSCL-Hancock on primitive variables,
monotonised-central slopes, the HLL5R solver, Powell's term in the induction equation only, parabolic cleaning),
and what ``finalize_config`` admits with it:

* the settings of the preset, and the combinations that are rejected, each with its message;
* the time step bounds the signal speeds of the solver, which exceed the fast speed of the cells;
* the Brio-Wu shock tube at first order stays positive and keeps its normal field;
* the isothermal Orszag-Tang vortex at 32^2 stays positive and conserves the mass, and the cleaning lowers the
  divergence error;
* a circularly polarised Alfven wave oblique to the grid converges at second order.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/mhd/test_reference_preset.py
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
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSTRAINED_TRANSPORT,
    DONOR_CELL,
    DOUBLE_MINMOD,
    FINITE_VOLUME,
    GLM,
    HLL,
    HLL5R,
    HLLD,
    IDEAL_GAS,
    INDUCTION_SOURCE,
    ISOTHERMAL,
    MUSCL_HANCOCK,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    PIECEWISE_LINEAR,
    POLYTROPIC,
    RK2_SSP,
    SPHERICAL,
    VL2,
)

# astronomix containers
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig
from astronomix.option_classes.presets import BEATTIE_REFERENCE
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    CoolingConfig,
    DiffusionConfig,
    GravityConfig,
    SimulationConfig,
    SnapshotSettings,
    StaticFloatVector,
    WindConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_volume._interface_frame import _barotropic_law, _interface_frame_indices, _mhd_face_state
from astronomix._finite_volume._riemann._hll5r import hll5r_signal_speed
from astronomix._finite_volume._timestep_estimation._timestep_estimator import _cfl_time_step
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import _prepare_padded_state, time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

COURANT_NUMBER = 0.8
NO_CLEANING = BEATTIE_REFERENCE._replace(parabolic_cleaning=False)
FIRST_ORDER = BEATTIE_REFERENCE._replace(reconstruction=DONOR_CELL)


def _configuration(shape, equation_of_state=ISOTHERMAL, solver_config=BEATTIE_REFERENCE, **settings):
    """A box of ``shape`` cells, periodic unless ``boundary`` says otherwise, that returns its final state."""
    boundary = settings.pop("boundary", PERIODIC_BOUNDARY)
    sides = BoundarySettings1D(boundary, boundary)
    box_size = settings.pop("box_size", 1.0)
    one_dimensional = len(shape) == 1 and "dimensionality" not in settings
    defaults = dict(
        solver_mode=FINITE_VOLUME,
        equation_of_state=equation_of_state,
        solver_config=solver_config,
        mhd=True,
        dimensionality=len(shape),
        box_size=box_size if one_dimensional else StaticFloatVector(box_size, box_size, box_size),
        boundary_settings=sides if one_dimensional else BoundarySettings(sides, sides, sides),
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    return SimulationConfig(**{**defaults, **settings})


def _finalized(shape=(16, 16, 16), **settings):
    """Finalise a configuration of the preset with single settings replaced."""
    config = _configuration(shape, **settings)
    num_variables = (7 if config.mhd else 4) + (config.equation_of_state == IDEAL_GAS)
    return finalize_config(config, (num_variables, *shape))


def _run(config, fields, t_end, gamma=5.0 / 3.0, sound_speed=1.0):
    """Initial state, final state and variable registry of a run to ``t_end`` at the Courant number 0.8."""
    registered_variables = get_registered_variables(config)
    initial_state = construct_primitive_state(config, registered_variables, **fields)
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(C_cfl=COURANT_NUMBER, t_end=t_end, gamma=gamma, isothermal_sound_speed=sound_speed)
    result = time_integration(initial_state, config, params, registered_variables)
    return np.asarray(initial_state), np.asarray(result.final_state), registered_variables


def _orszag_tang_fields(cells):
    """The isothermal Orszag-Tang vortex on the unit square, sampled at the cell centres."""
    centres = (np.arange(cells) + 0.5) / cells
    x, y = np.meshgrid(centres, centres, indexing="ij")
    field_strength = 1.0 / np.sqrt(4.0 * np.pi)
    return dict(
        density=jnp.full((cells, cells), 25.0 / (36.0 * np.pi)),
        velocity_x=jnp.asarray(-np.sin(2.0 * np.pi * y)),
        velocity_y=jnp.asarray(np.sin(2.0 * np.pi * x)),
        velocity_z=jnp.zeros((cells, cells)),
        magnetic_field_x=jnp.asarray(-field_strength * np.sin(2.0 * np.pi * y)),
        magnetic_field_y=jnp.asarray(field_strength * np.sin(4.0 * np.pi * x)),
        magnetic_field_z=jnp.zeros((cells, cells)),
    )


# -------------------------------------------------------------
# ================== ↓ Preset and rules ↓ =====================
# -------------------------------------------------------------


def test_settings_of_the_preset():
    assert BEATTIE_REFERENCE.time_integrator == MUSCL_HANCOCK
    assert BEATTIE_REFERENCE.reconstruction == PIECEWISE_LINEAR
    assert BEATTIE_REFERENCE.limiter == DOUBLE_MINMOD
    assert BEATTIE_REFERENCE.riemann_solver == HLL5R
    assert BEATTIE_REFERENCE.divergence_control == INDUCTION_SOURCE
    assert BEATTIE_REFERENCE.parabolic_cleaning
    assert not BEATTIE_REFERENCE.first_order_flux_correction


@pytest.mark.parametrize(
    "settings, message",
    [
        # The solver and the source are the two halves of one discretisation.
        (dict(solver_config=BEATTIE_REFERENCE._replace(riemann_solver=HLL)), "belong together"),
        (dict(solver_config=BEATTIE_REFERENCE._replace(divergence_control=GLM)), "belong together"),
        (
            dict(solver_config=BEATTIE_REFERENCE._replace(time_integrator=VL2, divergence_control=GLM)),
            "belong together",
        ),
        (
            dict(
                solver_config=BEATTIE_REFERENCE._replace(time_integrator=VL2, divergence_control=CONSTRAINED_TRANSPORT)
            ),
            "belong together",
        ),
        (
            dict(
                solver_config=BEATTIE_REFERENCE._replace(
                    time_integrator=VL2, riemann_solver=HLLD, divergence_control=GLM, parabolic_cleaning=True
                )
            ),
            "parabolic_cleaning is the divergence cleaning of divergence_control=INDUCTION_SOURCE",
        ),
        # The scheme is the MHD form of the split scheme on a Cartesian grid.
        (dict(mhd=False), "requires mhd=True"),
        (dict(solver_config=BEATTIE_REFERENCE._replace(time_integrator=VL2)), "use time_integrator=MUSCL_HANCOCK"),
        (dict(solver_config=BEATTIE_REFERENCE._replace(time_integrator=RK2_SSP)), "use time_integrator=MUSCL_HANCOCK"),
        # (For spherical geometry the framework sets the HLL solver; the source is rejected before that counts.)
        (dict(shape=(16,), geometry=SPHERICAL), "Cartesian geometry only"),
        (dict(equation_of_state=POLYTROPIC), "use equation_of_state=IDEAL_GAS or ISOTHERMAL"),
        (
            dict(solver_config=BEATTIE_REFERENCE._replace(first_order_flux_correction=True)),
            "switch first_order_flux_correction off",
        ),
        # Ideal MHD only.
        (dict(gravity_config=GravityConfig(self_gravity=True)), "gravity_config .* is not implemented"),
        (dict(cooling_config=CoolingConfig(cooling=True)), "cooling_config.cooling is not implemented"),
        (dict(diffusion_config=DiffusionConfig(viscosity=True)), "diffusion_config.viscosity is not implemented"),
        (dict(diffusion_config=DiffusionConfig(resistivity=True)), "diffusion_config.resistivity is not implemented"),
        (
            dict(diffusion_config=DiffusionConfig(thermal_conduction=True)),
            "diffusion_config.thermal_conduction is not implemented",
        ),
        (dict(wind_config=WindConfig(stellar_wind=True)), "wind_config.stellar_wind is not implemented"),
        (dict(wind_config=WindConfig(trace_wind_density=True)), "wind_config.trace_wind_density is not implemented"),
        (
            dict(turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=True)),
            "turbulent_forcing_config.turbulent_forcing is not implemented",
        ),
    ],
)
def test_rejected_with_the_induction_source(settings, message):
    equation_of_state = settings.pop("equation_of_state", IDEAL_GAS)
    with pytest.raises(ValueError, match=message):
        _finalized(equation_of_state=equation_of_state, **settings)


def test_split_scheme_without_the_induction_source_carries_no_field():
    solver_config = BEATTIE_REFERENCE._replace(riemann_solver=HLL, divergence_control=GLM, parabolic_cleaning=False)
    with pytest.raises(ValueError, match="preset BEATTIE_REFERENCE"):
        _finalized(solver_config=solver_config)


def test_accepted_with_the_induction_source():
    """Both equations of state in one to three dimensions, at first and second order, with and without cleaning."""
    for equation_of_state in (IDEAL_GAS, ISOTHERMAL):
        for shape in ((16,), (16, 16), (16, 16, 16)):
            for solver_config in (BEATTIE_REFERENCE, NO_CLEANING, FIRST_ORDER):
                config = _finalized(shape, equation_of_state=equation_of_state, solver_config=solver_config)
                assert config.solver_config.riemann_solver == HLL5R
                # One sweep and the cleaning reach two cells each; the ghost cells are refreshed in between.
                assert config.num_ghost_cells == 2
    assert _finalized((16, 16), boundary=OPEN_BOUNDARY).num_ghost_cells == 2


# -------------------------------------------------------------
# ====================== ↓ Time step ↓ ========================
# -------------------------------------------------------------


def _fast_speed(state, axis, sound_speed, registered_variables):
    density = state[registered_variables.density_index]
    field = state[registered_variables.magnetic_index.x : registered_variables.magnetic_index.x + 3]
    alfven_squared = jnp.sum(field**2, axis=0) / density
    normal_squared = field[axis - 1] ** 2 / density
    total = sound_speed**2 + alfven_squared
    return jnp.sqrt(0.5 * (total + jnp.sqrt(total**2 - 4.0 * sound_speed**2 * normal_squared)))


def test_time_step_bounds_the_signal_speeds_of_the_solver():
    cells, sound_speed = 32, 1.0
    config = _configuration((cells, cells))
    registered_variables = get_registered_variables(config)
    state = construct_primitive_state(config, registered_variables, **_orszag_tang_fields(cells))
    config = finalize_config(config, state.shape)
    params = SimulationParams(C_cfl=COURANT_NUMBER, isothermal_sound_speed=sound_speed)
    padded_state = _prepare_padded_state(state, config, params, registered_variables)
    time_step = float(_cfl_time_step(padded_state, config, params, registered_variables))

    law = _barotropic_law(params, config)
    largest_solver_speed, largest_fast_speed = 0.0, 0.0
    for axis in (1, 2):
        frame_indices = _interface_frame_indices(axis, config, registered_variables)
        left = _mhd_face_state(jnp.roll(state, 1, axis=axis), frame_indices, law)
        right = _mhd_face_state(state, frame_indices, law)
        solver_speed = hll5r_signal_speed(left, right, law.exponent, law.sound_speed_squared_of)
        fast_speed = jnp.abs(state[axis]) + _fast_speed(state, axis, sound_speed, registered_variables)
        largest_solver_speed = max(largest_solver_speed, float(solver_speed.max()))
        largest_fast_speed = max(largest_fast_speed, float(fast_speed.max()))

    assert time_step * largest_solver_speed * cells == pytest.approx(COURANT_NUMBER, rel=1e-12)
    # A step from the fast speed of the cells would be longer than the solver allows.
    assert largest_solver_speed > 1.02 * largest_fast_speed


# -------------------------------------------------------------
# ========================= ↓ Runs ↓ ==========================
# -------------------------------------------------------------


def test_first_order_shock_tube_stays_positive():
    """Brio-Wu at first order (Bouchut et al. 2010, Figure 4.4): 200 cells on [-1, 1], t = 0.2."""
    cells = 200
    left = (np.arange(cells) + 0.5) / cells < 0.5
    config = _configuration((cells,), IDEAL_GAS, FIRST_ORDER, boundary=OPEN_BOUNDARY, box_size=2.0)
    fields = dict(
        density=jnp.where(left, 1.0, 0.125),
        velocity_x=jnp.zeros(cells),
        velocity_y=jnp.zeros(cells),
        velocity_z=jnp.zeros(cells),
        magnetic_field_x=jnp.full(cells, 0.75),
        magnetic_field_y=jnp.where(left, 1.0, -1.0),
        magnetic_field_z=jnp.zeros(cells),
        gas_pressure=jnp.where(left, 1.0, 0.1),
    )
    _, final_state, registered_variables = _run(config, fields, t_end=0.2, gamma=2.0)
    density = final_state[registered_variables.density_index]
    pressure = final_state[registered_variables.pressure_index]
    assert np.all(np.isfinite(final_state))
    assert 0.11 < density.min() and density.max() <= 1.0 + 1e-12
    assert 0.08 < pressure.min() and pressure.max() <= 1.0 + 1e-12
    # A constant normal field has no divergence: the source vanishes and the field stays as it is.
    assert np.abs(final_state[registered_variables.magnetic_index.x] - 0.75).max() < 1e-13


def _relative_divergence(state, registered_variables):
    """Mean of ``h |div B| / |B|`` with central differences on a periodic 2D grid."""
    first = registered_variables.magnetic_index.x
    field = state[first : first + 3]
    differences = sum(np.roll(field[axis], -1, axis) - np.roll(field[axis], 1, axis) for axis in (0, 1))
    return float(np.mean(0.5 * np.abs(differences) / np.sqrt(np.sum(field**2, axis=0))))


def test_orszag_tang_vortex_and_what_the_cleaning_does():
    cells, sound_speed = 32, 1.0
    divergence = {}
    for solver_config in (BEATTIE_REFERENCE, NO_CLEANING):
        config = _configuration((cells, cells), solver_config=solver_config)
        initial_state, final_state, registered_variables = _run(
            config, _orszag_tang_fields(cells), t_end=0.5, sound_speed=sound_speed
        )
        density = final_state[registered_variables.density_index]
        assert np.all(np.isfinite(final_state))
        assert density.min() > 0.09
        assert density.mean() == pytest.approx(initial_state[0].mean(), rel=1e-13)
        divergence[solver_config.parabolic_cleaning] = _relative_divergence(final_state, registered_variables)
    # The cleaning lowers the mean divergence error by more than a factor of two.
    assert divergence[True] < 0.5 * divergence[False]
    assert divergence[True] < 0.03


def _oblique_alfven_wave(cells):
    """A circularly polarised Alfven wave with the wave vector 2 pi (1, 2), and its period."""
    centres = (np.arange(cells) + 0.5) / cells
    x, y = np.meshgrid(centres, centres, indexing="ij")
    wave_vector = np.array([1.0, 2.0])
    along = wave_vector / np.linalg.norm(wave_vector)
    across = np.array([-along[1], along[0]])
    phase = 2.0 * np.pi * (wave_vector[0] * x + wave_vector[1] * y)
    amplitude, mean_field = 0.1, 1.0
    swing, turn = amplitude * np.sin(phase), amplitude * np.cos(phase)
    fields = dict(
        density=np.ones_like(phase),
        velocity_x=swing * across[0],
        velocity_y=swing * across[1],
        velocity_z=turn,
        magnetic_field_x=mean_field * along[0] + swing * across[0],
        magnetic_field_y=mean_field * along[1] + swing * across[1],
        magnetic_field_z=turn,
    )
    period = 1.0 / (np.linalg.norm(wave_vector) * mean_field)
    return {name: jnp.asarray(value) for name, value in fields.items()}, period


def test_oblique_alfven_wave_converges_at_second_order():
    """
    Errors after one period at 16, 32 and 64 cells per side. The wave crosses the grid obliquely, so the normal
    field of every sweep varies and the Powell terms of the predictor act: with the sign printed in Waagan
    (2009), eq. 4.17, the order is one.
    """
    errors = []
    for cells in (16, 32, 64):
        fields, period = _oblique_alfven_wave(cells)
        initial_state, final_state, _ = _run(_configuration((cells, cells)), fields, t_end=period)
        errors.append(np.mean(np.abs(final_state - initial_state)))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.8), orders
    assert errors[-1] < 6e-4
