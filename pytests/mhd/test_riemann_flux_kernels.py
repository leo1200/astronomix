"""
The kernels of the finite-difference scheme with a Riemann flux (``FDSolverConfig.scheme = RIEMANN_FLUX``
with ``BackendConfig.backend = PALLAS``): the two flux kernels per axis, the constrained-transport
kernels, the energy-flux kernel and the stage combination.

The array form is the definition of the scheme, so every check compares the kernels with it, in float32
(the precision in which the kernels are compiled): one step from a rough state with periodic boundaries
and a few steps of a smooth state with open boundaries, through the interpreter on any device
(``BackendConfig.pallas_interpret``) and compiled where a GPU is present. The two forms are different
programs of the same arithmetic (a compiler contracts products and sums as it sees them), so the
comparison is to rounding, not bitwise. What the predicates refuse (it runs in the array form) and the
rule of ``finalize_config`` for ``pallas_ct`` are checked without a run.

The derivative of a kernel is the derivative of the array form (``diffable_pallas_call``), which
``test_riemann_flux_scheme.py`` checks; a derivative through the selected kernels compiles for several
minutes on a CPU and is therefore not part of this file.

    JAX_PLATFORMS=cpu python -m pytest pytests/mhd/test_riemann_flux_kernels.py
"""

# general
import functools

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    ISOTHERMAL,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    PALLAS,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix.option_classes.presets import FD_RIEMANN_IDEAL_GAS, FD_RIEMANN_ISOTHERMAL
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_difference._interface_fluxes._riemann_flux_pallas import _riemann_flux_pallas_supported
from astronomix._finite_difference._magnetic_update._flux_form_ct_pallas import _flux_form_ct_pallas_supported
from astronomix._spatial_operators._differencing import finite_difference_int6
from astronomix._spatial_operators._interpolate import interp_face_to_center
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

NUM_CELLS = 16
SOUND_SPEED = 0.1
GAMMA = 5.0 / 3.0
HAS_GPU = jax.default_backend() == "gpu"
BOTH_GASES = pytest.mark.parametrize("equation_of_state", [ISOTHERMAL, IDEAL_GAS], ids=["isothermal", "ideal_gas"])
BOTH_PATHS = pytest.mark.parametrize(
    "interpret",
    [True, pytest.param(False, marks=pytest.mark.skipif(not HAS_GPU, reason="the compiled kernels need a GPU"))],
    ids=["interpreted", "compiled"],
)

#: The end time of the run from the rough state: two steps at a CFL number of 0.5.
STEP_END_TIME = 0.01
#: Bound on the largest difference between the kernels and the array form after that run, relative to the
#: largest value of the state. Measured with the interpreted kernels on a CPU (float32, 16^3): 7.9e-06 for the
#: isothermal gas and 7.4e-07 for the ideal gas, where the two steps change the state by 0.14 and 0.13. The
#: bulk agrees to 1e-07; the largest entries are single cells at which the array form itself moves by as much
#: when the input changes by one unit in the last place (7.6e-06 in four draws, isothermal gas). The bound is
#: ten times that.
STEP_TOLERANCE = 1e-4
#: The same for the steps of the smooth state with open boundaries. Measured: 4.3e-06 at one entry (the array
#: form moves by 4.3e-06 at this entry under a change of the input by one unit in the last place), 1.8e-07
#: elsewhere.
OPEN_BOUNDARY_TOLERANCE = 1e-4
#: The ghost layers that ``finalize_config`` derives (the stage-1 test checks the two numbers), and a larger
#: count with which a block shape tiles the padded grid of 16^3 cells (32^3 and 48^3; 30^3 and 42^3 with the
#: derived counts are twice an odd number, which no block of at least a warp tiles).
DERIVED_GHOST_CELLS = {ISOTHERMAL: 7, IDEAL_GAS: 13}
TILED_GHOST_CELLS = {ISOTHERMAL: 8, IDEAL_GAS: 16}


def _config(equation_of_state, backend, interpret=False, boundary=PERIODIC_BOUNDARY, **backend_fields):
    """A 3D finite-difference MHD configuration with the Riemann-flux preset of the gas."""
    side = BoundarySettings1D(boundary, boundary)
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        mhd=True,
        equation_of_state=equation_of_state,
        num_cells=NUM_CELLS,
        box_size=1.0,
        boundary_settings=BoundarySettings(side, side, side),
        backend_config=BackendConfig(backend=backend, pallas_interpret=interpret, **backend_fields),
        solver_config=FD_RIEMANN_IDEAL_GAS if equation_of_state == IDEAL_GAS else FD_RIEMANN_ISOTHERMAL,
        progress_bar=False,
    )


def _params(equation_of_state, t_end):
    """The parameters of a run to ``t_end`` at a CFL number of 0.5."""
    if equation_of_state == IDEAL_GAS:
        return SimulationParams(C_cfl=0.5, gamma=GAMMA, minimum_density=1e-10, minimum_pressure=1e-12, t_end=t_end)
    return SimulationParams(C_cfl=0.5, minimum_density=1e-10, isothermal_sound_speed=SOUND_SPEED, t_end=t_end)


def _state(config, registered_variables, density, velocity, faces, pressure):
    """The primitive state with the cell-centred field interpolated from the interface fields."""
    fields = dict(
        density=jnp.asarray(density),
        velocity_x=jnp.asarray(velocity[0]),
        velocity_y=jnp.asarray(velocity[1]),
        velocity_z=jnp.asarray(velocity[2]),
        magnetic_field_x=interp_face_to_center(faces[0], 0),
        magnetic_field_y=interp_face_to_center(faces[1], 1),
        magnetic_field_z=interp_face_to_center(faces[2], 2),
        interface_magnetic_field_x=faces[0],
        interface_magnetic_field_y=faces[1],
        interface_magnetic_field_z=faces[2],
    )
    if config.equation_of_state == IDEAL_GAS:
        fields["gas_pressure"] = jnp.asarray(pressure)
    return construct_primitive_state(config=config, registered_variables=registered_variables, **fields)


def _smooth_fields():
    """Density, velocity, interface fields and pressure of a smooth state without a grid symmetry."""
    x = (np.arange(NUM_CELLS, dtype=np.float32) + 0.5) / NUM_CELLS
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    def s(a):
        return np.sin(2.0 * np.pi * a)

    def c(a):
        return np.cos(2.0 * np.pi * a)

    half = 0.5 / NUM_CELLS
    density = 1.0 + 0.2 * s(X + 0.07) * c(Y + 0.13) + 0.1 * s(Z + 0.11)
    velocity = [
        0.3 * s(Y + 0.21) * c(Z + 0.03) + 0.08 * s(X + 0.17),
        0.25 * s(Z + 0.41) * c(X + 0.09) + 0.08 * s(Y + 0.57),
        0.2 * s(X + 0.29) * c(Y + 0.19) + 0.08 * c(Z + 0.23),
    ]
    # Each interface component does not depend on its own coordinate, plus a pair whose
    # differences cancel: the interface divergence is zero.
    faces = [
        0.45 + 0.15 * s(Y + 0.12) * c(Z + 0.04) + 0.06 * s(X + half + 0.37) * c(Y + 0.27),
        0.2 + 0.15 * s(Z + 0.33) * c(X + 0.06) - 0.06 * c(X + 0.37) * s(Y + half + 0.27),
        0.1 + 0.15 * s(X + 0.71) * c(Y + 0.08),
    ]
    pressure = 0.6 * (1.0 + 0.2 * c(X + Y + 0.05) + 0.1 * s(Z + 0.15))
    return density, velocity, [jnp.asarray(face) for face in faces], pressure


def _smooth_state(config, registered_variables):
    """The smooth state (the state of the open-boundary test of the array form)."""
    return _state(config, registered_variables, *_smooth_fields())


def _rough_state(config, registered_variables, seed):
    """
    The smooth state with cell-to-cell roughness: density and pressure times a factor between 0.63 and 1.6,
    velocities plus a random part of rms 0.1 (the sound speed of the isothermal gas), interface fields plus
    the discrete curl of a random edge vector potential (interface divergence zero to rounding). The state
    is not smooth on the grid, so the nonlinear parts of the reconstruction act, and the step is still
    well-conditioned: a change of the input by one unit in the last place changes the result by less than
    1e-5 of its largest value. With density contrasts of a factor of 100 and Mach 10 it is not (single cells
    change by 1e-3 under such a change of the input, in the array form alone), and two programs cannot be
    compared to rounding there.
    """
    density, velocity, faces, pressure = _smooth_fields()
    keys = jax.random.split(jax.random.key(seed), 8)
    shape = (NUM_CELLS,) * 3
    spacing = 1.0 / NUM_CELLS
    density = density * 10.0 ** jax.random.uniform(keys[0], shape, minval=-0.2, maxval=0.2)
    velocity = [component + 0.1 * jax.random.normal(key, shape) for component, key in zip(velocity, keys[1:4])]
    pressure = pressure * 10.0 ** jax.random.uniform(keys[4], shape, minval=-0.2, maxval=0.2)
    a_x, a_y, a_z = (0.1 * spacing * jax.random.normal(key, shape) for key in keys[5:8])
    faces = (
        faces[0] + (finite_difference_int6(a_z, 1) - finite_difference_int6(a_y, 2)) / spacing,
        faces[1] + (finite_difference_int6(a_x, 2) - finite_difference_int6(a_z, 0)) / spacing,
        faces[2] + (finite_difference_int6(a_y, 0) - finite_difference_int6(a_x, 1)) / spacing,
    )
    return _state(config, registered_variables, density, velocity, faces, pressure)


def _probe(registered_variables, extent=NUM_CELLS):
    """An array of the shape of a state of ``extent`` cubed cells (the predicates read the shape only)."""
    return np.broadcast_to(np.float32(0.0), (registered_variables.num_vars,) + (extent,) * 3)


def _kernels_selected(state_like, config):
    """(flux kernels, constrained-transport kernels) run for ``state_like`` under ``config``."""
    return (
        bool(_riemann_flux_pallas_supported(state_like, config)),
        bool(_flux_form_ct_pallas_supported(state_like, config)),
    )


def _relative_difference(a, b):
    """max |a - b| relative to the largest value of b, and the number of differing entries."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b))), int(np.sum(a != b))


# ======================= one step, periodic =======================


@functools.lru_cache(maxsize=None)
def _rough_step(equation_of_state, backend, interpret):
    """(initial state, state at ``STEP_END_TIME``) from the rough state with periodic boundaries."""
    config = _config(equation_of_state, backend, interpret)
    registered_variables = get_registered_variables(config)
    state = _rough_state(config, registered_variables, seed=3)
    config = finalize_config(config, state.shape)
    expected = (True, True) if backend == PALLAS else (False, False)
    assert _kernels_selected(state, config) == expected
    final = time_integration(state, config, _params(equation_of_state, STEP_END_TIME), registered_variables)
    return np.asarray(state), np.asarray(final)


@BOTH_PATHS
@BOTH_GASES
def test_step_of_the_kernels_equals_the_array_form(equation_of_state, interpret):
    """Two steps from the rough state: every kernel of the stage against the array form, to rounding."""
    state, array_form = _rough_step(equation_of_state, NATIVE_JAX, False)
    _, kernels = _rough_step(equation_of_state, PALLAS, interpret)
    difference, differing = _relative_difference(kernels, array_form)
    change, _ = _relative_difference(array_form, state)
    print(f"kernel step eos {equation_of_state} interpret {interpret}: {differing} of {kernels.size} entries "
          f"differ, largest difference {difference:.2e} of the largest value (the step changes {change:.2e})")
    assert np.all(np.isfinite(kernels))
    assert change > 1e-2
    assert difference < STEP_TOLERANCE


# ======================= open boundaries =======================


@functools.lru_cache(maxsize=None)
def _open_boundary_run(equation_of_state, backend, interpret, ghost_cells):
    """The smooth state after a few steps with open boundaries and ``ghost_cells`` layers."""
    config = _config(equation_of_state, backend, interpret, boundary=OPEN_BOUNDARY)
    registered_variables = get_registered_variables(config)
    state = _smooth_state(config, registered_variables)
    config = finalize_config(config, state.shape)
    assert config.num_ghost_cells == DERIVED_GHOST_CELLS[equation_of_state]
    # finalize_config sets the count itself, so a larger one is set on the final configuration
    config = config._replace(num_ghost_cells=ghost_cells)
    padded = _probe(registered_variables, NUM_CELLS + 2 * ghost_cells)
    expected = (True, True) if backend == PALLAS else (False, False)
    assert _kernels_selected(padded, config) == expected
    final = time_integration(state, config, _params(equation_of_state, 0.015), registered_variables)
    return np.asarray(final)


@BOTH_PATHS
def test_open_boundaries_with_the_kernels(interpret):
    """
    Open boundaries, isothermal preset: the kernels on a padded grid that a block tiles (8 ghost layers)
    against the array form with the 7 layers that ``finalize_config`` derives. The array form does not
    depend on layers beyond the derived count (``test_riemann_flux_scheme.py``), so the difference is the
    difference of the two programs.
    """
    array_form = _open_boundary_run(ISOTHERMAL, NATIVE_JAX, False, DERIVED_GHOST_CELLS[ISOTHERMAL])
    kernels = _open_boundary_run(ISOTHERMAL, PALLAS, interpret, TILED_GHOST_CELLS[ISOTHERMAL])
    difference, differing = _relative_difference(kernels, array_form)
    print(f"kernels with open boundaries, interpret {interpret}: {differing} of {kernels.size} entries differ, "
          f"largest difference {difference:.2e} of the largest value")
    assert np.all(np.isfinite(kernels))
    assert difference < OPEN_BOUNDARY_TOLERANCE


# ======================= what runs in the array form =======================


@BOTH_GASES
def test_predicates_refuse_what_the_kernels_do_not_serve(equation_of_state):
    """
    The conditions under which a stage runs in the array form although the backend is Pallas: a grid that no
    block shape tiles (open boundaries with the derived ghost layers), a state that is not three-dimensional,
    and the compiled kernels on a CPU or in double precision. The interpreter takes any device and precision.
    """
    registered_variables = get_registered_variables(_config(equation_of_state, PALLAS))
    probe = _probe(registered_variables)
    shape = probe.shape

    def final(backend, interpret=False, **backend_fields):
        return finalize_config(_config(equation_of_state, backend, interpret, **backend_fields), shape)

    interpreted = final(PALLAS, interpret=True)
    assert _kernels_selected(probe, interpreted) == (True, True)
    assert _kernels_selected(probe, final(NATIVE_JAX)) == (False, False)
    assert _kernels_selected(probe, final(PALLAS)) == ((True, True) if HAS_GPU else (False, False))
    assert _kernels_selected(probe[:, 0], interpreted) == (False, False)
    with jax.enable_x64(True):
        assert _kernels_selected(probe, final(PALLAS)) == (False, False)
        assert _kernels_selected(probe, interpreted) == (True, True)
    padded_derived = _probe(registered_variables, NUM_CELLS + 2 * DERIVED_GHOST_CELLS[equation_of_state])
    padded_tiled = _probe(registered_variables, NUM_CELLS + 2 * TILED_GHOST_CELLS[equation_of_state])
    assert _kernels_selected(padded_derived, interpreted) == (False, False)
    assert _kernels_selected(padded_tiled, interpreted) == (True, True)
    # a block shape of the user that does not divide the grid
    assert _kernels_selected(probe, final(PALLAS, interpret=True, pallas_block_shape=(1, 1, 48))) == (False, False)


@BOTH_GASES
def test_constrained_transport_kernels_follow_the_backend(equation_of_state):
    """Under ``RIEMANN_FLUX`` ``finalize_config`` derives ``pallas_ct`` from the backend, whatever was set."""
    shape = _probe(get_registered_variables(_config(equation_of_state, PALLAS))).shape
    for backend in (PALLAS, NATIVE_JAX):
        for requested in (False, True):
            config = finalize_config(_config(equation_of_state, backend, pallas_ct=requested), shape)
            assert config.backend_config.pallas_ct == (backend == PALLAS)
