"""
The Pallas adjoint of the ideal-MHD WENO flux (``backend_config.pallas_adjoint``).

In reverse mode the ideal-MHD WENO flux of the Pallas backend is differentiated
by a Pallas kernel that runs ``jax.vjp`` of the forward kernel's window function
(``_weno_mhd_flux_from_window``); under positivity preservation the
recombination of the split face fluxes (with the joint inflow limiting of the
inflow reference) is differentiated in native JAX. What is checked, in float64
with WENO-Z weights, on the GPU (the Triton kernels) or in Pallas interpret mode
on the CPU:

* the Pallas VJP is the native VJP (state and inflow-reference cotangents), for
  the exact and the frozen (``weno_ad_frozen_weights``) linearisation, with
  and without the positivity-preserving recombination, on every axis;
* the window VJP, differentiated in stages, is the transpose of the Pallas
  forward kernel: a dot test against its central finite difference, and equal
  to ``jax.vjp`` of the whole window function;
* the gradient of a few positivity-preserving steps of 3D ideal MHD with
  constrained transport through ``time_integration`` with the Pallas adjoint
  is the one through the native tangent.

Run on a GPU (or, slowly, on the CPU with ``JAX_PLATFORMS=cpu``)::

    python -m pytest pytests/differentiability/test_pallas_weno_adjoint_mhd.py
"""

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# testing
import pytest

# astronomix constants
from astronomix import BACKWARDS, PALLAS, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    initialize_interface_fields,
    time_integration,
)
from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_native_for_axis
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _weno_flux_mhd_pallas_local,
    _weno_flux_mhd_pallas_vjp,
    _weno_flux_mhd_pallas_window_vjp_local,
)
from astronomix._finite_difference._interface_fluxes._weno_positivity import mhd_inflow_reference

pytestmark = [pytest.mark.x64, pytest.mark.slow]

NUM_CELLS = 8
# The Triton kernels on a GPU, their interpreted form on the CPU.
INTERPRET = jax.default_backend() == "cpu"
PARAMS = SimulationParams(gamma=5.0 / 3.0, minimum_density=1e-6, minimum_pressure=1e-8)


def flux_config(positivity_preserving, frozen):
    """
    A 3D ideal-MHD Pallas (interpret) configuration with WENO-Z weights.

    Args:
        positivity_preserving: The positivity-preserving WENO.
        frozen: The frozen-weights linearisation.

    Returns:
        The configuration and its registered variables.
    """
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        mhd=True,
        dimensionality=3,
        num_cells=NUM_CELLS,
        box_size=1.0,
        backend_config=BackendConfig(backend=PALLAS, pallas_interpret=INTERPRET, pallas_block_shape=(4, 4, 4)),
        solver_config=FDSolverConfig(
            weno_z=True,
            weno_positivity_preserving=positivity_preserving,
            weno_ad_frozen_weights=frozen,
        ),
    )
    return config, get_registered_variables(config)


def conserved_state(rng, registered_variables):
    """
    A random ideal-MHD conserved state (the eight cell-centred variables the
    WENO flux sees), with plasma beta of order one.

    Args:
        rng: The random generator.
        registered_variables: The registered variables.

    Returns:
        The conserved state, shape ``(8, N, N, N)``.
    """
    shape = (NUM_CELLS,) * 3
    density = np.exp(0.4 * rng.standard_normal(shape))
    velocity = 0.5 * rng.standard_normal((3,) + shape)
    magnetic_field = 0.6 * rng.standard_normal((3,) + shape) + np.array([0.4, 0.0, 0.2])[:, None, None, None]
    pressure = 0.5 * np.exp(0.4 * rng.standard_normal(shape))
    energy = (
        pressure / (PARAMS.gamma - 1.0)
        + 0.5 * density * np.sum(velocity ** 2, 0)
        + 0.5 * np.sum(magnetic_field ** 2, 0)
    )
    state = np.zeros((8,) + shape)
    state[registered_variables.density_index] = density
    for component in range(3):
        state[registered_variables.momentum_index[component]] = density * velocity[component]
        state[registered_variables.magnetic_index[component]] = magnetic_field[component]
    state[registered_variables.energy_index] = energy
    return jnp.asarray(state)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("positivity_preserving", [True, False])
def test_vjp_is_the_native_vjp(frozen, positivity_preserving):
    """The Pallas adjoint of the MHD flux equals the native VJP, on every axis."""
    rng = np.random.default_rng(0)
    config, registered_variables = flux_config(positivity_preserving, frozen)
    state = conserved_state(rng, registered_variables)
    cotangent = jnp.asarray(rng.standard_normal(state.shape))
    reference = mhd_inflow_reference(state, PARAMS, config, registered_variables) if positivity_preserving else None
    for axis in range(3):
        native = _weno_flux_native_for_axis(axis)
        if positivity_preserving:
            _, native_vjp = jax.vjp(
                lambda q, b, s: native(q, PARAMS, config, registered_variables, inflow_reference=(b, s)),
                state,
                *reference,
            )
            expected = native_vjp(cotangent)
            result = _weno_flux_mhd_pallas_vjp(
                state, cotangent, PARAMS, config, registered_variables, axis=axis, inflow_reference=reference)
        else:
            _, native_vjp = jax.vjp(lambda q: native(q, PARAMS, config, registered_variables), state)
            expected = native_vjp(cotangent)
            result = (_weno_flux_mhd_pallas_vjp(state, cotangent, PARAMS, config, registered_variables, axis=axis),)
        for name, value, reference_value in zip(("state", "reference", "speed sum"), result, expected):
            # (with frozen scalings the inflow reference has no derivative: both are zero)
            scale = max(float(jnp.linalg.norm(reference_value)), float(jnp.linalg.norm(expected[0])))
            error = float(jnp.linalg.norm(value - reference_value)) / scale
            # the native and the Pallas forward differ at round-off only
            assert error < 1e-9, (axis, name, error)


@pytest.mark.parametrize("positivity_preserving", [True, False])
def test_window_vjp_is_the_transpose_of_the_pallas_forward(positivity_preserving):
    """Dot test of the window adjoint against the Pallas forward kernel."""
    rng = np.random.default_rng(1)
    config, registered_variables = flux_config(positivity_preserving, False)
    state = conserved_state(rng, registered_variables)
    direction = jnp.asarray(rng.standard_normal(state.shape)) * state
    num_outputs = 2 * state.shape[0] + 1 if positivity_preserving else state.shape[0]
    cotangent = jnp.asarray(rng.standard_normal((num_outputs,) + state.shape[1:]))
    # small enough that few cells cross a kink of the stencil maxima
    step = 1e-7
    for axis in range(3):

        def forward(q):
            return _weno_flux_mhd_pallas_local(
                q, PARAMS, config, registered_variables, axis=axis, split_only=positivity_preserving)

        finite_difference = (forward(state + step * direction) - forward(state - step * direction)) / (2.0 * step)
        state_bar = _weno_flux_mhd_pallas_window_vjp_local(
            state, cotangent, PARAMS, config, registered_variables, axis=axis)
        adjoint_side = float(jnp.sum(state_bar * direction))
        tangent_side = float(jnp.sum(cotangent * finite_difference))
        assert abs(adjoint_side - tangent_side) < 1e-6 * abs(tangent_side), (axis, adjoint_side, tangent_side)
        if axis == 0:
            # the staged adjoint is jax.vjp of the whole window up to rounding
            # (one axis: the whole-window kernel takes minutes to compile)
            whole_window = _weno_flux_mhd_pallas_window_vjp_local(
                state, cotangent, PARAMS, config, registered_variables, axis=axis, staged=False)
            error = float(jnp.linalg.norm(state_bar - whole_window) / jnp.linalg.norm(whole_window))
            assert error < 1e-12, (axis, error)


def test_time_integration_gradient_matches_the_native_tangent():
    """The gradient of a few PP-WENO MHD steps equals that of the native tangent."""
    rng = np.random.default_rng(2)
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    shape = (NUM_CELLS,) * 3
    density = jnp.asarray(1.0 + 0.3 * rng.random(shape))
    velocity = [jnp.asarray(0.3 * rng.standard_normal(shape)) for _ in range(3)]
    pressure = jnp.asarray(0.5 + 0.2 * rng.random(shape))
    magnetic_field = [jnp.asarray(0.3 + 0.2 * rng.standard_normal(shape)) for _ in range(3)]
    interface_fields = initialize_interface_fields(*magnetic_field)
    params = SimulationParams(C_cfl=0.75, t_end=0.02, gamma=5.0 / 3.0, minimum_density=1e-6, minimum_pressure=1e-8)
    gradients = {}
    for pallas_adjoint in (False, True):
        config = SimulationConfig(
            solver_mode=FINITE_DIFFERENCE,
            mhd=True,
            dimensionality=3,
            num_cells=NUM_CELLS,
            box_size=1.0,
            backend_config=BackendConfig(
                backend=PALLAS,
                pallas_interpret=INTERPRET,
                pallas_block_shape=(4, 4, 4),
                pallas_adjoint=pallas_adjoint,
            ),
            boundary_settings=BoundarySettings(periodic, periodic, periodic),
            solver_config=FDSolverConfig(weno_z=True, weno_positivity_preserving=True),
            differentiation_mode=BACKWARDS,
            num_checkpoints=2,
        )
        registered_variables = get_registered_variables(config)
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=density,
            velocity_x=velocity[0],
            velocity_y=velocity[1],
            velocity_z=velocity[2],
            gas_pressure=pressure,
            magnetic_field_x=magnetic_field[0],
            magnetic_field_y=magnetic_field[1],
            magnetic_field_z=magnetic_field[2],
            interface_magnetic_field_x=interface_fields[0],
            interface_magnetic_field_y=interface_fields[1],
            interface_magnetic_field_z=interface_fields[2],
        )
        final_config = finalize_config(config, state.shape)

        def loss(initial):
            final = time_integration(initial, final_config, params, registered_variables)
            return jnp.sum(final[registered_variables.density_index] ** 2)

        gradients[pallas_adjoint] = np.asarray(jax.grad(loss)(state))
    native_tangent, pallas_adjoint = gradients[False], gradients[True]
    assert np.all(np.isfinite(pallas_adjoint))
    error = np.linalg.norm(pallas_adjoint - native_tangent) / np.linalg.norm(native_tangent)
    assert error < 1e-8, error
