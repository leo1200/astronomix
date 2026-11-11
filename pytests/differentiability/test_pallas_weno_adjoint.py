"""
The Pallas adjoint of the ideal-gas hydro WENO flux (``backend_config.pallas_adjoint``).

In reverse mode the hydro WENO flux of the Pallas backend is differentiated by a
Pallas kernel that runs ``jax.vjp`` of the forward kernel's window function
(``_weno_hydro_flux_from_window``) instead of transposing the native-JAX
tangent. What is checked, in Pallas interpret mode on the CPU (float64):

* without dual energy, the Pallas VJP is the native VJP, for the exact and the
  frozen (``weno_ad_frozen_weights``) linearisation, with and without the
  positivity-preserving recombination, on every axis;
* with dual energy, in a state where the switch takes the advected ``g`` in a
  third of the cells, the VJP (state and ``g`` cotangents) is the transpose of
  the Pallas forward: a dot test against its central finite difference (the
  native flux takes a slightly different value in those cells, so it is not the
  reference there);
* the gradient of a few positivity-preserving, dual-energy
  steps through ``time_integration`` with the Pallas adjoint is the one through
  the native tangent.

Run on the CPU::

    JAX_PLATFORMS=cpu python -m pytest pytests/differentiability/test_pallas_weno_adjoint.py
"""

# ==== GPU selection ====
import os
if os.environ.get("JAX_PLATFORMS") != "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# testing
import pytest

# astronomix constants
from astronomix import BACKWARDS, PALLAS, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import DOUBLE_PRECISION, FINITE_DIFFERENCE

# astronomix containers
from astronomix import (
    BackendConfig,
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
from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_native_for_axis
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _weno_flux_hydro_pallas_local,
    _weno_flux_hydro_pallas_vjp_local,
)

jax.config.update("jax_enable_x64", True)

NUM_CELLS = 8
PARAMS = SimulationParams(minimum_density=1e-6, minimum_pressure=1e-8)


def flux_config(positivity_preserving, dual_energy, frozen):
    """A 3D Pallas (interpret) configuration and the registry the WENO flux sees."""
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        num_cells=NUM_CELLS,
        box_size=1.0,
        backend_config=BackendConfig(backend=PALLAS, pallas_interpret=True, pallas_block_shape=(4, 4, 4)),
        weno_positivity_preserving=positivity_preserving,
        dual_energy=dual_energy,
        weno_ad_frozen_weights=frozen,
        numerical_precision=DOUBLE_PRECISION,
    )
    registered_variables = get_registered_variables(config)
    if dual_energy:
        # the WENO flux sees the state without the dual-energy row (_evolve_state)
        registered_variables = registered_variables._replace(
            num_vars=registered_variables.internal_energy_index,
            internal_energy_index=-1,
        )
    return config, registered_variables


def conserved_state(rng, cold_fraction=0.0):
    """A random conserved hydro state; ``cold_fraction`` of it kinetic-dominated."""
    shape = (NUM_CELLS,) * 3
    density = np.exp(0.5 * rng.standard_normal(shape))
    momentum = 0.7 * rng.standard_normal((3,) + shape) * density
    pressure = np.exp(0.8 * rng.standard_normal(shape)) * 0.1
    cold = rng.random(shape) < cold_fraction
    pressure = np.where(cold, 1e-6 * pressure, pressure)
    energy = pressure / (2.0 / 3.0) + 0.5 * np.sum(momentum ** 2, 0) / density
    state = np.concatenate([density[None], momentum, energy[None]])
    return jnp.asarray(state), jnp.asarray(pressure / (2.0 / 3.0)), cold


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("positivity_preserving", [True, False])
def test_vjp_is_the_native_vjp_without_dual_energy(frozen, positivity_preserving):
    rng = np.random.default_rng(0)
    config, registered_variables = flux_config(positivity_preserving, False, frozen)
    state, _, _ = conserved_state(rng, cold_fraction=0.3)
    cotangent = jnp.asarray(rng.standard_normal(state.shape))
    for axis in range(3):
        native = _weno_flux_native_for_axis(axis)
        _, native_vjp = jax.vjp(lambda q: native(q, PARAMS, config, registered_variables), state)
        (reference,) = native_vjp(cotangent)
        pallas = _weno_flux_hydro_pallas_vjp_local(
            state, cotangent, PARAMS, config, registered_variables, axis=axis)
        error = float(jnp.linalg.norm(pallas - reference) / jnp.linalg.norm(reference))
        # the exact derivative at the near-vacuum-pressure cells is noise at ~1e-6
        # (the two forwards differ there at 1e-12); warm states agree to 1e-14
        assert error < 1e-5, (axis, error)


@pytest.mark.parametrize("positivity_preserving", [True, False])
def test_dual_energy_vjp_is_the_transpose_of_the_pallas_forward(positivity_preserving):
    rng = np.random.default_rng(1)
    config, registered_variables = flux_config(positivity_preserving, True, False)
    state, internal_energy, cold = conserved_state(rng, cold_fraction=0.3)
    # g differs from the total-energy recovery in the cells where the switch takes it
    internal_energy = jnp.where(jnp.asarray(cold), 1e3 * internal_energy, internal_energy)
    state_direction = jnp.asarray(rng.standard_normal(state.shape)) * state
    energy_direction = jnp.asarray(rng.standard_normal(internal_energy.shape)) * internal_energy
    cotangent = jnp.asarray(rng.standard_normal(state.shape))
    # In the kinetic-dominated cells the total-energy pressure (which the
    # admissibility scaling uses) is a 1e-6 remainder of E - KE, so the
    # difference step must be far below that for the flux to be linear in it.
    step = 1e-10
    for axis in range(3):

        def forward(q, g):
            return _weno_flux_hydro_pallas_local(
                q, PARAMS, config, registered_variables, axis=axis, internal_energy_density=g[None])

        finite_difference = (
            forward(state + step * state_direction, internal_energy + step * energy_direction)
            - forward(state - step * state_direction, internal_energy - step * energy_direction)
        ) / (2.0 * step)
        state_bar, energy_bar = _weno_flux_hydro_pallas_vjp_local(
            state, cotangent, PARAMS, config, registered_variables, axis=axis,
            internal_energy_density=internal_energy[None])
        assert float(jnp.max(jnp.abs(energy_bar))) > 0.0
        adjoint_side = float(jnp.sum(state_bar * state_direction) + jnp.sum(energy_bar[0] * energy_direction))
        tangent_side = float(jnp.sum(cotangent * finite_difference))
        assert abs(adjoint_side - tangent_side) < 1e-4 * abs(tangent_side), (axis, adjoint_side, tangent_side)


def test_time_integration_gradient_matches_the_native_tangent():
    rng = np.random.default_rng(2)
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    shape = (NUM_CELLS,) * 3
    density = jnp.asarray(1.0 + 0.3 * rng.random(shape))
    velocity = [jnp.asarray(0.3 * rng.standard_normal(shape)) for _ in range(3)]
    pressure = jnp.asarray(0.5 + 0.2 * rng.random(shape))
    params = SimulationParams(C_cfl=0.3, t_end=0.02, minimum_density=1e-6, minimum_pressure=1e-8)
    gradients = {}
    for pallas_adjoint in (False, True):
        config = SimulationConfig(
            solver_mode=FINITE_DIFFERENCE,
            dimensionality=3,
            num_cells=NUM_CELLS,
            box_size=1.0,
            backend_config=BackendConfig(backend=PALLAS, pallas_interpret=True, pallas_block_shape=(4, 4, 4),
                                         pallas_adjoint=pallas_adjoint),
            boundary_settings=BoundarySettings(periodic, periodic, periodic),
            weno_positivity_preserving=True,
            dual_energy=True,
            weno_ad_frozen_weights=True,
            differentiation_mode=BACKWARDS,
            num_checkpoints=2,
            numerical_precision=DOUBLE_PRECISION,
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
