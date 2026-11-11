"""
Reverse-mode automatic differentiation through the finite-difference solver.

The problem is the 16^3 blast of ``_blast_setup``: FD/WENO hydro with the
positivity-preserving reconstruction, the dual-energy formalism, five
composition passive scalars with physical bounds and frozen WENO weights in
the tangent. Everything runs in float64 with the NATIVE_JAX backend, over a
few steps.

What is checked:

* ``jax.grad`` through ``time_integration`` works under ``BACKWARDS`` (the
  equinox-checkpointed time loop and the static, masked passive-scalar
  sub-cycling), and the gradient reaches the composition scalars;
* the dot-product test ``<J^T w, u> == w . (J u)``: on one fixed-step function
  (JVP and VJP of the same code), and between the ``FORWARDS`` JVP (while loop,
  traced sub-step count) and the ``BACKWARDS`` VJP (adaptive steps);
* the masked sub-cycling reproduces the dynamic one when the flow needs several
  sub-steps, and ``rematerialize_stages`` does not change the gradient;
* the bound clamp keeps derivative 1 on its bounds (``jnp.clip`` gives 0.5
  there, so a scalar sitting on its bound would lose half of its sensitivity
  every step);
* ``finalize_config`` refuses a passive-scalar sub-step cap below 1, and
  ``OPTIMAL_BACKEND`` resolves to NATIVE_JAX when JAX runs on the CPU.

Run on the CPU (the preamble then selects fast-compiling XLA flags)::

    JAX_PLATFORMS=cpu python -m pytest pytests/differentiability/test_fd_reverse_mode.py
"""

# ==== GPU selection ====
import os
if os.environ.get("JAX_PLATFORMS") == "cpu":
    # XLA:CPU compiles the unrolled step in seconds at optimisation level 0
    # (minutes and tens of GB otherwise); the numerics differ at round-off only.
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true",
    )
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
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
from astronomix import (
    BACKWARDS,
    NATIVE_JAX,
    OPTIMAL_BACKEND,
)

# astronomix containers
from astronomix import BackendConfig

# astronomix functions
from astronomix import (
    finalize_config,
    get_registered_variables,
    time_integration,
)
from astronomix._fluid_equations._passive_scalars import (
    _clip_keep_derivative,
    _substep_count,
    _velocity_components,
    advect_passive_scalars,
)

# shared blast setup (a sibling module of this test)
from _blast_setup import (
    DT_FIXED,
    EJECTA_FRACTION_SCALAR,
    NUM_CELLS,
    NUM_FIXED_STEPS,
    blast_config,
    blast_params,
    fluid_only_registry,
    get_fluid_state,
    setup_blast,
)

jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# ============================ ↓ Helpers ↓ ====================================
# -----------------------------------------------------------------------------


def _bits(array):
    """The bit pattern of a float64 array, for bitwise comparisons (NaN included)."""
    return np.asarray(array).view(np.int64)


def _smooth_direction(state, registered_variables, seed, scale=1e-3):
    """
    A smooth relative perturbation of density, velocity, pressure and every
    passive scalar.

    Each row is a product of low-wavenumber sines with random phases plus a
    cosine along z, scaled by the magnitude of the field (at least 1e-2). The
    dual-energy row is left at zero: the solver re-derives it from the pressure.

    Args:
        state: The primitive state to perturb.
        registered_variables: The registered variables.
        seed: The seed of the random phases.
        scale: The relative size of the perturbation.

    Returns:
        The perturbation, with the shape of ``state``.
    """
    rng = np.random.default_rng(seed)
    cell_phase = 2 * np.pi * np.arange(NUM_CELLS) / NUM_CELLS
    direction = np.zeros(state.shape)
    for row in range(state.shape[0]):
        if (
            registered_variables.internal_energy_active
            and row == registered_variables.internal_energy_index
        ):
            continue
        phase_x, phase_y, phase_z, phase_extra = rng.normal(size=4)
        direction[row] = (
            np.sin(phase_x + cell_phase)[:, None, None]
            * np.cos(phase_y + 2 * cell_phase)[None, :, None]
            * np.sin(phase_z + cell_phase)[None, None, :]
            + 0.3 * np.cos(phase_extra + 3 * cell_phase)[None, None, :]
        )
    direction = jnp.asarray(direction)
    field_magnitude = jnp.maximum(jnp.abs(state), 1e-2)
    return scale * direction * field_magnitude


def _jitted_gradient_and_final_state(config, params, registered_variables, initial_state, weights):
    """
    The gradient of ``weights . state(t_end)`` and the final state, each
    computed by its own jitted program.

    Args:
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        initial_state: The initial primitive state.
        weights: The weights of the objective, with the shape of the state.

    Returns:
        The gradient with respect to the initial state and the final state, as
        numpy arrays.
    """

    def objective(state):
        """The weighted sum of the final state."""
        return jnp.vdot(weights, time_integration(state, config, params, registered_variables))

    def evolve(state):
        """The final state."""
        return time_integration(state, config, params, registered_variables)

    gradient = jax.jit(jax.grad(objective))(initial_state)
    final_state = jax.jit(evolve)(initial_state)
    return np.asarray(gradient), np.asarray(final_state)


# -----------------------------------------------------------------------------
# ============================ ↑ Helpers ↑ ====================================
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# ============================= ↓ Tests ↓ =====================================
# -----------------------------------------------------------------------------


def test_bound_clamps_keep_derivative():
    """
    The tie-preserving clamp has the primal of ``jnp.clip`` bit for bit (signed
    zeros and NaN included) and derivative 1 on its bounds, where ``jnp.clip``
    splits the derivative evenly between value and bound.
    """
    values = jnp.array([-1.0, -0.0, 0.0, 0.3, 1.0, 1.0 + 1e-12, 2.0, jnp.nan])

    # The primal is jnp.clip, bit for bit.
    np.testing.assert_array_equal(
        _bits(_clip_keep_derivative(values, 0.0, 1.0)),
        _bits(jnp.clip(values, 0.0, 1.0)),
    )

    # Derivative 1 inside and on the bounds, 0 strictly outside.
    clip_gradient = jax.grad(lambda v: jnp.sum(_clip_keep_derivative(v, 0.0, 1.0)))(values)
    np.testing.assert_array_equal(np.asarray(clip_gradient), [0, 1, 1, 1, 1, 0, 0, 1])

    # What the tie-preserving clamp is for: jnp.clip halves the derivative on the bound.
    assert float(jax.grad(lambda v: jnp.clip(v, 0.0, 1.0))(0.0)) == 0.5

    # The bound's own derivative goes where the value comes from the bound: only
    # x = -1 lies strictly below the lower bound.
    lower_bound_gradient = jax.grad(
        lambda lower_bound: jnp.sum(_clip_keep_derivative(values, lower_bound, 1.0))
    )(0.0)
    assert float(lower_bound_gradient) == 1.0


def test_advection_keeps_bound_derivative():
    """
    At rest the passive-scalar advection is the identity, so its tangent must be
    the identity too, including in the cells holding exactly 0 or 1. A clamp with
    derivative 0.5 on the bound would leave 0.5^n there after n steps.
    """
    config, registered_variables, state = setup_blast()
    fluid_registry = fluid_only_registry(registered_variables)
    fluid_state = get_fluid_state(state, registered_variables)
    for velocity_index in registered_variables.velocity_index:
        fluid_state = fluid_state.at[velocity_index].set(0.0)
    passive_scalars = state[registered_variables.passive_scalar_index:]
    tangent = jnp.ones_like(passive_scalars)

    def advect(scalars):
        """One advection step of the passive scalars in the resting flow."""
        return advect_passive_scalars(
            scalars,
            fluid_state,
            1e-3,
            config.grid_spacing,
            config,
            fluid_registry,
        )

    _, advected_tangent = jax.jvp(advect, (passive_scalars,), (tangent,))
    ejecta_fraction = passive_scalars[EJECTA_FRACTION_SCALAR]
    on_bound = np.asarray((ejecta_fraction == 0.0) | (ejecta_fraction == 1.0))
    assert on_bound.mean() > 0.5
    np.testing.assert_allclose(
        np.asarray(advected_tangent[EJECTA_FRACTION_SCALAR])[on_bound],
        1.0,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(np.asarray(advected_tangent), 1.0, rtol=0, atol=1e-12)


def test_masked_substeps_reverse_vs_dynamic_jvp():
    """
    With several passive-scalar sub-steps (some taken, some masked off), the VJP
    of the masked sub-step loop (``BACKWARDS``) is the transpose of the JVP of
    the dynamic loop (``FORWARDS``) of the scalar advection.
    """
    sub_cycling = dict(passive_scalar_cfl=0.02, max_passive_scalar_substeps=6)
    dynamic_config, registered_variables, state = setup_blast(**sub_cycling)
    masked_config, _, _ = setup_blast(differentiation_mode=BACKWARDS, **sub_cycling)
    fluid_registry = fluid_only_registry(registered_variables)
    fluid_state = get_fluid_state(state, registered_variables)
    passive_scalars = state[registered_variables.passive_scalar_index:]
    dt = 4e-3

    num_substeps = int(
        _substep_count(
            _velocity_components(fluid_state, dynamic_config, fluid_registry),
            dynamic_config.grid_spacing,
            dt,
            dynamic_config,
        )
    )
    assert 1 < num_substeps < 6

    def make_scalar_step(scalar_config):
        """One advection step of the passive scalars under ``scalar_config``."""

        def scalar_step(fluid_state, passive_scalars, dt):
            return advect_passive_scalars(
                passive_scalars,
                fluid_state,
                dt,
                scalar_config.grid_spacing,
                scalar_config,
                fluid_registry,
            )

        return scalar_step

    rng = np.random.default_rng(0)
    fluid_direction = get_fluid_state(
        _smooth_direction(state, registered_variables, 5),
        registered_variables,
    )
    scalar_direction = jnp.asarray(1e-3 * rng.normal(size=passive_scalars.shape))
    dt_direction = 1e-4
    cotangent = jnp.asarray(rng.normal(size=passive_scalars.shape))

    dynamic_result, jacobian_direction = jax.jvp(
        make_scalar_step(dynamic_config),
        (fluid_state, passive_scalars, dt),
        (fluid_direction, scalar_direction, dt_direction),
    )
    masked_result, masked_vjp = jax.vjp(
        make_scalar_step(masked_config),
        fluid_state,
        passive_scalars,
        dt,
    )
    fluid_cotangent, scalar_cotangent, dt_cotangent = masked_vjp(cotangent)

    np.testing.assert_allclose(
        np.asarray(masked_result),
        np.asarray(dynamic_result),
        rtol=0,
        atol=1e-13 * float(jnp.abs(dynamic_result).max()),
    )
    lhs = float(
        jnp.vdot(fluid_cotangent, fluid_direction)
        + jnp.vdot(scalar_cotangent, scalar_direction)
        + dt_cotangent * dt_direction
    )
    rhs = float(jnp.vdot(cotangent, jacobian_direction))
    assert abs(lhs - rhs) <= 1e-12 * abs(rhs), (lhs, rhs)


def test_masked_substeps_match_dynamic():
    """
    With several passive-scalar sub-steps forced (a small ``passive_scalar_cfl``),
    a ``BACKWARDS`` run (the reverse-differentiable masked sub-step loop) gives
    the result of a ``FORWARDS`` run (the loop with a traced trip count),
    round-off apart.
    """
    sub_cycling = dict(passive_scalar_cfl=0.02, max_passive_scalar_substeps=6)
    dynamic_config, registered_variables, state = setup_blast(**sub_cycling)
    masked_config, _, _ = setup_blast(differentiation_mode=BACKWARDS, **sub_cycling)
    dynamic_state = np.asarray(
        time_integration(state, dynamic_config, blast_params(), registered_variables)
    )
    masked_state = np.asarray(
        time_integration(state, masked_config, blast_params(), registered_variables)
    )

    # The sub-cycling really happened: the result differs from a run with a
    # single sub-step.
    config, registered_variables, state = setup_blast(max_passive_scalar_substeps=1)
    single_substep_state = np.asarray(
        time_integration(state, config, blast_params(), registered_variables)
    )
    passive_scalar_index = registered_variables.passive_scalar_index
    substep_effect = (
        single_substep_state[passive_scalar_index:] - dynamic_state[passive_scalar_index:]
    )
    assert np.abs(substep_effect).max() > 1e-8

    # The same arithmetic, compiled in different contexts: round-off apart.
    for row in range(registered_variables.num_vars):
        np.testing.assert_allclose(
            masked_state[row],
            dynamic_state[row],
            rtol=0,
            atol=1e-13 * max(np.abs(dynamic_state[row]).max(), 1e-30),
        )


def test_grad_backwards_and_dot_product_fixed_step():
    """
    The dot-product test on one function: the JVP and the VJP of a fixed-step
    ``BACKWARDS`` run (masked sub-steps, scan time loop, stage rematerialisation
    on the reverse side, which the JVP does not see) satisfy
    ``<J^T w, u> = w . J u`` to round-off, and the gradient reaches the
    composition scalars.
    """
    config, registered_variables, state = setup_blast(
        fixed_timestep=True,
        num_timesteps=NUM_FIXED_STEPS,
        differentiation_mode=BACKWARDS,
        rematerialize_stages=True,
    )
    params = blast_params(t_end=NUM_FIXED_STEPS * DT_FIXED)

    def evolve(initial_state):
        """The state after the fixed steps."""
        return time_integration(initial_state, config, params, registered_variables)

    direction = _smooth_direction(state, registered_variables, 1)
    cotangent = jnp.asarray(np.random.default_rng(2).normal(size=state.shape))

    @jax.jit
    def jvp_and_vjp(initial_state, direction, cotangent):
        """The products ``J u`` and ``J^T w`` of the Jacobian of ``evolve``."""
        jacobian_direction = jax.jvp(evolve, (initial_state,), (direction,))[1]
        jacobian_transpose_cotangent = jax.vjp(evolve, initial_state)[1](cotangent)[0]
        return jacobian_direction, jacobian_transpose_cotangent

    jacobian_direction, jacobian_transpose_cotangent = jvp_and_vjp(state, direction, cotangent)
    assert bool(jnp.all(jnp.isfinite(jacobian_transpose_cotangent)))
    assert bool(jnp.all(jnp.isfinite(jacobian_direction)))
    lhs = float(jnp.vdot(jacobian_transpose_cotangent, direction))
    rhs = float(jnp.vdot(cotangent, jacobian_direction))
    assert abs(lhs - rhs) <= 1e-11 * max(abs(lhs), abs(rhs)), (lhs, rhs)

    # The gradient reaches the ejecta fraction, which mostly sits on its bounds.
    ejecta_fraction_index = registered_variables.passive_scalar_index + EJECTA_FRACTION_SCALAR
    assert float(jnp.abs(jacobian_transpose_cotangent[ejecta_fraction_index]).max()) > 0.0


def test_grad_adaptive_backwards_vs_forward_jvp():
    """
    ``jax.grad`` through the adaptive ``BACKWARDS`` loop (equinox checkpoints,
    masked sub-steps) agrees with the ``FORWARDS`` JVP (while loop, traced
    sub-step count) in two directions, and both give the same objective.
    """
    backwards_config, registered_variables, state = setup_blast(differentiation_mode=BACKWARDS)
    forwards_config, _, _ = setup_blast()
    params = blast_params()
    weights = jnp.asarray(np.random.default_rng(3).normal(size=state.shape))

    def make_objective(objective_config):
        """The objective ``weights . state(T_END)`` under ``objective_config``."""

        def objective(initial_state):
            final_state = time_integration(
                initial_state,
                objective_config,
                params,
                registered_variables,
            )
            return jnp.vdot(weights, final_state)

        return objective

    @jax.jit
    def forwards_objective_and_derivative(initial_state, direction):
        """The FORWARDS objective and its derivative along ``direction``."""
        return jax.jvp(make_objective(forwards_config), (initial_state,), (direction,))

    backwards_value, gradient = jax.jit(jax.value_and_grad(make_objective(backwards_config)))(state)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    ejecta_fraction_index = registered_variables.passive_scalar_index + EJECTA_FRACTION_SCALAR
    assert float(jnp.abs(gradient[ejecta_fraction_index]).max()) > 0.0

    for seed in (11, 12):
        direction = _smooth_direction(state, registered_variables, seed)
        forwards_value, directional_derivative = forwards_objective_and_derivative(
            state,
            direction,
        )
        gradient_projection = float(jnp.vdot(gradient, direction))
        directional_derivative = float(directional_derivative)
        assert (
            abs(float(forwards_value) - float(backwards_value))
            <= 1e-12 * abs(float(forwards_value))
        )
        assert (
            abs(gradient_projection - directional_derivative)
            <= 1e-10 * abs(directional_derivative)
        ), (gradient_projection, directional_derivative)


def test_remat_same_gradient():
    """
    ``rematerialize_stages`` changes what the backward pass stores, not what it
    computes: the gradient agrees with that of the plain reverse pass to
    round-off, and the forward result is the same bit for bit.
    """
    gradients = {}
    final_states = {}
    for rematerialize_stages in (False, True):
        config, registered_variables, state = setup_blast(
            fixed_timestep=True,
            num_timesteps=1,
            differentiation_mode=BACKWARDS,
            rematerialize_stages=rematerialize_stages,
        )
        weights = jnp.asarray(np.random.default_rng(4).normal(size=state.shape))
        gradient, final_state = _jitted_gradient_and_final_state(
            config,
            blast_params(t_end=DT_FIXED),
            registered_variables,
            state,
            weights,
        )
        gradients[rematerialize_stages] = gradient
        final_states[rematerialize_stages] = final_state

    scale = np.abs(gradients[False]).max()
    np.testing.assert_allclose(gradients[True], gradients[False], rtol=0, atol=1e-12 * scale)
    np.testing.assert_array_equal(final_states[True], final_states[False])


def test_config_guards():
    """
    ``finalize_config`` refuses a passive-scalar sub-step cap below 1, and
    ``OPTIMAL_BACKEND`` resolves to NATIVE_JAX when JAX runs on the CPU.
    """
    num_vars = get_registered_variables(blast_config()).num_vars
    state_shape = (num_vars, NUM_CELLS, NUM_CELLS, NUM_CELLS)
    with pytest.raises(ValueError, match="max_passive_scalar_substeps"):
        finalize_config(blast_config(max_passive_scalar_substeps=0), state_shape)
    if jax.devices()[0].platform == "cpu":
        config = finalize_config(
            blast_config(backend_config=BackendConfig(backend=OPTIMAL_BACKEND)),
            state_shape,
        )
        assert config.backend_config.backend == NATIVE_JAX


# -----------------------------------------------------------------------------
# ============================= ↑ Tests ↑ =====================================
# -----------------------------------------------------------------------------
