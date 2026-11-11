"""
Reverse-mode automatic differentiation through the finite-difference solver.

The problem is the 16^3 blast of ``_blast_setup``: FD/WENO hydro with the
positivity-preserving reconstruction, the dual-energy formalism, five
composition passive scalars with physical bounds, the library's shock history,
the cold-crush flux blend and frozen WENO weights in the tangent. Everything
runs in float64 with the NATIVE_JAX backend, over a few steps.

What is checked:

* ``jax.grad`` through ``time_integration`` works under ``BACKWARDS`` (the
  equinox-checkpointed time loop and the static, masked passive-scalar
  sub-cycling), and the gradient reaches the composition scalars;
* the dot-product test ``<J^T w, u> == w . (J u)``: on one fixed-step function
  (JVP and VJP of the same code), and between the ``FORWARDS`` JVP (while loop,
  traced sub-step count) and the ``BACKWARDS`` VJP (adaptive steps);
* the masked sub-cycling reproduces the dynamic one when the flow needs several
  sub-steps, and the reverse-mode memory options (``ad_remat``,
  ``ad_scalar_lean``) do not change the gradient;
* the bound clamps keep derivative 1 on their bounds (``jnp.clip`` gives 0.5
  there, so a scalar sitting on its bound would lose half of its sensitivity
  every step), and the smooth shock latch leaves the primal unchanged while
  giving the latch a derivative;
* ``finalize_config`` refuses an unknown ``ad_remat`` mode, and
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
from astronomix.option_classes.simulation_config import (
    AD_REMAT_AXIS,
    AD_REMAT_NONE,
    AD_REMAT_STAGE,
    SUBSTEPS_DYNAMIC,
    SUBSTEPS_MASKED,
)
from astronomix.variable_registry.registered_variables import (
    DENSITY_TIME_SLOT,
    ENTROPY_INITIAL_SLOT,
    SHOCKED_FRACTION_SLOT,
    TIME_SINCE_SHOCK_SLOT,
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
    _max_keep_derivative,
    _substep_count,
    _velocity_components,
    advect_passive_scalars,
    update_shock_history,
)

# shared blast setup (a sibling module of this test)
from _blast_setup import (
    DT_FIXED,
    EJECTA_FRACTION_SCALAR,
    GAMMA,
    NUM_CELLS,
    NUM_FIXED_STEPS,
    T_END,
    blast_config,
    blast_params,
    fluid_only_registry,
    get_fluid_state,
    get_shock_history,
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
    The tie-preserving clamps have the primal of ``jnp.clip`` / ``jnp.maximum``
    bit for bit (signed zeros and NaN included) and derivative 1 on their bounds,
    where ``jnp.clip`` splits the derivative evenly between value and bound.
    """
    values = jnp.array([-1.0, -0.0, 0.0, 0.3, 1.0, 1.0 + 1e-12, 2.0, jnp.nan])

    # The primal is jnp.clip / jnp.maximum, bit for bit.
    np.testing.assert_array_equal(
        _bits(_clip_keep_derivative(values, 0.0, 1.0)),
        _bits(jnp.clip(values, 0.0, 1.0)),
    )
    np.testing.assert_array_equal(
        _bits(_max_keep_derivative(values, 0.0)),
        _bits(jnp.maximum(values, 0.0)),
    )

    # Derivative 1 inside and on the bounds, 0 strictly outside.
    clip_gradient = jax.grad(lambda v: jnp.sum(_clip_keep_derivative(v, 0.0, 1.0)))(values)
    np.testing.assert_array_equal(np.asarray(clip_gradient), [0, 1, 1, 1, 1, 0, 0, 1])
    max_gradient = jax.grad(lambda v: jnp.sum(_max_keep_derivative(v, 0.0)))(values)
    np.testing.assert_array_equal(np.asarray(max_gradient), [0, 1, 1, 1, 1, 1, 1, 1])

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


def test_shock_history_clamps_keep_derivative():
    """
    A never-shocked parcel has ``time_since_shock = density_time = 0`` exactly;
    their clamp at 0 must pass the transported derivative through rather than
    halve it.
    """
    config, registered_variables, state = setup_blast()
    fluid_registry = fluid_only_registry(registered_variables)
    fluid_state = get_fluid_state(state, registered_variables)
    history = get_shock_history(state, registered_variables)
    dt = 1e-3

    def update(history):
        """One shock-history update in the initial flow."""
        return update_shock_history(
            history,
            fluid_state,
            dt,
            GAMMA,
            config.shock_entropy_jump,
            config,
            fluid_registry,
        )

    # A unit tangent on the shocked fraction and the two accumulators, none on
    # the entropy label.
    history_tangent = jnp.ones_like(history).at[ENTROPY_INITIAL_SLOT].set(0.0)
    updated_history, updated_tangent = jax.jvp(update, (history,), (history_tangent,))
    latched = np.asarray(updated_history[SHOCKED_FRACTION_SLOT]) == 1.0
    assert (~latched).mean() > 0.5

    # Where the latch did not fire, d(shocked fraction) is the carried one (where
    # it fired, the carried fraction is overwritten).
    np.testing.assert_array_equal(
        np.asarray(updated_tangent[SHOCKED_FRACTION_SLOT])[~latched],
        1.0,
    )
    # d(time since shock) = d(carried time since shock) + dt d(shocked fraction).
    np.testing.assert_allclose(
        np.asarray(updated_tangent[TIME_SINCE_SHOCK_SLOT])[~latched],
        1.0 + dt,
        rtol=1e-14,
    )


def test_smooth_shock_latch():
    """
    ``ad_smooth_shock_latch`` leaves the primal of the shock-history update
    unchanged bit for bit, gives the shocked-fraction latch a derivative near the
    shock (the boolean latch has none), and leaks no derivative into quiescent,
    never-shocked gas.
    """
    config, registered_variables, state = setup_blast()
    smooth_latch_config = config._replace(ad_smooth_shock_latch=True)
    fluid_registry = fluid_only_registry(registered_variables)
    pressure_index = registered_variables.pressure_index
    relative_pressure_perturbation = 1e-2

    def history_update_of_pressure(latch_config, fluid_state, history):
        """The shock-history update under ``latch_config`` as a function of the pressure."""

        def update(pressure):
            return update_shock_history(
                history,
                fluid_state.at[pressure_index].set(pressure),
                1e-3,
                GAMMA,
                latch_config.shock_entropy_jump,
                latch_config,
                fluid_registry,
            )

        return update

    # --------------- ↓ Near the shock ↓ ----------------

    # Two steps in, there is a real shock to latch on.
    evolved_state = time_integration(
        state,
        config,
        blast_params(t_end=2 * DT_FIXED),
        registered_variables,
    )
    evolved_fluid = get_fluid_state(evolved_state, registered_variables)
    evolved_history = get_shock_history(evolved_state, registered_variables)
    evolved_pressure = evolved_fluid[pressure_index]
    pressure_tangent = evolved_pressure * relative_pressure_perturbation

    hard_history, hard_tangent = jax.jvp(
        history_update_of_pressure(config, evolved_fluid, evolved_history),
        (evolved_pressure,),
        (pressure_tangent,),
    )
    smooth_history, smooth_tangent = jax.jvp(
        history_update_of_pressure(smooth_latch_config, evolved_fluid, evolved_history),
        (evolved_pressure,),
        (pressure_tangent,),
    )
    np.testing.assert_array_equal(_bits(hard_history), _bits(smooth_history))
    # The boolean latch has no derivative; the surrogate is sensitive near the shock.
    assert float(jnp.abs(hard_tangent[SHOCKED_FRACTION_SLOT]).max()) == 0.0
    assert float(jnp.abs(smooth_tangent[SHOCKED_FRACTION_SLOT]).max()) > 1e-4
    assert bool(jnp.all(jnp.isfinite(smooth_tangent)))

    # --------------- ↑ Near the shock ↑ ----------------

    # --------------- ↓ Quiescent gas ↓ ----------------

    # Quiescent, never-shocked gas (the initial ambient medium: entropy rise 0, at
    # rest) must not leak surrogate derivative, which would accumulate every step
    # in the carried fraction.
    initial_fluid = get_fluid_state(state, registered_variables)
    initial_history = get_shock_history(state, registered_variables)
    initial_pressure = initial_fluid[pressure_index]
    _, quiescent_tangent = jax.jvp(
        history_update_of_pressure(smooth_latch_config, initial_fluid, initial_history),
        (initial_pressure,),
        (initial_pressure * relative_pressure_perturbation,),
    )
    ejecta_fraction_index = registered_variables.passive_scalar_index + EJECTA_FRACTION_SCALAR
    ambient = np.asarray(state[ejecta_fraction_index] == 0.0)

    # The surrogate's response to the pressure perturbation in a resting cell at
    # the entropy threshold: the compression sigmoid is 1/2 at div v = 0 and the
    # entropy sigmoid has slope 1 / (4 width) there. With the default width the
    # quiescent leak is ~0.4 % of this; the test allows 1 %.
    compression_sigmoid_at_rest = 0.5
    entropy_sigmoid_slope_at_threshold = 0.25
    response_at_threshold = (
        compression_sigmoid_at_rest
        * entropy_sigmoid_slope_at_threshold
        / smooth_latch_config.ad_shock_latch_entropy_width
        * relative_pressure_perturbation
    )
    quiescent_leak = np.abs(np.asarray(quiescent_tangent[SHOCKED_FRACTION_SLOT]))[ambient]
    assert quiescent_leak.max() < 0.01 * response_at_threshold

    # --------------- ↑ Quiescent gas ↑ ----------------


def test_shock_arrival_tangent():
    """
    ``ad_shock_arrival`` leaves the primal of the shock-history update unchanged
    bit for bit, adds nothing in quiescent never-shocked gas, and gives the
    accumulators the derivative of the shock ARRIVAL time: on the blast, the
    derivative of the time since shock and of the ionization age with respect to
    the ejecta speed (which moves the arrival) matches central finite
    differences, where the boolean latch's tangent (transport only) misses most
    of it. The comparison uses the exact WENO-weight tangent: the frozen-weight
    linearisation misplaces the shock front itself (it gave 57 % of the
    finite-difference value for the time since shock).
    """
    config, registered_variables, state = setup_blast()
    arrival_config = config._replace(ad_shock_arrival=True)
    fluid_registry = fluid_only_registry(registered_variables)

    # --------------- ↓ The update: primal and quiescent gas ↓ ----------------

    evolved_state = time_integration(state, config, blast_params(t_end=2 * DT_FIXED), registered_variables)
    previous_fluid = get_fluid_state(state, registered_variables)
    evolved_fluid = get_fluid_state(evolved_state, registered_variables)
    evolved_history = get_shock_history(evolved_state, registered_variables)
    pressure_index = registered_variables.pressure_index

    def history_update(latch_config, fluid_state, previous, history):
        def update(pressure):
            return update_shock_history(
                history,
                fluid_state.at[pressure_index].set(pressure),
                1e-3,
                GAMMA,
                latch_config.shock_entropy_jump,
                latch_config,
                fluid_registry,
                previous_primitive_state=previous,
            )
        return update

    pressure = evolved_fluid[pressure_index]
    plain, plain_tangent = jax.jvp(history_update(config, evolved_fluid, previous_fluid, evolved_history),
                                   (pressure,), (pressure * 1e-2,))
    arrival, arrival_tangent = jax.jvp(history_update(arrival_config, evolved_fluid, previous_fluid,
                                                      evolved_history), (pressure,), (pressure * 1e-2,))
    np.testing.assert_array_equal(_bits(plain), _bits(arrival))
    assert bool(jnp.all(jnp.isfinite(arrival_tangent)))
    # the arrival term acts on the accumulators only, and only where a crossing happened
    changed = np.asarray(arrival_tangent[TIME_SINCE_SHOCK_SLOT] != plain_tangent[TIME_SINCE_SHOCK_SLOT])
    assert changed.any()
    np.testing.assert_array_equal(np.asarray(arrival_tangent[SHOCKED_FRACTION_SLOT]),
                                  np.asarray(plain_tangent[SHOCKED_FRACTION_SLOT]))

    # quiescent, never-shocked gas: the initial state, unchanged from the previous one
    initial_fluid = get_fluid_state(state, registered_variables)
    initial_history = get_shock_history(state, registered_variables)
    initial_pressure = initial_fluid[pressure_index]
    _, quiet_plain = jax.jvp(history_update(config, initial_fluid, initial_fluid, initial_history),
                             (initial_pressure,), (initial_pressure * 1e-2,))
    _, quiet_arrival = jax.jvp(history_update(arrival_config, initial_fluid, initial_fluid, initial_history),
                               (initial_pressure,), (initial_pressure * 1e-2,))
    np.testing.assert_array_equal(np.asarray(quiet_arrival), np.asarray(quiet_plain))

    # --------------- ↑ The update: primal and quiescent gas ↑ ----------------

    # --------------- ↓ The blast: arrival-time derivative vs FD ↓ ----------------

    weights = jnp.asarray(np.random.default_rng(0).random(state.shape[1:]))
    velocity_rows = [registered_variables.velocity_index.x, registered_variables.velocity_index.y,
                     registered_variables.velocity_index.z]
    params = blast_params(t_end=0.06)

    def accumulators(latch_config):
        def objective(speed_factor):
            scaled = state
            for row in velocity_rows:
                scaled = scaled.at[row].set(state[row] * speed_factor)
            history = get_shock_history(time_integration(scaled, latch_config, params, registered_variables),
                                        registered_variables)
            return jnp.stack([jnp.sum(weights * history[TIME_SINCE_SHOCK_SLOT]),
                              jnp.sum(weights * history[DENSITY_TIME_SLOT])])
        return objective

    exact = dict(weno_ad_frozen_weights=False)
    step = 0.02
    finite_difference = (accumulators(config._replace(**exact))(1.0 + step)
                         - accumulators(config._replace(**exact))(1.0 - step)) / (2 * step)
    _, latch_derivative = jax.jvp(accumulators(config._replace(**exact)), (1.0,), (1.0,))
    _, arrival_derivative = jax.jvp(accumulators(config._replace(ad_shock_arrival=True, **exact)), (1.0,), (1.0,))
    ratio = np.asarray(arrival_derivative / finite_difference)
    # 2026-10-12: 0.96 (time since shock), 1.07 (ionization age); the latch alone 0.06 / 0.34
    assert np.all((ratio > 0.85) & (ratio < 1.15)), ratio
    assert np.all(np.asarray(latch_derivative / finite_difference) < 0.5)

    # --------------- ↑ The blast: arrival-time derivative vs FD ↑ ----------------


def test_masked_substeps_reverse_vs_dynamic_jvp():
    """
    With several passive-scalar sub-steps (some taken, some masked off), the VJP
    of the masked sub-step loop is the transpose of the JVP of the dynamic loop,
    through the scalar advection and the smooth-latch shock-history update (whose
    custom JVP is transposed here).
    """
    configs = {}
    for substep_loop in (SUBSTEPS_DYNAMIC, SUBSTEPS_MASKED):
        configs[substep_loop], registered_variables, state = setup_blast(
            passive_scalar_cfl=0.02,
            max_passive_scalar_substeps=6,
            passive_scalar_substep_loop=substep_loop,
            ad_smooth_shock_latch=True,
        )
    fluid_registry = fluid_only_registry(registered_variables)
    fluid_state = get_fluid_state(state, registered_variables)
    passive_scalars = state[registered_variables.passive_scalar_index:]
    # Where the shock history starts within the passive-scalar block.
    history_offset = (
        registered_variables.shock_history_index - registered_variables.passive_scalar_index
    )
    dt = 4e-3

    dynamic_config = configs[SUBSTEPS_DYNAMIC]
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
        """One passive-scalar step (advection, then shock history) under ``scalar_config``."""

        def scalar_step(fluid_state, passive_scalars, dt):
            advected = advect_passive_scalars(
                passive_scalars,
                fluid_state,
                dt,
                scalar_config.grid_spacing,
                scalar_config,
                fluid_registry,
            )
            history = update_shock_history(
                advected[history_offset:],
                fluid_state,
                dt,
                GAMMA,
                scalar_config.shock_entropy_jump,
                scalar_config,
                fluid_registry,
            )
            return jnp.concatenate([advected[:history_offset], history], axis=0)

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
        make_scalar_step(configs[SUBSTEPS_DYNAMIC]),
        (fluid_state, passive_scalars, dt),
        (fluid_direction, scalar_direction, dt_direction),
    )
    masked_result, masked_vjp = jax.vjp(
        make_scalar_step(configs[SUBSTEPS_MASKED]),
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
    the reverse-differentiable masked sub-step loop gives the result of the loop
    with a traced trip count: round-off apart in the passive scalars, bit for bit
    in the fluid.
    """
    final_states = {}
    for substep_loop in (SUBSTEPS_DYNAMIC, SUBSTEPS_MASKED):
        config, registered_variables, state = setup_blast(
            passive_scalar_cfl=0.02,
            max_passive_scalar_substeps=6,
            passive_scalar_substep_loop=substep_loop,
        )
        final_states[substep_loop] = np.asarray(
            time_integration(state, config, blast_params(), registered_variables)
        )
    masked_state = final_states[SUBSTEPS_MASKED]
    dynamic_state = final_states[SUBSTEPS_DYNAMIC]

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
    for row in range(passive_scalar_index, registered_variables.num_vars):
        np.testing.assert_allclose(
            masked_state[row],
            dynamic_state[row],
            rtol=0,
            atol=1e-13 * max(np.abs(dynamic_state[row]).max(), 1e-30),
        )
    np.testing.assert_array_equal(
        masked_state[:passive_scalar_index],
        dynamic_state[:passive_scalar_index],
    )


def test_grad_backwards_and_dot_product_fixed_step():
    """
    The dot-product test on one function: the JVP and the VJP of a fixed-step
    ``BACKWARDS`` run (masked sub-steps, scan time loop, stage rematerialisation
    on the reverse side, which the JVP does not see) satisfy
    ``<J^T w, u> = w . J u`` to round-off, and the gradient reaches the
    composition scalars and the shock history.
    """
    config, registered_variables, state = setup_blast(
        fixed_timestep=True,
        num_timesteps=NUM_FIXED_STEPS,
        differentiation_mode=BACKWARDS,
        ad_remat=AD_REMAT_STAGE,
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

    # The gradient reaches the ejecta fraction, which mostly sits on its bounds,
    # and the shock history.
    ejecta_fraction_index = registered_variables.passive_scalar_index + EJECTA_FRACTION_SCALAR
    density_time_index = registered_variables.shock_history_index + DENSITY_TIME_SLOT
    assert float(jnp.abs(jacobian_transpose_cotangent[ejecta_fraction_index]).max()) > 0.0
    assert float(jnp.abs(jacobian_transpose_cotangent[density_time_index]).max()) > 0.0


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
    ``ad_remat`` changes what the backward pass stores, not what it computes.

    ``"stage"`` is covered by the dot-product test above. ``"axis"`` adds per-axis
    and per-scalar checkpoints; with ``ad_remat_chunks = 4`` each axis' increment
    is in addition split into four checkpointed z / y slabs, which leaves the
    forward pass bit for bit unchanged.
    """
    gradients = {}
    final_states = {}
    for remat_mode, remat_chunks in ((AD_REMAT_NONE, 1), (AD_REMAT_AXIS, 1), (AD_REMAT_AXIS, 4)):
        config, registered_variables, state = setup_blast(
            fixed_timestep=True,
            num_timesteps=1,
            differentiation_mode=BACKWARDS,
            ad_remat=remat_mode,
            ad_remat_chunks=remat_chunks,
        )
        weights = jnp.asarray(np.random.default_rng(4).normal(size=state.shape))
        gradient, final_state = _jitted_gradient_and_final_state(
            config,
            blast_params(t_end=DT_FIXED),
            registered_variables,
            state,
            weights,
        )
        gradients[remat_mode, remat_chunks] = gradient
        final_states[remat_mode, remat_chunks] = final_state

    scale = np.abs(gradients[AD_REMAT_NONE, 1]).max()
    np.testing.assert_allclose(
        gradients[AD_REMAT_AXIS, 1],
        gradients[AD_REMAT_NONE, 1],
        rtol=0,
        atol=1e-12 * scale,
    )
    np.testing.assert_allclose(
        gradients[AD_REMAT_AXIS, 4],
        gradients[AD_REMAT_AXIS, 1],
        rtol=0,
        atol=1e-12 * scale,
    )
    np.testing.assert_array_equal(final_states[AD_REMAT_AXIS, 4], final_states[AD_REMAT_AXIS, 1])


def test_scalar_lean_same_gradient():
    """
    ``ad_scalar_lean`` (the reverse-mode memory layout of the passive-scalar
    block: an equinox loop over the flow-derived sub-step count, checkpointed
    ratio recovery and shock history) changes what the backward pass stores, not
    what it computes, with sub-cycling active (several sub-steps), on the
    fixed-step and on the adaptive, equinox-checkpointed path.
    """
    for fixed_timestep in (True, False):
        if fixed_timestep:
            step_options = dict(fixed_timestep=True, num_timesteps=1)
            t_end = DT_FIXED
        else:
            step_options = {}
            t_end = T_END
        gradients = {}
        final_states = {}
        for scalar_lean in (False, True):
            config, registered_variables, state = setup_blast(
                differentiation_mode=BACKWARDS,
                ad_remat=AD_REMAT_AXIS,
                ad_scalar_lean=scalar_lean,
                passive_scalar_cfl=0.02,
                max_passive_scalar_substeps=6,
                **step_options,
            )
            weights = jnp.asarray(np.random.default_rng(5).normal(size=state.shape))
            gradients[scalar_lean], final_states[scalar_lean] = _jitted_gradient_and_final_state(
                config,
                blast_params(t_end=t_end),
                registered_variables,
                state,
                weights,
            )
        scale = np.abs(gradients[False]).max()
        assert np.all(np.isfinite(gradients[True]))
        np.testing.assert_allclose(gradients[True], gradients[False], rtol=0, atol=1e-11 * scale)
        np.testing.assert_allclose(final_states[True], final_states[False], rtol=1e-13, atol=0)


def test_config_guards():
    """
    ``finalize_config`` refuses an unknown ``ad_remat`` mode, and
    ``OPTIMAL_BACKEND`` resolves to NATIVE_JAX when JAX runs on the CPU.
    """
    num_vars = get_registered_variables(blast_config()).num_vars
    state_shape = (num_vars, NUM_CELLS, NUM_CELLS, NUM_CELLS)
    with pytest.raises(ValueError, match="ad_remat"):
        finalize_config(blast_config(ad_remat="everything"), state_shape)
    if jax.devices()[0].platform == "cpu":
        config = finalize_config(
            blast_config(backend_config=BackendConfig(backend=OPTIMAL_BACKEND)),
            state_shape,
        )
        assert config.backend_config.backend == NATIVE_JAX


# -----------------------------------------------------------------------------
# ============================= ↑ Tests ↑ =====================================
# -----------------------------------------------------------------------------
