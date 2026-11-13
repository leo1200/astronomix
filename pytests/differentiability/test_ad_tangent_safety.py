"""
The overflow-free derivative of the WENO-JS weights (``_weno_omega_weights_ad``).

JAX differentiates the weight quotient ``alpha_0 / alpha_sum`` through
``alpha_sum^-2``. Where every smoothness indicator of a stencil is large (a field
with an O(1e5) jump across the stencil) ``alpha_sum`` falls below ~5e-20 and
``alpha_sum^-2`` overflows float32, so even a ZERO cotangent gives
``0 * inf = NaN``. An unbounded passive scalar can develop such a jump in a
single collapsed cell; the NaN of the passive-scalar WENO backward pass then
spreads through the shared mass flux over the whole state gradient while the
objective stays finite.
``_weno_omega_weights_ad`` evaluates the same derivative as a log-derivative with
bounded factors; its primal is unchanged bit for bit.

What is checked:

* the primal of the overflow-free weights is that of the plain ones, bit for bit;
* their tangent is the exact derivative wherever autodiff of the plain weights
  is finite;
* on a recorded float32 stencil the plain VJP is NaN, the overflow-free one is
  finite and matches central differences;
* the VJP of the passive-scalar advection stays finite with that stencil in an
  unbounded passive scalar.

Run on the CPU (the conftest then selects fast-compiling XLA flags)::

    JAX_PLATFORMS=cpu python -m pytest pytests/differentiability/test_ad_tangent_safety.py
"""

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import BACKWARDS

# astronomix functions
from astronomix._finite_difference._interface_fluxes._weno_weights import (
    _weno_omega_weights,
    _weno_omega_weights_ad,
)
from astronomix._fluid_equations._passive_scalars import (
    _weno5_left_biased,
    advect_passive_scalars,
)

# shared blast setup (a sibling module of this test)
from _blast_setup import (
    NUM_CELLS,
    NUM_PASSIVE_SCALARS,
    PASSIVE_SCALAR_CONFIG,
    fluid_only_registry,
    get_fluid_state,
    setup_blast,
)

pytestmark = [pytest.mark.x64, pytest.mark.slow]


#: An unbounded passive scalar along x through a collapsed cell of a float32
#: remnant run, where the plain weights' VJP was NaN: an O(1e5) spike on a
#: background of about -20.
OVERFLOW_STENCIL = [
    -19.62940216064453,
    -19.648536682128906,
    -19.625972747802734,
    -19.603029251098633,
    -22.161638259887695,
    -778.392578125,
    -66914.5078125,
    -9.308792114257812,
    -18.60173988342285,
]

#: The regularisation ``epsilon`` added to the smoothness indicators.
WENO_EPSILON = 1e-7


def _reconstruct(values, omega_weights):
    """
    The left-biased WENO5 face values of the interior cells of a 1D stencil.

    Args:
        values: The cell values; the first and last two cells only serve as
            stencil support.
        omega_weights: The weight function, ``_weno_omega_weights`` or
            ``_weno_omega_weights_ad``.

    Returns:
        The values at the right faces of cells ``2, ..., len(values) - 3``.
    """
    face_values = [
        _weno5_left_biased(
            values[cell - 2],
            values[cell - 1],
            values[cell],
            values[cell + 1],
            values[cell + 2],
            WENO_EPSILON,
            omega_weights,
        )
        for cell in range(2, len(values) - 2)
    ]
    return jnp.stack(face_values)


def test_weno_weights_ad_primal_bitwise():
    """
    The overflow-free weights compute the primal of the plain ones bit for bit,
    in float32 and float64, for smoothness indicators spanning 24 decades and two
    alpha-sum floors, and so does a reconstruction of the overflow stencil.
    """
    rng = np.random.default_rng(0)
    for dtype in (jnp.float32, jnp.float64):
        smoothness_indicators = [
            jnp.asarray(10.0 ** rng.uniform(-12, 12, size=4096), dtype)
            for _ in range(3)
        ]
        for alpha_sum_floor in (1e-40, 1e-14):
            plain_weights = _weno_omega_weights(
                *smoothness_indicators,
                WENO_EPSILON,
                alpha_sum_floor,
            )
            overflow_free_weights = _weno_omega_weights_ad(
                *smoothness_indicators,
                WENO_EPSILON,
                alpha_sum_floor,
            )
            for plain_weight, overflow_free_weight in zip(plain_weights, overflow_free_weights):
                assert plain_weight.dtype == overflow_free_weight.dtype
                np.testing.assert_array_equal(
                    np.asarray(plain_weight).view(np.uint8),
                    np.asarray(overflow_free_weight).view(np.uint8),
                )

    stencil = jnp.asarray(OVERFLOW_STENCIL, jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(_reconstruct(stencil, _weno_omega_weights)),
        np.asarray(_reconstruct(stencil, _weno_omega_weights_ad)),
    )


def test_weno_weights_ad_tangent_exact():
    """
    The overflow-free tangent is the derivative of the plain weights by autodiff
    wherever that is finite (float64), for three indicator scales, and so is the
    derivative with respect to a traced epsilon.
    """

    def plain_weights(indicator_0, indicator_1, indicator_2):
        """The plain weights with the smallest alpha-sum floor."""
        return _weno_omega_weights(indicator_0, indicator_1, indicator_2, WENO_EPSILON, 1e-40)

    def overflow_free_weights(indicator_0, indicator_1, indicator_2):
        """The overflow-free weights with the smallest alpha-sum floor."""
        return _weno_omega_weights_ad(indicator_0, indicator_1, indicator_2, WENO_EPSILON, 1e-40)

    rng = np.random.default_rng(1)
    for scale in (1e-3, 1.0, 1e3):
        smoothness_indicators = []
        for _ in range(3):
            magnitude = np.abs(rng.normal(size=2000)) * scale ** 2
            spread = 10 ** rng.uniform(-6, 0, 2000)
            smoothness_indicators.append(jnp.asarray(magnitude * spread))
        indicator_tangents = [
            jnp.asarray(rng.normal(size=2000)) * smoothness_indicators[k]
            for k in range(3)
        ]
        _, plain_tangents = jax.jvp(plain_weights, smoothness_indicators, indicator_tangents)
        _, overflow_free_tangents = jax.jvp(
            overflow_free_weights,
            smoothness_indicators,
            indicator_tangents,
        )
        for plain_tangent, overflow_free_tangent in zip(plain_tangents, overflow_free_tangents):
            np.testing.assert_allclose(
                np.asarray(overflow_free_tangent),
                np.asarray(plain_tangent),
                rtol=1e-11,
                atol=1e-13 * float(jnp.max(jnp.abs(plain_tangent))),
            )

    # The derivative with respect to epsilon, on the indicators of the largest scale.
    def summed_plain_omega_0(epsilon):
        """The sum of the plain ``omega_0`` as a function of epsilon."""
        return _weno_omega_weights(*smoothness_indicators, epsilon, 1e-14)[0].sum()

    def summed_overflow_free_omega_0(epsilon):
        """The sum of the overflow-free ``omega_0`` as a function of epsilon."""
        return _weno_omega_weights_ad(*smoothness_indicators, epsilon, 1e-14)[0].sum()

    plain_epsilon_derivative = jax.grad(summed_plain_omega_0)(jnp.asarray(WENO_EPSILON))
    overflow_free_epsilon_derivative = jax.grad(summed_overflow_free_omega_0)(
        jnp.asarray(WENO_EPSILON)
    )
    np.testing.assert_allclose(
        float(overflow_free_epsilon_derivative),
        float(plain_epsilon_derivative),
        rtol=1e-10,
    )


def test_weno_weights_ad_no_nan_on_overflow_stencil():
    """
    On the overflow stencil (float32) the plain weights' VJP is NaN even for a
    zero cotangent; the overflow-free one is finite (and zero for a zero
    cotangent) and matches central differences of the plain reconstruction in
    float64.
    """

    def plain_reconstruction(values):
        """The WENO5 face values with the plain weights."""
        return _reconstruct(values, _weno_omega_weights)

    def overflow_free_reconstruction(values):
        """The WENO5 face values with the overflow-free weights."""
        return _reconstruct(values, _weno_omega_weights_ad)

    # --------------- ↓ float32 ↓ ----------------

    stencil = jnp.asarray(OVERFLOW_STENCIL, jnp.float32)
    face_values, plain_vjp = jax.vjp(plain_reconstruction, stencil)
    # The plain weights' VJP is NaN even for a zero cotangent.
    assert not np.all(np.isfinite(np.asarray(plain_vjp(jnp.zeros_like(face_values))[0])))
    face_values, overflow_free_vjp = jax.vjp(overflow_free_reconstruction, stencil)
    np.testing.assert_array_equal(
        np.asarray(overflow_free_vjp(jnp.zeros_like(face_values))[0]),
        0.0,
    )
    float32_gradient = np.asarray(overflow_free_vjp(jnp.ones_like(face_values))[0])
    assert np.all(np.isfinite(float32_gradient))

    # --------------- ↑ float32 ↑ ----------------

    # --------------- ↓ float64 central differences ↓ ----------------

    # The overflow-free VJP is the derivative: compare it with central
    # differences of the plain reconstruction in float64 on the same stencil.
    stencil_64 = jnp.asarray(OVERFLOW_STENCIL, jnp.float64)
    face_values_64, vjp_64 = jax.vjp(overflow_free_reconstruction, stencil_64)
    float64_gradient = np.asarray(vjp_64(jnp.ones_like(face_values_64))[0])
    relative_step = 1e-6
    central_differences = []
    for cell, value in enumerate(OVERFLOW_STENCIL):
        step = relative_step * abs(value)
        forward_sum = float(jnp.sum(plain_reconstruction(stencil_64.at[cell].add(step))))
        backward_sum = float(jnp.sum(plain_reconstruction(stencil_64.at[cell].add(-step))))
        central_differences.append((forward_sum - backward_sum) / (2 * relative_step * abs(value)))
    np.testing.assert_allclose(
        float64_gradient,
        np.array(central_differences),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(float32_gradient, float64_gradient, rtol=2e-3, atol=2e-4)

    # --------------- ↑ float64 central differences ↑ ----------------


def test_passive_scalar_advection_vjp_finite_with_overflow_stencil():
    """
    End to end through ``advect_passive_scalars`` in float32: with the overflow
    stencil in an unbounded passive scalar, the advected scalars and the VJP
    are finite, and the VJP is zero for a zero cotangent.
    """
    # BACKWARDS selects the masked (static) sub-step loop. Bounds are declared
    # for all but the last passive scalar, which is therefore unbounded.
    config, registered_variables, state = setup_blast(
        differentiation_mode=BACKWARDS,
        passive_scalar_config=PASSIVE_SCALAR_CONFIG._replace(
            bounds=tuple((0.0, 1.0) for _ in range(NUM_PASSIVE_SCALARS - 1)),
        ),
    )
    state = state.astype(jnp.float32)

    # The unbounded scalar: uniform at the stencil's background value, with the
    # overflow stencil along x through the centre of the box.
    unbounded_scalar_index = registered_variables.passive_scalar_index + NUM_PASSIVE_SCALARS - 1
    stencil_start = 5
    stencil_stop = stencil_start + len(OVERFLOW_STENCIL)
    box_centre = NUM_CELLS // 2
    unbounded_scalar = jnp.full(state.shape[1:], -19.6, jnp.float32)
    unbounded_scalar = unbounded_scalar.at[stencil_start:stencil_stop, box_centre, box_centre].set(
        jnp.asarray(OVERFLOW_STENCIL, jnp.float32)
    )
    state = state.at[unbounded_scalar_index].set(unbounded_scalar)

    fluid_registry = fluid_only_registry(registered_variables)
    fluid_state = get_fluid_state(state, registered_variables)
    passive_scalars = state[registered_variables.passive_scalar_index:]

    def advect(scalars, fluid):
        """One advection step of the passive scalars."""
        return advect_passive_scalars(
            scalars,
            fluid,
            jnp.float32(2e-3),
            config.grid_spacing,
            config,
            fluid_registry,
        )

    advected, advection_vjp = jax.vjp(advect, passive_scalars, fluid_state)
    assert np.all(np.isfinite(np.asarray(advected)))

    scalar_gradient, fluid_gradient = advection_vjp(jnp.zeros_like(advected))
    np.testing.assert_array_equal(np.asarray(scalar_gradient), 0.0)
    np.testing.assert_array_equal(np.asarray(fluid_gradient), 0.0)

    scalar_gradient, fluid_gradient = advection_vjp(jnp.ones_like(advected))
    assert np.all(np.isfinite(np.asarray(scalar_gradient)))
    assert np.all(np.isfinite(np.asarray(fluid_gradient)))
