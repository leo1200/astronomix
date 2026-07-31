"""WENO nonlinear weights, shared by the native and Pallas WENO kernels.

Lives in its own module because ``_weno.py`` imports ``_weno_pallas.py`` at
module level (backend dispatch), so a helper used by BOTH must sit below the
two of them in the import graph.
"""

# jax
import jax.numpy as jnp


def _weno_omega_weights(IS0, IS1, IS2, epsilon, tiny):
    """WENO weights ``(omega_0, omega_2)`` from the smoothness indicators.

    Mathematically ``omega_i = alpha_i / max(alpha_0 + alpha_1 + alpha_2, tiny)``
    with ``alpha_i = c_i / (epsilon + IS_i)^2``, ``(c_0, c_1, c_2) = (1, 6, 3)``.

    In f64 the classic form's five IEEE divisions per call dominate the GPU
    kernel cost (an f64 divide expands to ~15-20 SASS ops), so the quotient is
    cleared to a single reciprocal: multiplying numerator and denominator by
    ``(t_0 t_1 t_2)^2`` (``t_i = epsilon + IS_i``) gives
    ``omega_0 = u_0 / max(u_0 + u_1 + u_2, tiny * (t_0 t_1 t_2)^2)`` with
    ``u_0 = (t_1 t_2)^2``, ``u_1 = 6 (t_0 t_2)^2``, ``u_2 = 3 (t_0 t_1)^2`` —
    the same function including the ``tiny`` clamp, up to rounding.

    f32 keeps the classic form: its divisions are cheap (``div.full.f32``),
    and the ``u`` products would overflow f32 for characteristic-flux jumps
    beyond ~1e4 (t^4 scaling), where the classic form still degrades
    gracefully to ``omega -> 0``.
    """
    t0 = epsilon + IS0
    t1 = epsilon + IS1
    t2 = epsilon + IS2
    if t0.dtype == jnp.float64:
        u0 = (t1 * t2) ** 2
        u1 = 6.0 * (t0 * t2) ** 2
        u2 = 3.0 * (t0 * t1) ** 2
        prod = t0 * t1 * t2
        inv = 1.0 / jnp.maximum(u0 + u1 + u2, tiny * prod * prod)
        return u0 * inv, u2 * inv
    alpha0 = 1.0 / t0 ** 2
    alpha1 = 6.0 / t1 ** 2
    alpha2 = 3.0 / t2 ** 2
    alpha_sum = jnp.maximum(alpha0 + alpha1 + alpha2, tiny)
    return alpha0 / alpha_sum, alpha2 / alpha_sum


def _weno_omega_weights_z(IS0, IS1, IS2, epsilon, tiny):
    """WENO-Z weights ``(omega_0, omega_2)`` (Borges et al. 2008, p = 1).

    ``alpha_k = c_k (1 + tau_5 / (epsilon + IS_k))`` with the global smoothness
    indicator ``tau_5 = |IS_0 - IS_2|``. Two properties matter here:

    * **Scale invariance.** The nonlinearity is driven by the RATIO
      ``tau_5 / IS_k``, so it does not depend on the magnitude of the
      characteristic variables. The classic JS form
      ``alpha_k = c_k / (epsilon + IS_k)^2`` instead compares ``IS_k`` against
      an ABSOLUTE ``epsilon``, so with O(100) code variables it is always in
      the fully-nonlinear regime and discriminates between stencils even in
      perfectly smooth flow.
    * **Accuracy at smooth extrema.** In smooth regions ``tau_5`` is a higher
      order small quantity than ``IS_k``, so ``alpha_k -> c_k`` and the scheme
      recovers its optimal (5th-order) linear weights — including at critical
      points, where JS drops order and adds dissipation. That dissipation
      preferentially erases small-amplitude extrema (e.g. the incipient cold
      condensations of a thermal instability).
    """
    t0 = epsilon + IS0
    t1 = epsilon + IS1
    t2 = epsilon + IS2
    tau = jnp.abs(IS0 - IS2)
    alpha0 = 1.0 * (1.0 + tau / t0)
    alpha1 = 6.0 * (1.0 + tau / t1)
    alpha2 = 3.0 * (1.0 + tau / t2)
    alpha_sum = jnp.maximum(alpha0 + alpha1 + alpha2, tiny)
    return alpha0 / alpha_sum, alpha2 / alpha_sum


def _weno_omega_weights_adjoint(IS0, IS1, IS2, omega0_bar, omega2_bar, epsilon, tiny):
    """Hand transpose of :func:`_weno_omega_weights`.

    Returns ``(omega0, omega2, IS0_bar, IS1_bar, IS2_bar)`` — the recomputed
    forward weights plus the cotangents of the smoothness indicators.  The
    clamp branch follows the established hand-adjoint convention (ties route
    the gradient to the clamp side, matching the pre-existing
    ``jnp.where(s > tiny, ...)`` rule).
    """
    t0 = epsilon + IS0
    t1 = epsilon + IS1
    t2 = epsilon + IS2
    if t0.dtype == jnp.float64:
        q0 = t1 * t2
        q1 = t0 * t2
        q2 = t0 * t1
        u0 = q0 ** 2
        u1 = 6.0 * q1 ** 2
        u2 = 3.0 * q2 ** 2
        S = u0 + u1 + u2
        prod = t0 * q0
        clamp = tiny * prod * prod
        inv = 1.0 / jnp.maximum(S, clamp)
        omega0 = u0 * inv
        omega2 = u2 * inv
        u0_bar = omega0_bar * inv
        u2_bar = omega2_bar * inv
        inv_bar = omega0_bar * u0 + omega2_bar * u2
        denom_bar = -inv_bar * inv * inv
        S_bar = jnp.where(S > clamp, denom_bar, 0.0)
        prod_bar = jnp.where(S > clamp, 0.0, denom_bar) * tiny * 2.0 * prod
        u0_bar += S_bar
        u1_bar = S_bar
        u2_bar += S_bar
        q0_bar = 2.0 * u0_bar * q0
        q1_bar = 12.0 * u1_bar * q1
        q2_bar = 6.0 * u2_bar * q2
        t0_bar = q1_bar * t2 + q2_bar * t1 + prod_bar * q0
        t1_bar = q0_bar * t2 + q2_bar * t0 + prod_bar * q1
        t2_bar = q0_bar * t1 + q1_bar * t0 + prod_bar * q2
        return omega0, omega2, t0_bar, t1_bar, t2_bar
    alpha0 = 1.0 / t0 ** 2
    alpha1 = 6.0 / t1 ** 2
    alpha2 = 3.0 / t2 ** 2
    s3 = alpha0 + alpha1 + alpha2
    alpha_sum = jnp.maximum(s3, tiny)
    omega0 = alpha0 / alpha_sum
    omega2 = alpha2 / alpha_sum
    alpha0_bar = omega0_bar / alpha_sum
    alpha_sum_bar = omega0_bar * (-alpha0 / alpha_sum ** 2)
    alpha2_bar = omega2_bar / alpha_sum
    alpha_sum_bar += omega2_bar * (-alpha2 / alpha_sum ** 2)
    s3_bar = jnp.where(s3 > tiny, alpha_sum_bar, 0.0)
    alpha0_bar += s3_bar
    alpha1_bar = s3_bar
    alpha2_bar += s3_bar
    IS0_bar = alpha0_bar * (-2.0) * t0 ** (-3)
    IS1_bar = alpha1_bar * 6.0 * (-2.0) * t1 ** (-3)
    IS2_bar = alpha2_bar * 3.0 * (-2.0) * t2 ** (-3)
    return omega0, omega2, IS0_bar, IS1_bar, IS2_bar
