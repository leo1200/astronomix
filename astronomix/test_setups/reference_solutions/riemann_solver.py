"""
Exact Riemann solver for the 1D ideal-gas Euler equations.

Implements the standard star-region pressure iteration of Toro (2009, ch. 4)
in pure JAX: bracket-and-bisect for the contact pressure ``p_star``, recover
the contact velocity ``u_star``, then sample the resulting self-similar fan at
each requested position. The fixed-iteration loops keep the routine jit- and
grad-compatible. The single-letter symbols (``g1..g7``, ``S``, ``f_K``) follow
Toro's notation so the code maps directly onto the reference equations.
"""

# jax
import jax
import jax.numpy as jnp

# Bisection iteration count. 60 halvings push the bracket below 2^-60 ~ 1e-18
# relative width, well past float64 machine epsilon.
_BISECT_ITERS = 60
# Bracket-expansion iteration count. Each step is a conditional *10
# multiplication; 20 steps allow up to 10^20 expansion.
_EXPAND_ITERS = 20


def _exact_riemann_ideal_gas(
    rho_L, u_L, p_L,
    rho_R, u_R, p_R,
    gamma,
    x: jnp.ndarray,
    t,
    x0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Exact solution of the 1D Riemann problem for an ideal gas with constant
    adiabatic index ``gamma``, in pure JAX.

    Implements the standard procedure of Toro (2009), chapter 4: bracket-solve
    for the star-region pressure on

        f(p) = f_L(p; W_L) + f_R(p; W_R) + (u_R - u_L) = 0,

    where each f_K is the shock or rarefaction branch depending on whether
    p is above or below p_K, recover u* from the consistency relation, then
    sample the resulting self-similar solution at S = (x - x0) / t.

    The function is jit- and grad-compatible. Vacuum-generation cases
    (G4 * (a_L + a_R) <= u_R - u_L) are not handled and will produce garbage.

    Args:
        rho_L, u_L, p_L: left state primitive variables.
        rho_R, u_R, p_R: right state primitive variables.
        gamma: ratio of specific heats.
        x: positions at which to evaluate the solution.
        t: time at which to evaluate the solution (must be > 0).
        x0: position of the initial diaphragm.

    Returns:
        rho, u, p: arrays of the primitive variables at the requested points.
    """
    x = jnp.asarray(x)
    a_L = jnp.sqrt(gamma * p_L / rho_L)
    a_R = jnp.sqrt(gamma * p_R / rho_R)

    # Toro's gamma-dependent constants (eqs. 4.x); named g1..g7 to match the text.
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)
    g3 = 2.0 * gamma / (gamma - 1.0)
    g4 = 2.0 / (gamma - 1.0)
    g5 = 2.0 / (gamma + 1.0)
    g6 = (gamma - 1.0) / (gamma + 1.0)
    g7 = (gamma - 1.0) / 2.0

    # -- Per-side pressure function (Toro eqs. 4.6 & 4.21) --------------------
    # Both branches are evaluated and selected with `where`; the unused branch
    # must remain finite to keep grads NaN-free, hence the p_safe clamp.
    def f_K(p, p_K, rho_K, a_K):
        p_safe = jnp.maximum(p, 1e-300)
        A_K = g5 / rho_K
        B_K = g6 * p_K
        shock = (p_safe - p_K) * jnp.sqrt(A_K / (p_safe + B_K))
        rare  = g4 * a_K * ((p_safe / p_K) ** g1 - 1.0)
        return jnp.where(p > p_K, shock, rare)

    def f(p):
        return f_K(p, p_L, rho_L, a_L) + f_K(p, p_R, rho_R, a_R) + (u_R - u_L)

    # -- Bracket expansion ---------------------------------------------------
    # f is monotonically increasing in p; grow p_hi (each step a conditional
    # *10) until f(p_hi) > 0. Fixed-iteration so reverse-mode autodiff works.
    def expand_body(_, state):
        p_hi, f_hi = state
        mult = jnp.where(f_hi < 0.0, 10.0, 1.0)
        p_hi_new = p_hi * mult
        return (p_hi_new, f(p_hi_new))

    p_hi_init = 10.0 * jnp.maximum(p_L, p_R)
    p_hi, _ = jax.lax.fori_loop(
        0, _EXPAND_ITERS, expand_body, (p_hi_init, f(p_hi_init))
    )
    p_lo = jnp.asarray(1e-14, dtype=p_hi.dtype)

    # -- Bisection -----------------------------------------------------------
    def bisect_body(_, state):
        lo, hi = state
        mid   = 0.5 * (lo + hi)
        f_lo  = f(lo)
        f_mid = f(mid)
        same  = f_lo * f_mid > 0.0
        return (jnp.where(same, mid, lo), jnp.where(same, hi, mid))

    p_lo, p_hi = jax.lax.fori_loop(0, _BISECT_ITERS, bisect_body, (p_lo, p_hi))
    p_star = 0.5 * (p_lo + p_hi)

    u_star = 0.5 * (u_L + u_R) + 0.5 * (
        f_K(p_star, p_R, rho_R, a_R) - f_K(p_star, p_L, rho_L, a_L)
    )

    # -- Sample at each x ----------------------------------------------------
    # S is the self-similar coordinate; the wave structure is constant along S.
    S = (x - x0) / t

    # ============= Left of the contact ==============
    # Shock sub-case: piecewise constant in S
    S_L_shock      = u_L - a_L * jnp.sqrt(g2 * p_star / p_L + g1)
    rho_sL_shock   = rho_L * (p_star / p_L + g6) / (g6 * p_star / p_L + 1.0)
    rho_left_shock = jnp.where(S < S_L_shock, rho_L, rho_sL_shock)
    u_left_shock   = jnp.where(S < S_L_shock, u_L,   u_star)
    p_left_shock   = jnp.where(S < S_L_shock, p_L,   p_star)

    # Rarefaction sub-case: head, fan, tail, star.
    # Clamping `inner_L` keeps the unused-branch power finite for autodiff.
    a_sL          = a_L * (p_star / p_L) ** g1
    S_HL          = u_L    - a_L
    S_TL          = u_star - a_sL
    rho_sL_rare   = rho_L * (p_star / p_L) ** (1.0 / gamma)
    inner_L       = jnp.maximum(g5 + g6 / a_L * (u_L - S), 1e-30)
    rho_fan_L     = rho_L * inner_L ** g4
    u_fan_L       = g5 * (a_L + g7 * u_L + S)
    p_fan_L       = p_L   * inner_L ** g3
    rho_left_rare = jnp.where(S < S_HL, rho_L,
                              jnp.where(S > S_TL, rho_sL_rare, rho_fan_L))
    u_left_rare   = jnp.where(S < S_HL, u_L,
                              jnp.where(S > S_TL, u_star,      u_fan_L))
    p_left_rare   = jnp.where(S < S_HL, p_L,
                              jnp.where(S > S_TL, p_star,      p_fan_L))

    is_shock_L = p_star > p_L
    rho_left = jnp.where(is_shock_L, rho_left_shock, rho_left_rare)
    u_left   = jnp.where(is_shock_L, u_left_shock,   u_left_rare)
    p_left   = jnp.where(is_shock_L, p_left_shock,   p_left_rare)

    # ============= Right of the contact ==============
    S_R_shock       = u_R + a_R * jnp.sqrt(g2 * p_star / p_R + g1)
    rho_sR_shock    = rho_R * (p_star / p_R + g6) / (g6 * p_star / p_R + 1.0)
    rho_right_shock = jnp.where(S > S_R_shock, rho_R, rho_sR_shock)
    u_right_shock   = jnp.where(S > S_R_shock, u_R,   u_star)
    p_right_shock   = jnp.where(S > S_R_shock, p_R,   p_star)

    a_sR           = a_R * (p_star / p_R) ** g1
    S_HR           = u_R    + a_R
    S_TR           = u_star + a_sR
    rho_sR_rare    = rho_R * (p_star / p_R) ** (1.0 / gamma)
    inner_R        = jnp.maximum(g5 - g6 / a_R * (u_R - S), 1e-30)
    rho_fan_R      = rho_R * inner_R ** g4
    u_fan_R        = g5 * (-a_R + g7 * u_R + S)
    p_fan_R        = p_R   * inner_R ** g3
    rho_right_rare = jnp.where(S > S_HR, rho_R,
                               jnp.where(S < S_TR, rho_sR_rare, rho_fan_R))
    u_right_rare   = jnp.where(S > S_HR, u_R,
                               jnp.where(S < S_TR, u_star,      u_fan_R))
    p_right_rare   = jnp.where(S > S_HR, p_R,
                               jnp.where(S < S_TR, p_star,      p_fan_R))

    is_shock_R = p_star > p_R
    rho_right = jnp.where(is_shock_R, rho_right_shock, rho_right_rare)
    u_right   = jnp.where(is_shock_R, u_right_shock,   u_right_rare)
    p_right   = jnp.where(is_shock_R, p_right_shock,   p_right_rare)

    # ============= Combine across the contact ==============
    left = S <= u_star
    rho = jnp.where(left, rho_left, rho_right)
    u   = jnp.where(left, u_left,   u_right)
    p   = jnp.where(left, p_left,   p_right)
    return rho, u, p
