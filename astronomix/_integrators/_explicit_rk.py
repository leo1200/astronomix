"""
Generic explicit Runge-Kutta time-integration schemes.

These drivers are agnostic to the spatial discretisation and to the state
layout: the state ``u`` is an arbitrary JAX pytree (a single conserved-state
array for the hydro solvers, or the ``(q, bx, by, bz)`` tuple for the MHD
constrained-transport solver) and the *model* is supplied as a small set of
closures.  The finite-difference (WENO / CT) and finite-volume (unsplit)
solvers build those closures and call into the scheme; the heavy,
discretisation-specific work lives there, not here.

Model contract (all closures operate on the state pytree ``u``):

    pre_stage(u) -> u
        Hooks applied at the start of every stage (positivity flooring,
        boundary handling, ...).  Defaults to the identity.

    rhs(u, dt_stage) -> du
        The stage increment ``dt_stage * L(u)``, a pytree matching ``u``.

    finalize(u) -> u
        Hooks applied once after the last stage (e.g. recomputing the
        cell-centered MHD fields from the interface fields, final
        positivity).  Defaults to the identity.

    lsrk_increment(u, du, a_coef, dt) -> du   [optional, low-storage only]
        Returns the new low-storage register ``a_coef * du + dt * L(u)``.
        Used by ``lsrk4`` to let a model fuse the ``a_coef * du`` accumulate
        into its flux-divergence kernel (memory optimisation).  When omitted,
        ``lsrk4`` falls back to ``a_coef * du + rhs(u, dt)``.

Schemes:
    ssprk4   - 5-stage 4th-order Spiteri-Ruuth SSPRK (3-register).
    lsrk4    - 5-stage 4th-order Carpenter-Kennedy 2N-storage low-storage RK.
    rk2_ssp  - 2-stage 2nd-order SSP RK (Heun).
"""

# jax
import jax
import jax.numpy as jnp


def _identity(u):
    """Identity hook: return the state pytree unchanged (the default stage hook)."""
    return u


def _tree_axpby(a, x, b, y):
    """Return the pytree ``a * x + b * y`` (``a``, ``b`` scalars)."""
    return jax.tree_util.tree_map(lambda x_, y_: a * x_ + b * y_, x, y)


def _tree_add(x, y):
    """Return the elementwise pytree sum ``x + y``."""
    return jax.tree_util.tree_map(lambda x_, y_: x_ + y_, x, y)


# ---------------------------------------------------------------------------
# SSPRK4 — Spiteri & Ruuth (2002) 5-stage, 4th-order, strong-stability-
# preserving Runge-Kutta.  Three registers: the initial state ``u0``, the
# running stage state ``u_curr`` and the accumulating ``u_final``.
# ---------------------------------------------------------------------------
_SSPRK4_K0 = (1.0, 0.44437049406734, 0.62010185138540, 0.17807995410773, -2.081261929715610e-02)
_SSPRK4_KRHS = (0.39175222700392, 0.36841059262959, 0.25189177424738, 0.54497475021237, 0.22600748319395)
_SSPRK4_KCURR = (0.0, 0.55562950593266, 0.37989814861460, 0.82192004589227, 5.03580947213895e-01)
_SSPRK4_FINAL = (-2.081261929715610e-02, 0.0, 0.51723167208978, -6.518979800418380e-12, 5.03580947213895e-01)


def ssprk4(u0, dt, *, rhs, pre_stage=_identity, finalize=_identity):
    """5-stage 4th-order SSPRK (Spiteri-Ruuth).

    Args:
        u0: Initial state pytree at ``t = t_n``.
        dt: Full time step.
        rhs: ``rhs(u, dt_stage) -> du`` stage increment.
        pre_stage: per-stage hook, defaults to identity.
        finalize: post-integration hook, defaults to identity.

    Returns:
        The state pytree at ``t = t_n + dt``.
    """
    k0_s = jnp.asarray(_SSPRK4_K0)
    krhs_s = jnp.asarray(_SSPRK4_KRHS)
    kcurr_s = jnp.asarray(_SSPRK4_KCURR)
    final_s = jnp.asarray(_SSPRK4_FINAL)

    def stage(stage_idx, carry):
        u_curr, u_final = carry

        u_curr = pre_stage(u_curr)

        k0 = k0_s[stage_idx]
        kcurr = kcurr_s[stage_idx]
        ff = final_s[stage_idx + 1]

        du = rhs(u_curr, krhs_s[stage_idx] * dt)

        # u_curr <- k0 * u0 + kcurr * u_curr + du
        u_curr = jax.tree_util.tree_map(
            lambda q0_, c_, r_: k0 * q0_ + kcurr * c_ + r_, u0, u_curr, du
        )
        # u_final <- u_final + ff * u_curr
        u_final = jax.tree_util.tree_map(
            lambda f_, c_: f_ + ff * c_, u_final, u_curr
        )

        return (u_curr, u_final)

    u_final_init = jax.tree_util.tree_map(lambda x: final_s[0] * x, u0)

    # Stages 0..3 in a loop; the 5th stage uses a different final combination
    # (it adds its raw increment to ``u_final`` without the k0/kcurr update and
    # without a pre_stage hook), matching the Spiteri-Ruuth construction.
    u4, u_final = jax.lax.fori_loop(0, 4, stage, (u0, u_final_init))

    du4 = rhs(u4, krhs_s[4] * dt)
    u_final = _tree_add(u_final, du4)

    return finalize(u_final)


# ---------------------------------------------------------------------------
# LSRK4 — Carpenter & Kennedy (1994) 2N-storage, 5-stage, 4th-order low-storage
# Runge-Kutta.  Two registers: the state ``u`` and the accumulator ``du``:
#     du <- A[i] * du + dt * L(u);   u <- u + B[i] * du
# with A[0] = 0 so the first stage is a plain forward-Euler micro-step.
# ---------------------------------------------------------------------------
_LSRK4_A = (
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
)
_LSRK4_B = (
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
)


def lsrk4(u0, dt, *, pre_stage=_identity, finalize=_identity, rhs=None, lsrk_increment=None):
    """5-stage 4th-order Carpenter-Kennedy 2N-storage low-storage RK4.

    Either ``rhs`` or ``lsrk_increment`` must be supplied.  ``lsrk_increment``
    lets the model fuse the ``a_coef * du`` accumulate into its flux kernel;
    when only ``rhs`` is given the accumulate is done here via ``tree_map``.
    """
    if lsrk_increment is None and rhs is None:
        raise ValueError("lsrk4 requires either 'rhs' or 'lsrk_increment'.")

    dtype = jax.tree_util.tree_leaves(u0)[0].dtype
    a_s = jnp.asarray(_LSRK4_A, dtype=dtype)
    b_s = jnp.asarray(_LSRK4_B, dtype=dtype)

    def stage(stage_idx, carry):
        u, du = carry

        u = pre_stage(u)

        a_coef = a_s[stage_idx]
        b_coef = b_s[stage_idx]

        if lsrk_increment is not None:
            du = lsrk_increment(u, du, a_coef, dt)
        else:
            du = jax.tree_util.tree_map(
                lambda d_, r_: a_coef * d_ + r_, du, rhs(u, dt)
            )

        u = jax.tree_util.tree_map(lambda u_, d_: u_ + b_coef * d_, u, du)
        return (u, du)

    du0 = jax.tree_util.tree_map(jnp.zeros_like, u0)
    u_final, _ = jax.lax.fori_loop(0, 5, stage, (u0, du0))

    return finalize(u_final)


# ---------------------------------------------------------------------------
# RK2-SSP — 2-stage, 2nd-order strong-stability-preserving Runge-Kutta
# (Heun's method):  u1 = u0 + dt L(u0);  u2 = u1 + dt L(u1);  u = (u0 + u2)/2.
# ---------------------------------------------------------------------------
def rk2_ssp(u0, dt, *, rhs, pre_stage=_identity, finalize=_identity):
    """2-stage 2nd-order SSP RK (Heun)."""
    u = pre_stage(u0)
    u1 = _tree_add(u, rhs(u, dt))

    u1 = pre_stage(u1)
    u2 = _tree_add(u1, rhs(u1, dt))

    u_final = _tree_axpby(0.5, u0, 0.5, u2)
    return finalize(u_final)
