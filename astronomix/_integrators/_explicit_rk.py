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
        Hook applied at the start of every stage (e.g. boundary handling).
        Defaults to the identity.

    rhs(u, dt_stage) -> du
        The stage increment ``dt_stage * L(u)``, a pytree matching ``u``.

    post_stage(u) -> u   [ssprk4, ssprk33]
        Hook applied to each new stage state as soon as it is formed, before
        it is accumulated or used as the base of the next increment (e.g.
        rebuilding the cell-centered MHD fields from the interface fields).
        The final SSPRK4 combination keeps the ``_SSPRK4_FINAL_RAW`` share of
        each stage on the state as formed, so a modifying hook leaves that
        combination convex. Defaults to the identity.

    finalize(u) -> u
        Hook applied once after the last stage (e.g. recomputing the
        cell-centered MHD fields from the interface fields).  Defaults to the
        identity.

    lsrk_increment(u, du, a_coef, dt) -> du   [optional, low-storage only]
        Returns the new low-storage register ``a_coef * du + dt * L(u)``.
        Used by ``lsrk4`` to let a model fuse the ``a_coef * du`` accumulate
        into its flux-divergence kernel (memory optimisation).  When omitted,
        ``lsrk4`` falls back to ``a_coef * du + rhs(u, dt)``.

Schemes:
    ssprk4   - 5-stage 4th-order Spiteri-Ruuth SSPRK (3-register).
    ssprk33  - 3-stage 3rd-order Shu-Osher SSPRK (2-register).
    lsrk4    - 5-stage 4th-order Carpenter-Kennedy 2N-storage low-storage RK.
"""

# general
from fractions import Fraction

# jax
import jax
import jax.numpy as jnp


def _identity(u):
    """Identity hook: return the state pytree unchanged (the default stage hook)."""
    return u


def _tree_add(x, y):
    """Return the elementwise pytree sum ``x + y``."""
    return jax.tree_util.tree_map(lambda x_, y_: x_ + y_, x, y)


# ---------------------------------------------------------------------------
# SSPRK4 — Spiteri & Ruuth (2002) 5-stage, 4th-order, strong-stability-
# preserving Runge-Kutta.  Three registers: the initial state ``u0``, the
# running stage state ``u_curr`` and the accumulating ``u_final``.
# ---------------------------------------------------------------------------

#: The Shu-Osher form of SSPRK(5,4) to 15 digits (Gottlieb, Ketcheson & Shu
#: 2009): stage ``i`` is ``u_i = sum_j a_ij u_j + dt sum_j b_ij L(u_j)`` with
#: the nonzero entries listed here. All weights are non-negative, which is what
#: makes the scheme strong-stability preserving.
_SSPRK4_SHU_OSHER = dict(
    b10=0.391752226571890,
    a20=0.444370493651235,
    a21=0.555629506348765,
    b21=0.368410593050371,
    a30=0.620101851488403,
    a32=0.379898148511597,
    b32=0.251891774271694,
    a40=0.178079954393132,
    a43=0.821920045606868,
    b43=0.544974750228521,
    a52=0.517231671970585,
    a53=0.096059710526147,
    b53=0.063692468666290,
    a54=0.386708617503269,
    b54=0.226007483236906,
)


def _weights_summing_to_one(weights):
    """
    Adjust the weight of smallest magnitude so that the float64 weights sum
    to exactly one.

    The state weights of every stage sum to one in exact arithmetic, but not
    after rounding to float64; the excess then scales the whole state once per
    step, an error that grows linearly with the number of steps and caps the
    conservation of quadratic quantities such as the gravitational energy.
    The adjustment is a few units in the last place.

    Args:
        weights: The state weights of one stage.

    Returns:
        The weights, with the smallest one replaced by one minus the others
        (exactly, in rational arithmetic).
    """
    weights = list(weights)
    slack_index = min(range(len(weights)), key=lambda index: abs(weights[index]))
    others = sum(
        Fraction(weight) for index, weight in enumerate(weights) if index != slack_index
    )
    weights[slack_index] = float(1 - others)
    if sum(Fraction(weight) for weight in weights) != 1:
        raise ValueError("the stage weights cannot be made to sum to one exactly")
    return tuple(weights)


def _three_register_coefficients():
    """
    The SSPRK(5,4) coefficients in the three-register form of ``ssprk4``.

    The final Shu-Osher stage also needs ``L(u_3)``, which the three registers
    no longer hold. It is eliminated through
    ``u_4 = a40 u_0 + a43 u_3 + b43 dt L(u_3)``: with ``r = b53 / b43``,

        u_5 = -r a40 u_0 + a52 u_2 + (a53 - r a43) u_3 + (a54 + r) u_4 + b54 dt L(u_4).

    Returns:
        ``(k0, krhs, kcurr, final, final_raw)``: per stage, the weight of
        ``u_0``, the time-step weight of the increment and the weight of the
        current stage state; the weights of ``u_0 ... u_4`` in the final
        combination; and the part ``r`` of the ``u_4`` weight that stands for
        the eliminated ``b53 dt L(u_3)`` (indexed by the loop stage 0..3 that
        forms the state).
    """
    table = _SSPRK4_SHU_OSHER
    ratio = table["b53"] / table["b43"]
    second = _weights_summing_to_one((table["a20"], table["a21"]))
    third = _weights_summing_to_one((table["a30"], table["a32"]))
    fourth = _weights_summing_to_one((table["a40"], table["a43"]))
    final = _weights_summing_to_one(
        (
            -ratio * table["a40"],
            table["a52"],
            table["a53"] - ratio * table["a43"],
            table["a54"] + ratio,
        )
    )
    k0 = (1.0, second[0], third[0], fourth[0], final[0])
    krhs = (table["b10"], table["b21"], table["b32"], table["b43"], table["b54"])
    kcurr = (0.0, second[1], third[1], fourth[1], final[3])
    final_weights = (final[0], 0.0, final[1], final[2], final[3])
    final_raw = (0.0, 0.0, 0.0, ratio)
    return k0, krhs, kcurr, final_weights, final_raw


#: The final SSPRK4 combination carries a negative ``u_0`` weight
#: (``-r a40``): the eliminated term ``b53 dt L(u_3)`` stands in for part of
#: ``u_4``. ``_SSPRK4_FINAL_RAW`` is that part ``r`` of the ``u_4`` weight; it
#: must multiply ``u_4`` exactly as formed, so that a ``post_stage`` that
#: changes ``u_4`` leaves the final combination convex.
(
    _SSPRK4_K0,
    _SSPRK4_KRHS,
    _SSPRK4_KCURR,
    _SSPRK4_FINAL,
    _SSPRK4_FINAL_RAW,
) = _three_register_coefficients()


def ssprk4(u0, dt, *, rhs, pre_stage=_identity, post_stage=_identity, finalize=_identity):
    """5-stage 4th-order SSPRK (Spiteri-Ruuth).

    Args:
        u0: Initial state pytree at ``t = t_n``.
        dt: Full time step.
        rhs: ``rhs(u, dt_stage) -> du`` stage increment.
        pre_stage: Per-stage hook, defaults to the identity.
        post_stage: Applied to each new stage state as soon as it is formed,
            before it is accumulated or used as the base of the next increment;
            defaults to the identity.
        finalize: Post-integration hook, defaults to the identity.

    Returns:
        The state pytree at ``t = t_n + dt``.
    """
    k0_s = jnp.asarray(_SSPRK4_K0)
    krhs_s = jnp.asarray(_SSPRK4_KRHS)
    kcurr_s = jnp.asarray(_SSPRK4_KCURR)
    final_s = jnp.asarray(_SSPRK4_FINAL)
    final_raw_s = jnp.asarray(_SSPRK4_FINAL_RAW)

    def stage(stage_idx, carry):
        u_curr, u_final = carry

        u_curr = pre_stage(u_curr)

        k0 = k0_s[stage_idx]
        kcurr = kcurr_s[stage_idx]
        ff = final_s[stage_idx + 1]

        du = rhs(u_curr, krhs_s[stage_idx] * dt)

        # u_curr <- k0 * u0 + kcurr * u_curr + du
        u_formed = jax.tree_util.tree_map(
            lambda q0_, c_, r_: k0 * q0_ + kcurr * c_ + r_, u0, u_curr, du
        )
        if post_stage is _identity:
            u_curr = u_formed
            # u_final <- u_final + ff * u_curr
            u_final = jax.tree_util.tree_map(
                lambda f_, c_: f_ + ff * c_, u_final, u_curr
            )
        else:
            u_curr = post_stage(u_formed)
            # u_final <- u_final + ff * u_curr (the raw share on u as formed)
            ff_raw = final_raw_s[stage_idx]
            u_final = jax.tree_util.tree_map(
                lambda f_, c_, r_: f_ + (ff - ff_raw) * c_ + ff_raw * r_,
                u_final,
                u_curr,
                u_formed,
            )

        return (u_curr, u_final)

    u_final_init = jax.tree_util.tree_map(lambda x: final_s[0] * x, u0)

    # Stages 0..3 in a loop; the 5th stage uses a different final combination
    # (it adds its raw increment to ``u_final`` without the k0/kcurr update),
    # matching the Spiteri-Ruuth construction. Its right-hand side is evaluated
    # after the pre_stage hook like every other one, so the boundary is filled
    # before each of the five right-hand sides.
    u4, u_final = jax.lax.fori_loop(0, 4, stage, (u0, u_final_init))

    du4 = rhs(pre_stage(u4), krhs_s[4] * dt)
    u_final = _tree_add(u_final, du4)

    return finalize(u_final)


# ---------------------------------------------------------------------------
# The third-order SSPRK(3,3) scheme, in which every stage is built from the
# initial state and the previous stage only,
#
#     u_i = k0_i u_0 + kcurr_i u_{i-1} + krhs_i dt L(u_{i-1}),   i = 1 .. s,
#
# so two registers (``u_0`` and the running stage state) suffice. Every stage
# is a convex combination of ``u_0`` and one forward-Euler step of size
# ``(krhs_i / kcurr_i) dt`` (``krhs_i dt`` where ``kcurr_i`` is listed as 0:
# the first stage, ``u_1 = u_0 + krhs_1 dt L(u_0)``); the SSP coefficient is
# the smallest ``kcurr_i / krhs_i``.
# ---------------------------------------------------------------------------

#: SSPRK(3,3) of Shu & Osher (1988), SSP coefficient 1:
#:     u_1 = u_0 + dt L(u_0)
#:     u_2 = 3/4 u_0 + 1/4 (u_1 + dt L(u_1))
#:     u_3 = 1/3 u_0 + 2/3 (u_2 + dt L(u_2))
#: as ``(k0, kcurr, krhs)`` per stage.
_SSPRK33_STAGES = (
    (1.0, 0.0, 1.0),
    _weights_summing_to_one((3.0 / 4.0, 1.0 / 4.0)) + (1.0 / 4.0,),
    _weights_summing_to_one((1.0 / 3.0, 2.0 / 3.0)) + (2.0 / 3.0,),
)


def _two_register_ssprk(u0, dt, stages, *, rhs, pre_stage, post_stage, finalize):
    """
    Advance ``u0`` by ``dt`` with a scheme of the form

        u_i = k0_i u_0 + kcurr_i u_{i-1} + krhs_i dt L(u_{i-1}),   i = 1 .. s.

    ``pre_stage`` is applied to the state every right-hand side is evaluated
    at, ``post_stage`` to every intermediate stage state ``u_1 .. u_{s-1}`` as
    soon as it is formed, and ``finalize`` to ``u_s``. Every increment is
    evaluated at the state it is added to, and the weights of ``u_0`` and of
    the (hooked) previous stage are non-negative, so a ``post_stage`` that
    changes the stage states leaves every stage a convex combination of
    forward-Euler steps.

    Args:
        u0: Initial state pytree at ``t = t_n``.
        dt: Full time step.
        stages: Per stage, ``(k0, kcurr, krhs)``.
        rhs: ``rhs(u, dt_stage) -> du`` stage increment.
        pre_stage: Per-stage hook.
        post_stage: Hook on each intermediate stage state.
        finalize: Post-integration hook.

    Returns:
        The state pytree at ``t = t_n + dt``.
    """
    k0_s = jnp.asarray([stage[0] for stage in stages])
    kcurr_s = jnp.asarray([stage[1] for stage in stages])
    krhs_s = jnp.asarray([stage[2] for stage in stages])
    last = len(stages) - 1

    def formed(stage_idx, u_curr):
        u_curr = pre_stage(u_curr)
        du = rhs(u_curr, krhs_s[stage_idx] * dt)
        k0 = k0_s[stage_idx]
        kcurr = kcurr_s[stage_idx]
        # k0 * u0 + kcurr * u_curr + du
        return jax.tree_util.tree_map(
            lambda q0_, c_, r_: k0 * q0_ + kcurr * c_ + r_, u0, u_curr, du
        )

    def stage(stage_idx, u_curr):
        return post_stage(formed(stage_idx, u_curr))

    # The intermediate stages in a loop (one compiled instance of the
    # right-hand side), the last one outside it: it is finalized, not hooked.
    u_last = jax.lax.fori_loop(0, last, stage, u0)
    return finalize(formed(last, u_last))


def ssprk33(u0, dt, *, rhs, pre_stage=_identity, post_stage=_identity, finalize=_identity):
    """3-stage 3rd-order SSPRK of Shu & Osher (1988), SSP coefficient 1.

        u_1 = u_0 + dt L(u_0)
        u_2 = 3/4 u_0 + 1/4 (u_1 + dt L(u_1))
        u_3 = 1/3 u_0 + 2/3 (u_2 + dt L(u_2))

    Args:
        u0: Initial state pytree at ``t = t_n``.
        dt: Full time step.
        rhs: ``rhs(u, dt_stage) -> du`` stage increment.
        pre_stage: Per-stage hook, defaults to the identity.
        post_stage: Applied to the stage states ``u_1`` and ``u_2`` as soon as
            they are formed; defaults to the identity.
        finalize: Post-integration hook, defaults to the identity.

    Returns:
        The state pytree at ``t = t_n + dt``.
    """
    return _two_register_ssprk(
        u0, dt, _SSPRK33_STAGES, rhs=rhs, pre_stage=pre_stage, post_stage=post_stage, finalize=finalize
    )


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

    Args:
        u0: Initial state pytree at ``t = t_n``.
        dt: Full time step.
        pre_stage: Per-stage hook, defaults to the identity.
        finalize: Post-integration hook, defaults to the identity.
        rhs: ``rhs(u, dt) -> du`` increment, or None.
        lsrk_increment: ``lsrk_increment(u, du, a_coef, dt) -> du`` fused
            register update, or None.

    Returns:
        The state pytree at ``t = t_n + dt``.
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
