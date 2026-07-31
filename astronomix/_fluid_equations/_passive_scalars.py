"""
Passive scalars (advected per-parcel labels) for the finite-difference solver.

A passive scalar ``C`` is a label carried by the fluid without acting back on it:

    d(rho C)/dt + div(rho C v) = 0            <=>      dC/dt + v.grad C = 0

They are what turns a single-fluid remnant simulation into something that can be
compared with a spectrum rather than with a photograph: chemical stratification
per element (Orlando et al.'s ``C_el``), an ejecta-versus-circumstellar
discriminator, and — through the shock bookkeeping below — the non-equilibrium
ionization age ``n_e t`` and the electron/ion temperature relaxation.

**Why this is operator-split rather than an extra characteristic field.**
The WENO flux is characteristic-wise: adding scalars to it means extending the
eigenstructure (they ride the entropy wave at speed ``u``) in the native kernel,
in the hand-written adjoints and in the Pallas kernels, all of which must stay
bit-compatible with each other. The dual-energy density ``g`` set the precedent
for carrying an extra advected field outside that machinery, and the same choice
is made here — at the cost of an O(dt) splitting error in the scalars only,
which for a tracer is far cheaper than the risk.

Unlike ``g``, though, first-order upwind will not do. Every use of these scalars
is about a *contact discontinuity* — the ejecta/CSM interface, the boundary of an
Fe-rich knot — and first-order upwind smears a contact over ~sqrt(N_steps)
cells, which over the ~10^4 steps of a remnant run erases exactly the structure
the scalars exist to track. So the advection here is 5th-order WENO in space with
SSP-RK3 in time, the same combination the hydro solver uses.

**Boundedness comes from a CONSISTENT MASS FLUX.** A companion density ``rho~``
is advected alongside, and the scalar flux is built as

    F_s = F_rho * C_face

— the *same* numerical mass flux that updates ``rho~``, multiplied by the upwind
reconstructed value of the RATIO. The update of ``s`` is then a combination
weighted consistently with the update of ``rho~``, so ``C = s / rho~`` inherits a
maximum principle. Reconstructing ``rho C`` independently — the obvious
implementation, and the one tried first here — does not: the two reconstructions
use different data and therefore different nonlinear weights, the ratio has no
bound, and at a PERSISTENT contact discontinuity (the ejecta/circumstellar
interface sits in the same cells for the whole run) the error is one-signed and
RATCHETS. Measured: the ejecta fraction climbed ~0.8% per step, 1.0 -> 1.85 in
40 steps, overflowing float32 into NaN over a full run and taking every
composition field with it. Sharing the WENO weights across the stack was tried
as a fix and only slowed the ratchet to ~0.3% per step, because WENO5's
reconstruction has negative coefficients and is not monotone whatever weights it
is given. The consistency has to come from the flux.

Reconstructing the ratio rather than ``rho C`` is better physics too: ``C`` is
smooth across a shock, where ``rho`` is not.

Two lessons are worth recording because they nearly hid this. A SHORT
integration cannot distinguish a bounded overshoot from a slow ratchet — the
first 50 steps of this bug read as a harmless 0.7% WENO overshoot, and were
documented as such. And the validation suite ran in float64 while production
runs in float32, so the overflow it ends in could not appear in testing. The
suite now integrates 4x longer and checks that the excursion does not grow.

**Shock bookkeeping (``config.track_shock_history``).** Four further scalars are
managed by the library, implementing the ionization-age proxy of Dwarkadas, Dewey
& Bauer (2010) as used by Orlando et al. (2015). Rather than integrating an
ionization network, each parcel carries how long it has been shocked and how much
electron column it has swept:

* ``entropy_initial`` — the parcel's specific entropy ``log(p / rho^gamma)`` at
  ``t = 0``, advected and never rewritten. Entropy is constant along particle
  paths in smooth adiabatic flow and rises only across shocks, so comparing the
  current entropy against the parcel's own initial value is a clean, dt- and
  resolution-independent test of "has this parcel been through a shock", with no
  need to store a per-step history or tune a per-step threshold. Since the
  Rankine-Hugoniot jump fixes the entropy rise as a function of Mach number
  alone, the threshold ``config.shock_entropy_jump`` is really a minimum shock
  strength: the default ``log(2)`` corresponds to Mach 3.3 at ``gamma = 5/3``,
  against Mach ~100 (7.1 nats) for a young remnant's shocks and 0.06 nats for a
  Sod tube. Weak compressions and sound waves, which carry no ionization, are
  correctly ignored.
* ``shocked_fraction`` — how much of the parcel has been through a shock,
  advected and saturating at 1. It is what latches a parcel as shocked after it
  stops converging, and it is a FRACTION rather than a boolean flag on purpose:
  see :func:`update_shock_history`.
* ``time_since_shock`` — accumulates ``shocked_fraction * dt``; this is Orlando's
  ``Delta t_j = t - t_sh,j``, which drives the Coulomb electron/ion relaxation.
* ``density_time`` — accumulates ``shocked_fraction * rho dt``. In code units
  this is the ionization age up to a constant: ``n_e t = density_time *
  unit_density * unit_time / (mu_e m_p)``. Accumulating the integral directly
  avoids having to carry a shock *time* stamp, which mixes badly under advection
  (averaging a never-shocked parcel with a long-shocked one would produce a
  meaningless intermediate stamp).

All three are *history* variables that ride with the parcel, not instantaneous
flags, and that distinction matters when interpreting them: a parcel shocked at
an earlier step keeps its accumulated record, so later mixing with unshocked
material can leave a cell holding a positive ionization age while its *current*
entropy contrast has fallen back below the threshold. That is the intended
behaviour — it is what "this material has been through a shock" means once the
material moves and mixes — but it means a test of the form "every flagged cell
is currently above the threshold" will fail, and should.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    STATE_TYPE,
    SimulationConfig,
)

# astronomix containers
from astronomix.variable_registry.registered_variables import (
    NUM_SHOCK_HISTORY_SCALARS,
    RegisteredVariables,
)

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix._finite_difference._interface_fluxes._weno_weights import (
    _weno_omega_weights,
    _weno_omega_weights_z,
)

#: the WENO smoothness-indicator floor, matching the hydro kernels
_TINY = 1e-40


def _velocity_components(primitive_state, config, rv):
    """The velocity components of the primitive state, as a list per dimension."""
    if config.dimensionality == 1:
        return [primitive_state[rv.velocity_index]]
    if config.dimensionality == 2:
        return [primitive_state[rv.velocity_index.x],
                primitive_state[rv.velocity_index.y]]
    return [primitive_state[rv.velocity_index.x],
            primitive_state[rv.velocity_index.y],
            primitive_state[rv.velocity_index.z]]


def specific_entropy(primitive_state, gamma, registered_variables):
    """``log(p / rho^gamma)`` — constant along particle paths except across shocks."""
    rho = jnp.maximum(primitive_state[registered_variables.density_index], 1e-30)
    p = jnp.maximum(primitive_state[registered_variables.pressure_index], 1e-30)
    return jnp.log(p) - gamma * jnp.log(rho)


def _weno5_left_biased(q0, q1, q2, q3, q4, epsilon, omega_weights):
    """WENO5 reconstruction at the right face of the middle cell.

    ``q0..q4`` are the values at ``i-2 .. i+2``; the result is the state at
    ``i+1/2`` reconstructed from the left, i.e. the upwind value for a positive
    face velocity. The optimal linear weights are ``(1, 6, 3)/10``, matching the
    convention of :func:`_weno_omega_weights`, which returns ``(omega_0,
    omega_2)``.
    """
    p0 = (2.0 * q0 - 7.0 * q1 + 11.0 * q2) / 6.0
    p1 = (-q1 + 5.0 * q2 + 2.0 * q3) / 6.0
    p2 = (2.0 * q2 + 5.0 * q3 - q4) / 6.0

    # smoothness indicators (Jiang & Shu 1996)
    c13 = 13.0 / 12.0
    IS0 = c13 * (q0 - 2.0 * q1 + q2) ** 2 + 0.25 * (q0 - 4.0 * q1 + 3.0 * q2) ** 2
    IS1 = c13 * (q1 - 2.0 * q2 + q3) ** 2 + 0.25 * (q1 - q3) ** 2
    IS2 = c13 * (q2 - 2.0 * q3 + q4) ** 2 + 0.25 * (3.0 * q2 - 4.0 * q3 + q4) ** 2

    w0, w2 = omega_weights(IS0, IS1, IS2, epsilon, _TINY)
    w1 = 1.0 - w0 - w2
    return w0 * p0 + w1 * p1 + w2 * p2


def _advection_rhs(fields, velocities, grid_spacing, epsilon, omega_weights):
    """``-div(fields * v)`` for a stack of conserved densities.

    ``fields`` has a leading field axis, so spatial axis ``a`` is array axis
    ``a + 1``. The face velocity is the mean of the two neighbours and the
    interface value is upwinded on its sign, both sides reconstructed to 5th
    order.
    """
    # `fields` holds the CONSERVED stack (rho~, s = rho~ C); the flux needs the
    # ratio, so recover it here at every Runge-Kutta stage
    density = fields[0]
    safe = jnp.where(density > 0.0, density, 1.0)
    ratios = fields[1:] / safe[None, ...]
    rhs_rho = jnp.zeros_like(density)
    rhs_s = jnp.zeros_like(ratios)

    def faces(q_stack, sa, vf, bcast):
        """Upwind WENO5 face value of a (possibly stacked) field."""
        q = [_shift(q_stack, sh, axis=sa) for sh in (2, 1, 0, -1, -2, -3)]
        left = _weno5_left_biased(q[0], q[1], q[2], q[3], q[4], epsilon, omega_weights)
        right = _weno5_left_biased(q[5], q[4], q[3], q[2], q[1], epsilon, omega_weights)
        return jnp.where(bcast >= 0.0, left, right)

    for axis, v in enumerate(velocities):
        vf = 0.5 * (v + _shift(v, -1, axis=axis))       # face velocity at i+1/2

        # ONE mass flux, built from the density reconstruction ...
        rho_face = faces(density, axis, vf, vf)
        flux_rho = vf * rho_face

        # ... and the scalar flux is that SAME mass flux times the upwind value
        # of the RATIO. This is what makes the recovered C bounded, and it is
        # why reconstructing rho*C independently (the obvious implementation)
        # does not work: with independent reconstructions the update of s is not
        # a convex combination weighted consistently with the update of rho, so
        # C = s/rho~ has no maximum principle. At a persistent contact the error
        # is one-signed and RATCHETS -- measured at ~0.8% per step on the
        # ejecta fraction, reaching NaN over a full run. Sharing the WENO
        # weights alone only slowed it to ~0.3% per step; the consistency has to
        # come from the flux, not the weights.
        #
        # Reconstructing the ratio rather than rho*C is also better physics: C
        # is smooth across a shock, where rho is not.
        c_face = faces(ratios, axis + 1, vf, vf[None, ...])
        flux_s = flux_rho[None, ...] * c_face

        rhs_rho = rhs_rho - (flux_rho - _shift(flux_rho, 1, axis=axis)) / grid_spacing
        rhs_s = rhs_s - (flux_s - _shift(flux_s, 1, axis=axis + 1)) / grid_spacing

    return jnp.concatenate([rhs_rho[None, ...], rhs_s], axis=0)


def _substep_count(velocities, grid_spacing, dt, config):
    """How many sub-steps this advection needs, from the flow itself.

    Two conditions have to hold over a sub-step ``h``:

    * the advection CFL, ``max|u| h / dx <= cfl``;
    * positivity of the companion density, ``h max|div v| <= 0.5`` — the
      explicit update ``rho~ - h div(rho v)`` is what goes negative in a strong
      compression, and a negative denominator is what makes the recovered ratio
      diverge.

    Returned as a traced integer so ``lax.fori_loop`` runs exactly once on a
    benign step; the cap keeps a pathological cell from stalling the run.
    """
    inv_dx = 1.0 / grid_spacing
    vmax = jnp.max(jnp.stack([jnp.max(jnp.abs(v)) for v in velocities]))
    div_v = sum(0.5 * (_shift(v, -1, axis=a) - _shift(v, 1, axis=a)) * inv_dx
                for a, v in enumerate(velocities))
    divmax = jnp.max(jnp.abs(div_v))

    tiny = 1e-30
    h_adv = config.passive_scalar_cfl * grid_spacing / jnp.maximum(vmax, tiny)
    h_div = 0.5 / jnp.maximum(divmax, tiny)
    h_safe = jnp.minimum(h_adv, h_div)
    n = jnp.ceil(dt / jnp.maximum(h_safe, tiny))
    return jnp.clip(n, 1, config.max_passive_scalar_substeps).astype(jnp.int32)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def advect_passive_scalars(
    scalars: Float[Array, "..."],
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    grid_spacing: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Advance the passive scalars by one operator-split step.

    Args:
        scalars: The per-parcel labels, shape ``(n_scalars,) + grid``.
        primitive_state: The primitive state at the start of the step, whose
            density and velocity define the advecting flow (frozen over ``dt``).
        dt: The hydro time step. It is NOT automatically safe for this
            advection — see :func:`_substep_count` — so the step is sub-cycled
            as the flow requires.
        grid_spacing: The cell width.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The updated scalars, same shape as ``scalars``.
    """
    epsilon = config.weno_epsilon
    omega_weights = _weno_omega_weights_z if config.weno_z else _weno_omega_weights

    density = primitive_state[registered_variables.density_index]
    velocities = _velocity_components(primitive_state, config, registered_variables)

    # Advect (rho~, rho~ C_0, ..., rho~ C_{n-1}) as one stack through the
    # identical operator; see the module docstring for why the companion density
    # rather than the hydro solver's updated density is what keeps C bounded.
    stacked = jnp.concatenate([density[None, ...], scalars * density[None, ...]], axis=0)

    def L(u):
        return _advection_rhs(u, velocities, grid_spacing, epsilon, omega_weights)

    def ssprk3(u, h):
        """One SSP-RK3 (Shu & Osher 1988) step, the companion to WENO5."""
        u1 = u + h * L(u)
        u2 = 0.75 * u + 0.25 * (u1 + h * L(u1))
        return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + h * L(u2))

    # SUB-CYCLING. The hydro timestep is limited by |u| + c, which bounds this
    # advection's |u| but NOT the companion density's positivity: the explicit
    # update stays positive only while dt |div v| < 1, and at the hydro's
    # C_cfl = 0.3 the margin is only ~0.6. Where radiative cooling and a fast
    # ejecta piston compress the same cells that margin is spent, rho~ crosses
    # zero, and the recovered ratio runs away (measured: an ejecta fraction
    # reaching -87; removing cooling, the piston, OR dropping to C_cfl = 0.05
    # each cured it on its own).
    #
    # The substep count is computed from the flow rather than configured, so a
    # benign step costs exactly what it did before (n = 1 and the loop runs
    # once); only the violent steps pay.
    n_sub = _substep_count(velocities, grid_spacing, dt, config)
    h = dt / n_sub.astype(stacked.dtype)
    u_new = jax.lax.fori_loop(0, n_sub, lambda _, u: ssprk3(u, h), stacked)

    rho_new = u_new[0]

    # The recovered ratio ``s / rho~`` is meaningless wherever the companion
    # density has collapsed, and a supernova remnant's interior IS near vacuum
    # (rho ~ 3e-6 against ~1 in the shell). There numerator and denominator are
    # both tiny, WENO overshoot dominates their ratio, and the result feeds back
    # through ``s = rho C`` on the next step, so it amplifies. Unguarded this
    # destroyed a 256^3 remnant run: every composition scalar reached NaN and the
    # ionization age hit 1e17 in a few cells, while its median stayed perfectly
    # correct -- a failure that survives a glance at a figure.
    #
    # The guard is deliberately TARGETED. ``rho~`` is advected from ``rho`` over
    # a single step by a consistent operator, so it cannot legitimately fall by
    # six orders of magnitude; a cell where it has is pathological, and only
    # there is the high-order result replaced by the global range bound (which
    # advection cannot exceed anyway). Applying that bound everywhere instead
    # looks harmless and is not: it clips smooth extrema every step, a one-sided
    # error that accumulates and dropped the measured convergence order from
    # 5.0 to 1.8.
    rho_floor = 1e-6 * jnp.abs(density)
    safe = jnp.maximum(rho_new, rho_floor)
    safe = jnp.where(safe > 0.0, safe, 1.0)
    out = u_new[1:] / safe[None, ...]

    lo = jnp.min(scalars, axis=tuple(range(1, scalars.ndim)), keepdims=True)
    hi = jnp.max(scalars, axis=tuple(range(1, scalars.ndim)), keepdims=True)
    untrustworthy = (rho_new < rho_floor)[None, ...]
    out = jnp.where(untrustworthy,
                    jnp.clip(jnp.nan_to_num(out, nan=0.0), lo, hi),
                    out)

    # Physical bounds, where the caller has declared them. This is the backstop
    # that actually holds in production: the guard above only fires when the
    # companion density has collapsed to 1e-6 of the density it was advected
    # from, and a ratio is already meaningless well before that. A mass fraction
    # is bounded in [0, 1] by definition, so enforcing it costs nothing —
    # unlike clipping to a scalar's own current range, which clips smooth
    # extrema and destroys the order.
    bounds = _scalar_bounds(config, scalars.shape[0])
    if bounds is not None:
        out = jnp.clip(out, bounds[0], bounds[1])

    # The two shock-history ACCUMULATORS have no declarable upper bound -- they
    # grow with the run -- but they do have an exact one available here:
    # advection cannot raise a field's global maximum, only the source term can,
    # and the source is applied separately in `update_shock_history`. Clipping
    # them to their own pre-advection global maximum is therefore free of
    # physical content and is what stops the same s/rho~ pathology that the
    # declared bounds stop for the mass fractions. Without it a production run
    # ends with an ionization age of 1e10 in cells whose true value is O(1),
    # while every bounded field looks perfect.
    #
    # This is safe here for the reason the same clip was NOT safe applied to
    # every scalar: these are monotone diagnostics, so they have no smooth
    # extremum whose clipping would bias the solution.
    if config.track_shock_history:
        n_acc = 2                                   # time_since_shock, density_time
        acc = out[-n_acc:]
        hi_acc = jnp.max(scalars[-n_acc:],
                         axis=tuple(range(1, scalars.ndim)), keepdims=True)
        out = jnp.concatenate([out[:-n_acc], jnp.clip(acc, 0.0, hi_acc)], axis=0)
    return out


def _scalar_bounds(config, n_total):
    """``(lo, hi)`` arrays broadcastable over the scalar stack, or ``None``.

    User scalars take their bounds from ``config.passive_scalar_bounds``; the
    library-managed shock-history block knows its own — the shocked fraction is
    in [0, 1] and both accumulators are non-negative, while the initial-entropy
    label is unbounded.

    ``n_total`` comes from the scalar array itself, NOT from the registry: by
    the time this runs the registry has had the scalar block stripped off it, so
    its counts read zero.
    """
    n_hist = NUM_SHOCK_HISTORY_SCALARS if config.track_shock_history else 0
    n_user = n_total - n_hist

    declared = tuple(config.passive_scalar_bounds or ())
    if not declared and n_hist == 0:
        return None

    inf = jnp.inf
    los, his = [], []
    for k in range(n_user):
        b = declared[k] if k < len(declared) else None
        los.append(-inf if b is None else b[0])
        his.append(inf if b is None else b[1])
    if n_hist:
        # entropy_initial, shocked_fraction, time_since_shock, density_time
        los += [-inf, 0.0, 0.0, 0.0]
        his += [inf, 1.0, inf, inf]

    ndim = int(config.dimensionality)
    shape = (n_total,) + (1,) * ndim
    return jnp.asarray(los).reshape(shape), jnp.asarray(his).reshape(shape)


@partial(jax.jit, static_argnames=["config"])
def _fill_scalar_ghost_cells(scalars, config: SimulationConfig):
    """Fill the ghost zones of the passive-scalar block.

    The generic boundary handler cannot be reused: it negates the variable whose
    index equals the axis (the normal velocity), which for a scalar block would
    silently flip a composition field. Passive scalars are true scalars, so all
    three boundary types reduce to a copy — mirrored for reflective, wrapped for
    periodic, edge-extended for open — with no sign change anywhere.
    """
    ng = config.num_ghost_cells
    if ng == 0:
        return scalars

    bs = config.boundary_settings
    per_axis = (bs,) if config.dimensionality == 1 else (bs.x, bs.y, bs.z)

    for axis, axis_bs in enumerate(per_axis):
        sa = axis + 1                    # spatial axis within the scalar block

        def sl(start, stop):
            out = [slice(None)] * scalars.ndim
            out[sa] = slice(start, stop)
            return tuple(out)

        left, right = axis_bs.left_boundary, axis_bs.right_boundary
        if left == PERIODIC_BOUNDARY and right == PERIODIC_BOUNDARY:
            scalars = scalars.at[sl(0, ng)].set(scalars[sl(-2 * ng, -ng)])
            scalars = scalars.at[sl(-ng, None)].set(scalars[sl(ng, 2 * ng)])
            continue

        if left == OPEN_BOUNDARY:
            scalars = scalars.at[sl(0, ng)].set(scalars[sl(ng, ng + 1)])
        elif left == REFLECTIVE_BOUNDARY:
            scalars = scalars.at[sl(0, ng)].set(
                jnp.flip(scalars[sl(ng, 2 * ng)], axis=sa))

        if right == OPEN_BOUNDARY:
            scalars = scalars.at[sl(-ng, None)].set(scalars[sl(-ng - 1, -ng)])
        elif right == REFLECTIVE_BOUNDARY:
            scalars = scalars.at[sl(-ng, None)].set(
                jnp.flip(scalars[sl(-2 * ng, -ng)], axis=sa))

    return scalars


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def update_shock_history(
    history: Float[Array, "..."],
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    entropy_jump: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Advance the three shock-history scalars after the hydro update.

    ``history`` holds, in order, ``(entropy_initial, shocked_fraction,
    time_since_shock, density_time)`` — see the module docstring. A parcel counts as shocked once
    its specific entropy has risen by more than ``entropy_jump`` nats above the
    value it carried at ``t = 0``; ``entropy_jump = log(2)`` corresponds to a
    factor-two entropy rise, comfortably above numerical noise and well below
    the ~10 nats a strong SNR shock produces.

    Args:
        history: The three scalars, shape ``(3,) + grid``.
        primitive_state: The primitive state AFTER the hydro update.
        dt: The time step just taken.
        gamma: The adiabatic index.
        entropy_jump: The entropy-rise threshold, in nats.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The updated history scalars.
    """
    entropy_now = specific_entropy(primitive_state, gamma, registered_variables)

    # Two conditions for a FRESH shock crossing, then a latch.
    #
    # The entropy test alone over-triggers. ``entropy_initial`` is advected, so
    # wherever the initial state itself has an entropy discontinuity the
    # reconstruction smears it, and material sitting next to high-entropy gas
    # inherits a too-low label and looks shocked. In a strong shock tube that
    # flagged 96% of the box at a median rise of 0.26 nats -- i.e. almost
    # everything, and almost none of it above the threshold.
    #
    # Requiring the flow to be CONVERGING as well removes it: rarefaction fans
    # diverge and contact discontinuities have div(v) ~ 0, so only genuine
    # compressions can arm the flag. This is the standard shock criterion
    # (Pfrommer et al. 2017 use converging flow plus aligned temperature and
    # density gradients).
    #
    # The latch is what makes it a *history*: a parcel is no longer converging
    # once it is well downstream, but it is still shocked material, and its
    # ionization age must keep accumulating. ``time_since_shock > 0`` is itself
    # the latch, and it advects with the parcel, so the record follows the
    # material rather than the grid.
    velocities = _velocity_components(primitive_state, config, registered_variables)
    div_v = sum(0.5 * (_shift(v, -1, axis=a) - _shift(v, 1, axis=a))
                for a, v in enumerate(velocities))
    newly_shocked = ((entropy_now - history[0]) > entropy_jump) & (div_v < 0.0)

    # The latch is a FRACTION, not a boolean, and this matters more than it
    # looks. A boolean latch on "has this cell any accumulated time?" is
    # triggered by the infinitesimal advective leakage the scheme spreads into
    # neighbouring cells: a cell holding 1e-18 from numerical diffusion would
    # then accumulate at the FULL rate, and within a hundred steps the whole box
    # is flagged (measured: 100%). Carrying the shocked fraction as its own
    # advected scalar instead makes the accumulation proportional, so leakage
    # contributes in proportion to how much shocked material actually arrived.
    shocked_fraction = jnp.clip(
        jnp.maximum(history[1], jnp.where(newly_shocked, 1.0, 0.0)), 0.0, 1.0)

    density = primitive_state[registered_variables.density_index]
    # Both accumulators are monotone non-decreasing along a particle path by
    # construction, so any negative value is WENO ringing where the advected
    # field meets its own sharp edge at the shock front (measured: ~1e-4 of the
    # peak). Clamp it: a negative ionization age or a negative time since
    # shocking is meaningless downstream, and the clamp cannot mask a real
    # effect because the true value can never be below zero.
    time_since_shock = jnp.maximum(history[2] + shocked_fraction * dt, 0.0)
    density_time = jnp.maximum(history[3] + shocked_fraction * density * dt, 0.0)
    # entropy_initial is a t = 0 label: advected, never rewritten
    return jnp.stack([history[0], shocked_fraction, time_since_shock, density_time],
                     axis=0)
