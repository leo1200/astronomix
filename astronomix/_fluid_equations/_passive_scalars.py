"""
Passive scalars (advected per-parcel labels), in every scheme.

A passive scalar ``C`` is a label carried by the fluid without acting back on it:

    d(rho C)/dt + div(rho C v) = 0            <=>      dC/dt + v.grad C = 0

Passive scalars are what lets a single-fluid simulation be compared with a
spectrum rather than only with an image: chemical stratification per element
(Orlando et al.'s ``C_el``) or an ejecta-versus-circumstellar discriminator.

**Operator splitting, the same for every scheme.** The block of scalars is
split off the end of the state before the solver step and advected with the
flow at the start of the step (``split_off_passive_scalars``), the solver
evolves the state without it, and the block is reattached after the step
(``reattach_passive_scalars``). The advection needs only the density and the
velocity, so the finite-difference and the finite-volume schemes share it.

**Why this is operator-split rather than an extra characteristic field.**
The WENO flux is characteristic-wise: adding scalars to it means extending the
eigenstructure (they ride the entropy wave at speed ``u``) in the native kernel,
in the hand-written adjoints and in the Pallas kernels, all of which must stay
bit-compatible with each other. The dual-energy density ``g`` set the precedent
for carrying an extra advected field outside that machinery, and the same choice
is made here, at the cost of an O(dt) splitting error in the scalars only.

Unlike for ``g``, first-order upwind is not good enough here. Every use of these
scalars is about a *contact discontinuity* (the ejecta / circumstellar
interface, the boundary of a metal-rich knot), and first-order upwind smears a
contact over ~sqrt(N_steps) cells, which over the ~10^4 steps of a long run
erases exactly the structure the scalars exist to track. So the advection here
is fifth-order WENO in space with SSP-RK3 in time, the third-order strong
stability preserving companion of WENO5 (Shu & Osher 1988), which is enough for
an operator-split tracer.

**Boundedness comes from a consistent mass flux.** A companion density ``rho~``
is advected alongside, and the scalar flux is built as

    F_s = F_rho * C_face

i.e. the *same* numerical mass flux that updates ``rho~``, multiplied by the
upwind reconstructed value of the RATIO. The update of ``s = rho~ C`` is then a
combination weighted consistently with the update of ``rho~``, so ``C = s /
rho~`` inherits a maximum principle. Reconstructing ``rho C`` independently does
not bound ``C``: the two reconstructions use different data and therefore
different nonlinear weights, and at a persistent contact discontinuity (one that
sits in the same cells for the whole run) the resulting error is one-signed and
accumulates step after step until it overflows. Sharing the WENO weights between
the two reconstructions does not cure this, because the WENO5 reconstruction has
negative coefficients and is not monotone whatever weights it is given; the
consistency has to come from the flux.

Reconstructing the ratio rather than ``rho C`` is better physics too: ``C`` is
smooth across a shock, where ``rho`` is not.

**Differentiability.** Two things make this module differentiable in reverse
mode as well as forward mode, both without changing the primal:

* the sub-cycling loop has a traced trip count, which reverse mode cannot
  differentiate; under ``differentiation_mode == BACKWARDS`` it becomes a
  static, masked loop (see :func:`advect_passive_scalars`);
* every bound clamp keeps derivative 1 ON its bound (``jnp.clip`` gives 0.5 at
  a tie, and these fields sit exactly on their bounds almost everywhere, so a
  plain clamp would halve the tangent every step).
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import (
    Array,
    Float,
)

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_DIFFERENCE,
    GHOST_CELLS,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix._finite_difference._interface_fluxes._weno_weights import (
    WENO_EPSILON,
    _weno_omega_weights_ad,
    _weno_omega_weights_z,
)

#: Floor on the sum of the WENO alphas (the ``tiny`` argument of
#: ``_weno_omega_weights*``). It only keeps the weight quotient finite where
#: every smoothness indicator of a stencil is huge.
_WENO_ALPHA_SUM_FLOOR = 1e-40

#: Floor keeping densities, pressures, speeds and divergences strictly positive
#: before they are divided by or logged.
_POSITIVE_FLOOR = 1e-30


# -----------------------------------------------------------------------------
# ========= ↓ Bound clamp whose derivative survives at the bound ↓ ===========
# -----------------------------------------------------------------------------

# ``jnp.clip`` splits the derivative evenly at a tie:
# ``grad(clip)(x = lo) = 0.5``. For these scalars a tie is the common case, not
# an edge case: in a remnant most cells hold a mass fraction of EXACTLY 0 or
# exactly 1. Advection keeps them exactly on the bound (the ratio of identical
# operators is exact), so the clamp ties again every step and the tangent /
# cotangent of those cells is halved every step: 0.5^30 ~ 1e-9 over a 30-step
# window. The clamp below has the same primal (the very same ``jnp`` call, so
# bit for bit) and passes the derivative of ``x`` through whenever ``x`` is
# inside the bounds OR on them; only a strictly out-of-bounds value takes the
# bound's.

@jax.custom_jvp
def _clip_keep_derivative(x, lo, hi):
    """``jnp.clip(x, lo, hi)`` with derivative 1 inside and ON the bounds."""
    return jnp.clip(x, lo, hi)


@_clip_keep_derivative.defjvp
def _clip_keep_derivative_jvp(primals, tangents):
    x, lo, hi = primals
    dx, dlo, dhi = tangents
    out = jnp.clip(x, lo, hi)
    dout = jnp.where(x < lo, dlo, jnp.where(x > hi, dhi, dx))
    return out, jnp.broadcast_to(dout, out.shape).astype(out.dtype)

# -----------------------------------------------------------------------------
# ========= ↑ Bound clamp whose derivative survives at the bound ↑ ===========
# -----------------------------------------------------------------------------


def _velocity_components(primitive_state, config, registered_variables):
    """The velocity components along the spatial axes, as a list per dimension (an MHD layout carries all three
    components in every dimensionality)."""
    velocity_index = registered_variables.velocity_index
    if isinstance(velocity_index, int):
        return [primitive_state[velocity_index]]
    return [primitive_state[index] for index in tuple(velocity_index)[: config.dimensionality]]


def _split_companion_stack(conserved_stack):
    """
    Split the advected conserved stack into its two parts.

    The stack is ``(rho~, rho~ C_0, ..., rho~ C_{n-1})`` along the leading
    axis: the companion density first, then the scalar densities.

    Returns:
        ``(companion_density, scalar_densities)``.
    """
    return conserved_stack[0], conserved_stack[1:]


def _weno5_left_biased(q0, q1, q2, q3, q4, epsilon, omega_weights):
    """
    WENO5 reconstruction at the right face of the middle cell.

    ``q0..q4`` are the values at ``i-2 .. i+2``; the result is the state at
    ``i+1/2`` reconstructed from the left, i.e. the upwind value for a positive
    face velocity. The optimal linear weights are ``(1, 6, 3)/10``, matching the
    convention of :func:`_weno_omega_weights`, which returns ``(omega_0,
    omega_2)``.

    Args:
        q0, q1, q2, q3, q4: The cell values at ``i-2 .. i+2``.
        epsilon: The WENO epsilon of the smoothness denominators.
        omega_weights: The nonlinear-weight function (one of the
            ``_weno_omega_weights*`` variants).

    Returns:
        The reconstructed value at ``i+1/2``.
    """
    # The three third-order candidate reconstructions at i+1/2.
    candidate_0 = (2.0 * q0 - 7.0 * q1 + 11.0 * q2) / 6.0
    candidate_1 = (-q1 + 5.0 * q2 + 2.0 * q3) / 6.0
    candidate_2 = (2.0 * q2 + 5.0 * q3 - q4) / 6.0

    # Smoothness indicators of the three candidate stencils (Jiang & Shu 1996).
    thirteen_twelfths = 13.0 / 12.0
    smoothness_0 = (
        thirteen_twelfths * (q0 - 2.0 * q1 + q2) ** 2
        + 0.25 * (q0 - 4.0 * q1 + 3.0 * q2) ** 2
    )
    smoothness_1 = (
        thirteen_twelfths * (q1 - 2.0 * q2 + q3) ** 2
        + 0.25 * (q1 - q3) ** 2
    )
    smoothness_2 = (
        thirteen_twelfths * (q2 - 2.0 * q3 + q4) ** 2
        + 0.25 * (3.0 * q2 - 4.0 * q3 + q4) ** 2
    )

    weight_0, weight_2 = omega_weights(
        smoothness_0,
        smoothness_1,
        smoothness_2,
        epsilon,
        _WENO_ALPHA_SUM_FLOOR,
    )
    weight_1 = 1.0 - weight_0 - weight_2
    return weight_0 * candidate_0 + weight_1 * candidate_1 + weight_2 * candidate_2


def _advection_rhs(
    conserved_stack,
    velocities,
    grid_spacing,
    epsilon,
    omega_weights,
):
    """
    ``-div(conserved_stack * v)`` for the advected conserved stack.

    The stack carries a leading field axis, so spatial axis ``a`` is array axis
    ``a + 1``. The face velocity is the mean of the two neighbours and the
    interface value is upwinded on its sign, both sides reconstructed to fifth
    order. The scalar fluxes are the companion density's mass flux times the
    upwind value of the ratio (see the module docstring).

    Args:
        conserved_stack: The conserved stack ``(rho~, rho~ C_0, ...)``.
        velocities: The velocity components, one field per dimension.
        grid_spacing: The cell width.
        epsilon: The WENO epsilon.
        omega_weights: The nonlinear-weight function.

    Returns:
        The right-hand side, same shape as ``conserved_stack``.
    """
    # The flux needs the ratio, so it is recovered from the conserved stack at
    # every Runge-Kutta stage.
    companion_density, scalar_densities = _split_companion_stack(conserved_stack)
    safe_density = jnp.where(companion_density > 0.0, companion_density, 1.0)
    ratios = scalar_densities / safe_density[None, ...]
    density_rhs = jnp.zeros_like(companion_density)
    scalar_rhs = jnp.zeros_like(ratios)

    def upwind_face_values(stack, array_axis, face_velocity):
        """Upwind WENO5 value at i+1/2 of a (possibly stacked) field."""
        q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3 = [
            _shift(stack, shift, axis=array_axis) for shift in (2, 1, 0, -1, -2, -3)
        ]
        from_left = _weno5_left_biased(
            q_im2,
            q_im1,
            q_i,
            q_ip1,
            q_ip2,
            epsilon,
            omega_weights,
        )
        from_right = _weno5_left_biased(
            q_ip3,
            q_ip2,
            q_ip1,
            q_i,
            q_im1,
            epsilon,
            omega_weights,
        )
        return jnp.where(face_velocity >= 0.0, from_left, from_right)

    def flux_differences(companion_density, ratios, velocity, grid_spacing, axis):
        """This axis' ``(F_{i+1/2} - F_{i-1/2}) / dx`` for rho~ and for s."""
        face_velocity = 0.5 * (velocity + _shift(velocity, -1, axis=axis))

        # ONE mass flux, built from the density reconstruction ...
        density_face = upwind_face_values(companion_density, axis, face_velocity)
        mass_flux = face_velocity * density_face

        # ... and the scalar flux is that SAME mass flux times the upwind value
        # of the RATIO. This consistent mass flux is what bounds the recovered
        # C; see the module docstring.
        density_flux_difference = (
            mass_flux - _shift(mass_flux, 1, axis=axis)
        ) / grid_spacing
        ratio_face = upwind_face_values(ratios, axis + 1, face_velocity[None, ...])
        scalar_flux = mass_flux[None, ...] * ratio_face

        return density_flux_difference, (
            scalar_flux - _shift(scalar_flux, 1, axis=axis + 1)
        ) / grid_spacing

    for axis, velocity in enumerate(velocities):
        density_flux_difference, scalar_flux_difference = flux_differences(
            companion_density,
            ratios,
            velocity,
            grid_spacing,
            axis,
        )
        density_rhs = density_rhs - density_flux_difference
        scalar_rhs = scalar_rhs - scalar_flux_difference

    return jnp.concatenate([density_rhs[None, ...], scalar_rhs], axis=0)


def _substep_count(velocities, grid_spacing, dt, config):
    """
    How many sub-steps this advection needs, from the flow itself.

    Two conditions have to hold over a sub-step ``h``:

    * the advection CFL, ``max|u| h / dx <= cfl``;
    * positivity of the companion density, ``h max|div v| <= 0.5``: the
      explicit update ``rho~ - h div(rho v)`` is what goes negative in a strong
      compression, and a negative denominator is what makes the recovered ratio
      diverge.

    Returned as a traced integer so ``lax.fori_loop`` runs exactly once on a
    benign step; the cap ``config.passive_scalar_config.max_substeps`` keeps a
    pathological cell from stalling the run.

    Args:
        velocities: The velocity components, one field per dimension.
        grid_spacing: The cell width.
        dt: The hydro time step.
        config: The simulation configuration.

    Returns:
        The number of sub-steps, an int32 scalar in
        ``[1, config.passive_scalar_config.max_substeps]``.
    """
    inverse_dx = 1.0 / grid_spacing
    max_speed = jnp.max(jnp.stack([jnp.max(jnp.abs(velocity)) for velocity in velocities]))
    velocity_divergence = sum(
        0.5 * (_shift(velocity, -1, axis=axis) - _shift(velocity, 1, axis=axis)) * inverse_dx
        for axis, velocity in enumerate(velocities)
    )
    max_abs_divergence = jnp.max(jnp.abs(velocity_divergence))

    advective_substep = (
        config.passive_scalar_config.cfl * grid_spacing / jnp.maximum(max_speed, _POSITIVE_FLOOR)
    )
    compressive_substep = 0.5 / jnp.maximum(max_abs_divergence, _POSITIVE_FLOOR)
    safe_substep = jnp.minimum(advective_substep, compressive_substep)
    num_substeps = jnp.ceil(dt / jnp.maximum(safe_substep, _POSITIVE_FLOOR))
    return jnp.clip(num_substeps, 1, config.passive_scalar_config.max_substeps).astype(jnp.int32)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def advect_passive_scalars(
    scalars: Float[Array, "..."],
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    grid_spacing: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Advance the passive scalars by one operator-split step.

    Args:
        scalars: The per-parcel labels, shape ``(n_scalars,) + grid``.
        primitive_state: The primitive state at the start of the step, whose
            density and velocity define the advecting flow (frozen over ``dt``).
        dt: The hydro time step. It is NOT automatically safe for this
            advection (see :func:`_substep_count`), so the step is sub-cycled
            as the flow requires.
        grid_spacing: The cell width.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The updated scalars, same shape as ``scalars``.
    """

    # -------------------------------------------------------------
    # ======= ↓ WENO weights and the advected conserved stack ↓ ====
    # -------------------------------------------------------------

    epsilon = WENO_EPSILON
    # The overflow-free ``_weno_omega_weights_ad`` gives the same weights bit
    # for bit, but a derivative that cannot overflow. These weights are always
    # differentiated, and an unbounded label with a large jump across the
    # stencil makes the float32 Jiang-Shu weight VJP NaN even for a zero
    # cotangent.
    weno_z = config.solver_mode == FINITE_DIFFERENCE and config.solver_config.weno_z
    omega_weights = _weno_omega_weights_z if weno_z else _weno_omega_weights_ad

    density = primitive_state[registered_variables.density_index]
    velocities = _velocity_components(primitive_state, config, registered_variables)

    # Advect (rho~, rho~ C_0, ..., rho~ C_{n-1}) as one stack through the
    # identical operator; see the module docstring for why the companion density
    # rather than the hydro solver's updated density is what keeps C bounded.
    conserved_stack = jnp.concatenate(
        [density[None, ...], scalars * density[None, ...]],
        axis=0,
    )

    # -------------------------------------------------------------
    # ======= ↑ WENO weights and the advected conserved stack ↑ ====
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =================== ↓ RHS rematerialisation ↓ ================
    # -------------------------------------------------------------

    # Reverse-mode rematerialisation of the right-hand side
    # (``config.rematerialize_stages``): the backward pass keeps only each
    # Runge-Kutta stage's input and recomputes the WENO internals.
    @jax.checkpoint
    def checkpointed_advection_rhs(stack, velocities, grid_spacing):
        return _advection_rhs(stack, velocities, grid_spacing, epsilon, omega_weights)

    def rematerialised_advection_rhs(stack):
        return checkpointed_advection_rhs(stack, velocities, grid_spacing)

    if config.rematerialize_stages:
        advection_rhs = rematerialised_advection_rhs
    else:
        def advection_rhs(stack):
            return _advection_rhs(stack, velocities, grid_spacing, epsilon, omega_weights)

    # -------------------------------------------------------------
    # =================== ↑ RHS rematerialisation ↑ ================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ Sub-cycled SSP-RK3 update ↓ ==============
    # -------------------------------------------------------------

    def ssprk3_step(stack, substep_dt, rhs=None):
        """One SSP-RK3 step (Shu & Osher 1988), the companion of WENO5."""
        rhs = advection_rhs if rhs is None else rhs
        stage_1 = stack + substep_dt * rhs(stack)
        stage_2 = 0.75 * stack + 0.25 * (stage_1 + substep_dt * rhs(stage_1))
        return (1.0 / 3.0) * stack + (2.0 / 3.0) * (stage_2 + substep_dt * rhs(stage_2))

    # SUB-CYCLING. The hydro time step is limited by |u| + c, which bounds this
    # advection's |u| but NOT the companion density's positivity: the explicit
    # update stays positive only while dt |div v| < 1, and at a typical hydro
    # C_cfl = 0.3 the margin is only ~0.6. Where strong compression (e.g. a
    # radiatively cooling shell hit by a fast piston) spends that margin, rho~
    # crosses zero and the recovered ratio runs away.
    #
    # The sub-step count is computed from the flow rather than configured, so a
    # benign step costs exactly one SSP-RK3 step (the loop runs once); only the
    # violent steps pay.
    #
    # The count is a traced integer, so ``fori_loop(0, num_substeps)`` is a
    # while loop, which reverse-mode AD cannot differentiate. Under
    # ``differentiation_mode == BACKWARDS`` the loop instead runs to the
    # static cap ``passive_scalar_config.max_substeps`` and skips the surplus sub-steps
    # with a ``lax.cond`` (only the taken branch executes): the same arithmetic
    # on the same data. With a cap of 1 this is just one straight-line sub-step.
    # Otherwise every sub-step, the first included, sits in ONE scan whose body
    # is ``jax.checkpoint``-ed, with its right-hand side rematerialised too,
    # whatever ``config.rematerialize_stages`` says: XLA allocates the buffers
    # of a conditional and of a loop body whether or not they run, and adds
    # them to the rest of the step's, so a separately differentiated first
    # sub-step plus a masked loop for the rest would need nearly twice the
    # backward memory of the single rematerialised scan. The price is one extra
    # evaluation of the advection in the backward pass. The count itself
    # carries no derivative (it is an integer; the flow it is derived from is
    # held constant for it explicitly).
    if config.differentiation_mode == BACKWARDS:
        num_substeps = _substep_count(
            [jax.lax.stop_gradient(velocity) for velocity in velocities],
            jax.lax.stop_gradient(grid_spacing),
            jax.lax.stop_gradient(dt),
            config,
        )
        substep_dt = dt / num_substeps.astype(conserved_stack.dtype)
        max_substeps = int(config.passive_scalar_config.max_substeps)
        if max_substeps == 1:
            advected_stack = ssprk3_step(conserved_stack, substep_dt)
        else:
            @jax.checkpoint
            def masked_substep(substep_index, stack):
                def take_substep(stack):
                    return ssprk3_step(stack, substep_dt, rematerialised_advection_rhs)

                def skip_substep(stack):
                    return stack

                return jax.lax.cond(
                    substep_index < num_substeps,
                    take_substep,
                    skip_substep,
                    stack,
                )

            # Python-int bounds make this fori_loop a scan, not a while loop,
            # so reverse mode can differentiate it.
            advected_stack = jax.lax.fori_loop(
                0,
                max_substeps,
                masked_substep,
                conserved_stack,
            )
    else:
        num_substeps = _substep_count(velocities, grid_spacing, dt, config)
        substep_dt = dt / num_substeps.astype(conserved_stack.dtype)
        advected_stack = jax.lax.fori_loop(
            0,
            num_substeps,
            lambda _, stack: ssprk3_step(stack, substep_dt),
            conserved_stack,
        )

    # -------------------------------------------------------------
    # ================= ↑ Sub-cycled SSP-RK3 update ↑ ==============
    # -------------------------------------------------------------

    return _recover_ratios(advected_stack, density, scalars, config=config)


def split_off_passive_scalars(
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Split the block of passive scalars off the end of the state and advect it with the flow at the start of the
    step, so that the solver step sees the layout without it.

    Args:
        primitive_state: The primitive state at the start of the step, scalars included.
        dt: The time step.
        config: The simulation configuration.
        registered_variables: The registered variables of the full state.

    Returns:
        The state without the scalars, the advected scalars and the registered variables of the state without
        the scalars.
    """
    passive_scalar_index = registered_variables.passive_scalar_index
    passive_scalars = primitive_state[passive_scalar_index:]
    primitive_state = primitive_state[:passive_scalar_index]
    registered_variables = registered_variables._replace(
        num_vars=passive_scalar_index,
        passive_scalar_index=-1,
        num_passive_scalars=0,
        passive_scalars_active=False,
    )
    # ``advect_passive_scalars`` sub-cycles internally when the flow requires it.
    passive_scalars = advect_passive_scalars(
        passive_scalars,
        primitive_state,
        dt,
        config.grid_spacing,
        config,
        registered_variables,
    )
    if config.differentiation_mode == BACKWARDS:
        # Reverse-mode memory: the scalars never feed back on the gas, so the backward pass of their advection is
        # independent of the solver's and XLA is free to schedule the two together, adding the transient memory
        # of the scalar sub-steps to the solver's peak. Tying the two through a barrier (a numerical no-op, whose
        # transpose is a barrier on the cotangents) makes the scalar backward pass wait for the solver's. Only on
        # the reverse-mode (masked) path, so the FORWARDS program is untouched.
        passive_scalars, primitive_state = jax.lax.optimization_barrier((passive_scalars, primitive_state))
    return primitive_state, passive_scalars, registered_variables


def reattach_passive_scalars(primitive_state: STATE_TYPE, passive_scalars, config: SimulationConfig) -> STATE_TYPE:
    """
    Fill the ghost cells of the advected scalars and attach the block to the end of the state after the solver
    step.

    Args:
        primitive_state: The primitive state after the solver step, without the scalars.
        passive_scalars: The advected scalars.
        config: The simulation configuration.

    Returns:
        The full primitive state.
    """
    if config.boundary_handling == GHOST_CELLS:
        passive_scalars = _fill_scalar_ghost_cells(passive_scalars, config)
    return jnp.concatenate([primitive_state, passive_scalars], axis=0)


def _recover_ratios(advected_stack, density, scalars, *, config):
    """
    The scalar ratios ``s / rho~`` of an advected conserved stack, guarded where
    the companion density collapsed and clipped to the known bounds.

    Args:
        advected_stack: The advected conserved stack ``(rho~, rho~ C_0, ...)``.
        density: The density the stack was built from (start of the step).
        scalars: The scalars before the advection, whose global ranges bound
            the guarded cells.
        config: The simulation configuration.

    Returns:
        The advected scalars, same shape as ``scalars``.
    """
    advected_density, advected_scalar_densities = _split_companion_stack(advected_stack)

    # -------------------------------------------------------------
    # ==================== ↓ Collapse guard ↓ ======================
    # -------------------------------------------------------------

    # The recovered ratio ``s / rho~`` is meaningless wherever the companion
    # density has collapsed, as it does in near-vacuum regions (e.g. blast-wave
    # interiors, many orders of magnitude below the shell density). There
    # numerator and denominator are both tiny, WENO overshoot dominates their
    # ratio, and the result feeds back through ``s = rho C`` on the next step,
    # so it amplifies until the scalars are NaN in a few cells, while their
    # median stays perfectly correct (a failure that survives a glance at a
    # figure).
    #
    # The guard is deliberately TARGETED. ``rho~`` is advected from ``rho`` over
    # a single step by a consistent operator, so it cannot legitimately fall by
    # six orders of magnitude; a cell where it has is pathological, and only
    # there is the high-order result replaced by the global range bound (which
    # advection cannot exceed anyway). Applying that bound everywhere instead
    # looks harmless and is not: it clips smooth extrema every step, a one-sided
    # error that accumulates and destroys the convergence order.
    density_floor = 1e-6 * jnp.abs(density)
    safe_density = jnp.maximum(advected_density, density_floor)
    safe_density = jnp.where(safe_density > 0.0, safe_density, 1.0)
    recovered = advected_scalar_densities / safe_density[None, ...]

    spatial_axes = tuple(range(1, scalars.ndim))
    range_min = jnp.min(scalars, axis=spatial_axes, keepdims=True)
    range_max = jnp.max(scalars, axis=spatial_axes, keepdims=True)
    untrustworthy = (advected_density < density_floor)[None, ...]
    recovered = jnp.where(
        untrustworthy,
        _clip_keep_derivative(jnp.nan_to_num(recovered, nan=0.0), range_min, range_max),
        recovered,
    )

    # -------------------------------------------------------------
    # ==================== ↑ Collapse guard ↑ ======================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ==================== ↓ Declared bounds ↓ =====================
    # -------------------------------------------------------------

    # Physical bounds, where the caller has declared them. This is the backstop
    # that actually holds in practice: the guard above only fires when the
    # companion density has collapsed to 1e-6 of the density it was advected
    # from, and a ratio is already meaningless well before that. A mass fraction
    # is bounded in [0, 1] by definition, so enforcing it costs nothing, unlike
    # clipping to a scalar's own current range, which clips smooth extrema and
    # destroys the order. ``_clip_keep_derivative`` has the primal of
    # ``jnp.clip``, but a scalar sitting exactly on its bound (most of the grid)
    # keeps derivative 1.
    bounds = _scalar_bounds(config, scalars.shape[0])
    if bounds is not None:
        lower_bounds, upper_bounds = bounds
        recovered = _clip_keep_derivative(recovered, lower_bounds, upper_bounds)

    # -------------------------------------------------------------
    # ==================== ↑ Declared bounds ↑ =====================
    # -------------------------------------------------------------

    return recovered


def _scalar_bounds(config, num_scalars):
    """
    ``(lower, upper)`` bound arrays broadcastable over the scalar stack, or
    ``None`` when no scalar has a bound.

    The bounds are those declared in ``config.passive_scalar_config.bounds``.

    ``num_scalars`` comes from the scalar array itself, NOT from the registry:
    by the time this runs the registry has had the scalar block stripped off
    it, so its counts read zero.

    Args:
        config: The simulation configuration.
        num_scalars: The number of scalars in the stack.

    Returns:
        The bounds, each of shape ``(num_scalars,) + (1,) * dimensionality``,
        or ``None``.
    """
    declared = tuple(config.passive_scalar_config.bounds or ())
    if not declared:
        return None

    lower_bounds, upper_bounds = [], []
    for scalar_number in range(num_scalars):
        declared_bound = declared[scalar_number] if scalar_number < len(declared) else None
        lower_bounds.append(-jnp.inf if declared_bound is None else declared_bound[0])
        upper_bounds.append(jnp.inf if declared_bound is None else declared_bound[1])

    shape = (num_scalars,) + (1,) * int(config.dimensionality)
    return jnp.asarray(lower_bounds).reshape(shape), jnp.asarray(upper_bounds).reshape(shape)


@partial(jax.jit, static_argnames=["config"])
def _fill_scalar_ghost_cells(scalars, config: SimulationConfig):
    """
    Fill the ghost zones of the passive-scalar block.

    The generic boundary handler cannot be reused: it negates the variable whose
    index equals the axis (the normal velocity), which for a scalar block would
    silently flip a composition field. Passive scalars are true scalars, so all
    three boundary types reduce to a copy (mirrored for reflective, wrapped for
    periodic, edge-extended for open) with no sign change anywhere. Other
    boundary types leave the ghost cells untouched.

    Args:
        scalars: The scalar block, shape ``(n_scalars,) + padded grid``.
        config: The simulation configuration.

    Returns:
        The scalar block with filled ghost cells.
    """
    num_ghost = config.num_ghost_cells
    if num_ghost == 0:
        return scalars

    boundary_settings = config.boundary_settings
    if config.dimensionality == 1:
        per_axis_settings = (boundary_settings,)
    else:
        per_axis_settings = (
            boundary_settings.x,
            boundary_settings.y,
            boundary_settings.z,
        )[:config.dimensionality]

    for axis, axis_settings in enumerate(per_axis_settings):
        # The spatial axis within the scalar block (axis 0 holds the scalars).
        array_axis = axis + 1

        def axis_slice(start, stop):
            index = [slice(None)] * scalars.ndim
            index[array_axis] = slice(start, stop)
            return tuple(index)

        left = axis_settings.left_boundary
        right = axis_settings.right_boundary
        if left == PERIODIC_BOUNDARY and right == PERIODIC_BOUNDARY:
            scalars = scalars.at[axis_slice(0, num_ghost)].set(
                scalars[axis_slice(-2 * num_ghost, -num_ghost)]
            )
            scalars = scalars.at[axis_slice(-num_ghost, None)].set(
                scalars[axis_slice(num_ghost, 2 * num_ghost)]
            )
            continue

        if left == OPEN_BOUNDARY:
            scalars = scalars.at[axis_slice(0, num_ghost)].set(
                scalars[axis_slice(num_ghost, num_ghost + 1)]
            )
        elif left == REFLECTIVE_BOUNDARY:
            scalars = scalars.at[axis_slice(0, num_ghost)].set(
                jnp.flip(scalars[axis_slice(num_ghost, 2 * num_ghost)], axis=array_axis)
            )

        if right == OPEN_BOUNDARY:
            scalars = scalars.at[axis_slice(-num_ghost, None)].set(
                scalars[axis_slice(-num_ghost - 1, -num_ghost)]
            )
        elif right == REFLECTIVE_BOUNDARY:
            scalars = scalars.at[axis_slice(-num_ghost, None)].set(
                jnp.flip(scalars[axis_slice(-2 * num_ghost, -num_ghost)], axis=array_axis)
            )

    return scalars
