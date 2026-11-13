"""Positivity-preserving recombination of the WENO split fluxes.

The finite-difference WENO flux at an interface is the sum of two
reconstructed split fluxes, ``F_hat = f_hat^+ + f_hat^-``, upwinded from the
left and from the right. Take ``alpha`` = the largest wave speed on the
stencil. Splitting characteristic field ``s`` with speed ``alpha_s <= alpha``
makes each split flux a scaled vector

    f^+- = +-(alpha / 2) w^+-,      w^+- = w~^+- + z^+-,
    w~^+- = q +- F(q) / alpha,      z^+- = sum_s (alpha_s / alpha - 1) R_s (L_s q).

``w~`` is a physically admissible state (positive density, and pressure for
an ideal gas) whenever ``alpha >= |v_n| + c`` for the Euler equations. For
ideal MHD it is not (Wu 2018), and the scalings act on the cell's own
mirror pairs and on its inflow states summed over the axes instead (see
``_joint_inflow_scalings``). Write ``theta`` for the
scaling of the WENO face value toward its upwind cell. Following Zhang & Shu
(2012, J. Comput. Phys. 231, 2245), the forward-Euler update of a cell is a
convex combination of admissible states if all of these hold:

    w + theta d - z'  admissible  (the face value, which flows into the neighbour),
    w~ - theta d      admissible  (the mirror about the unshifted state),
    lambda (alpha_left + alpha_right) <= 1,

with ``d = w_hat - w`` and ``z'`` the neighbour's own shift at the same
interface. The cell's own flux cancels between the two faces, so face-local
speeds are allowed. The outflow of the neighbour carries ``-z'``, which pairs
with the inflow ``w + theta d``: the first-order inflow state is
``w~ + z - z'``, shifted by the dissipation of the jump of the mass-free
fields. For shear waves this only raises its pressure (with the
momentum-weighted interface velocity and ``alpha_s >= |v_n|``). Two
scalings in [0, 1] enforce the conditions, both inside the reconstruction:

* every field that carries mass is split with the common speed ``alpha``,
  the stencil's spectral radius. That makes the frozen-basis splitting
  monotone for every cell of the stencil, however differently the basis
  represents it (a low-density cell with a large fast speed next to dense
  gas). Per-field speeds chosen only to keep the split states admissible
  are positive but not robust in high-Mach MHD turbulence. For a
  barotropic gas, where admissibility asks only for a positive density, the
  fields that carry no mass (shear waves, MHD Alfven waves) keep their own
  speed, so vortical modes keep the default dissipation; their ``z`` leaves
  the density unchanged. For an ideal gas every field is split with
  ``alpha`` (mass_free_modes). The shear waves of an ideal gas could keep
  their own speed as well, since the shift jump above only raises the
  pressure, but that is not used;
* the face value is pulled toward its upwind state, ``w_hat -> w + theta d``,
  by the largest ``theta`` meeting the two conditions above. ``theta < 1``
  mixes the first-order candidate into the WENO combination with a weight set
  by admissibility instead of smoothness; ``theta = 1`` in smooth flow.

Both scalings use the same primitive: the largest ``t in [0, 1]`` with
``base + t * step`` admissible. Density is linear in ``t``. The pressure is
concave along any line in conserved space, so the chord between the two ends
lies below it and the chord's root is admissible: one closed-form step.

The module holds two forms of the same operations: an array form for the
native-JAX kernel (variable axis leading) and a form on per-cell tuples of
local components for the Pallas kernels.
"""

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equation_of_state import is_barotropic
from astronomix._stencil_operations._stencil_operations import _shift


#: Guard of the divisions by splitting speeds, densities and pressure
#: margins. It lies far below any physical value of these quantities, so it
#: only keeps a division by an exact zero finite.
DIVISION_FLOOR = 1e-30


def mass_free_modes(config: SimulationConfig) -> tuple:
    """
    Indices of the characteristic fields that keep their own splitting speed,
    in the mode order of the eigensystem modules: for a barotropic gas the
    fields that carry no mass, i.e. the shear waves of hydrodynamics and the
    Alfven waves of MHD (admissibility asks only for a positive density, which
    their shifts leave unchanged). For an ideal gas every field is split with
    the common speed.

    Args:
        config: The simulation configuration.

    Returns:
        A tuple of mode indices.
    """
    if not is_barotropic(config):
        return ()
    if config.mhd:
        return (1, 4)
    return tuple(range(1, config.dimensionality))


def stencil_maximum(cell_field, axis: int = 0):
    """
    Maximum of a cell field over the six-point WENO stencil of each interface
    (cells i - 2 ... i + 3 for the interface at i + 1/2).

    Args:
        cell_field: A field without a variable axis.
        axis: The spatial axis of the stencil.

    Returns:
        The stencil maximum per interface, aligned with cell i.
    """
    return jnp.max(
        jnp.stack([_shift(cell_field, offset, axis=axis) for offset in (2, 1, 0, -1, -2, -3)]),
        axis=0,
    )


# -------------------------------------------------------------
# ==== ↓ Array form (native kernel; variable axis leading) ↓ ===
# -------------------------------------------------------------


@jax.custom_jvp
def _floored_ratio(numerator, denominator):
    """
    Return ``numerator / max(denominator, DIVISION_FLOOR)``.

    The forward value is exactly the guarded division it replaces; only the
    derivative differs (see ``_floored_ratio_jvp``). ``_ratio_rules._floored_ratio``
    of the scheme with a Riemann flux has the same values but another derivative
    (the true slope down to the floor), so one definition for both would change
    the derivatives of one of the two schemes.
    """
    return numerator / jnp.maximum(denominator, DIVISION_FLOOR)


@_floored_ratio.defjvp
def _floored_ratio_jvp(primals, tangents):
    """
    Derivative of the guarded division that stays finite in single precision.

    JAX differentiates ``x / y`` through ``-x y**-2``. For the 1e-30 guard (in
    fact for any ``y`` below ~5e-20) ``y**-2`` overflows float32 to infinity,
    and the tangent multiplying it is exactly zero wherever ``jnp.maximum``
    picks the guard or a later ``where`` / ``clip`` discards the lane, so the
    product is ``0 * inf = NaN``, which reverse mode then spreads. The
    derivative is therefore only formed where ``y**-2`` is finite; elsewhere
    it is the guard's own derivative, zero.
    """
    numerator, denominator = primals
    numerator_tangent, denominator_tangent = tangents
    ratio = numerator / jnp.maximum(denominator, DIVISION_FLOOR)
    smallest_safe_denominator = 2.0 * float(
        np.sqrt(1.0 / np.finfo(jnp.result_type(denominator)).max)
    )
    differentiable = denominator > smallest_safe_denominator
    inverse_denominator = 1.0 / jnp.where(differentiable, denominator, 1.0)
    ratio_tangent = jnp.where(
        differentiable,
        (numerator_tangent - jnp.where(differentiable, ratio, 0.0) * denominator_tangent)
        * inverse_denominator,
        0.0,
    )
    return ratio, ratio_tangent


def _gas_pressure(
    state,
    gamma,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Gas pressure of a conserved-state-like vector (no floors).

    Args:
        state: A conserved-state-like array (variables first).
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The gas pressure.
    """
    density = state[registered_variables.density_index]
    energy = state[registered_variables.energy_index]

    if config.dimensionality == 1 and not config.mhd:
        momentum_squared = state[registered_variables.momentum_index] ** 2
    else:
        momentum_squared = state[registered_variables.momentum_index.x] ** 2
        if config.dimensionality >= 2 or config.mhd:
            momentum_squared = momentum_squared + state[registered_variables.momentum_index.y] ** 2
        if config.dimensionality == 3 or config.mhd:
            momentum_squared = momentum_squared + state[registered_variables.momentum_index.z] ** 2

    internal_energy = energy - _floored_ratio(0.5 * momentum_squared, density)
    if config.mhd:
        internal_energy = internal_energy - 0.5 * (
            state[registered_variables.magnetic_index.x] ** 2
            + state[registered_variables.magnetic_index.y] ** 2
            + state[registered_variables.magnetic_index.z] ** 2
        )
    return (gamma - 1.0) * internal_energy


def _admissible_fraction(
    base,
    step,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Largest t in [0, 1] with ``base + t * step`` admissible.

    The floors are the configured minima, capped at half of ``base``'s own
    value so a state sitting below a floor is not forced to t = 0. A
    non-admissible ``base`` gives t = 0.

    Args:
        base: The starting state (an admissible split state).
        step: The direction of the segment.
        params: The simulation parameters (floors, gamma).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The fraction t, one per interface.
    """

    density_index = registered_variables.density_index
    base_density = base[density_index]
    density_floor = jnp.minimum(params.minimum_density, 0.5 * base_density)
    density_step = step[density_index]
    fraction = jnp.where(
        density_step < 0.0,
        _floored_ratio(base_density - density_floor, -density_step),
        1.0,
    )
    fraction = jnp.clip(jnp.where(base_density > 0.0, fraction, 0.0), 0.0, 1.0)

    if config.equation_of_state != IDEAL_GAS:
        return fraction

    gamma = params.gamma
    base_pressure = _gas_pressure(base, gamma, config, registered_variables)
    pressure_floor = jnp.minimum(params.minimum_pressure, 0.5 * jnp.maximum(base_pressure, 0.0))
    base_margin = base_pressure - pressure_floor
    end_pressure = _gas_pressure(
        base + fraction[None] * step,
        gamma,
        config,
        registered_variables,
    )
    end_margin = end_pressure - pressure_floor
    chord_root = _floored_ratio(fraction * base_margin, base_margin - end_margin)
    fraction = jnp.where(end_margin >= 0.0, fraction, chord_root)
    return jnp.where(base_margin > 0.0, fraction, 0.0)


def _upwind_split_states(conserved_state, cell_flux, common_speed, axis=0):
    """
    The unshifted split states ``w~`` of the two upwind cells of each
    interface: ``q_i + F_i / alpha`` (for f^+) and ``q_{i+1} - F_{i+1} / alpha``
    (for f^-).

    Args:
        conserved_state: The conserved state (variables first).
        cell_flux: The physical flux at the cell centres.
        common_speed: The splitting speed of each interface.
        axis: The spatial sweep axis.

    Returns:
        The plus and minus split states, aligned with cell i.
    """
    alpha = jnp.maximum(common_speed, DIVISION_FLOOR)[None]
    plus_state = conserved_state + cell_flux / alpha
    minus_state = (
        _shift(conserved_state, -1, axis=axis + 1) - _shift(cell_flux, -1, axis=axis + 1) / alpha
    )
    return plus_state, minus_state


# Ideal MHD: the split states q +- F / alpha of one cell are often not
# admissible at the fast speed (Wu 2018), but the per-cell decomposition of
# the forward-Euler update only needs two weighted PAIRS to be:
#
#   q_i^{n+1} = (1 - lambda S) q_i + (lambda S / 2) (own_i + in_i),   S = alpha_L + alpha_R,
#   own_i = q_i - (alpha_R theta+_{i+1/2} d+_{i+1/2} + alpha_L theta-_{i-1/2} d-_{i-1/2}) / S,
#   in_i  = (alpha_L w+_{i-1} + alpha_R w-_{i+1}) / S
#           + (alpha_L theta+_{i-1/2} d+_{i-1/2} + alpha_R theta-_{i+1/2} d-_{i+1/2}) / S.
#
# The cell's own flux cancels in own_i, whose base is q_i itself; in_i's base is
# the first-order Lax-Friedrichs inflow, in which the magnetic-tension terms of
# the two neighbours cancel up to their B_n difference (Wu's generalized
# splitting property; random low-beta pairs fail at the fast speed in ~1e-4
# of cases, single split states in ~70 %). Each theta enters one pair of each
# of its two cells; a pair is admissible for every theta in [0, t]^2 once the
# corners (t, 0), (0, t), (t, t) are (convexity), and a face takes the
# smaller t of its two pairs.


def _pair_fraction(base, first, second, params, config, registered_variables):
    """
    Largest t with ``base + a first + b second`` admissible for all a, b in
    [0, t] (the corners (t, 0), (0, t), (t, t) suffice by convexity).

    Args:
        base: The base state of the pair.
        first: The first step.
        second: The second step.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The fraction t.
    """
    return jnp.minimum(
        jnp.minimum(
            _admissible_fraction(base, first, params, config, registered_variables),
            _admissible_fraction(base, second, params, config, registered_variables),
        ),
        _admissible_fraction(base, first + second, params, config, registered_variables),
    )


def _paired_scalings(
    conserved_state,
    common_speed,
    plus_unshifted,
    minus_unshifted,
    plus_step,
    minus_step,
    params,
    config,
    registered_variables,
    axis=0,
):
    """
    theta+ and theta- per interface from the own and inflow pairs of the cells
    on both sides, with the inflow pairs per axis. This is the ideal-MHD
    fallback when no ``inflow_reference`` is passed (dual energy, 1D, and the
    exposed right-hand side of ``analysis_helpers``).

    Args:
        conserved_state: The conserved state (variables first).
        common_speed: The splitting speed of each interface.
        plus_unshifted: The unshifted plus split states ``w~^+``.
        minus_unshifted: The unshifted minus split states ``w~^-``.
        plus_step: The plus face steps ``d^+``.
        minus_step: The minus face steps ``d^-``.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The spatial sweep axis.

    Returns:
        ``(theta_plus, theta_minus)``, aligned with cell i.
    """
    alpha_right = jnp.maximum(common_speed, DIVISION_FLOOR)
    alpha_left = _shift(alpha_right, 1, axis=axis)
    weight_sum = alpha_left + alpha_right
    left_weight = (alpha_left / weight_sum)[None]
    right_weight = (alpha_right / weight_sum)[None]

    own_plus = -right_weight * plus_step
    own_minus = -left_weight * _shift(minus_step, 1, axis=axis + 1)
    own_fraction = _pair_fraction(
        conserved_state,
        own_plus,
        own_minus,
        params,
        config,
        registered_variables,
    )

    inflow_base = (
        left_weight * _shift(plus_unshifted, 1, axis=axis + 1) + right_weight * minus_unshifted
    )
    inflow_plus = left_weight * _shift(plus_step, 1, axis=axis + 1)
    inflow_minus = right_weight * minus_step
    inflow_fraction = _pair_fraction(
        inflow_base,
        inflow_plus,
        inflow_minus,
        params,
        config,
        registered_variables,
    )

    plus_theta = jnp.minimum(own_fraction, _shift(inflow_fraction, -1, axis=axis))
    minus_theta = jnp.minimum(inflow_fraction, _shift(own_fraction, -1, axis=axis))
    return plus_theta, minus_theta


# The per-axis inflow pair above is not the right object in multi-D. Its
# base fails wherever B_n varies along the axis (~7 % of the cells of
# Mach-20 turbulence), although that variation is mostly divergence-free and
# cancels between the axes. What the update needs is the cell's inflow summed
# over all axes,
#
#   in_i = B_i + sum_k theta_k s_k,  B_i = sum_d S_d base_d / sum_d S_d,
#   s_k = alpha_face d_face / sum_d S_d   (the 2 dim inflow faces),
#
# whose first-order part B_i was admissible in every cell of the failing
# states. Splitting it convexly, in_i = sum_k (B_i + n theta_k s_k) / n with
# n = 2 dim, gives each inflow face its own bound theta_k <= frac(B_i, n s_k).
# The own pairs stay per axis (base q_i).


def mhd_physical_flux(
    conserved_state,
    gamma,
    registered_variables: RegisteredVariables,
    axis: int,
):
    """
    Ideal-MHD flux along spatial ``axis`` in the registered variable layout.

    No floors are applied, like the cell fluxes of the Pallas MHD kernel
    without dual energy (with dual energy that kernel uses the switched
    pressure, which this function does not).

    Args:
        conserved_state: The conserved state (variables first).
        gamma: The adiabatic index.
        registered_variables: The registered variables.
        axis: The spatial axis of the flux.

    Returns:
        The physical flux, an array shaped like the state.
    """
    density = conserved_state[registered_variables.density_index]
    momentum = [conserved_state[index] for index in registered_variables.momentum_index]
    field = [conserved_state[index] for index in registered_variables.magnetic_index]
    energy = conserved_state[registered_variables.energy_index]
    velocity = [component / density for component in momentum]
    magnetic_pressure = 0.5 * sum(component * component for component in field)
    kinetic_energy = 0.5 * sum(
        momentum_component * velocity_component
        for momentum_component, velocity_component in zip(momentum, velocity)
    )
    total_pressure = (gamma - 1.0) * (
        energy - kinetic_energy - magnetic_pressure
    ) + magnetic_pressure
    velocity_dot_field = sum(
        velocity_component * field_component
        for velocity_component, field_component in zip(velocity, field)
    )
    flux = jnp.zeros_like(conserved_state)
    flux = flux.at[registered_variables.density_index].set(momentum[axis])
    for component, (momentum_index, magnetic_index) in enumerate(
        zip(registered_variables.momentum_index, registered_variables.magnetic_index)
    ):
        momentum_flux = momentum[component] * velocity[axis] - field[axis] * field[component]
        if component == axis:
            momentum_flux = momentum_flux + total_pressure
        flux = flux.at[momentum_index].set(momentum_flux)
        flux = flux.at[magnetic_index].set(
            velocity[axis] * field[component] - field[axis] * velocity[component]
        )
    return flux.at[registered_variables.energy_index].set(
        (energy + total_pressure) * velocity[axis] - field[axis] * velocity_dot_field
    )


def mhd_inflow_reference(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The axis-summed first-order inflow state of every cell and the summed
    splitting speeds (ideal MHD, untransposed layout):

        B_i = sum_d [alpha_L q_{i-1} + alpha_R q_{i+1} + F_d(q_{i-1}) - F_d(q_{i+1})] / sum_d S_d,

    with alpha_L, alpha_R the splitting speeds of the faces i -+ 1/2 along d
    (stencil maxima of |v_d| + c_f, as in the kernels) and S_d = alpha_L + alpha_R.

    Args:
        conserved_state: The conserved state (variables first).
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        ``(B, sum_d S_d)``.
    """
    gamma = params.gamma
    density = conserved_state[registered_variables.density_index]
    floored_density = jnp.maximum(density, params.minimum_density)
    pressure = jnp.maximum(
        _gas_pressure(conserved_state, gamma, config, registered_variables),
        params.minimum_pressure,
    )
    field = [conserved_state[index] for index in registered_variables.magnetic_index]
    field_squared = sum(component * component for component in field) / floored_density
    sound_squared = gamma * pressure / floored_density
    numerator = jnp.zeros_like(conserved_state)
    speed_sum = jnp.zeros_like(density)
    for axis in range(config.dimensionality):
        discriminant = (
            (sound_squared + field_squared) ** 2
            - 4.0 * sound_squared * field[axis] ** 2 / floored_density
        )
        fast_speed = jnp.sqrt(
            0.5 * (sound_squared + field_squared + jnp.sqrt(jnp.maximum(discriminant, 0.0)))
        )
        normal_momentum = conserved_state[registered_variables.momentum_index[axis]]
        radius = jnp.abs(normal_momentum / density) + fast_speed
        alpha_right = stencil_maximum(radius, axis=axis)
        alpha_left = _shift(alpha_right, 1, axis=axis)
        flux = mhd_physical_flux(conserved_state, gamma, registered_variables, axis)
        numerator = numerator + (
            alpha_left[None] * _shift(conserved_state, 1, axis=axis + 1)
            + alpha_right[None] * _shift(conserved_state, -1, axis=axis + 1)
            + _shift(flux, 1, axis=axis + 1) - _shift(flux, -1, axis=axis + 1)
        )
        speed_sum = speed_sum + alpha_left + alpha_right
    return numerator / speed_sum[None], speed_sum


def _joint_inflow_scalings(
    conserved_state,
    common_speed,
    plus_step,
    minus_step,
    inflow_reference,
    params,
    config,
    registered_variables,
    axis=0,
):
    """
    theta+ and theta- per interface: the own pairs per axis, the inflow faces
    against the cell's axis-summed inflow.

    Args:
        conserved_state: The conserved state (variables first).
        common_speed: The splitting speed of each interface.
        plus_step: The plus face steps ``d^+``.
        minus_step: The minus face steps ``d^-``.
        inflow_reference: ``(B, sum S)`` of ``mhd_inflow_reference`` in the
            sweep's layout.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The spatial sweep axis.

    Returns:
        ``(theta_plus, theta_minus)``, aligned with cell i.
    """
    reference_state, speed_sum = inflow_reference
    alpha_right = jnp.maximum(common_speed, DIVISION_FLOOR)
    alpha_left = _shift(alpha_right, 1, axis=axis)
    own_total = alpha_left + alpha_right
    own_plus = -(alpha_right / own_total)[None] * plus_step
    own_minus = -(alpha_left / own_total)[None] * _shift(minus_step, 1, axis=axis + 1)
    own_fraction = _pair_fraction(
        conserved_state,
        own_plus,
        own_minus,
        params,
        config,
        registered_variables,
    )

    share = (2.0 * config.dimensionality / speed_sum)[None]
    left_inflow = share * alpha_left[None] * _shift(plus_step, 1, axis=axis + 1)
    right_inflow = share * alpha_right[None] * minus_step
    left_fraction = _admissible_fraction(
        reference_state,
        left_inflow,
        params,
        config,
        registered_variables,
    )
    right_fraction = _admissible_fraction(
        reference_state,
        right_inflow,
        params,
        config,
        registered_variables,
    )

    plus_theta = jnp.minimum(own_fraction, _shift(left_fraction, -1, axis=axis))
    minus_theta = jnp.minimum(right_fraction, _shift(own_fraction, -1, axis=axis))
    return plus_theta, minus_theta


def positivity_preserving_interface_flux(
    conserved_state,
    cell_flux,
    common_speed,
    plus_face_flux,
    minus_face_flux,
    plus_shift,
    minus_shift,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int = 0,
    inflow_reference=None,
):
    """
    Assemble the interface flux from admissibility-scaled split fluxes.

    Args:
        conserved_state: Conserved state (variables first).
        cell_flux: Physical flux at the cell centres.
        common_speed: The largest wave speed on each interface's stencil.
        plus_face_flux: The reconstructed ``f_hat^+`` (central part included).
        minus_face_flux: The reconstructed ``f_hat^-``.
        plus_shift: ``z^+`` at the speeds actually used.
        minus_shift: ``z^-`` likewise.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The spatial sweep axis (0 for the native kernel, which moves the
            active axis to the front).
        inflow_reference: Ideal MHD: ``mhd_inflow_reference`` in the sweep's
            layout; the inflow faces are then limited jointly per cell. Without
            it, the per-axis pairs are used.

    Returns:
        The interface flux at i + 1/2, aligned with cell i.
    """

    alpha = jnp.maximum(common_speed, DIVISION_FLOOR)[None]
    plus_unshifted, minus_unshifted = _upwind_split_states(
        conserved_state,
        cell_flux,
        common_speed,
        axis,
    )
    plus_owner = plus_unshifted + plus_shift
    minus_owner = minus_unshifted + minus_shift
    plus_step = 2.0 * plus_face_flux / alpha - plus_owner
    minus_step = -2.0 * minus_face_flux / alpha - minus_owner

    def face_scaling(owner, unshifted, step, neighbour_shift):
        # The inflow state of the neighbour's update carries the shift jump
        # z - z' (see the module docstring), the mirror the unshifted state.
        return jnp.minimum(
            _admissible_fraction(owner - neighbour_shift, step, params, config, registered_variables),
            _admissible_fraction(unshifted, -step, params, config, registered_variables),
        )

    if config.mhd and config.equation_of_state == IDEAL_GAS and inflow_reference is not None:
        plus_theta, minus_theta = _joint_inflow_scalings(
            conserved_state,
            common_speed,
            plus_step,
            minus_step,
            inflow_reference,
            params,
            config,
            registered_variables,
            axis,
        )
    elif config.mhd and config.equation_of_state == IDEAL_GAS:
        plus_theta, minus_theta = _paired_scalings(
            conserved_state,
            common_speed,
            plus_unshifted,
            minus_unshifted,
            plus_step,
            minus_step,
            params,
            config,
            registered_variables,
            axis,
        )
    else:
        plus_theta = face_scaling(plus_owner, plus_unshifted, plus_step, minus_shift)
        minus_theta = face_scaling(minus_owner, minus_unshifted, minus_step, plus_shift)
    if config.solver_config.weno_ad_frozen_weights:
        # The scaling theta is a nonlinear limiter weight like the WENO
        # omegas, so it is frozen together with them.
        plus_theta = jax.lax.stop_gradient(plus_theta)
        minus_theta = jax.lax.stop_gradient(minus_theta)

    plus_flux = 0.5 * alpha * (plus_owner + plus_theta[None] * plus_step)
    minus_flux = -0.5 * alpha * (minus_owner + minus_theta[None] * minus_step)
    # MHD: the normal-field component is the Lax-Friedrichs diffusion of B_n.
    # It must stay: the convex decomposition behind positivity includes it,
    # and with B_n held the energy would carry the magnetic energy of a B_n
    # never applied (at low beta enough to make p negative). CT ignores this
    # component and the cell-centred B is rebuilt from the faces, pressure held.
    return plus_flux + minus_flux


# -------------------------------------------------------------
# ==== ↑ Array form (native kernel; variable axis leading) ↑ ===
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===== ↓ Pallas-kernel form (per-cell local tuples) ↓ ========
# -------------------------------------------------------------
# The Pallas kernels work on tuples of the LOCAL components (rho, m_normal,
# m_tangential..., [B_normal, B_tangential...], [E]) of one cell column; these
# mirror the array form operation for operation. The positions in these
# tuples are fixed by the kernels' layout functions (the registry is resolved
# once there), so they are named here rather than looked up.

#: Slot of the density in every local tuple. The total energy, when carried,
#: is the last slot.
DENSITY_SLOT = 0

MAGNETIC_SLOTS = (4, 5, 6)


def _local_gas_pressure(state, gamma_minus_one, magnetic_slots=()):
    """
    Ideal-gas pressure of a local (rho, m..., [B...], E) tuple (no floors).

    Args:
        state: The local component tuple; the energy is its last slot.
        gamma_minus_one: The adiabatic index minus one.
        magnetic_slots: The slots of the magnetic field components (MHD).

    Returns:
        The gas pressure.
    """
    momentum_slots = [slot for slot in range(1, len(state) - 1) if slot not in magnetic_slots]
    momentum_squared = state[momentum_slots[0]] * state[momentum_slots[0]]
    for slot in momentum_slots[1:]:
        momentum_squared = momentum_squared + state[slot] * state[slot]
    # the guarded division with the float32-safe derivative, as the array form
    internal_energy = state[-1] - _floored_ratio(0.5 * momentum_squared, state[DENSITY_SLOT])
    for slot in magnetic_slots:
        internal_energy = internal_energy - 0.5 * state[slot] * state[slot]
    return gamma_minus_one * internal_energy


def _local_admissible_fraction(
    base,
    step,
    gamma_minus_one,
    minimum_density,
    minimum_pressure,
    ideal_gas,
    magnetic_slots,
):
    """
    ``_admissible_fraction`` for local tuples (density only when not an ideal
    gas).

    Args:
        base: The local tuple of the starting state.
        step: The local tuple of the direction of the segment.
        gamma_minus_one: The adiabatic index minus one (unused when not an
            ideal gas).
        minimum_density: The density floor.
        minimum_pressure: The pressure floor (unused when not an ideal gas).
        ideal_gas: Whether the pressure constraint applies.
        magnetic_slots: The slots of the magnetic field components (MHD).

    Returns:
        The fraction t.
    """
    base_density = base[DENSITY_SLOT]
    density_step = step[DENSITY_SLOT]
    density_floor = jnp.minimum(minimum_density, 0.5 * base_density)
    fraction = jnp.where(
        density_step < 0.0,
        _floored_ratio(base_density - density_floor, -density_step),
        1.0,
    )
    fraction = jnp.clip(jnp.where(base_density > 0.0, fraction, 0.0), 0.0, 1.0)
    if not ideal_gas:
        return fraction

    base_pressure = _local_gas_pressure(base, gamma_minus_one, magnetic_slots)
    pressure_floor = jnp.minimum(minimum_pressure, 0.5 * jnp.maximum(base_pressure, 0.0))
    base_margin = base_pressure - pressure_floor
    end_state = tuple(base[slot] + fraction * step[slot] for slot in range(len(step)))
    end_margin = _local_gas_pressure(end_state, gamma_minus_one, magnetic_slots) - pressure_floor
    chord_root = _floored_ratio(fraction * base_margin, base_margin - end_margin)
    fraction = jnp.where(end_margin >= 0.0, fraction, chord_root)
    return jnp.where(base_margin > 0.0, fraction, 0.0)


def _local_upwind_split_states(left_state, right_state, left_flux, right_flux, common_speed):
    """
    ``_upwind_split_states`` for local tuples.

    Args:
        left_state: The local tuple of cell i.
        right_state: The local tuple of cell i + 1.
        left_flux: The physical flux of cell i.
        right_flux: The physical flux of cell i + 1.
        common_speed: The splitting speed of the interface.

    Returns:
        The plus and minus split states as tuples.
    """
    alpha = jnp.maximum(common_speed, DIVISION_FLOOR)
    plus_state = tuple(
        left_state[slot] + left_flux[slot] / alpha for slot in range(len(left_state))
    )
    minus_state = tuple(
        right_state[slot] - right_flux[slot] / alpha for slot in range(len(right_state))
    )
    return plus_state, minus_state


def positivity_preserving_flux_local(
    left_state,
    right_state,
    left_flux,
    right_flux,
    plus_face_flux,
    minus_face_flux,
    plus_shift,
    minus_shift,
    common_speed,
    gamma_minus_one,
    minimum_density,
    minimum_pressure,
    ideal_gas=True,
    magnetic_slots=(),
    frozen_theta=False,
):
    """
    ``positivity_preserving_interface_flux`` for one interface of a
    Pallas WENO kernel.

    Args:
        left_state: The local conserved tuple of cell i.
        right_state: The local conserved tuple of cell i + 1.
        left_flux: The physical flux of cell i.
        right_flux: The physical flux of cell i + 1.
        plus_face_flux: The reconstructed plus split flux (central part
            included).
        minus_face_flux: The reconstructed minus split flux.
        plus_shift: ``z^+`` at the speeds actually used.
        minus_shift: ``z^-`` likewise.
        common_speed: The largest wave speed on the stencil.
        gamma_minus_one: The adiabatic index minus one (unused when not an
            ideal gas).
        minimum_density: The density floor.
        minimum_pressure: The pressure floor (unused when not an ideal gas).
        ideal_gas: Whether the pressure constraint applies.
        magnetic_slots: Local slots of the magnetic field (MHD), excluded from
            the kinetic energy in the pressure. The normal-field component
            keeps its Lax-Friedrichs diffusion (see
            ``positivity_preserving_interface_flux``).
        frozen_theta: Freeze the admissibility scalings for differentiation,
            as ``weno_ad_frozen_weights`` does in the array form (the adjoint
            kernels); the value is unchanged.

    Returns:
        The interface flux as a list of local components.
    """
    num_components = len(left_state)
    alpha = jnp.maximum(common_speed, DIVISION_FLOOR)
    plus_unshifted, minus_unshifted = _local_upwind_split_states(
        left_state,
        right_state,
        left_flux,
        right_flux,
        common_speed,
    )
    plus_owner = tuple(
        plus_unshifted[slot] + plus_shift[slot] for slot in range(num_components)
    )
    minus_owner = tuple(
        minus_unshifted[slot] + minus_shift[slot] for slot in range(num_components)
    )
    plus_step = tuple(
        2.0 * plus_face_flux[slot] / alpha - plus_owner[slot] for slot in range(num_components)
    )
    minus_step = tuple(
        -2.0 * minus_face_flux[slot] / alpha - minus_owner[slot] for slot in range(num_components)
    )

    def fraction(base, step):
        return _local_admissible_fraction(
            base,
            step,
            gamma_minus_one,
            minimum_density,
            minimum_pressure,
            ideal_gas,
            magnetic_slots,
        )

    def face_scaling(owner, unshifted, step, neighbour_shift):
        mirror_step = tuple(-component for component in step)
        inflow_base = tuple(o - z for o, z in zip(owner, neighbour_shift))
        return jnp.minimum(fraction(inflow_base, step), fraction(unshifted, mirror_step))

    plus_theta = face_scaling(plus_owner, plus_unshifted, plus_step, minus_shift)
    minus_theta = face_scaling(minus_owner, minus_unshifted, minus_step, plus_shift)
    if frozen_theta:
        plus_theta = jax.lax.stop_gradient(plus_theta)
        minus_theta = jax.lax.stop_gradient(minus_theta)
    # The normal-field component stays, as in positivity_preserving_interface_flux.
    interface_flux = [
        0.5 * alpha * (plus_owner[slot] + plus_theta * plus_step[slot])
        - 0.5 * alpha * (minus_owner[slot] + minus_theta * minus_step[slot])
        for slot in range(num_components)
    ]
    return interface_flux


# -------------------------------------------------------------
# ===== ↑ Pallas-kernel form (per-cell local tuples) ↑ ========
# -------------------------------------------------------------
