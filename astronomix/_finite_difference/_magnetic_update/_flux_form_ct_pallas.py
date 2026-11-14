"""
Pallas kernels of the flux-form constrained transport of ``_flux_form_ct.py``
(``FDSolverConfig.scheme = RIEMANN_FLUX``).

``_edge_emfs`` and the curl of its result run here as three kernels with
stacked channels between them:

    face velocities  the conserved state -> 7 channels: the six transverse
                     velocities at the faces (sixth-order midpoint
                     interpolation along the face normal) and the cold-plasma
                     speed |B| / sqrt(rho) of the cell
    edge EMFs        the stacked interface fields and the 7 channels -> the
                     edge EMFs, each the difference of two two-state HLL
                     fluxes of seventh-order WENO-Z point values
    curl             the edge EMFs -> the increments of the interface fields
                     (``_ct_plain_curl_pallas_local``, the sixth-order
                     staggered curl of ``_constrained_transport_pallas.py``)

The functions carry the names and the order of the array form. A kernel
function takes the values of the stencil cells where the array form shifts
whole arrays. Two places are rearranged to save divisions, each stated where
it stands and equal to the array form up to rounding: the momenta are
multiplied by one reciprocal density per cell, and the left- and right-biased
point values of a stencil are evaluated together (``_weno7z_point_pair``).

The EMFs are stacked in the channel order of the curl kernel, Omega_z,
Omega_x, Omega_y, where the array form returns (E_x, E_y, E_z).

The kernels are never differentiated: ``_flux_form_ct.py`` calls them through
``diffable_pallas_call_n``, which takes the tangent and the cotangent from the
array form. The quotients written out below therefore carry the forward rule
only; the derivative rules stay in ``_ratio_rules.py``.

Every kernel goes through ``_pallas_call_sharded`` (one halo exchange per
kernel on a device mesh); the ``*_local`` builds derive every shape from
their own arguments, and each input may carry its own halo around the cells
written (``_shard_cells``). The stage on a device mesh calls the ``*_local``
builds on a padded shard instead (``_time_integrators/_padded_stage.py``).
"""

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._pallas_helpers import _pallas_call_sharded, _shard_cells, pl
from astronomix._finite_difference._interface_fluxes._riemann_flux_pallas import _riemann_flux_pallas_supported
from astronomix._finite_difference._magnetic_update._constrained_transport_pallas import (
    _block_cell_indices,
    _compiler_kwargs,
    _ct_block_and_grid,
    _ct_plain_curl_pallas_local,
    _wrapped_index,
)

# The constants of ``_flux_form_ct.py`` and the floor of ``_ratio_rules.py``.
# They are repeated here because the array module imports this one at its
# bottom (the pattern of the package).
EPSILON_FLOOR = 1e-30
CT_WENO_EPSILON_RELATIVE = 1e-6
DIVISION_FLOOR = 1e-30

#: Channel of the velocity v_along at the faces of ``component``, keyed by
#: (transported face-field component, transport axis).
_FACE_VELOCITY_CHANNEL = {(0, 1): 0, (0, 2): 1, (1, 0): 2, (1, 2): 3, (2, 0): 4, (2, 1): 5}
#: Channel of the cold-plasma speed of the cell, and the channel count.
_COLD_SPEED_CHANNEL = 6
_NUM_CHANNELS = 7

#: Reach of the face-velocity and of the curl kernel per axis; the point
#: values of the EMF kernel reach one cell further.
_HALO = 3
_EMF_HALO = 4


def _flux_form_ct_pallas_supported(conserved_state, config: SimulationConfig) -> bool:
    """
    Whether the kernels can run in place of ``_edge_emfs`` and the curl:
    ``backend_config.pallas_ct``, which ``finalize_config`` sets under
    ``scheme = RIEMANN_FLUX`` to "the backend is Pallas", and the conditions
    of the flux kernels (``_riemann_flux_pallas_supported``).
    """
    return bool(config.backend_config.pallas_ct) and _riemann_flux_pallas_supported(conserved_state, config)


def _interp_center_to_face_6(value_at):
    """
    Sixth-order midpoint interpolation to i + 1/2 of the values
    ``value_at(offset)``: (3, -25, 150, 150, -25, 3) / 256 over the cell
    offsets -2 ... 3. The division by 256 is written as the multiplication
    by 1 / 256 (a power of two: the same result).
    """
    return (
        3.0 * value_at(-2)
        - 25.0 * value_at(-1)
        + 150.0 * value_at(0)
        + 150.0 * value_at(1)
        - 25.0 * value_at(2)
        + 3.0 * value_at(3)
    ) * (1.0 / 256.0)


def _weno7z_point_pair(values, relative_epsilon, epsilon_floor):
    """
    The left- and right-biased seventh-order WENO-Z point values at +1/2 of
    the stencil values ``values[-3] ... values[4]``, with the epsilon
    ``relative_epsilon`` times the largest squared value plus
    ``epsilon_floor``: ``_weno7z_point_interpolation`` of values[-3 ... 3] and
    of the mirrored stencil values[4 ... -2] (the two calls of
    ``_flux_form_ct._weno7z_point_pair``), evaluated together. Equal to the
    two calls up to rounding; rearranged so that the two sides share work and
    need 5 instead of 13 divisions each:

      - The first differences d_k = v_(k+1) - v_k are formed once. The
        mirrored stencil has the differences -d_(-k), and the sign drops out
        of its smoothness indicators (sums of squares).
      - The indicators are written in the second differences
        s_k = d_k - d_(k-1) and the third differences t_k = s_(k+1) - s_k
        (mirror images s_(1-k) and -t_(-k)): with u_j the differences of one
        stencil, u_0 - 3 u_1 + 2 u_2 = 2 s_(-1) - s_(-2), u_3 - u_2 = s_0,
        2 u_3 - 3 u_4 + u_5 = s_2 - 2 s_1, and
        u_(j-1) - 2 u_j + u_(j+1) = t_(j-3). The squares of t_(-1), t_0 and
        t_1 enter both sides.
      - The indicators are kept as 36 beta_k (first term without its
        division by 6; 13/12 and 781/720 become 39 and 781/20), with
        36 epsilon next to them: tau / (beta_k + epsilon) is unchanged.
      - The candidate values and the linear weights are kept without their
        denominators 16 and 64; the weighted sum is divided by 16 times the
        sum of the weights instead (powers of two: exact).
      - The central candidate, (-v_(-1) + 9 v_0 + 9 v_1 - v_2) / 16, is its
        own mirror image and is formed once.

    Args:
        values: The stencil values, indexed by the offset -3 ... 4.
        relative_epsilon: ``CT_WENO_EPSILON_RELATIVE``, typed like the values.
        epsilon_floor: ``EPSILON_FLOOR``, typed like the values.

    Returns:
        The left-biased and the right-biased point value.
    """
    largest_square = None
    for shift in range(-3, 5):
        squared = values[shift] * values[shift]
        largest_square = squared if largest_square is None else jnp.maximum(largest_square, squared)
    epsilon = relative_epsilon * largest_square + epsilon_floor
    first_difference = {k: values[k + 1] - values[k] for k in range(-3, 4)}
    second_difference = {k: first_difference[k] - first_difference[k - 1] for k in range(-2, 4)}
    third_term = {}
    for k in range(-2, 3):
        third_difference = second_difference[k + 1] - second_difference[k]
        third_term[k] = (781.0 / 20.0) * (third_difference * third_difference)
    central = 9.0 * (values[0] + values[1]) - (values[-1] + values[2])
    epsilon36 = 36.0 * epsilon

    def biased_value(mirrored):
        """The point value of the stencil values[-3 ... 3], or of its mirror
        image about the half point."""

        def v(k):
            return values[1 - k] if mirrored else values[k]

        def d(k):
            return first_difference[-k] if mirrored else first_difference[k]

        def s(k):
            return second_difference[1 - k] if mirrored else second_difference[k]

        def t2(k):
            return third_term[-k] if mirrored else third_term[k]

        # 16 times the candidate values p_0 ... p_3 (p_2 is ``central``).
        candidate0 = -5.0 * v(-3) + 21.0 * v(-2) - 35.0 * v(-1) + 35.0 * v(0)
        candidate1 = v(-2) - 5.0 * v(-1) + 15.0 * v(0) + 5.0 * v(1)
        candidate3 = 5.0 * v(0) + 15.0 * v(1) - 5.0 * v(2) + v(3)
        # 36 times the smoothness indicators beta_0 ... beta_3.
        first0 = 2.0 * d(-3) - 7.0 * d(-2) + 11.0 * d(-1)
        first1 = -d(-2) + 5.0 * d(-1) + 2.0 * d(0)
        first2 = 2.0 * d(-1) + 5.0 * d(0) - d(1)
        first3 = 11.0 * d(0) - 7.0 * d(1) + 2.0 * d(2)
        second0 = 2.0 * s(-1) - s(-2)
        second3 = s(2) - 2.0 * s(1)
        second_central = 39.0 * (s(0) * s(0))
        beta0 = first0 * first0 + 39.0 * (second0 * second0) + t2(-2)
        beta1 = first1 * first1 + second_central + t2(-1)
        beta2 = first2 * first2 + second_central + t2(0)
        beta3 = first3 * first3 + 39.0 * (second3 * second3) + t2(1)
        tau = jnp.abs(beta0 + 3.0 * beta1 - 3.0 * beta2 - beta3)
        # 64 times the nonlinear weights alpha_0 ... alpha_3.
        alpha0 = 1.0 + tau / (beta0 + epsilon36)
        alpha1 = 21.0 * (1.0 + tau / (beta1 + epsilon36))
        alpha2 = 35.0 * (1.0 + tau / (beta2 + epsilon36))
        alpha3 = 7.0 * (1.0 + tau / (beta3 + epsilon36))
        weighted = alpha0 * candidate0 + alpha1 * candidate1 + alpha2 * central + alpha3 * candidate3
        return weighted / (16.0 * (alpha0 + alpha1 + alpha2 + alpha3))

    return biased_value(False), biased_value(True)


def _hll_transport_flux(face_velocity, field, cold_speed, relative_epsilon, epsilon_floor):
    """
    The two-state HLL transport flux of a face field at an edge, from the
    stencil values ``face_velocity[-3] ... [4]`` and ``field[-3] ... [4]``
    along the transport axis and the cold speed of the edge.
    """
    velocity_left, velocity_right = _weno7z_point_pair(face_velocity, relative_epsilon, epsilon_floor)
    field_left, field_right = _weno7z_point_pair(field, relative_epsilon, epsilon_floor)
    zero = cold_speed - cold_speed
    a_plus = jnp.maximum(zero, jnp.maximum(velocity_left, velocity_right) + cold_speed)
    a_minus = jnp.maximum(zero, cold_speed - jnp.minimum(velocity_left, velocity_right))
    total = a_plus + a_minus
    transport_left = velocity_left * field_left
    transport_right = velocity_right * field_right
    flux = (
        a_plus * transport_left + a_minus * transport_right - a_plus * a_minus * (field_right - field_left)
    ) / jnp.maximum(total, DIVISION_FLOOR)
    return jnp.where(total > 0.0, flux, 0.5 * (transport_left + transport_right))


def _edge_emfs_pallas(
    conserved_state,
    interface_fields,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    ``_flux_form_ct._edge_emfs`` through the face-velocity and the EMF kernel.

    Args:
        conserved_state: The conserved state.
        interface_fields: The stacked ``(3, nx, ny, nz)`` interface fields.
        params: The simulation parameters (the density floor).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The stacked ``(3, nx, ny, nz)`` EMFs Omega_z, Omega_x, Omega_y.
    """
    dtype = interface_fields.dtype
    _, block_shape, _ = _ct_block_and_grid(interface_fields.shape[1:], config)

    def build_face_velocities(state_local, minimum_density):
        return _cell_velocity_and_cold_speed_pallas_local(
            state_local, minimum_density, config=config, registered_variables=registered_variables
        )

    auxiliary = _pallas_call_sharded(
        build_face_velocities,
        state_inputs=(conserved_state,),
        other_args=(jnp.asarray(params.minimum_density, dtype=dtype),),
        halo=(_HALO,) * 3,
        block_shape=block_shape,
        local_output=True,
    )

    def build_emfs(fields_local, auxiliary_local, relative_epsilon, epsilon_floor):
        return _edge_emfs_pallas_local(fields_local, auxiliary_local, relative_epsilon, epsilon_floor, config=config)

    return _pallas_call_sharded(
        build_emfs,
        state_inputs=(interface_fields, auxiliary),
        other_args=(
            jnp.asarray(CT_WENO_EPSILON_RELATIVE, dtype=dtype),
            jnp.asarray(EPSILON_FLOOR, dtype=dtype),
        ),
        halo=(_EMF_HALO,) * 3,
        block_shape=block_shape,
        local_output=True,
    )


def _edge_emf_curl_pallas(emfs, dt_over_dx, config: SimulationConfig):
    """
    ``_energy_flux.edge_emf_curl`` of the stacked EMFs (Omega_z, Omega_x,
    Omega_y) with the sixth-order curl kernel.

    Args:
        emfs: The stacked ``(3, nx, ny, nz)`` EMFs.
        dt_over_dx: The (stage) time step over the cell size.
        config: The simulation configuration.

    Returns:
        The stacked ``(3, nx, ny, nz)`` increments of the interface fields.
    """
    _, block_shape, _ = _ct_block_and_grid(emfs.shape[1:], config)

    def build_curl(emfs_local, time_step_over_spacing):
        return _ct_plain_curl_pallas_local(emfs_local, time_step_over_spacing, config)

    return _pallas_call_sharded(
        build_curl,
        state_inputs=(emfs,),
        other_args=(jnp.asarray(dt_over_dx, dtype=emfs.dtype),),
        halo=(_HALO,) * 3,
        block_shape=block_shape,
        local_output=True,
    )


def _cell_velocity_and_cold_speed_pallas_local(
    state, minimum_density, *, config: SimulationConfig, registered_variables: RegisteredVariables
):
    """
    Single-shard build of the face-velocity kernel: the velocities of
    ``_flux_form_ct._cell_velocity_and_cold_speed`` interpolated to the faces
    (channels 0 to 5, see ``_FACE_VELOCITY_CHANNEL``) and the cold-plasma
    speed of the cell (channel 6).

    Args:
        state: The conserved state (possibly halo-padded).
        minimum_density: The density floor (scalar array).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The stacked ``(7, nx, ny, nz)`` channels.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(state.shape[1:], config)
    extents = tuple(spatial_shape)
    density_index = registered_variables.density_index
    momentum_indices = (
        registered_variables.momentum_index.x,
        registered_variables.momentum_index.y,
        registered_variables.momentum_index.z,
    )
    magnetic_indices = (
        registered_variables.magnetic_index.x,
        registered_variables.magnetic_index.y,
        registered_variables.magnetic_index.z,
    )

    state_spec = pl.BlockSpec(state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())
    out_spec = pl.BlockSpec((_NUM_CHANNELS,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(state_ref, minimum_density_ref, out_ref):
        cell = _block_cell_indices(block_shape, spatial_shape)

        def value(variable, axis, shift):
            """The state variable at the cell shifted by ``shift`` along ``axis``."""
            index = list(cell)
            index[axis] = _wrapped_index(cell[axis], shift, extents[axis])
            return state_ref[(variable,) + tuple(index)]

        def inverse_density(axis, shift):
            """1 / max(rho, rho_min) of the shifted cell. The array form
            divides every momentum by the density; here the reciprocal is
            taken once per cell (the compiler merges the equal expressions of
            the velocity components) and multiplied, equal to rounding."""
            return 1.0 / jnp.maximum(value(density_index, axis, shift), minimum_density_ref[()])

        def velocity(component, axis, shift):
            return value(momentum_indices[component], axis, shift) * inverse_density(axis, shift)

        for (component, along), channel in _FACE_VELOCITY_CHANNEL.items():
            out_ref[channel, ...] = _interp_center_to_face_6(
                lambda shift, velocity_component=along, face_axis=component: velocity(
                    velocity_component, face_axis, shift
                )
            )

        field = [value(index, 0, 0) for index in magnetic_indices]
        field_squared = field[0] * field[0] + field[1] * field[1] + field[2] * field[2]
        out_ref[_COLD_SPEED_CHANNEL, ...] = jnp.sqrt(field_squared * inverse_density(0, 0))

    out_shape = jax.ShapeDtypeStruct((_NUM_CHANNELS,) + _shard_cells(spatial_shape)[1], state.dtype)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[state_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="flux_form_ct_face_velocity",
        **_compiler_kwargs(config),
    )(state, minimum_density)


def _edge_emfs_pallas_local(
    interface_fields, auxiliary, relative_epsilon, epsilon_floor, *, config: SimulationConfig
):
    """
    Single-shard build of the EMF kernel: the edge EMFs, stacked as channels
    0 = Omega_z, 1 = Omega_x, 2 = Omega_y. Each is the difference of two HLL
    transport fluxes (``_hll_transport_flux``) with the cold speed the
    largest of the four cells around the edge.

    Args:
        interface_fields: The stacked interface fields (possibly halo-padded).
        auxiliary: The channels of the face-velocity kernel (possibly halo-padded).
        relative_epsilon: ``CT_WENO_EPSILON_RELATIVE`` (scalar array).
        epsilon_floor: ``EPSILON_FLOOR`` (scalar array).
        config: The simulation configuration.

    Returns:
        The stacked ``(3, nx, ny, nz)`` EMFs.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(interface_fields.shape[1:], config)
    extents = tuple(spatial_shape)
    auxiliary_extents = tuple(int(extent) for extent in auxiliary.shape[1:])

    field_spec = pl.BlockSpec(interface_fields.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    auxiliary_spec = pl.BlockSpec(auxiliary.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())
    out_spec = pl.BlockSpec((3,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(field_ref, auxiliary_ref, relative_epsilon_ref, epsilon_floor_ref, out_ref):
        cell = _block_cell_indices(block_shape, spatial_shape)
        # On a padded shard the two inputs may carry different halos (each around the same cells).
        auxiliary_cell = cell
        if auxiliary_extents != extents:
            auxiliary_cell = _block_cell_indices(block_shape, auxiliary_extents)

        def read(ref, channel, offset):
            cells, cell_extents = (cell, extents) if ref is field_ref else (auxiliary_cell, auxiliary_extents)
            index = tuple(_wrapped_index(cells[axis], offset[axis], cell_extents[axis]) for axis in range(3))
            return ref[(channel,) + index]

        def transport_flux(component, along):
            """G[b_component transported along ``along``] at the edge."""

            def offset(shift_along, shift_component=0):
                shifted = [0, 0, 0]
                shifted[along] += shift_along
                shifted[component] += shift_component
                return tuple(shifted)

            velocity_channel = _FACE_VELOCITY_CHANNEL[(component, along)]
            field = {shift: read(field_ref, component, offset(shift)) for shift in range(-3, 5)}
            face_velocity = {shift: read(auxiliary_ref, velocity_channel, offset(shift)) for shift in range(-3, 5)}
            cold_speed = None
            for step_along in (0, 1):
                for step_component in (0, 1):
                    candidate = read(auxiliary_ref, _COLD_SPEED_CHANNEL, offset(step_along, step_component))
                    cold_speed = candidate if cold_speed is None else jnp.maximum(cold_speed, candidate)
            return _hll_transport_flux(
                face_velocity, field, cold_speed, relative_epsilon_ref[()], epsilon_floor_ref[()]
            )

        out_ref[0, ...] = transport_flux(0, 1) - transport_flux(1, 0)
        out_ref[1, ...] = transport_flux(1, 2) - transport_flux(2, 1)
        out_ref[2, ...] = transport_flux(2, 0) - transport_flux(0, 2)

    out_shape = jax.ShapeDtypeStruct((3,) + _shard_cells(spatial_shape)[1], interface_fields.dtype)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[field_spec, auxiliary_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="flux_form_ct_emf",
        **_compiler_kwargs(config),
    )(interface_fields, auxiliary, relative_epsilon, epsilon_floor)
