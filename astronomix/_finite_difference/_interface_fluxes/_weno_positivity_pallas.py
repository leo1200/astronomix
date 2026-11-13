"""Pallas kernels for the positivity-preserving WENO recombination of ideal MHD.

The array forms in ``_weno_positivity.py`` (``mhd_inflow_reference`` and the
joint-inflow branch of ``positivity_preserving_interface_flux``) are
elementwise but full of shifted 8-component arrays; XLA materialises them, and
at 256^3 they cost ~4x the WENO kernel and ~9 GB of temporaries. These two
kernels compute the same quantities in one pass each:

* ``mhd_inflow_reference_pallas``: per cell, the axis-summed first-order
  inflow B_i and the summed splitting speeds (channels: the state's variables,
  then sum_d S_d);
* ``mhd_joint_recombination_pallas``: per interface, the limited flux from the
  split face fluxes of the WENO kernel (three neighbouring interfaces), the
  state at cells i, i + 1 and the reference at cells i, i + 1; its arithmetic
  is the window function ``_joint_recombination_window``. (The reverse mode of
  ``BackendConfig.pallas_adjoint`` differentiates the array form instead, see
  ``_weno_flux_mhd_pallas_vjp``.)

Both read their inputs with periodic (modular) indexing inside the local
array, like the WENO kernels, so they run unchanged inside the multi-GPU
``shard_map`` + halo wrap.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    _local_admissible_fraction,
    _local_gas_pressure,
)
from astronomix._pallas_helpers import _as_3tuple_block_shape, _pallas_compiler_params, pl

MAGNETIC_SLOTS = (4, 5, 6)


def _mhd_local_indices(registered_variables: RegisteredVariables, axis: int):
    """(rho, m_n, m_t1, m_t2, B_n, B_t1, B_t2, E) indices for a sweep along ``axis``."""
    rv = registered_variables
    momentum = [int(rv.momentum_index.x), int(rv.momentum_index.y), int(rv.momentum_index.z)]
    field = [int(rv.magnetic_index.x), int(rv.magnetic_index.y), int(rv.magnetic_index.z)]
    tangential = [k for k in range(3) if k != axis]
    return (
        int(rv.density_index), momentum[axis], momentum[tangential[0]], momentum[tangential[1]],
        field[axis], field[tangential[0]], field[tangential[1]], int(rv.energy_index),
    )


def _local_mhd_flux(state, gm1):
    """Ideal-MHD flux along the normal of a local (rho, m_n, m_t1, m_t2, B_n,
    B_t1, B_t2, E) tuple (no floors, as the WENO kernel's cell fluxes)."""
    density, mn, mt1, mt2, bn, bt1, bt2, energy = state
    inverse_density = 1.0 / density
    vn, vt1, vt2 = mn * inverse_density, mt1 * inverse_density, mt2 * inverse_density
    total_pressure = _local_gas_pressure(state, gm1, MAGNETIC_SLOTS) + 0.5 * (bn * bn + bt1 * bt1 + bt2 * bt2)
    v_dot_b = vn * bn + vt1 * bt1 + vt2 * bt2
    return (
        mn,
        mn * vn + total_pressure - bn * bn,
        mt1 * vn - bn * bt1,
        mt2 * vn - bn * bt2,
        0.0 * bn,
        vn * bt1 - bn * vt1,
        vn * bt2 - bn * vt2,
        (energy + total_pressure) * vn - bn * v_dot_b,
    )


def _grid_and_indices(spatial_shape, config: SimulationConfig):
    """The kernel grid and the block shape (bx, by, bz) of a spatial shape."""
    nx, ny, nz = spatial_shape
    bx, by, bz = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, 3, spatial_shape=spatial_shape)
    return (nx // bx, ny // by, nz // bz), (bx, by, bz)


def _block_indices(block, spatial_shape):
    """The periodic x, y, z cell indices of the current kernel block."""
    bx, by, bz = block
    nx, ny, nz = spatial_shape
    ii = (pl.program_id(0) * bx + jnp.arange(bx)[:, None, None]) % nx
    jj = (pl.program_id(1) * by + jnp.arange(by)[None, :, None]) % ny
    kk = (pl.program_id(2) * bz + jnp.arange(bz)[None, None, :]) % nz
    return ii, jj, kk


def _reader(ref, ii, jj, kk, spatial_shape):
    """A reader at(var, axis, offset) of the cell offset along axis from the block cells (periodic)."""
    nx, ny, nz = spatial_shape

    def at(var, axis, offset):
        if axis == 0:
            return ref[var, (ii + offset) % nx, jj, kk]
        if axis == 1:
            return ref[var, ii, (jj + offset) % ny, kk]
        return ref[var, ii, jj, (kk + offset) % nz]
    return at


def _signal_speed_pallas(conserved_state, params: SimulationParams, config: SimulationConfig,
                         registered_variables: RegisteredVariables):
    """|v_d| + c_f,d of every cell for d = x, y, z (3 channels), computed once
    so the reference kernel only takes stencil maxima."""
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    grid, block = _grid_and_indices(spatial_shape, config)
    sweeps = [_mhd_local_indices(registered_variables, axis) for axis in range(3)]

    def kernel(q_ref, gamma_ref, rhomin_ref, pgmin_ref, out_ref):
        ii, jj, kk = _block_indices(block, spatial_shape)
        at = _reader(q_ref, ii, jj, kk, spatial_shape)
        gamma = gamma_ref[()]
        state = tuple(at(var, 0, 0) for var in sweeps[0])
        density = state[0]
        floored_density = jnp.maximum(density, rhomin_ref[()])
        pressure = jnp.maximum(_local_gas_pressure(state, gamma - 1.0, MAGNETIC_SLOTS), pgmin_ref[()])
        sound_squared = gamma * pressure / floored_density
        field = (state[4], state[5], state[6])
        field_squared = (field[0] * field[0] + field[1] * field[1] + field[2] * field[2]) / floored_density
        inverse_density = 1.0 / density
        for axis in range(3):
            discriminant = ((sound_squared + field_squared) ** 2
                            - 4.0 * sound_squared * field[axis] * field[axis] / floored_density)
            fast = jnp.sqrt(0.5 * (sound_squared + field_squared + jnp.sqrt(jnp.maximum(discriminant, 0.0))))
            out_ref[axis, ...] = jnp.abs(state[1 + axis] * inverse_density) + fast

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())
    dtype = conserved_state.dtype
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((3,) + spatial_shape, dtype),
        grid=grid,
        in_specs=[pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))] + [scalar_spec] * 3,
        out_specs=pl.BlockSpec((3,) + block, lambda bi, bj, bk: (0, bi, bj, bk)),
        interpret=config.backend_config.pallas_interpret,
        name="mhd_signal_speed",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(params.gamma, dtype=dtype),
        jnp.asarray(params.minimum_density, dtype=dtype),
        jnp.asarray(params.minimum_pressure, dtype=dtype),
    )


def mhd_inflow_reference_pallas(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Pallas form of ``mhd_inflow_reference`` (3D ideal MHD): the signal
    speeds once per cell, then one pass for the stencil maxima and the inflow.

    Returns:
        ``(B, sum_d S_d)``.
    """
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    grid, block = _grid_and_indices(spatial_shape, config)
    sweeps = [_mhd_local_indices(registered_variables, axis) for axis in range(3)]
    used = set(sweeps[0])
    radius = _signal_speed_pallas(conserved_state, params, config, registered_variables)

    def kernel(q_ref, radius_ref, gamma_ref, out_ref):
        ii, jj, kk = _block_indices(block, spatial_shape)
        at = _reader(q_ref, ii, jj, kk, spatial_shape)
        radius_at = _reader(radius_ref, ii, jj, kk, spatial_shape)
        gm1 = gamma_ref[()] - 1.0
        numerator = [None] * 8
        speed_sum = None
        for axis in range(3):
            radii = {offset: radius_at(axis, axis, offset) for offset in range(-3, 4)}
            # faces i + 1/2 (cells i-2..i+3) and i - 1/2 (cells i-3..i+2)
            inner = radii[-2]
            for offset in range(-1, 3):
                inner = jnp.maximum(inner, radii[offset])
            alpha_right = jnp.maximum(inner, radii[3])
            alpha_left = jnp.maximum(inner, radii[-3])
            left = tuple(at(var, axis, -1) for var in sweeps[axis])
            right = tuple(at(var, axis, 1) for var in sweeps[axis])
            left_flux = _local_mhd_flux(left, gm1)
            right_flux = _local_mhd_flux(right, gm1)
            for slot in range(8):
                term = (alpha_left * left[slot] + alpha_right * right[slot]
                        + left_flux[slot] - right_flux[slot])
                # local slot -> the sweep's registry variable -> x-sweep slot
                x_slot = sweeps[0].index(sweeps[axis][slot])
                numerator[x_slot] = term if numerator[x_slot] is None else numerator[x_slot] + term
            total = alpha_left + alpha_right
            speed_sum = total if speed_sum is None else speed_sum + total
        inverse = 1.0 / speed_sum
        for var in range(nvars):
            if var in used:
                out_ref[var, ...] = numerator[sweeps[0].index(var)] * inverse
            else:
                out_ref[var, ...] = at(var, 0, 0)
        out_ref[nvars, ...] = speed_sum

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params
    dtype = conserved_state.dtype
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((nvars + 1,) + spatial_shape, dtype),
        grid=grid,
        in_specs=[pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0)),
                  pl.BlockSpec(radius.shape, lambda bi, bj, bk: (0, 0, 0, 0)),
                  pl.BlockSpec((), lambda bi, bj, bk: ())],
        out_specs=pl.BlockSpec((nvars + 1,) + block, lambda bi, bj, bk: (0, bi, bj, bk)),
        interpret=config.backend_config.pallas_interpret,
        name="mhd_inflow_reference",
        **kwargs,
    )(conserved_state, radius, jnp.asarray(params.gamma, dtype=dtype))
    return out[:nvars], out[nvars]


def _joint_recombination_window(
    q_left,
    q_right,
    plus_here,
    plus_right,
    minus_left,
    minus_here,
    speeds,
    reference_left,
    reference_right,
    speed_sum_left,
    speed_sum_right,
    gm1,
    rhomin,
    pgmin,
    faces,
):
    """
    The limited ideal-MHD interface flux at i + 1/2 from its window (the
    arithmetic of ``mhd_joint_recombination_pallas`` for one interface).

    Args:
        q_left: The local state of cell i (``(rho, m_n, m_t1, m_t2, B_n, B_t1,
            B_t2, E)``).
        q_right: That of cell i + 1.
        plus_here: The plus split face flux of the interface i + 1/2.
        plus_right: That of the interface i + 3/2.
        minus_left: The minus split face flux of the interface i - 1/2.
        minus_here: That of the interface i + 1/2.
        speeds: The splitting speeds of the interfaces i - 1/2, i + 1/2,
            i + 3/2 (the kernel output, before the floor).
        reference_left: The inflow reference state of cell i.
        reference_right: That of cell i + 1.
        speed_sum_left: The summed splitting speeds of cell i.
        speed_sum_right: Those of cell i + 1.
        gm1: gamma - 1.
        rhomin: The density floor.
        pgmin: The pressure floor.
        faces: The number of faces of a cell, ``2 * dimensionality``.

    Returns:
        The interface flux as a list of the eight local components.
    """

    def fraction(base, step):
        return _local_admissible_fraction(base, step, gm1, rhomin, pgmin, True, MAGNETIC_SLOTS)

    def pair_fraction(base, first, second):
        both = tuple(a + b for a, b in zip(first, second))
        return jnp.minimum(jnp.minimum(fraction(base, first), fraction(base, second)), fraction(base, both))

    f_left, f_right = _local_mhd_flux(q_left, gm1), _local_mhd_flux(q_right, gm1)
    alpha = {offset: jnp.maximum(speed, 1e-30) for offset, speed in zip((-1, 0, 1), speeds)}
    # One reciprocal per splitting speed instead of a division per
    # component: divisions are long instruction sequences on the GPU (and
    # in double precision especially), the scalings need only round-off
    # accuracy.
    inverse_alpha = {offset: 1.0 / alpha[offset] for offset in (-1, 0, 1)}
    plus = {0: plus_here, 1: plus_right}
    minus = {-1: minus_left, 0: minus_here}

    def plus_step(offset, state, flux):
        inverse = inverse_alpha[offset]
        return tuple(2.0 * plus[offset][s] * inverse - (state[s] + flux[s] * inverse) for s in range(8))

    def minus_step(offset, state, flux):
        inverse = inverse_alpha[offset]
        return tuple(-2.0 * minus[offset][s] * inverse - (state[s] - flux[s] * inverse) for s in range(8))

    step_plus_here = plus_step(0, q_left, f_left)       # face i+1/2, from cell i
    step_minus_here = minus_step(0, q_right, f_right)   # face i+1/2, from cell i+1
    step_minus_left = minus_step(-1, q_left, f_left)    # face i-1/2, from cell i
    step_plus_right = plus_step(1, q_right, f_right)    # face i+3/2, from cell i+1

    def own(state, step_plus, alpha_plus, step_minus, alpha_minus):
        total = alpha_plus + alpha_minus
        return pair_fraction(
            state,
            tuple(-(alpha_plus / total) * x for x in step_plus),
            tuple(-(alpha_minus / total) * x for x in step_minus),
        )

    own_left = own(q_left, step_plus_here, alpha[0], step_minus_left, alpha[-1])
    own_right = own(q_right, step_plus_right, alpha[1], step_minus_here, alpha[0])
    share_left = faces * alpha[0] / speed_sum_left
    share_right = faces * alpha[0] / speed_sum_right
    # cell i's right inflow face and cell i+1's left inflow face are this one
    right_inflow = fraction(reference_left, tuple(share_left * x for x in step_minus_here))
    left_inflow = fraction(reference_right, tuple(share_right * x for x in step_plus_here))
    theta_plus = jnp.minimum(own_left, left_inflow)
    theta_minus = jnp.minimum(right_inflow, own_right)

    a = alpha[0]
    inverse_a = inverse_alpha[0]
    flux = []
    for slot in range(8):
        plus_flux = 0.5 * a * (q_left[slot] + f_left[slot] * inverse_a + theta_plus * step_plus_here[slot])
        minus_flux = -0.5 * a * (q_right[slot] - f_right[slot] * inverse_a + theta_minus * step_minus_here[slot])
        flux.append(plus_flux + minus_flux)
    return flux


def mhd_joint_recombination_pallas(
    conserved_state,
    split_flux,
    reference_state,
    speed_sum,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Limited interface fluxes along ``axis`` (3D ideal MHD, joint inflow).

    Args:
        conserved_state: The (local) conserved state.
        split_flux: The WENO kernel's output: plus split face fluxes (state
            channels), minus split face fluxes, splitting speed.
        reference_state: ``B`` of ``mhd_inflow_reference``.
        speed_sum: ``sum_d S_d`` of ``mhd_inflow_reference``.

    Returns:
        The interface flux at i + 1/2, aligned with cell i.
    """
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    grid, block = _grid_and_indices(spatial_shape, config)
    local_indices = _mhd_local_indices(registered_variables, axis)
    faces = 2.0 * config.dimensionality

    def kernel(q_ref, split_ref, reference_ref, speed_sum_ref, gamma_ref, rhomin_ref, pgmin_ref, out_ref):
        ii, jj, kk = _block_indices(block, spatial_shape)
        q_at = _reader(q_ref, ii, jj, kk, spatial_shape)
        split_at = _reader(split_ref, ii, jj, kk, spatial_shape)
        reference_at = _reader(reference_ref, ii, jj, kk, spatial_shape)
        speed_sum_at = _reader(speed_sum_ref, ii, jj, kk, spatial_shape)
        gm1 = gamma_ref[()] - 1.0
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]

        def local(read, offset, base=0):
            return tuple(read(base + var, axis, offset) for var in local_indices)

        flux = _joint_recombination_window(
            local(q_at, 0),
            local(q_at, 1),
            local(split_at, 0),
            local(split_at, 1),
            local(split_at, -1, nvars),
            local(split_at, 0, nvars),
            tuple(split_at(2 * nvars, axis, offset) for offset in (-1, 0, 1)),
            local(reference_at, 0),
            local(reference_at, 1),
            speed_sum_at(0, axis, 0),
            speed_sum_at(0, axis, 1),
            gm1,
            rhomin,
            pgmin,
            faces,
        )
        zero = flux[0] * 0.0
        for var in range(nvars):
            if var not in local_indices:
                out_ref[var, ...] = zero
        for slot, var in enumerate(local_indices):
            out_ref[var, ...] = flux[slot]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())
    dtype = conserved_state.dtype

    def full(array):
        return pl.BlockSpec(array.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    speed_sum = speed_sum[None]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, dtype),
        grid=grid,
        in_specs=[full(conserved_state), full(split_flux), full(reference_state), full(speed_sum)] + [scalar_spec] * 3,
        out_specs=pl.BlockSpec((nvars,) + block, lambda bi, bj, bk: (0, bi, bj, bk)),
        interpret=config.backend_config.pallas_interpret,
        name=f"mhd_joint_recombination_axis_{axis}",
        **kwargs,
    )(
        conserved_state, split_flux, reference_state, speed_sum,
        jnp.asarray(params.gamma, dtype=dtype),
        jnp.asarray(params.minimum_density, dtype=dtype),
        jnp.asarray(params.minimum_pressure, dtype=dtype),
    )


def mhd_inflow_reference_dispatch(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """``mhd_inflow_reference`` with the Pallas kernel where the ideal-MHD
    Pallas WENO runs (3D), the array form elsewhere. Like the Pallas WENO
    fluxes, the Pallas path sits behind ``diffable_pallas_call``, so its
    tangent (and the reverse sweep) runs the array form, in the same
    ``shard_map`` + halo exchange as the primal."""
    from astronomix._finite_difference._interface_fluxes._weno_pallas import _mhd_pallas_flux_supported
    from astronomix._finite_difference._interface_fluxes._weno_positivity import mhd_inflow_reference
    from astronomix._pallas_helpers import _pallas_call_sharded, diffable_pallas_call

    if config.dimensionality != 3 or not _mhd_pallas_flux_supported(conserved_state, config):
        return mhd_inflow_reference(conserved_state, params, config, registered_variables)
    nvars = int(conserved_state.shape[0])
    _, block = _grid_and_indices(tuple(int(x) for x in conserved_state.shape[1:]), config)

    def sharded(reference_fn, block_shape):
        def run(state, params):
            def build(state_local):
                reference, speed_sum = reference_fn(state_local, params, config, registered_variables)
                return jnp.concatenate([reference, speed_sum[None]], axis=0)

            # the face speeds reach cells i -+ 3 along every axis
            out = _pallas_call_sharded(build, state_inputs=(state,), halo=(4, 4, 4), block_shape=block_shape)
            return out[:nvars], out[nvars]

        return run

    return diffable_pallas_call(
        conserved_state,
        params,
        pallas_branch=sharded(mhd_inflow_reference_pallas, block),
        native_branch=sharded(mhd_inflow_reference, (1, 1, 1)),
    )
