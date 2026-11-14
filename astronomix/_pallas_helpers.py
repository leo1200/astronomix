"""
Shared Pallas-backend utilities used across the finite-difference and
finite-volume paths.

This module is the single place that

- imports Pallas / Triton (``pl is None`` / ``pltriton is None`` when they are
  unavailable),
- normalises ``config.backend_config.pallas_block_shape`` to a 3-tuple and
  builds the Triton ``CompilerParams`` from the configuration,
- exposes the ``backend == PALLAS`` predicate,
- holds the active device mesh and state ``PartitionSpec`` of a sharded run
  (``pallas_mesh_context``),
- provides the multi-GPU wrapper ``_pallas_call_sharded``, which turns an
  opaque ``pl.pallas_call`` into a ``shard_map`` body with a ppermute halo
  exchange, ``sharded_roll``, the same halo exchange for the periodic rolls
  of the native-JAX stencils, and ``shard_map_on_padded_shards``, which pads
  every shard once with the halo of a whole computation (the stage of the
  Riemann-flux scheme) and runs it there,
- pairs every Pallas kernel with a native-JAX tangent so that the Pallas paths
  are differentiable (``diffable_pallas_call``).

Every Pallas kernel module under ``astronomix`` imports from here, so new knobs
and fallbacks only need to be added once.
"""

# general
import contextvars
from contextlib import contextmanager

# jax
import jax
import jax.numpy as jnp
from jax.sharding import (
    Mesh,
    NamedSharding,
    PartitionSpec,
)

try:
    from jax.shard_map import shard_map  # jax >= 0.8
except ImportError:  # jax < 0.8
    from jax.experimental.shard_map import shard_map

# astronomix constants
from astronomix.option_classes.simulation_config import (
    PAIR_ALL_TO_ALL,
    PALLAS,
    PPERMUTE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig

# Pallas and Triton are optional: a CPU-only or older JAX install may lack one
# or both. They are imported defensively so that the rest of the module loads
# (callers gate on ``pl is None`` / ``pltriton is None``) instead of failing at
# import time.
try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - Pallas optional
    pl = None

try:
    from jax.experimental.pallas import triton as pltriton
except Exception:  # pragma: no cover - Triton GPU backend optional
    pltriton = None


def _backend_is_pallas(config: SimulationConfig) -> bool:
    """Return whether the configured backend is the Pallas/Triton GPU backend."""
    return config.backend_config.backend == PALLAS


#: Cells per program of the default Pallas block: one cell per thread at the
#: default ``pallas_num_warps = 4``.
DEFAULT_BLOCK_CELLS = 128


def _default_pallas_block_shape(ndim: int, spatial_shape=None) -> tuple[int, int, int]:
    """
    Return the default Pallas block shape ``(bx, by, bz)`` for ``ndim`` spatial
    dimensions (inactive dimensions set to 1): ``DEFAULT_BLOCK_CELLS`` cells,
    filled from the contiguous (last) spatial axis, i.e. ``(1, 1, 128)`` in 3D
    and ``(1, 128)`` in 2D on production grids.

    The state is stored as (variable, x, y, z) with z contiguous, and a memory
    transaction moves 128 contiguous bytes, so long rows along the last axis
    read the stencils with the fewest transactions; 128 cells keep one cell
    per thread at four warps. On a GH200 this was the fastest or within ~2 %
    of the fastest block for every kernel family (finite-difference WENO
    hydrodynamics, ideal and isothermal MHD with and without the
    positivity-preserving WENO, Pallas CT, VL2) in both precisions: blocks with
    short rows, e.g. (4, 4, 8), were 7-38 % slower, and in 2D the earlier
    (16, 16) up to 2x slower (256 cells on four warps spill registers). The
    block extent of 1 along x also keeps the block-rounded halo of an
    x-sharded run at the stencil reach.

    Args:
        ndim: The number of spatial dimensions.
        spatial_shape: Optional spatial shape of the array to be tiled. With
            it, every axis gets the largest power of two that fits the
            remaining cell budget and divides the extent (so a 64^3 grid gets
            (1, 2, 64)).

    Returns:
        The block shape as a 3-tuple.
    """
    if spatial_shape is None:
        extents = (DEFAULT_BLOCK_CELLS,) * ndim
    else:
        extents = tuple(int(extent) for extent in spatial_shape)
    parts = [1, 1, 1]
    remaining_cells = DEFAULT_BLOCK_CELLS
    for axis in reversed(range(ndim)):
        parts[axis] = _largest_tiling_block_size(remaining_cells, extents[axis])
        remaining_cells //= parts[axis]
    if _too_small_for_a_warp(parts, extents):
        # No efficient tiling exists (e.g. an odd number of interior cells plus
        # ghost cells): keep the full-size block, which does not tile the grid,
        # so that the support predicates select the native kernels, as before.
        return tuple(DEFAULT_BLOCK_CELLS if axis == ndim - 1 else 1 for axis in range(3))
    return tuple(parts)


def _too_small_for_a_warp(parts, extents) -> bool:
    """
    Whether a resolved block holds fewer cells than one warp although the grid
    has more: such a block would leave most threads of a program idle.
    """
    block_cells = parts[0] * parts[1] * parts[2]
    grid_cells = 1
    for extent in extents:
        grid_cells *= int(extent)
    return block_cells < min(32, grid_cells)


def _largest_tiling_block_size(requested_size: int, extent: int) -> int:
    """
    The largest power of two that is at most ``requested_size`` and divides
    ``extent``.

    Triton tiles must be powers of two, and the kernels tile the grid exactly,
    so a block size must be both. Clamping a block to a smaller grid extent
    (``min(32, 24) = 24``) would break the first rule, and keeping a block that
    does not divide the extent (32 on 48 cells) the second, which makes the
    support predicates drop the whole run to the native backend.

    Args:
        requested_size: The configured (or default) block size.
        extent: The grid extent along the axis.

    Returns:
        The block size along the axis (1 if no larger power of two fits).
    """
    block_size = 1
    while 2 * block_size <= requested_size and extent % (2 * block_size) == 0:
        block_size *= 2
    return block_size


def _as_3tuple_block_shape(block_shape, ndim: int, spatial_shape=None) -> tuple[int, int, int]:
    """
    Normalise a user-supplied block shape (None, a string or a tuple) to
    ``(bx, by, bz)`` with the inactive dimensions set to 1. The Pallas grid
    construction relies on this tuple being canonical.

    When ``spatial_shape`` is given, each active block dimension is reduced to
    the largest power of two that is at most the requested size and divides
    the grid extent (see ``_largest_tiling_block_size``), so that any block
    stays a valid Triton tiling of any grid (``bz = 32`` becomes 16 on 48
    cells and 8 on 24) instead of failing the grid-divisibility support
    predicates, which would silently drop the whole run to the native
    backend. ``None`` selects ``_default_pallas_block_shape``.

    Args:
        block_shape: The configured block shape (None, "bx,by,bz" or a tuple).
        ndim: The number of spatial dimensions.
        spatial_shape: Optional spatial shape of the array to be tiled.

    Returns:
        The block shape as a 3-tuple.
    """
    if block_shape is None:
        return _default_pallas_block_shape(ndim, spatial_shape)
    if isinstance(block_shape, str):
        parts = tuple(int(part.strip()) for part in block_shape.split(",") if part.strip())
    else:
        parts = tuple(int(size) for size in block_shape)
    if len(parts) == 0:
        return _default_pallas_block_shape(ndim, spatial_shape)
    parts = (tuple(parts[:ndim]) + (1, 1, 1))[:3]
    parts = tuple(parts[axis] if axis < ndim else 1 for axis in range(3))
    if spatial_shape is not None:
        extents = tuple(int(extent) for extent in spatial_shape) + (1, 1)
        resolved = tuple(
            _largest_tiling_block_size(block_size, extent)
            for block_size, extent in zip(parts, extents[:3])
        )
        # Without an efficient tiling the configured block is kept; it does not
        # tile the grid, so the support predicates select the native kernels.
        if not _too_small_for_a_warp(resolved, extents[:ndim]):
            parts = resolved
    return parts


def _pallas_compiler_params(config: SimulationConfig):
    """
    Return the Triton ``CompilerParams``, or None if the Triton backend is not
    available or the user opted out via ``pallas_use_triton=False``.
    """
    use_triton = config.backend_config.pallas_use_triton
    if use_triton and pltriton is not None:
        return pltriton.CompilerParams(
            num_warps=config.backend_config.pallas_num_warps,
        )
    return None


# -------------------------------------------------------------
# =========== ↓ Multi-GPU shard_map + halo wrapper ↓ ==========
# -------------------------------------------------------------
#
# Every Pallas kernel in this codebase passes its state-shaped input(s) to
# ``pl.pallas_call`` via ``BlockSpec(state.shape, lambda ...: (0, 0, 0, 0))``.
# That tells Pallas/Triton that each block program may read anywhere in the
# array, which is the correct (and fast) layout on a single device, but it
# makes the call entirely opaque to GSPMD. When the input is sharded across a
# device mesh, XLA's only legal lowering is to all-gather the whole state on
# every device before each ``pallas_call``, which dominates every kernel and
# removes any strong scaling.
#
# Instead, each ``pl.pallas_call`` is wrapped in a ``shard_map`` body that
#   1. ppermutes a halo of ``stencil_reach`` cells from each neighbour shard
#      along every sharded spatial axis (periodic ring),
#   2. concatenates [left_halo, local, right_halo] on each sharded axis,
#   3. calls the unchanged kernel on the halo-padded local shard (its modular
#      indexing wraps within the padded shape; the halo cells provide the
#      correct neighbour values for the interior reads),
#   4. strips the halo from the output.
# With ``local_output`` the kernel writes the shard's cells only (no strip),
# with ``separate_halos`` it also reads the halos apart from the shard (no pad).
# ``shard_map_on_padded_shards`` exchanges once for a whole computation; the
# kernels it runs write a region set with ``_shard_context`` (the shard widened
# by a margin) from inputs that each carry their own halo around it.
#
# The user-facing knob is ``pallas_mesh_context(mesh, spec)`` around the JIT
# trace, plus each kernel calling ``_pallas_call_sharded`` instead of
# ``pl.pallas_call(...)(args)`` directly. No kernel arithmetic changes.


_pallas_mesh_ctx: contextvars.ContextVar = contextvars.ContextVar(
    "astronomix_pallas_mesh",
    default=None,
)
_pallas_spec_ctx: contextvars.ContextVar = contextvars.ContextVar(
    "astronomix_pallas_state_spec",
    default=None,
)
_pallas_halo_exchange_ctx: contextvars.ContextVar = contextvars.ContextVar(
    "astronomix_pallas_halo_exchange",
    default=PAIR_ALL_TO_ALL,
)
#: The cells the build writes: inside ``_pallas_call_sharded`` with ``local_output`` the shard (its spatial shape,
#: the halos of its inputs with ``separate_halos`` or None), inside ``shard_map_on_padded_shards`` a region.
_pallas_shard_ctx: contextvars.ContextVar = contextvars.ContextVar("astronomix_pallas_shard", default=None)


@contextmanager
def _shard_context(shard):
    """
    Set the cells a kernel build writes, ``(spatial shape, halos)``: the shard (or a region of a padded shard,
    with ``halos`` None) centred in every input; None writes the whole array.
    """
    token = _pallas_shard_ctx.set(shard)
    try:
        yield
    finally:
        _pallas_shard_ctx.reset(token)


def _shard_cells(spatial_shape):
    """
    ``(first cells, extents)`` per axis of the cells a kernel build writes from an input of ``spatial_shape``:
    on a shard (``local_output``) its own cells, which an input padded by a halo holds after it; else all.
    """
    spatial_shape = tuple(int(extent) for extent in spatial_shape)
    shard = _pallas_shard_ctx.get()
    if shard is None:
        return (0,) * len(spatial_shape), spatial_shape
    return tuple((extent - cells) // 2 for extent, cells in zip(spatial_shape, shard[0], strict=True)), shard[0]


def _shifted_index(index, offset: int):
    """``index + offset``, and ``index`` itself for a zero offset (the program of one device stays as it is)."""
    return index if offset == 0 else index + offset


def _shard_halos():
    """
    With ``separate_halos``: per kernel input and spatial axis, ``(left, right)`` (the cells of the two
    neighbouring shards, as wide as the exchange) or None for an axis without one; None elsewhere.
    """
    shard = _pallas_shard_ctx.get()
    return None if shard is None else shard[1]


@contextmanager
def pallas_mesh_context(mesh, spec=None, halo_exchange=PAIR_ALL_TO_ALL):
    """
    Set the active mesh (and state PartitionSpec) for Pallas kernel sharding.

    ``time_integration`` enters this context around the JIT trace whenever the
    user supplies a ``sharding`` argument. Inside the context every Pallas
    kernel that calls ``_pallas_call_sharded`` runs as a ``shard_map`` with a
    ppermute halo exchange instead of a bare ``pl.pallas_call``.

    Inside a JIT trace the kernel inputs are tracers, which carry no
    ``.sharding`` under ``AxisType.Auto`` meshes, so the wrapper needs ``spec``
    to know which spatial axes are split; the mesh axis names alone do not say
    that (a 1-axis ``("x",)`` mesh with ``P(None, "x")`` splits x, which
    ``P(*mesh.axis_names)`` would not).

    Callers that differentiate a sharded ``time_integration`` must hold this
    context around the outer trace as well: reverse-mode rules (custom_jvp
    primals, the backward sweep of the checkpointed loop) are traced after
    ``time_integration`` has returned.

    When ``mesh`` is ``None`` (single-device run) or has size 1, the context has
    no effect and the kernels run unwrapped.

    Args:
        mesh: The device mesh, or None.
        spec: The PartitionSpec of the ``(var, x, y, z)`` state (the user's
            ``sharding.spec``), or None.
        halo_exchange: How the halos are exchanged along a split axis
            (``BackendConfig.halo_exchange``).
    """
    if halo_exchange not in (PPERMUTE, PAIR_ALL_TO_ALL):
        raise ValueError(f"Unknown halo_exchange {halo_exchange!r}.")
    token = _pallas_mesh_ctx.set(mesh)
    token_spec = _pallas_spec_ctx.set(spec)
    token_halo_exchange = _pallas_halo_exchange_ctx.set(halo_exchange)
    try:
        yield
    finally:
        _pallas_halo_exchange_ctx.reset(token_halo_exchange)
        _pallas_spec_ctx.reset(token_spec)
        _pallas_mesh_ctx.reset(token)


def _current_pallas_mesh():
    """Return the mesh of the active Pallas mesh context (or None)."""
    return _pallas_mesh_ctx.get()


def _current_pallas_spec():
    """Return the state PartitionSpec of the active Pallas mesh context (or None)."""
    return _pallas_spec_ctx.get()


def sharded_roll(x, shift: int, axis: int):
    """
    Periodic roll along the split axis of the active Pallas mesh, computed as a
    halo exchange.

    Inside a multi-device ``pallas_mesh_context``, ``jnp.roll(x, shift, axis)``
    along the axis that the state PartitionSpec distributes over devices is
    computed in a ``shard_map`` in which every shard passes its ``|shift|``
    boundary planes to its neighbour by ppermute. GSPMD partitions the slice +
    concatenate of a periodic roll along a split axis as two all-to-all
    reshards of the whole local block instead, which costs far more
    communication than the halo. The roll is a pure data movement, so the
    result is bitwise identical to the single-device roll.

    In a reverse sweep even one ppermute per stencil shift is costly, which is
    why the native WENO tangent runs shard-local as a whole
    (``_weno._native_tangent_sharded``).

    The split axis is read from the context spec: an array with as many
    dimensions as the ``(var, x, y, z)`` state spec uses the spec's own axis
    position, an array with one dimension fewer (a single field ``(x, y, z)``)
    the position shifted by one.

    Args:
        x: The array to roll.
        shift: The signed number of positions to roll by.
        axis: The axis along which to roll.

    Returns:
        The rolled array, or ``None`` when the caller should roll as usual: no
        multi-device context, ``axis`` is not the split axis, the axis does not
        divide evenly over the devices, ``|shift|`` exceeds one shard, or the
        split axis has only two devices (see the note in the body).
    """
    mesh = _current_pallas_mesh()
    spec = _current_pallas_spec()
    if mesh is None or mesh.size <= 1 or spec is None:
        return None

    # --------------- ↓ Locate the split axis ↓ ----------------
    spec_axis_names = list(spec)
    split_positions = [
        position for position, axis_name in enumerate(spec_axis_names) if axis_name is not None
    ]
    if len(split_positions) != 1:
        return None
    split_axis = split_positions[0]
    mesh_axis_name = spec_axis_names[split_axis]
    if isinstance(mesh_axis_name, tuple):
        if len(mesh_axis_name) != 1:
            return None
        mesh_axis_name = mesh_axis_name[0]
    num_dims = x.ndim
    if num_dims == len(spec_axis_names) - 1 and split_axis >= 1:
        split_axis -= 1
    elif num_dims != len(spec_axis_names):
        return None
    axis = axis % num_dims
    if axis != split_axis:
        return None
    # --------------- ↑ Locate the split axis ↑ ----------------

    num_devices = mesh.shape[mesh_axis_name]
    axis_length = x.shape[axis]
    if num_devices <= 1 or axis_length % num_devices:
        return None

    # NOTE: with two devices along the split axis, native-JAX tangents that are
    # not shard-local (currently the MHD WENO fluxes) round differently under
    # this roll than under GSPMD's, and the non-smooth WENO tangent amplifies
    # the difference to percent-level changes of reverse-mode gradients at a
    # few cells. Such meshes therefore keep GSPMD's roll.
    if num_devices == 2:
        return None

    local_length = axis_length // num_devices
    signed_shift = int(shift) % axis_length
    if signed_shift > axis_length // 2:
        signed_shift -= axis_length
    if signed_shift == 0:
        return x
    if abs(signed_shift) > local_length:
        return None

    array_spec = [None] * num_dims
    array_spec[split_axis] = mesh_axis_name
    array_spec = PartitionSpec(*array_spec)
    send_to_right = [(device, (device + 1) % num_devices) for device in range(num_devices)]
    send_to_left = [(device, (device - 1) % num_devices) for device in range(num_devices)]

    def roll_local_block(local_block):
        """Roll one shard, receiving the wrapped planes from its neighbour."""
        if signed_shift > 0:
            # out[i] = in[i - shift]: the first ``shift`` planes come from the
            # left neighbour.
            received_planes = jax.lax.ppermute(
                jax.lax.slice_in_dim(
                    local_block,
                    local_length - signed_shift,
                    local_length,
                    axis=axis,
                ),
                mesh_axis_name,
                perm=send_to_right,
            )
            return jax.lax.concatenate(
                [
                    received_planes,
                    jax.lax.slice_in_dim(local_block, 0, local_length - signed_shift, axis=axis),
                ],
                axis,
            )
        # out[i] = in[i + |shift|]: the last ``|shift|`` planes come from the
        # right neighbour.
        planes_from_right = -signed_shift
        received_planes = jax.lax.ppermute(
            jax.lax.slice_in_dim(local_block, 0, planes_from_right, axis=axis),
            mesh_axis_name,
            perm=send_to_left,
        )
        return jax.lax.concatenate(
            [
                jax.lax.slice_in_dim(local_block, planes_from_right, local_length, axis=axis),
                received_planes,
            ],
            axis,
        )

    return shard_map(
        roll_local_block,
        mesh=mesh,
        in_specs=(array_spec,),
        out_specs=array_spec,
        check_rep=False,
    )(x)


def _round_halo_up_to_block(halo, block_shape) -> tuple[int, ...]:
    """
    Round each natural halo width up to a multiple of the corresponding Pallas
    block size. The kernel's internal ``grid = (nx // bx, ...)`` must remain
    block-divisible after the halo padding, so the halo always grows to the
    nearest block multiple.

    Args:
        halo: The natural halo width per spatial axis.
        block_shape: The Pallas block size per spatial axis.

    Returns:
        The block-rounded halo width per spatial axis.
    """
    rounded_halo = []
    for width, block_size in zip(halo, block_shape, strict=False):
        width = int(width)
        block_size = max(int(block_size), 1)
        if width <= 0:
            rounded_halo.append(0)
        else:
            num_blocks, remainder = divmod(width, block_size)
            rounded_halo.append(block_size * (num_blocks + (1 if remainder else 0)))
    return tuple(rounded_halo)


def _spatial_sharded_axes(mesh, pspec, ndim):
    """
    List every spatial array axis that is split across more than one device.
    The variable axis (array axis 0) is always skipped.

    Args:
        mesh: The device mesh.
        pspec: The PartitionSpec of the ``(var, x, y, z)`` array.
        ndim: The number of spatial dimensions.

    Returns:
        A list of ``(array_axis, mesh_axis_name, num_devices)`` tuples.
    """
    sharded_axes = []
    for array_axis in range(1, ndim + 1):
        if array_axis >= len(pspec):
            break
        mesh_axis_entry = pspec[array_axis]
        if mesh_axis_entry is None:
            continue
        if isinstance(mesh_axis_entry, tuple):
            for mesh_axis_name in mesh_axis_entry:
                num_devices = mesh.shape[mesh_axis_name]
                if num_devices > 1:
                    sharded_axes.append((array_axis, mesh_axis_name, num_devices))
        else:
            num_devices = mesh.shape[mesh_axis_entry]
            if num_devices > 1:
                sharded_axes.append((array_axis, mesh_axis_entry, num_devices))
    return sharded_axes


def _default_state_pspec(mesh, ndim) -> PartitionSpec:
    """
    Fallback PartitionSpec for an array whose sharding is unknown (for example
    an intermediate inside a JIT trace with no ``pallas_mesh_context`` spec).
    Assumes the standard ``(VARAXIS, XAXIS, YAXIS, ZAXIS)`` mesh built by
    ``pytests/_benchmark_utils.py::_build_sharding`` and by callers that mirror
    it.

    Args:
        mesh: The device mesh.
        ndim: The number of spatial dimensions.

    Returns:
        ``PartitionSpec(*mesh.axis_names[: 1 + ndim])``.
    """
    axis_names = tuple(mesh.axis_names)
    return PartitionSpec(*axis_names[: 1 + ndim])


def _resolve_state_pspec(state, mesh) -> PartitionSpec:
    """
    Return the PartitionSpec of a (var, x, y, z) state array on ``mesh``.

    Concrete arrays carry their own ``NamedSharding``. Tracers carry no
    ``.sharding`` under ``AxisType.Auto`` meshes, so inside a traced
    ``time_integration(sharding=...)`` the spec registered with
    ``pallas_mesh_context`` is used, and the default ``(var, x, y, z)`` spec
    otherwise.

    Args:
        state: A state-shaped array (leading variable axis).
        mesh: The active Pallas mesh.

    Returns:
        The PartitionSpec of ``state``.
    """
    try:
        sharding = getattr(state, "sharding", None)
    except Exception:
        # Reading ``.sharding`` of a tracer raises ("use jax.typeof(x)"), and
        # not always as an AttributeError.
        sharding = None
    context_spec = _current_pallas_spec()
    if isinstance(sharding, NamedSharding):
        return sharding.spec
    if context_spec is not None and len(context_spec) <= state.ndim:
        return PartitionSpec(*context_spec, *((None,) * (state.ndim - len(context_spec))))
    return _default_state_pspec(mesh, state.ndim - 1)


def _pallas_mesh_splits_axis(state, spatial_axis: int) -> bool:
    """
    Whether the active Pallas mesh distributes ``spatial_axis`` (0 = x) of
    ``state`` over more than one device.

    This is the condition under which ``_pallas_call_sharded`` exchanges halos
    along that axis; code paths that rely on such an exchange (for example a
    kept output halo) must be gated on it rather than on the number of
    visible devices, which says nothing about how the state is split.

    Args:
        state: A state-shaped array (leading variable axis).
        spatial_axis: The spatial axis, 0-based.

    Returns:
        True if the axis is split over several devices.
    """
    mesh = _current_pallas_mesh()
    if mesh is None or mesh.size <= 1:
        return False
    num_spatial_dims = state.ndim - 1
    pspec = _resolve_state_pspec(state, mesh)
    split_array_axes = {
        array_axis for array_axis, _, _ in _spatial_sharded_axes(mesh, pspec, num_spatial_dims)
    }
    return spatial_axis + 1 in split_array_axes


def _per_axis_halo(halo, num_spatial_dims: int) -> tuple:
    """
    Normalise a halo specification to one non-negative width per spatial axis.

    Short tuples are padded with zeros and long ones truncated, so ``(2,)``
    becomes ``(2, 0, 0)`` in 3D and ``(2, 1, 4)`` becomes ``(2, 1)`` in 2D.
    """
    widths = tuple(int(width) for width in halo) + (0,) * max(0, num_spatial_dims - len(halo))
    widths = widths[:num_spatial_dims]
    if any(width < 0 for width in widths):
        raise ValueError("Halo widths must be non-negative.")
    return widths


def _normalize_input_halos(input_halos, num_inputs: int, num_spatial_dims: int):
    """
    One per-axis halo width tuple per state input, or ``None`` if every input
    exchanges the full (block-rounded) halo.
    """
    if input_halos is None:
        return None
    if len(input_halos) != num_inputs:
        raise ValueError("input_halos must give one halo per state input.")
    return tuple(_per_axis_halo(halo, num_spatial_dims) for halo in input_halos)


def _normalize_output_halos(output_halo, num_outputs: int, num_spatial_dims: int):
    """
    One per-axis kept-halo tuple per state output. A single per-axis tuple
    applies to every output; ``None`` keeps no halo.
    """
    if output_halo is None:
        return tuple((0,) * num_spatial_dims for _ in range(num_outputs))
    if all(isinstance(width, int) for width in output_halo):
        output_halos = tuple(output_halo for _ in range(num_outputs))
    else:
        if len(output_halo) != num_outputs:
            raise ValueError("output_halo must give one halo per state output.")
        output_halos = tuple(output_halo)
    return tuple(_per_axis_halo(halo, num_spatial_dims) for halo in output_halos)


def _local_edge_padding(array, width: int, axis: int, *, left: bool):
    """
    ``width`` copies of the local edge plane of ``array`` along ``axis``.

    These fill the part of the block-rounded padding that an input does not
    exchange with its neighbours (see ``input_halos`` of
    ``_pallas_call_sharded``); kernels must never read them into a kept
    output.
    """
    if width <= 0:
        return None
    size = array.shape[axis]
    if left:
        edge = jax.lax.slice_in_dim(array, 0, 1, axis=axis)
    else:
        edge = jax.lax.slice_in_dim(array, size - 1, size, axis=axis)
    repetitions = [1] * array.ndim
    repetitions[axis] = int(width)
    return jnp.tile(edge, repetitions)


def _neighbour_pairs(num_devices: int, phase: int) -> list:
    """
    The disjoint neighbour pairs of one phase of the pairwise halo exchange:
    ``(0, 1), (2, 3), ...`` in phase 0 and ``(1, 2), ..., (N - 1, 0)`` in
    phase 1, each pair ordered left neighbour first.
    """
    return [[left, (left + 1) % num_devices] for left in range(phase, num_devices, 2)]


def _pairwise_halos(left_edge, right_edge, mesh_axis_name, num_devices: int):
    """
    The halos received from both ring neighbours, exchanged in two phases of
    all-to-alls over disjoint neighbour pairs instead of two permutes over the
    whole ring.

    A collective over the whole ring runs on a communicator that spans every
    device of the axis, and its cost grows with the device count; each
    all-to-all here spans one pair. In a pair the left device sends its right
    edge and receives the right device's left edge; every device is the left
    member of the pair of one phase and the right member of the other. The
    data that crosses devices is exactly the ring exchange's, so the halos are
    bit-identical. Requires an even ``num_devices`` (the phases would overlap
    otherwise) and a string ``mesh_axis_name`` (JAX rejects
    ``axis_index_groups`` for integer axis names).

    Args:
        left_edge: This shard's left boundary planes.
        right_edge: This shard's right boundary planes.
        mesh_axis_name: The (string) mesh axis the array axis is split over.
        num_devices: The number of devices along that axis (even).

    Returns:
        ``(left_halo, right_halo)``: the right edge of the left neighbour and
        the left edge of the right neighbour.
    """
    device_index = jax.lax.axis_index(mesh_axis_name)
    zero_planes = jnp.zeros_like(left_edge)
    left_halo = zero_planes
    right_halo = zero_planes
    for phase in (0, 1):
        is_left_member = (device_index % 2) == phase
        # Slot 0 travels to the left member, slot 1 to the right member.
        payload = jnp.stack(
            (
                jnp.where(is_left_member, zero_planes, left_edge),
                jnp.where(is_left_member, right_edge, zero_planes),
            ),
            axis=0,
        )
        received = jax.lax.all_to_all(
            payload,
            mesh_axis_name,
            split_axis=0,
            concat_axis=0,
            axis_index_groups=_neighbour_pairs(num_devices, phase),
            tiled=False,
        )
        # The left member received its right neighbour's left edge in slot 1,
        # the right member its left neighbour's right edge in slot 0.
        right_halo = jnp.where(is_left_member, received[1], right_halo)
        left_halo = jnp.where(is_left_member, left_halo, received[0])
    return left_halo, right_halo


def _string_axis_name(mesh_axis_name) -> str:
    """A string version of a (possibly integer) mesh axis name."""
    return mesh_axis_name if isinstance(mesh_axis_name, str) else f"mesh_axis_{mesh_axis_name}"


def _with_string_axis_names(mesh, pspec):
    """
    The same device mesh and PartitionSpec with string axis names, for the
    grouped collectives that JAX supports only on named axes.
    """
    named_mesh = Mesh(
        mesh.devices,
        tuple(_string_axis_name(name) for name in mesh.axis_names),
        axis_types=mesh.axis_types,
    )

    def rename(entry):
        if entry is None:
            return None
        if isinstance(entry, tuple):
            return tuple(_string_axis_name(name) for name in entry)
        return _string_axis_name(entry)

    named_pspec = PartitionSpec(*(rename(entry) for entry in pspec))
    return named_mesh, named_pspec


def _exchanged_halos(array, axis, width, mesh_axis_name, num_devices, pairwise: bool):
    """
    The ``width`` cells along ``axis`` of the left and of the right neighbouring shard (periodic ring), received
    by permutes over the ring or, with ``pairwise``, by ``_pairwise_halos``.
    """
    size = array.shape[axis]
    left_edge = jax.lax.slice_in_dim(array, 0, width, axis=axis)
    right_edge = jax.lax.slice_in_dim(array, size - width, size, axis=axis)
    return _received_halos(left_edge, right_edge, mesh_axis_name, num_devices, pairwise)


def _received_halos(left_edge, right_edge, mesh_axis_name, num_devices, pairwise: bool):
    """
    The right edge of the left neighbouring shard and the left edge of the right one (periodic ring), for the
    edges ``left_edge`` and ``right_edge`` of this shard: by permutes over the ring or, with ``pairwise``, by
    ``_pairwise_halos``.
    """
    if pairwise:
        return _pairwise_halos(left_edge, right_edge, mesh_axis_name, num_devices)
    to_left = [(device, (device - 1) % num_devices) for device in range(num_devices)]
    to_right = [(device, (device + 1) % num_devices) for device in range(num_devices)]
    # Each device sends its right edge to the right neighbour, which installs
    # it as its left halo; symmetrically for the right halo.
    left_halo = jax.lax.ppermute(right_edge, mesh_axis_name, perm=to_right)
    right_halo = jax.lax.ppermute(left_edge, mesh_axis_name, perm=to_left)
    return left_halo, right_halo


def _pad_axis_with_halo(
    array,
    axis,
    padded_width,
    exchanged_width,
    mesh_axis_name,
    num_devices,
    pairwise: bool = False,
):
    """
    Pad ``array`` by ``padded_width`` cells on both sides of ``axis``: the
    innermost ``exchanged_width`` cells come from the neighbouring shards
    (periodic ring), the rest are local edge copies.

    Args:
        array: The local shard.
        axis: The array axis to pad.
        padded_width: The (block-rounded) padding per side.
        exchanged_width: How many of those cells are real neighbour data.
        mesh_axis_name: The mesh axis the array axis is split over.
        num_devices: The number of devices along that mesh axis.
        pairwise: Exchange over disjoint neighbour pairs
            (``_pairwise_halos``) instead of permutes over the whole ring.

    Returns:
        The padded local array.
    """
    if exchanged_width > padded_width:
        raise ValueError("An input halo cannot exceed the padded shape halo.")

    pieces = []
    left_padding = _local_edge_padding(array, padded_width - exchanged_width, axis, left=True)
    if left_padding is not None:
        pieces.append(left_padding)

    if exchanged_width > 0:
        left_halo, right_halo = _exchanged_halos(array, axis, exchanged_width, mesh_axis_name, num_devices, pairwise)
        pieces.extend((left_halo, array, right_halo))
    else:
        pieces.append(array)

    right_padding = _local_edge_padding(array, padded_width - exchanged_width, axis, left=False)
    if right_padding is not None:
        pieces.append(right_padding)

    return jnp.concatenate(pieces, axis=axis)


def _pallas_call_sharded(
    kernel_build_fn,
    state_inputs,
    other_args=(),
    *,
    halo,
    block_shape,
    num_state_outputs: int = 1,
    input_halos=None,
    output_halo=None,
    local_output: bool = False,
    separate_halos: bool = False,
):
    """
    Optionally wrap a Pallas kernel build-and-call in ``shard_map``.

    Args:
        kernel_build_fn:
            Callable ``(state_inputs_local_padded..., other_args...) -> out``
            whose body builds and calls ``pl.pallas_call``. When the call
            runs inside a ``shard_map`` body, each invocation sees the
            *local* (halo-padded) shape and the kernel's internal
            ``grid``/``BlockSpec`` are built for that shape automatically.
        state_inputs:
            Tuple of state-shape arrays that all share the same sharding
            (same ``PartitionSpec``). Each one is padded with halo cells
            from neighbour shards along every sharded spatial axis.
        other_args:
            Tuple of replicated arrays (scalar dt, scalar gamma, ...)
            passed through as ``PartitionSpec()``.
        halo:
            Per-spatial-axis natural stencil reach ``(hx, hy, hz)``.
            Pointwise kernels pass ``(0, 0, 0)``; they still get the
            ``shard_map`` (so the kernel runs locally on each shard),
            just with no ppermute.
        block_shape:
            Per-spatial-axis Pallas block size ``(bx, by, bz)``. The
            halo is rounded up to the nearest block multiple so the
            padded shard remains block-divisible.
        num_state_outputs:
            Number of state-shape outputs of ``kernel_build_fn`` (1 for
            most kernels; more for kernels that return tuples of state-shaped
            arrays).
        input_halos:
            Optional per-input halo widths ``((hx, hy, hz), ...)``. Every input
            is still padded to the same block-rounded shape, but only the
            requested cells are exchanged with the neighbours; the rest are
            local edge copies. The kernel must not read those copies into any
            output it keeps. ``None`` exchanges the full padding for every
            input.
        output_halo:
            Optional halo width to keep on the state-shaped outputs, one
            per-axis tuple for all outputs or one per output. The default
            strips all padding. A kept halo lets a following local stencil
            reuse cells this kernel already computed instead of exchanging
            them again.
        local_output:
            The build writes only the cells of the shard (``_shard_cells``),
            and every input is padded by its own exchanged halo only (an input
            that exchanges nothing is the local shard itself, which an
            in-place output may alias). No padded output and no strip.
            Excludes ``output_halo``.
        separate_halos:
            ``local_output`` without padded copies: every input is the local
            shard, and the cells received from the neighbours along each split
            axis come as separate arrays (``_shard_halos``), from the unpadded
            shards, so a kernel may read along one split axis at a time only.

    Returns:
        Either ``kernel_build_fn(*state_inputs, *other_args)`` directly
        (single-device path) or the equivalent ``shard_map``-wrapped
        result with the halo stripped from each state-shape output.
    """
    mesh = _current_pallas_mesh()
    if mesh is None or mesh.size <= 1:
        return kernel_build_fn(*state_inputs, *other_args)

    first_state_input = state_inputs[0]
    ndim = first_state_input.ndim - 1
    pspec = _resolve_state_pspec(first_state_input, mesh)

    sharded_axes = _spatial_sharded_axes(mesh, pspec, ndim)
    if not sharded_axes:
        return kernel_build_fn(*state_inputs, *other_args)

    # The pairwise exchange needs an even device count along every split axis
    # and string axis names; the shard_map then runs on the same devices under
    # string names.
    pairwise = _pallas_halo_exchange_ctx.get() == PAIR_ALL_TO_ALL and all(
        num_devices % 2 == 0 for _, _, num_devices in sharded_axes
    )
    if pairwise:
        mesh, pspec = _with_string_axis_names(mesh, pspec)
        sharded_axes = _spatial_sharded_axes(mesh, pspec, ndim)

    block_shape_3d = tuple(block_shape) + (1,) * max(0, 3 - len(block_shape))
    halo_3d = tuple(halo) + (0,) * max(0, 3 - len(halo))
    requested_input_halos = _normalize_input_halos(input_halos, len(state_inputs), ndim)
    kept_output_halos = _normalize_output_halos(output_halo, int(num_state_outputs), ndim)

    # ``shape_halo`` is the block-rounded padding of every local array handed
    # to the kernel; ``exchanged_halos`` are the widths actually received from
    # the neighbours, per input. With ``local_output`` each input is padded by
    # its own exchanged width (the kernel grid covers the shard only, so no
    # rounding is needed) and nothing is stripped.
    local_output = local_output or separate_halos
    if requested_input_halos is None:
        shape_halo = _round_halo_up_to_block(halo_3d[:ndim], block_shape_3d[:ndim])
        exchanged_halos = tuple(halo_3d[:ndim] if local_output else shape_halo for _ in state_inputs)
    else:
        largest_halo = tuple(
            max([int(halo_3d[axis])] + [input_halo[axis] for input_halo in requested_input_halos])
            for axis in range(ndim)
        )
        shape_halo = _round_halo_up_to_block(largest_halo, block_shape_3d[:ndim])
        exchanged_halos = requested_input_halos
    padded_halos = exchanged_halos if local_output else tuple(shape_halo for _ in state_inputs)
    if local_output:
        shape_halo = (0,) * ndim
        if output_halo is not None:
            raise ValueError("output_halo is not available with local_output.")

    for kept_halo in kept_output_halos:
        if any(kept > padded for kept, padded in zip(kept_halo, shape_halo)):
            raise ValueError("output_halo cannot exceed the padded shape halo.")

    def body(*all_args):
        """Pad the local shards, run the kernel build, strip the halo."""
        state_arrays = list(all_args[: len(state_inputs)])
        replicated_args = all_args[len(state_inputs):]
        shard_shape = tuple(int(extent) for extent in state_arrays[0].shape[1:])
        halos = [[None] * ndim for _ in state_arrays]

        for array_axis, mesh_axis_name, num_devices in sharded_axes:
            spatial_axis = array_axis - 1
            for input_index, array in enumerate(state_arrays):
                padded_width = padded_halos[input_index][spatial_axis]
                if padded_width <= 0:
                    continue
                if separate_halos:
                    halos[input_index][spatial_axis] = _exchanged_halos(
                        array, array_axis, padded_width, mesh_axis_name, num_devices, pairwise
                    )
                    continue
                state_arrays[input_index] = _pad_axis_with_halo(
                    array,
                    array_axis,
                    padded_width,
                    exchanged_halos[input_index][spatial_axis],
                    mesh_axis_name,
                    num_devices,
                    pairwise=pairwise,
                )

        shard = (shard_shape, tuple(map(tuple, halos)) if separate_halos else None) if local_output else None
        # Re-enter the wrapper with mesh=None so the recursive
        # ``kernel_build_fn`` call goes through the no-wrap path. Without
        # this, a kernel that calls ``_pallas_call_sharded`` from its body
        # would wrap itself forever.
        with pallas_mesh_context(None), _shard_context(shard):
            out = kernel_build_fn(*state_arrays, *replicated_args)

        def strip_halo(output, kept_halo):
            """Remove all but ``kept_halo`` of the padding from one output."""
            for array_axis, _, _ in sharded_axes:
                spatial_axis = array_axis - 1
                if spatial_axis >= len(shape_halo):
                    continue
                stripped_width = shape_halo[spatial_axis] - kept_halo[spatial_axis]
                if stripped_width <= 0:
                    continue
                size = output.shape[array_axis]
                output = jax.lax.slice_in_dim(
                    output,
                    stripped_width,
                    size - stripped_width,
                    axis=array_axis,
                )
            return output

        if isinstance(out, tuple):
            return tuple(
                strip_halo(output, kept_halo)
                for output, kept_halo in zip(out, kept_output_halos, strict=True)
            )
        return strip_halo(out, kept_output_halos[0])

    state_specs = tuple(pspec for _ in state_inputs)
    other_specs = tuple(PartitionSpec() for _ in other_args)
    if num_state_outputs > 1:
        out_specs = tuple(pspec for _ in range(num_state_outputs))
    else:
        out_specs = pspec

    wrapped = shard_map(
        body,
        mesh=mesh,
        in_specs=state_specs + other_specs,
        out_specs=out_specs,
        check_rep=False,
    )
    return wrapped(*state_inputs, *other_args)


def _mesh_split_axes(state) -> tuple:
    """
    ``(spatial axis, number of devices)`` of every spatial axis (0 = x) of ``state`` that the active Pallas mesh
    splits, in the order of the axes; empty without a mesh.
    """
    mesh = _current_pallas_mesh()
    if mesh is None or mesh.size <= 1:
        return ()
    sharded_axes = _spatial_sharded_axes(mesh, _resolve_state_pspec(state, mesh), state.ndim - 1)
    return tuple((array_axis - 1, num_devices) for array_axis, _, num_devices in sharded_axes)


def _padded_shards(arrays, width: int, sharded_axes, pairwise: bool) -> list:
    """
    The local shards ``arrays`` (variables first, one spatial shape) padded by ``width`` cells on both sides of
    every split axis with the cells of the neighbouring shards (periodic ring). All arrays share one exchange per
    split axis; the exchange along a later axis sends the padded edges of the earlier ones, so the corner cells
    arrive as well.

    Args:
        arrays: The local shards.
        width: The halo width per side.
        sharded_axes: ``(array axis, mesh axis name, number of devices)`` of every split axis.
        pairwise: Exchange over disjoint neighbour pairs (``_pairwise_halos``) instead of the ring.

    Returns:
        The padded shards.
    """
    num_rows = [int(array.shape[0]) for array in arrays]
    padding = [(0, 0)] * arrays[0].ndim
    for array_axis, _, _ in sharded_axes:
        padding[array_axis] = (width, width)
    padded = [jnp.pad(array, padding) for array in arrays]
    for array_axis, mesh_axis_name, num_devices in sharded_axes:
        size = padded[0].shape[array_axis]
        left_edges = jnp.concatenate(
            [jax.lax.slice_in_dim(array, width, 2 * width, axis=array_axis) for array in padded]
        )
        right_edges = jnp.concatenate(
            [jax.lax.slice_in_dim(array, size - 2 * width, size - width, axis=array_axis) for array in padded]
        )
        left_halos, right_halos = _received_halos(left_edges, right_edges, mesh_axis_name, num_devices, pairwise)
        first_row = 0
        for index, rows in enumerate(num_rows):
            for halos, start in ((left_halos, 0), (right_halos, size - width)):
                halo = jax.lax.slice_in_dim(halos, first_row, first_row + rows, axis=0)
                padded[index] = jax.lax.dynamic_update_slice_in_dim(padded[index], halo, start, axis=array_axis)
            first_row += rows
    return padded


def shard_map_on_padded_shards(body, state_inputs, other_args=(), *, width: int, num_outputs: int):
    """
    Run ``body`` on every shard of the active Pallas mesh with its state inputs padded by ``width`` cells on both
    sides of every split axis (``_padded_shards``: one halo exchange per split axis for all inputs together).

    ``body(*padded_inputs, *other_args, split_axes=..., global_maximum=...)`` runs with the mesh context cleared,
    so the kernels it calls build for the arrays they are given (the region a kernel writes is set with
    ``_shard_context``); ``split_axes`` are the split spatial axes (0 = x) and ``global_maximum(value)`` is the
    elementwise maximum of ``value`` over all shards. ``body`` returns ``num_outputs`` arrays with the spatial
    shape of the shard (variables first), which are assembled over the mesh like the first state input.

    Args:
        body: The computation on one padded shard.
        state_inputs: Arrays split like the state (variables first, one spatial shape).
        other_args: Replicated arguments (scalars or pytrees of them).
        width: The halo width per side.
        num_outputs: The number of outputs of ``body``.

    Returns:
        The tuple of the outputs of ``body``.
    """
    mesh = _current_pallas_mesh()
    pspec = _resolve_state_pspec(state_inputs[0], mesh)
    num_spatial_dims = state_inputs[0].ndim - 1
    sharded_axes = _spatial_sharded_axes(mesh, pspec, num_spatial_dims)
    pairwise = _pallas_halo_exchange_ctx.get() == PAIR_ALL_TO_ALL and all(
        num_devices % 2 == 0 for _, _, num_devices in sharded_axes
    )
    # String axis names for the grouped exchange and for the maximum (an integer name would be read as an array
    # axis by ``pmax``).
    mesh, pspec = _with_string_axis_names(mesh, pspec)
    sharded_axes = _spatial_sharded_axes(mesh, pspec, num_spatial_dims)
    split_axes = tuple(array_axis - 1 for array_axis, _, _ in sharded_axes)
    mesh_axis_names = tuple(mesh.axis_names)

    def global_maximum(value):
        return jax.lax.pmax(value, mesh_axis_names)

    def local_body(*arguments):
        padded = _padded_shards(arguments[: len(state_inputs)], width, sharded_axes, pairwise)
        with pallas_mesh_context(None):
            return tuple(
                body(*padded, *arguments[len(state_inputs) :], split_axes=split_axes, global_maximum=global_maximum)
            )

    return shard_map(
        local_body,
        mesh=mesh,
        in_specs=tuple(pspec for _ in state_inputs) + tuple(PartitionSpec() for _ in other_args),
        out_specs=tuple(pspec for _ in range(num_outputs)),
        check_rep=False,
    )(*state_inputs, *other_args)


# -------------------------------------------------------------
# =========== ↑ Multi-GPU shard_map + halo wrapper ↑ ==========
# -------------------------------------------------------------


# -------------------------------------------------------------
# == ↓ Differentiability: native-JAX tangents for Pallas ↓ ====
# -------------------------------------------------------------
#
# Pallas kernels in this codebase use ``input_output_aliases`` for memory
# efficiency. JAX cannot transpose an aliased ``pl.pallas_call`` ("JVP with
# aliasing not supported"), so any path that hits a Pallas kernel is
# non-differentiable by default. The gap is bridged with a ``jax.custom_jvp``
# whose primal still calls the (aliased, fast) Pallas branch and whose tangent
# rule delegates to the equivalent native-JAX branch, which is
# JVP-differentiable. Reverse mode (``jax.grad``) is then derived by JAX via
# transposition.
#
# Forward simulation performance is unaffected: outside of AD the custom_jvp
# rule is not invoked and the call collapses to the bare Pallas branch.
#
# Both branches must produce the same pytree-structured output. The Pallas
# kernels reproduce their native counterparts to round-off, so the gradient
# obtained by transposing the native JVP at the Pallas-evaluated inputs is the
# gradient of the Pallas operation.
#
# A hand-written Pallas adjoint can replace the native tangent branch of a
# single kernel without changing its call sites (``pallas_vjp_call``).


def diffable_pallas_call(state, params, *, pallas_branch, native_branch):
    """
    Run ``pallas_branch(state, params)`` behind a custom_jvp boundary that
    routes the tangent computation through ``native_branch``.

    Both branches must accept the same positional ``(state, params)`` pair
    and produce the same pytree structure. Anything static (config,
    registered_variables, axis index, ...) should be closed over.

    Outside of AD the call collapses to ``pallas_branch(state, params)``
    directly, with no overhead. Under ``jax.jvp`` / ``jax.jacfwd`` /
    ``jax.grad`` / ``jax.vjp`` / ``jax.jacrev`` the custom rule fires and
    the tangent goes through ``native_branch``.

    Args:
        state: The (differentiable) state argument.
        params: The (differentiable) parameter argument.
        pallas_branch: The Pallas implementation, ``(state, params) -> out``.
        native_branch: The native-JAX implementation with the same signature.

    Returns:
        ``pallas_branch(state, params)``.
    """

    @jax.custom_jvp
    def pallas_with_native_tangent(state_argument, params_argument):
        return pallas_branch(state_argument, params_argument)

    @pallas_with_native_tangent.defjvp
    def native_tangent(primals, tangents):
        primal_out = pallas_branch(*primals)
        _, tangent_out = jax.jvp(native_branch, primals, tangents)
        return primal_out, tangent_out

    return pallas_with_native_tangent(state, params)


def diffable_pallas_call_n(primals, *, pallas_branch, native_branch):
    """
    Same as :func:`diffable_pallas_call` for a tuple of arbitrary
    differentiable primals (callers with more than two differentiable
    arguments, e.g. extra right-hand-side / accumulator buffers).

    Args:
        primals: The tuple of differentiable arguments.
        pallas_branch: The Pallas implementation, ``(*primals) -> out``.
        native_branch: The native-JAX implementation with the same signature.

    Returns:
        ``pallas_branch(*primals)``.
    """

    @jax.custom_jvp
    def pallas_with_native_tangent(*arguments):
        return pallas_branch(*arguments)

    @pallas_with_native_tangent.defjvp
    def native_tangent(arguments, tangents):
        primal_out = pallas_branch(*arguments)
        _, tangent_out = jax.jvp(native_branch, arguments, tangents)
        return primal_out, tangent_out

    return pallas_with_native_tangent(*primals)


def pallas_vjp_call(state, aux, *, pallas_forward, pallas_backward):
    """
    Run ``pallas_forward(state, aux)`` behind a ``jax.custom_vjp`` boundary
    whose reverse rule is a hand-written Pallas adjoint kernel
    ``pallas_backward``.

    Unlike :func:`diffable_pallas_call` (which routes the tangent, and hence
    the transposed gradient, through native JAX), this keeps the entire
    backward pass on the Pallas/GPU backend: ``pallas_backward(state, aux,
    cotangent)`` returns the input cotangent ``d(loss)/d(state)`` directly.

    Differentiates with respect to ``state`` only. ``aux`` (e.g. the traced
    ``SimulationParams``) is threaded through the boundary and given a zero
    cotangent; it must be passed explicitly rather than closed over because
    ``jax.custom_vjp`` cannot capture traced values in its forward/backward
    closures (only static data such as config or axis may be closed over by
    the two branches). Treating the physical constants as non-differentiable
    matches the inverse-problem regime (gradients with respect to the state,
    not the parameters).

    NOTE: ``jax.custom_vjp`` supports reverse mode only; ``jax.jvp`` /
    forward-mode AD on this boundary raises. For forward mode use
    :func:`diffable_pallas_call`.

    Args:
        state: The differentiable state argument.
        aux: Non-differentiable traced data (e.g. the simulation parameters).
        pallas_forward: The Pallas forward kernel, ``(state, aux) -> out``.
        pallas_backward: The Pallas adjoint kernel,
            ``(state, aux, cotangent) -> state_cotangent``.

    Returns:
        ``pallas_forward(state, aux)``.
    """

    @jax.custom_vjp
    def pallas_with_pallas_adjoint(state_argument, aux_argument):
        return pallas_forward(state_argument, aux_argument)

    def forward_with_residuals(state_argument, aux_argument):
        return pallas_forward(state_argument, aux_argument), (state_argument, aux_argument)

    def backward(residuals, cotangent):
        state_argument, aux_argument = residuals
        state_cotangent = pallas_backward(state_argument, aux_argument, cotangent)

        def zero_cotangent(leaf):
            """Correctly typed zero cotangent (float0 for non-inexact leaves)."""
            leaf = jnp.asarray(leaf)
            if jnp.issubdtype(leaf.dtype, jnp.inexact):
                return jnp.zeros_like(leaf)
            return jnp.zeros(leaf.shape, dtype=jax.dtypes.float0)

        return (state_cotangent, jax.tree_util.tree_map(zero_cotangent, aux_argument))

    pallas_with_pallas_adjoint.defvjp(forward_with_residuals, backward)
    return pallas_with_pallas_adjoint(state, aux)


# -------------------------------------------------------------
# == ↑ Differentiability: native-JAX tangents for Pallas ↑ ====
# -------------------------------------------------------------
