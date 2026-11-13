"""
FFT-based Poisson solver for the gravitational potential.

Solves Poisson's equation for the gravitational potential in Fourier space.
Fully periodic domains use the spectral Green's function directly (with the
Jeans swindle subtracting the mean density), or optionally
(``GravityConfig.poisson_discrete_laplacian``) the Green's function of the
second-order discrete Laplacian, as Athena++ does. Non-periodic (open) boundaries use
the Hockney & Eastwood method, zero-padding the domain to twice its size and
convolving with the isolated-system Green's function.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn
from jax.sharding import PartitionSpec

try:  # the stable alias (newer jax)
    from jax import shard_map as _shard_map
except ImportError:  # pragma: no cover - older jax
    from jax.experimental.shard_map import shard_map as _shard_map

# astronomix helpers
from astronomix._pallas_helpers import _current_pallas_mesh

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    GHOST_CELLS,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig


def _active_x_sharded_mesh():
    """The active ``(mesh, x_axis_name)`` when tracing under a multi-device
    1D X-decomposed state mesh, else ``None``.

    ``time_integration`` sets the mesh contextvar around the JIT trace
    whenever the caller supplies a sharding, with the standard
    ``(VARAXIS, XAXIS, YAXIS, ZAXIS)`` axis layout.  Only a pure-X
    decomposition is supported by the slab-transposed FFT below; anything
    else falls back to the replicated ``fftn`` path.
    """
    mesh = _current_pallas_mesh()
    if mesh is None:
        return None
    names = tuple(mesh.axis_names)
    if len(names) < 4:
        return None
    x_name = names[1]
    if mesh.shape[x_name] <= 1:
        return None
    other_sharded = mesh.shape[names[0]] > 1 or any(
        mesh.shape[name] > 1 for name in names[2:]
    )
    if other_sharded:
        return None
    return mesh, x_name


def _negative_laplacian_symbol(
    wavenumbers,
    grid_spacing: float,
    discrete_laplacian: bool,
):
    """
    The Fourier symbol of minus the Laplacian along one axis: ``k^2`` for the
    spectral operator, or ``(2 - 2 cos(k dx)) / dx^2`` for the second-order
    (three-point) discrete Laplacian, as in Athena++'s FFT gravity
    (``FFTGravity::ApplyKernel``, mode 0).

    Args:
        wavenumbers: The angular wavenumbers along the axis.
        grid_spacing: The grid spacing.
        discrete_laplacian: Whether to use the discrete Laplacian.

    Returns:
        The symbol, non-negative and zero only for ``k = 0``.
    """
    if discrete_laplacian:
        return (2.0 - 2.0 * jnp.cos(wavenumbers * grid_spacing)) / grid_spacing**2
    return wavenumbers**2


def _periodic_poisson_3d_sharded(
    gas_density,
    grid_spacing: float,
    G,
    mesh,
    x_name,
    discrete_laplacian: bool = False,
):
    """Distributed periodic Poisson solve for an X-sharded 3D density.

    The SPMD partitioner replicates the operand of an FFT along its
    transform dimensions, so the plain ``fftn`` path costs *global*-grid
    memory and work per device.  This version keeps everything slab-local:
    FFT over (y, z) batched over the local X slab, an ``all_to_all``
    transpose of the slab decomposition from X to Y, the X-axis FFT, the
    Green's-function multiply on the local k-slab, and the mirrored
    inverse.  Per-device memory and FLOPs then stay proportional to the
    local shard, which is what makes self-gravity weak scaling flat.

    Mathematically identical to the replicated branch (same discrete
    transform, Green's function and regularisation); only the data layout
    differs.
    """
    num_cells_x, num_cells_y, num_cells_z = gas_density.shape
    num_slabs = mesh.shape[x_name]
    local_cells_y = num_cells_y // num_slabs

    # The state mesh uses *integer* axis names, which JAX's named
    # collectives (all_to_all, axis_index) would misread as positional
    # array axes.  Alias the same devices (same order, so no data movement)
    # under string names for the shard_map body.
    named_mesh = jax.sharding.Mesh(mesh.devices, ("vars", "x", "y", "z"))

    k_base_x = jnp.fft.fftfreq(num_cells_x, d=grid_spacing) * 2 * jnp.pi
    k_base_y = jnp.fft.fftfreq(num_cells_y, d=grid_spacing) * 2 * jnp.pi
    k_base_z = jnp.fft.fftfreq(num_cells_z, d=grid_spacing) * 2 * jnp.pi

    def _local_solve(density_block):
        # density_block: the local X slab, (num_cells_x / P, ny, nz).
        density_k = jnp.fft.fftn(density_block, axes=(1, 2))
        # Transpose the decomposition X -> Y (device p ends up with full X
        # and the p-th Y slab), then transform the now-local X axis.
        density_k = jax.lax.all_to_all(
            density_k, "x", split_axis=1, concat_axis=0, tiled=True
        )
        density_k = jnp.fft.fft(density_k, axis=0)

        # Green's function on this device's k-slab (same regularisation as
        # the replicated branch: the k = 0 mode maps to -1/(4 pi), harmless
        # because the mean density was already subtracted).
        slab_index = jax.lax.axis_index("x")
        k_local_y = jax.lax.dynamic_slice(
            k_base_y, (slab_index * local_cells_y,), (local_cells_y,)
        )
        k_squared = (
            _negative_laplacian_symbol(k_base_x, grid_spacing, discrete_laplacian).reshape(-1, 1, 1)
            + _negative_laplacian_symbol(k_local_y, grid_spacing, discrete_laplacian).reshape(1, -1, 1)
            + _negative_laplacian_symbol(k_base_z, grid_spacing, discrete_laplacian).reshape(1, 1, -1)
        )
        k_squared = jnp.where(k_squared == 0, 1e-12, k_squared)
        greens_function = jnp.where(
            k_squared > 1e-12, -4 * jnp.pi * G / k_squared, -1 / (4 * jnp.pi)
        )
        potential_k = greens_function * density_k

        # Mirrored inverse: X-axis inverse FFT, transpose Y -> X back, then
        # the (y, z) inverse on the local X slab.
        potential_k = jnp.fft.ifft(potential_k, axis=0)
        potential_k = jax.lax.all_to_all(
            potential_k, "x", split_axis=0, concat_axis=1, tiled=True
        )
        potential = jnp.fft.ifftn(potential_k, axes=(1, 2))
        return jnp.real(potential)

    slab_spec = PartitionSpec("x", None, None)
    return _shard_map(
        _local_solve,
        mesh=named_mesh,
        in_specs=(slab_spec,),
        out_specs=slab_spec,
    )(gas_density)


@partial(jax.jit, static_argnames=["grid_spacing", "config"])
def _compute_gravitational_potential(
    gas_density: FIELD_TYPE,
    grid_spacing: float,
    config: SimulationConfig,
    G: Union[float, Float[Array, ""]] = 1.0,
) -> FIELD_TYPE:
    """
    Compute the gravitational potential using FFT to solve Poisson's equation for
    periodic and open boundaries (via the Hockney & Eastwood method).

    A field with ghost cells is solved for the mass of the domain alone: a
    periodic domain on its interior, the ghost cells wrapped around; an open
    domain with the ghost density set to zero, so that the ghost cells receive
    the potential of the domain's mass.

    Args:
        gas_density: The gas density field (with or without ghost cells).
        grid_spacing: The grid spacing.
        config: The simulation configuration.
        G: The gravitational constant.

    Returns:
        The gravitational potential, shaped like the density.

    """

    dimensionality = config.dimensionality
    discrete_laplacian = config.gravity_config.poisson_discrete_laplacian

    # The open-boundary branch is only taken when *no* boundary is periodic; a
    # mix of periodic and non-periodic boundaries is not yet supported, so the
    # domain is treated as either fully periodic or fully open.
    # TODO: support mixed boundary conditions.
    non_periodic_boundaries = False

    if dimensionality == 1:
        if not (
            config.boundary_settings.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.right_boundary == PERIODIC_BOUNDARY
        ):
            non_periodic_boundaries = True
    elif dimensionality == 2:
        if not (
            config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY
        ):
            non_periodic_boundaries = True
    elif dimensionality == 3:
        if not (
            config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.z.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.z.right_boundary == PERIODIC_BOUNDARY
        ):
            non_periodic_boundaries = True

    if config.gravity_config.poisson_manual_open_boundaries:
        non_periodic_boundaries = True

    interior_shape = tuple(config.num_cells)[:dimensionality]
    if config.boundary_handling == GHOST_CELLS and gas_density.shape != interior_shape:
        num_ghost_cells = config.num_ghost_cells
        interior = (slice(num_ghost_cells, -num_ghost_cells),) * dimensionality
        if not non_periodic_boundaries:
            potential = _compute_gravitational_potential(gas_density[interior], grid_spacing, config, G)
            return jnp.pad(potential, num_ghost_cells, mode="wrap")
        gas_density = jnp.zeros_like(gas_density).at[interior].set(gas_density[interior])

    # The Jeans swindle: only a meaningful periodic solution exists once the
    # (unphysical) mean density is removed, so subtract it for periodic domains.
    if not non_periodic_boundaries:
        gas_density = gas_density - jnp.mean(gas_density)

    if not non_periodic_boundaries:
        # -------------------------------------------------------------
        # ============= ↓ Periodic boundaries version ↓ ==============
        # -------------------------------------------------------------

        # Under a multi-device X-decomposed mesh, route the 3D solve
        # through the slab-transposed distributed FFT -- the plain fftn
        # below would be replicated per device by the SPMD partitioner
        # (global-grid memory and work on every GPU).  Requires the slab
        # counts to divide the decomposition; otherwise fall through.
        if dimensionality == 3:
            active = _active_x_sharded_mesh()
            if active is not None:
                mesh, x_name = active
                num_slabs = mesh.shape[x_name]
                divisible = (
                    gas_density.shape[0] % num_slabs == 0
                    and gas_density.shape[1] % num_slabs == 0
                )
                if divisible:
                    return _periodic_poisson_3d_sharded(
                        gas_density,
                        grid_spacing,
                        G,
                        mesh,
                        x_name,
                        discrete_laplacian,
                    )

        # Transform the density to Fourier space.
        density_k = fftn(gas_density)

        # The symbol of minus the Laplacian, sum_d k_d^2 (spectral) or its
        # discrete-Laplacian counterpart, broadcast over the grid. It is
        # called k_squared in both cases.
        k_squared = jnp.zeros(gas_density.shape, dtype=gas_density.real.dtype)
        for axis, num_cells in enumerate(gas_density.shape):
            axis_wavenumbers = jnp.fft.fftfreq(num_cells, d=grid_spacing) * 2 * jnp.pi
            broadcast_shape = [1] * dimensionality
            broadcast_shape[axis] = num_cells
            k_squared = k_squared + _negative_laplacian_symbol(
                axis_wavenumbers,
                grid_spacing,
                discrete_laplacian,
            ).reshape(broadcast_shape)

        # Regularise the k = 0 mode to avoid dividing by zero; its Green's-
        # function value is the finite constant -1 / (4 pi). This is harmless
        # because the Jeans swindle above already removed the mean density, so
        # density_k[0] ~ 0.
        k_squared = jnp.where(k_squared == 0, 1e-12, k_squared)
        greens_function = jnp.where(
            k_squared > 1e-12, -4 * jnp.pi * G / k_squared, -1 / (4 * jnp.pi)
        )

        # Apply the Green's function in Fourier space and transform back.
        potential_k = greens_function * density_k
        gravitational_potential = jnp.real(ifftn(potential_k))

        return gravitational_potential

    else:
        # -------------------------------------------------------------
        # ====== ↓ Open boundaries (Hockney & Eastwood) version ↓ ====
        # -------------------------------------------------------------
        # TODO: check that this works for differing cell counts per dimension.
        #
        # (a) Extend the domain to twice the size in each dimension and embed
        #     the original density in the (0, ..., 0) corner; the zero padding
        #     is what makes the periodic FFT convolution behave as an isolated
        #     (open-boundary) one.
        original_shape = gas_density.shape
        extended_shape = tuple(2 * s for s in original_shape)

        extended_density = jnp.zeros(extended_shape, dtype=gas_density.dtype)
        slices = tuple(slice(0, s) for s in original_shape)
        extended_density = extended_density.at[slices].set(gas_density)

        # (b) Construct the Green's function on the extended grid.
        #
        # The Hockney-Eastwood prescription computes, for each dimension,
        #     pos = [0, 1, 2, ..., n-1, 2n - n, ..., 1] * grid_spacing,
        # i.e. pos = arange(2*n); pos = where(pos < n, pos, 2*n - pos), which
        # yields the minimum-image distances from a source placed at the origin.
        grids = []
        for s in original_shape:
            n = s
            extended_n = 2 * n
            pos = jnp.arange(extended_n)
            pos = jnp.where(pos < n, pos, 2 * n - pos)
            pos = pos * grid_spacing
            grids.append(pos)

        # Build the radial distance array r on the extended grid.
        if dimensionality == 1:
            r = grids[0]  # already nonnegative
        elif dimensionality == 2:
            x, y = jnp.meshgrid(grids[0], grids[1], indexing="ij")
            r = jnp.sqrt(x**2 + y**2)
        elif dimensionality == 3:
            x, y, z = jnp.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
            r = jnp.sqrt(x**2 + y**2 + z**2)

        # Replace any zero distance with grid_spacing to avoid the singularity
        # at the origin.
        r_safe = jnp.where(r == 0, grid_spacing, r)

        # (c) Isolated (open-boundary) Green's function for each dimensionality.
        if dimensionality == 1:
            # 1D: solving phi'' = 4 pi G delta(x) gives phi = -2 pi G |x|.
            kernel = -2 * jnp.pi * G * r
        elif dimensionality == 2:
            # 2D: phi = -2 G log(r) (up to an additive constant).
            kernel = -2 * G * jnp.log(r_safe)
        elif dimensionality == 3:
            # 3D: the isolated potential is phi = -G / r.
            kernel = -G / r_safe

        # (d) FFT-convolve the extended density with the Green's function.
        density_k_ext = fftn(extended_density)
        kernel_k_ext = fftn(kernel)
        potential_ext = jnp.real(ifftn(density_k_ext * kernel_k_ext))

        # (e) Extract the portion of the potential covering the original grid;
        #     the grid_spacing**dim factor accounts for the discrete convolution
        #     measure.
        gravitational_potential = potential_ext[slices]
        return gravitational_potential * grid_spacing**dimensionality
