"""
The discrete Fourier transform of 3D fields split over a device mesh, without gathering the cube.

A field of the state is split along x over the mesh axis ``XAXIS`` and along y over ``YAXIS`` (the decompositions
(px, py, 1) and (px, 1, 1) of the state mesh), so every device holds a block (Nx / px, Ny / py, Nz). The forward
transform transforms the axes a device holds whole, then exchanges the blocks within a row of the mesh
(``all_to_all``) so that the next axis becomes whole:

    (Nx/px, Ny/py, Nz)  -- fft z, all_to_all over y -->  (Nx/px, Ny, Nz/py)  -- fft y, all_to_all over x -->
    (Nx, Ny/px, Nz/py)  -- fft x

and leaves the spectrum in this spectral layout: kx whole, ky split over the mesh axis x, kz split over the mesh axis
y. On a (px, 1, 1) mesh y and z are transformed together and only the exchange over x is needed. The inverse runs the
same steps backwards. Every device keeps a few blocks of its own size; nothing of the size of the cube is formed.

``fft_block`` and ``ifft_block`` transform the local block inside a ``shard_map`` over :func:`spectral_mesh`; the
periodic Poisson solver of self-gravity and the shell spectra of ``analysis_helpers.sharded_spectra`` are built on
them. ``distributed_fft`` and ``distributed_ifft`` are the same transforms for a whole field.
"""

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec

try:  # the stable alias (newer jax)
    from jax import shard_map as _shard_map
except ImportError:  # pragma: no cover - older jax
    from jax.experimental.shard_map import shard_map as _shard_map

#: The names of the axes of the state mesh (variables, x, y, z) inside the transforms. The state mesh uses integer
#: axis names, which the collectives would read as array axes.
MESH_AXIS_NAMES = ("vars", "x", "y", "z")
X_NAME, Y_NAME = MESH_AXIS_NAMES[1], MESH_AXIS_NAMES[2]

#: A field in the layout of the state: x split over the mesh axis x, y over the mesh axis y.
FIELD_SPEC = PartitionSpec(X_NAME, Y_NAME, None)

#: A spectrum in the spectral layout: kx whole, ky split over the mesh axis x, kz over the mesh axis y.
SPECTRUM_SPEC = PartitionSpec(None, X_NAME, Y_NAME)


def spectral_mesh(mesh):
    """
    The state mesh under the axis names of the transforms (the same devices in the same order).

    Args:
        mesh: The state mesh (variables, x, y, z) of shape (1, px, py, 1), or ``None`` for one device.

    Returns:
        The mesh with the axis names ``MESH_AXIS_NAMES``.
    """
    if mesh is None:
        return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1), MESH_AXIS_NAMES)
    devices = np.asarray(mesh.devices)
    if devices.ndim != 4 or devices.shape[0] != 1 or devices.shape[3] != 1:
        raise ValueError(f"the distributed FFT needs a state mesh of shape (1, px, py, 1), got {devices.shape}")
    return Mesh(devices, MESH_AXIS_NAMES)


def check_decomposition(shape, mesh):
    """
    Raise if a field of this shape does not split into equal blocks in both layouts of the transform.

    Args:
        shape: The global shape (Nx, Ny, Nz).
        mesh: A mesh from :func:`spectral_mesh`.
    """
    num_x, num_y = mesh.shape[X_NAME], mesh.shape[Y_NAME]
    num_cells_x, num_cells_y, num_cells_z = shape
    if num_cells_x % num_x or num_cells_y % num_y or num_cells_y % num_x or num_cells_z % num_y:
        raise ValueError(
            f"the distributed FFT of a {shape} field on a ({num_x}, {num_y}, 1) mesh needs Nx and Ny divisible by "
            f"px, Ny and Nz divisible by py"
        )


def fft_block(block):
    """
    The forward transform of the local block of a field, inside a ``shard_map`` over :func:`spectral_mesh`.

    Args:
        block: The local block (Nx / px, Ny / py, Nz) of a field in the layout ``FIELD_SPEC``.

    Returns:
        The local block (Nx, Ny / px, Nz / py) of its discrete Fourier transform in the layout ``SPECTRUM_SPEC``.
    """
    if jax.lax.axis_size(Y_NAME) == 1:
        spectrum = jnp.fft.fftn(block, axes=(1, 2))
    else:
        spectrum = jnp.fft.fft(block, axis=2)
        spectrum = jax.lax.all_to_all(spectrum, Y_NAME, split_axis=2, concat_axis=1, tiled=True)
        spectrum = jnp.fft.fft(spectrum, axis=1)
    spectrum = jax.lax.all_to_all(spectrum, X_NAME, split_axis=1, concat_axis=0, tiled=True)
    return jnp.fft.fft(spectrum, axis=0)


def ifft_block(spectrum):
    """
    The inverse transform of the local block of a spectrum, inside a ``shard_map`` over :func:`spectral_mesh`.

    Args:
        spectrum: The local block (Nx, Ny / px, Nz / py) of a spectrum in the layout ``SPECTRUM_SPEC``.

    Returns:
        The real part of the local block (Nx / px, Ny / py, Nz) of the inverse transform, layout ``FIELD_SPEC``.
    """
    field = jnp.fft.ifft(spectrum, axis=0)
    field = jax.lax.all_to_all(field, X_NAME, split_axis=0, concat_axis=1, tiled=True)
    if jax.lax.axis_size(Y_NAME) == 1:
        return jnp.real(jnp.fft.ifftn(field, axes=(1, 2)))
    field = jnp.fft.ifft(field, axis=1)
    field = jax.lax.all_to_all(field, Y_NAME, split_axis=1, concat_axis=2, tiled=True)
    return jnp.real(jnp.fft.ifft(field, axis=2))


def local_part(values, axis_name):
    """
    The part of a 1D array along a split axis that this device holds, inside a ``shard_map``.

    Args:
        values: The values along the whole axis (e.g. the wavenumbers of ky).
        axis_name: The mesh axis the array axis is split over (``X_NAME`` for ky, ``Y_NAME`` for kz).

    Returns:
        The device's contiguous part.
    """
    size = values.shape[0] // jax.lax.axis_size(axis_name)
    return jax.lax.dynamic_slice(values, (jax.lax.axis_index(axis_name) * size,), (size,))


def distributed_fft(field, mesh=None):
    """
    The discrete Fourier transform of a real 3D field on a device mesh: ``jnp.fft.fftn(field)`` up to rounding.

    Args:
        field: The field (Nx, Ny, Nz), split like a field of the state (x over ``XAXIS``, y over ``YAXIS``).
        mesh: The state mesh, shape (1, px, py, 1); ``None`` for one device.

    Returns:
        The complex spectrum (Nx, Ny, Nz) in the spectral layout ``SPECTRUM_SPEC`` of ``spectral_mesh(mesh)``.
    """
    mesh = spectral_mesh(mesh)
    check_decomposition(field.shape, mesh)
    return _shard_map(fft_block, mesh=mesh, in_specs=FIELD_SPEC, out_specs=SPECTRUM_SPEC)(field)


def distributed_ifft(spectrum, mesh=None):
    """
    The inverse of :func:`distributed_fft`: ``jnp.real(jnp.fft.ifftn(spectrum))`` up to rounding.

    Args:
        spectrum: The spectrum (Nx, Ny, Nz) in the spectral layout.
        mesh: The state mesh, shape (1, px, py, 1); ``None`` for one device.

    Returns:
        The real field (Nx, Ny, Nz) in the layout of the state.
    """
    mesh = spectral_mesh(mesh)
    check_decomposition(spectrum.shape, mesh)
    return _shard_map(ifft_block, mesh=mesh, in_specs=SPECTRUM_SPEC, out_specs=FIELD_SPEC)(spectrum)
