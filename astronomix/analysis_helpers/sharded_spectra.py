"""
Shell-binned spectra of 3D fields split over a device mesh, computed without gathering the cube.

Every device transforms its block with the distributed FFT (``astronomix.parallel.distributed_fft``), bins the modes
of its spectral block into shells and adds its shell sums to those of the other devices; only the 1D shell arrays
leave the devices. Two binnings, each equal to a single-device diagnostic of the project:

* ``INTEGER_SHELLS``: ``n = rint(|m|)`` for ``n = 0 .. N/2`` (modes beyond are dropped), with ``m`` the integer mode
  vector, the shells of the dynamo study's estimator (``examples/scripts/forward/mhd/turbulence/_mhd_spectral.py``);
* ``ANGULAR_SHELLS``: unit-width bins of the angular wavenumber ``2 pi |m|`` of a unit box, the shells of
  ``energy_spectrum.PHYSICAL_BINNING``.

On a grid with unequal cell counts the mode numbers count the fundamental of the longest axis.

The ``local_*`` functions and ``shell_sum`` work on the local blocks inside a ``shard_map`` over
``spectral_mesh(mesh)``, so that a reduction of many spectra shares one pass (the estimator's reduction is built
from them); :func:`sharded_energy_spectrum` is the energy spectrum of a whole field.
"""

# general
import math

# jax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

try:  # the stable alias (newer jax)
    from jax import shard_map as _shard_map
except ImportError:  # pragma: no cover - older jax
    from jax.experimental.shard_map import shard_map as _shard_map

# astronomix functions
from astronomix.parallel.distributed_fft import (
    FIELD_SPEC,
    X_NAME,
    Y_NAME,
    check_decomposition,
    fft_block,
    local_part,
    spectral_mesh,
)

INTEGER_SHELLS = 0
ANGULAR_SHELLS = 1


def local_axes(values_x, values_y, values_z):
    """
    The parts of three per-axis arrays that belong to the local spectral block, shaped to broadcast against it.

    Args:
        values_x, values_y, values_z: The values along the whole x, y and z axis of the spectrum.

    Returns:
        ``(x, y, z)`` of shapes ``(Nx, 1, 1)``, ``(1, Ny / px, 1)`` and ``(1, 1, Nz / py)``.
    """
    return (
        values_x[:, None, None],
        local_part(values_y, X_NAME)[None, :, None],
        local_part(values_z, Y_NAME)[None, None, :],
    )


def local_wavevectors(num_cells, grid_spacing):
    """The angular wavevector components ``2 pi fftfreq(N_i, dx)`` of the local spectral block."""
    return local_axes(*[2.0 * jnp.pi * jnp.fft.fftfreq(cells, d=grid_spacing) for cells in num_cells])


def number_of_shells(num_cells, binning):
    """``N/2 + 1`` integer shells, or the angular bins up to the corner mode of the grid (plus one)."""
    longest = max(num_cells)
    if binning == INTEGER_SHELLS:
        return longest // 2 + 1
    largest_mode = math.sqrt(sum((cells // 2 * longest / cells) ** 2 for cells in num_cells))
    return int(2.0 * math.pi * largest_mode) + 2


def local_shells(num_cells, binning):
    """
    The shell of every mode of the local spectral block.

    Args:
        num_cells: The global shape (Nx, Ny, Nz).
        binning: ``INTEGER_SHELLS`` or ``ANGULAR_SHELLS``.

    Returns:
        ``(shells, num_shells)``: the int32 shell indices of the block and the number of shells.
    """
    longest = max(num_cells)
    mode_x, mode_y, mode_z = local_axes(*[jnp.fft.fftfreq(cells, d=1.0 / longest) for cells in num_cells])
    magnitude = jnp.sqrt(mode_x**2 + mode_y**2 + mode_z**2)
    if binning == INTEGER_SHELLS:
        shells = jnp.rint(magnitude).astype(jnp.int32)
    else:
        shells = (2.0 * jnp.pi * magnitude).astype(jnp.int32)
    return shells, number_of_shells(num_cells, binning)


def shell_centres(num_cells, binning):
    """The shell centres: ``n = 0 .. N/2`` (``INTEGER_SHELLS``) or angular wavenumbers ``0.5, 1.5, ..``."""
    centres = jnp.arange(number_of_shells(num_cells, binning))
    return centres if binning == INTEGER_SHELLS else centres + 0.5


def shell_sum(values, shells, num_shells):
    """
    The sum of a per-mode quantity over every shell and over the whole mesh, inside a ``shard_map``.

    Args:
        values: The per-mode quantity on the local spectral block.
        shells: Its shells (:func:`local_shells`); shells from ``num_shells`` on are dropped.
        num_shells: The number of shells.

    Returns:
        The shell sums, the same on every device.
    """
    local = jnp.zeros(num_shells, values.dtype).at[shells].add(values, mode="drop")
    return jax.lax.psum(local, (X_NAME, Y_NAME))


def squared_magnitude(spectrum):
    """``|f|^2`` of every mode of a spectrum."""
    return jnp.real(spectrum) ** 2 + jnp.imag(spectrum) ** 2


def sharded_energy_spectrum(fx, fy, fz, mesh=None, energy_coeff=0.5, binning=ANGULAR_SHELLS):
    """
    The shell-summed energy spectrum ``c |f|^2`` of a 3D vector field on a device mesh, normalised so that the sum
    over the shells is ``mean(c |f|^2)`` (the convention of ``energy_spectrum.vector_field_energy_spectrum``).

    The kinetic energy spectrum is that of ``sqrt(rho) v`` with ``c = 1/2``, the magnetic one that of ``B``.

    Args:
        fx, fy, fz: The components (Nx, Ny, Nz), split like fields of the state.
        mesh: The state mesh, shape (1, px, py, 1); ``None`` for one device.
        energy_coeff: The coefficient ``c``.
        binning: ``ANGULAR_SHELLS`` (``energy_spectrum.PHYSICAL_BINNING``) or ``INTEGER_SHELLS``.

    Returns:
        ``(shell_centres, E)``.
    """
    num_cells = fx.shape
    mesh = spectral_mesh(mesh)
    check_decomposition(num_cells, mesh)
    normalisation = float(math.prod(num_cells)) ** 2

    def spectrum_block(*components):
        shells, num_shells = local_shells(num_cells, binning)
        energy = energy_coeff * sum(squared_magnitude(fft_block(component)) for component in components)
        return shell_sum(energy / normalisation, shells, num_shells)

    energy = _shard_map(spectrum_block, mesh=mesh, in_specs=(FIELD_SPEC,) * 3, out_specs=PartitionSpec())(fx, fy, fz)
    return shell_centres(num_cells, binning), energy
