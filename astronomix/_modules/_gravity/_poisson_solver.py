"""
FFT-based Poisson solver for the gravitational potential.

Solves Poisson's equation for the gravitational potential in Fourier space.
Fully periodic domains use the spectral Green's function directly (with the
Jeans swindle subtracting the mean density). Non-periodic (open) boundaries use
the Hockney & Eastwood method, zero-padding the domain to twice its size and
convolving with the isolated-system Green's function.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig


# @jaxtyped(typechecker=typechecker)
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

    Args:
        gas_density: The gas density field.
        grid_spacing: The grid spacing.
        config: The simulation configuration.
        G: The gravitational constant.

    Returns:
        The gravitational potential.

    """

    # TODO: remove ghost cells in this computation, which currently treats the
    # padded field directly.

    dimensionality = config.dimensionality

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

    # The Jeans swindle: only a meaningful periodic solution exists once the
    # (unphysical) mean density is removed, so subtract it for periodic domains.
    if not non_periodic_boundaries:
        gas_density = gas_density - jnp.mean(gas_density)

    if not non_periodic_boundaries:
        # -------------------------------------------------------------
        # ============= ↓ Periodic boundaries version ↓ ==============
        # -------------------------------------------------------------

        # Transform the density to Fourier space.
        density_k = fftn(gas_density)

        # Build the squared wavenumber magnitude for the active dimensionality.
        if dimensionality == 1:
            num_cells_x = gas_density.shape[0]
            k_base_x = jnp.fft.fftfreq(num_cells_x, d=grid_spacing) * 2 * jnp.pi
            k = k_base_x
            k_squared = k**2
        elif dimensionality == 2:
            num_cells_x, num_cells_y = gas_density.shape
            k_base_x = jnp.fft.fftfreq(num_cells_x, d=grid_spacing) * 2 * jnp.pi
            k_base_y = jnp.fft.fftfreq(num_cells_y, d=grid_spacing) * 2 * jnp.pi
            kx, ky = jnp.meshgrid(k_base_x, k_base_y, indexing="ij")
            k_squared = kx**2 + ky**2
        elif dimensionality == 3:
            num_cells_x, num_cells_y, num_cells_z = gas_density.shape
            k_base_x = jnp.fft.fftfreq(num_cells_x, d=grid_spacing) * 2 * jnp.pi
            k_base_y = jnp.fft.fftfreq(num_cells_y, d=grid_spacing) * 2 * jnp.pi
            k_base_z = jnp.fft.fftfreq(num_cells_z, d=grid_spacing) * 2 * jnp.pi
            kx, ky, kz = jnp.meshgrid(k_base_x, k_base_y, k_base_z, indexing="ij")
            k_squared = kx**2 + ky**2 + kz**2

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
