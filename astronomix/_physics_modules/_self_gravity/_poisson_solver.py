# general
from functools import partial
import jax
import jax.numpy as jnp

# typing
from beartype import beartype as typechecker
from typing import Union
from jaxtyping import Array, Float, jaxtyped

# jf1uid constants
from astronomix.option_classes.simulation_config import FIELD_TYPE, PERIODIC_BOUNDARY

# astronomix classes
from astronomix.option_classes.simulation_config import SimulationConfig

# fft
from jax.numpy.fft import fftn, ifftn


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

    # TODO: remove ghost cells in this computations (?)

    dimensionality = config.dimensionality
    num_cells = gas_density.shape[0]

    # we only use outflow if not all boundaries are periodic
    # SO THERES EITHER ALL PERIODIC OR NONE
    # TODO: improve

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

    # if periodic boundaries
    if not non_periodic_boundaries:
        # jeans swindle (?)
        gas_density = gas_density - jnp.mean(gas_density)

    if not non_periodic_boundaries:
        # -----------------------------
        # Periodic boundaries version
        # -----------------------------
        # Compute FFT of the density field.
        density_k = fftn(gas_density)

        # Build the k–vector (note that fftfreq returns cycles/unit; multiply by 2π).
        k_base = jnp.fft.fftfreq(num_cells, d=grid_spacing) * 2 * jnp.pi

        if dimensionality == 1:
            k = k_base  # 1D case.
            k_squared = k**2
        elif dimensionality == 2:
            kx, ky = jnp.meshgrid(k_base, k_base, indexing="ij")
            k_squared = kx**2 + ky**2
        elif dimensionality == 3:
            kx, ky, kz = jnp.meshgrid(k_base, k_base, k_base, indexing="ij")
            k_squared = kx**2 + ky**2 + kz**2

        # Avoid division by zero (k = 0 mode).
        k_squared = jnp.where(k_squared == 0, 1e-12, k_squared)
        greens_function = jnp.where(
            k_squared > 1e-12, -4 * jnp.pi * G / k_squared, -1 / (4 * jnp.pi)
        )

        # Multiply in Fourier space and invert.
        potential_k = greens_function * density_k
        gravitational_potential = jnp.real(ifftn(potential_k))

        return gravitational_potential

    else:
        # ----------------------------------------------------
        # Open boundaries version via Hockney & Eastwood method
        # ----------------------------------------------------
        #
        # (a) Extend the domain to twice the size in each dimension.
        original_shape = gas_density.shape
        extended_shape = tuple(2 * s for s in original_shape)

        # Embed the original density in the (0,...,0) corner of the extended grid.
        extended_density = jnp.zeros(extended_shape, dtype=gas_density.dtype)
        slices = tuple(slice(0, s) for s in original_shape)
        extended_density = extended_density.at[slices].set(gas_density)

        # (b) Construct the Green's function on the extended grid.
        #
        # The Hockney–Eastwood prescription is to compute, for each dimension,
        #     pos = [0, 1, 2, ..., n-1, 2n - n, ..., 1] * grid_spacing,
        # which is equivalent to:
        #
        #    pos = np.arange(2*n);  pos = where(pos < n, pos, 2*n - pos)
        #
        # This yields the “minimum–image” distances from the source placed at the origin.
        grids = []
        for s in original_shape:
            n = s
            extended_n = 2 * n
            pos = jnp.arange(extended_n)
            pos = jnp.where(pos < n, pos, 2 * n - pos)
            pos = pos * grid_spacing
            grids.append(pos)

        # Now create the distance array r on the extended grid.
        if dimensionality == 1:
            r = grids[0]  # Already nonnegative.
        elif dimensionality == 2:
            x, y = jnp.meshgrid(grids[0], grids[1], indexing="ij")
            r = jnp.sqrt(x**2 + y**2)
        elif dimensionality == 3:
            x, y, z = jnp.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
            r = jnp.sqrt(x**2 + y**2 + z**2)

        # Replace any zero distance with grid_spacing (to avoid singularity).
        r_safe = jnp.where(r == 0, grid_spacing, r)

        # (c) Define the isolated (open–boundary) Green's function.
        if dimensionality == 1:
            # For 1D (solving φ'' = 4πGδ(x)), the solution is φ = -2πG |x|
            kernel = -2 * jnp.pi * G * r
        elif dimensionality == 2:
            # In 2D, φ = -2G log(r)  (up to an additive constant).
            kernel = -2 * G * jnp.log(r_safe)
        elif dimensionality == 3:
            # In 3D, the isolated potential is φ = -G/r.
            kernel = -G / r_safe

        # (d) FFT–convolve: Multiply the FFTs of the extended density and the Green's function.
        density_k_ext = fftn(extended_density)
        kernel_k_ext = fftn(kernel)
        potential_ext = jnp.real(ifftn(density_k_ext * kernel_k_ext))

        # (e) Extract the portion of the potential corresponding to the original grid.
        gravitational_potential = potential_ext[slices]
        return gravitational_potential / num_cells**dimensionality
