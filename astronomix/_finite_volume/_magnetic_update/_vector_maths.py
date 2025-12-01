import jax
import jax.numpy as jnp


@jax.jit
def cross(a, b):
    result = jnp.zeros_like(a)
    result = result.at[0, ...].set(a[1, ...] * b[2, ...] - a[2, ...] * b[1, ...])
    result = result.at[1, ...].set(a[2, ...] * b[0, ...] - a[0, ...] * b[2, ...])
    result = result.at[2, ...].set(a[0, ...] * b[1, ...] - a[1, ...] * b[0, ...])
    return result


@jax.jit
def divergence2D(field, grid_spacing: float):
    """Calculate the divergence of a 3-component field on a 2D grid.
    field.shape == (3, nx, ny)
    """
    divergence = jnp.zeros_like(field)

    # d/dx of component 0: roll along x -> axis 0 of field[0]
    dfx = (jnp.roll(field[0], -1, axis=0) - jnp.roll(field[0], 1, axis=0)) / (2 * grid_spacing)
    divergence = divergence.at[0].add(dfx)

    # d/dy of component 1: roll along y -> axis 1 of field[1]
    dfy = (jnp.roll(field[1], -1, axis=1) - jnp.roll(field[1], 1, axis=1)) / (2 * grid_spacing)
    divergence = divergence.at[1].add(dfy)

    return jnp.sum(divergence, axis=0)


@jax.jit
def divergence3D(field, grid_spacing: float):
    """Calculate the divergence of a 3-component field on a 3D grid.
    field.shape == (3, nx, ny, nz)
    """
    divergence = jnp.zeros_like(field)

    # d/dx of component 0: roll along x -> axis 0 of field[0]
    dfx = (jnp.roll(field[0], -1, axis=0) - jnp.roll(field[0], 1, axis=0)) / (2 * grid_spacing)
    divergence = divergence.at[0].add(dfx)

    # d/dy of component 1: roll along y -> axis 1 of field[1]
    dfy = (jnp.roll(field[1], -1, axis=1) - jnp.roll(field[1], 1, axis=1)) / (2 * grid_spacing)
    divergence = divergence.at[1].add(dfy)

    # d/dz of component 2: roll along z -> axis 2 of field[2]
    dfz = (jnp.roll(field[2], -1, axis=2) - jnp.roll(field[2], 1, axis=2)) / (2 * grid_spacing)
    divergence = divergence.at[2].add(dfz)

    return jnp.sum(divergence, axis=0)


@jax.jit
def curl3D(field, grid_spacing: float):
    """Calculate the curl of a 3-component field on a 3D grid.
    field.shape == (3, nx, ny, nz)
    """
    curl = jnp.zeros_like(field)

    # curl_x = dFz/dy - dFy/dz
    dFz_dy = 0.5 * (jnp.roll(field[2], -1, axis=1) - jnp.roll(field[2], 1, axis=1)) / grid_spacing
    dFy_dz = 0.5 * (jnp.roll(field[1], -1, axis=2) - jnp.roll(field[1], 1, axis=2)) / grid_spacing
    curl = curl.at[0].add(dFz_dy - dFy_dz)

    # curl_y = dFx/dz - dFz/dx
    dFx_dz = 0.5 * (jnp.roll(field[0], -1, axis=2) - jnp.roll(field[0], 1, axis=2)) / grid_spacing
    dFz_dx = 0.5 * (jnp.roll(field[2], -1, axis=0) - jnp.roll(field[2], 1, axis=0)) / grid_spacing
    curl = curl.at[1].add(dFx_dz - dFz_dx)

    # curl_z = dFy/dx - dFx/dy
    dFy_dx = 0.5 * (jnp.roll(field[1], -1, axis=0) - jnp.roll(field[1], 1, axis=0)) / grid_spacing
    dFx_dy = 0.5 * (jnp.roll(field[0], -1, axis=1) - jnp.roll(field[0], 1, axis=1)) / grid_spacing
    curl = curl.at[2].add(dFy_dx - dFx_dy)

    return curl


@jax.jit
def curl2D(field, grid_spacing: float):
    """Calculate the curl of a 3-component field on a 2D grid.
    field.shape == (3, nx, ny)
    (assumes grid_spacing is dy in some of your original docstring contexts)
    """
    curl = jnp.zeros_like(field)

    # curl_x = dFz/dy  (roll y -> axis=1 of field[2])
    dFz_dy = 0.5 * (jnp.roll(field[2], -1, axis=1) - jnp.roll(field[2], 1, axis=1)) / grid_spacing
    curl = curl.at[0].add(dFz_dy)

    # curl_y = - dFz/dx  (roll x -> axis=0 of field[2])
    dFz_dx = 0.5 * (jnp.roll(field[2], -1, axis=0) - jnp.roll(field[2], 1, axis=0)) / grid_spacing
    curl = curl.at[1].add(-dFz_dx)

    # curl_z = dFy/dx - dFx/dy
    dFy_dx = 0.5 * (jnp.roll(field[1], -1, axis=0) - jnp.roll(field[1], 1, axis=0)) / grid_spacing
    dFx_dy = 0.5 * (jnp.roll(field[0], -1, axis=1) - jnp.roll(field[0], 1, axis=1)) / grid_spacing
    curl = curl.at[2].add(dFy_dx - dFx_dy)

    return curl
