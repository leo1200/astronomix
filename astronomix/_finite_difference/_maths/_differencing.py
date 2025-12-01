import jax
import jax.numpy as jnp


from functools import partial


@partial(jax.jit, static_argnames=["axis"])
def finite_difference_int6(f_int, axis):
    """
    High-order FD derivative, assumed f defined at interfaces,
    and the i-th index corresponds to the i+1/2 interface.

    The finite difference formula is:
        df/dx at i = c1 * (f_{i+1/2} - f_{i-1/2})
                     + c2 * (f_{i+3/2} - f_{i-3/2})
                     + c3 * (f_{i+5/2} - f_{i-5/2})

    Note that the i+1/2 interface corresponds
    to index i in the array.

    6th order: c1 = 75/64, c2 = -25/384, c3 = 3/640
    4th order: c1 = 9/8, c2 = -1/24, c3 = 0
    """
    c1, c2, c3 = 75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0
    return (
        c1 * (f_int - jnp.roll(f_int, 1, axis=axis))
        + c2 * (jnp.roll(f_int, -1, axis=axis) - jnp.roll(f_int, 2, axis=axis))
        + c3 * (jnp.roll(f_int, -2, axis=axis) - jnp.roll(f_int, 3, axis=axis))
    )

@jax.jit
def _interface_field_divergence(bx_int, by_int, bz_int, grid_spacing):
    """
    Compute the divergence of the magnetic field
    defined at interfaces using finite difference.

    Args:
        bx_int: Magnetic field in x-direction at x interfaces.
        by_int: Magnetic field in y-direction at y interfaces.
        bz_int: Magnetic field in z-direction at z interfaces.
        grid_spacing: Grid spacing (assumed uniform in all directions).

    Returns:
        Divergence of the magnetic field at cell centers.
    """
    d_bx_dx = finite_difference_int6(bx_int, axis=0) / grid_spacing
    d_by_dy = finite_difference_int6(by_int, axis=1) / grid_spacing
    d_bz_dz = finite_difference_int6(bz_int, axis=2) / grid_spacing

    div_b = d_bx_dx + d_by_dy + d_bz_dz
    return div_b