import jax
import jax.numpy as jnp


def _cosine_taper(k, k0, k1):
    """Returns taper factor in [0,1]: 1 for k<=k0, smooth down to 0 at k>=k1."""
    # If k1 == k0, use a hard cutoff
    small = 1e-12
    if k1 <= k0 + small:
        return jnp.where(k <= k0, 1.0, 0.0)
    t = (k - k0) / (k1 - k0)
    t = jnp.clip(t, 0.0, 1.0)
    return jnp.where(k <= k0, 1.0, 0.5 * (1.0 + jnp.cos(jnp.pi * t)))


# k_max <= int(0.7 * Ndim/2)


def create_turb_field(
    Ndim,
    A0,
    slope,
    kmin,
    kmax,
    key,
    sharding=None,
    kroll_frac=0.85,
    zero_mean=True,
):
    """
    Generate a real Gaussian random field with target amplitude scaling.

    This version is optimized for memory by avoiding explicit meshgrid/indices
    arrays and using broadcasting and efficient array manipulations instead.
    """
    # Build integer wavenumbers [-N/2..N/2-1]
    k1d = jnp.fft.fftfreq(Ndim, d=1.0) * Ndim

    # --- Memory Optimization 1: Broadcasting instead of meshgrid ---
    # Calculate k-space magnitude without creating full kx, ky, kz arrays.
    # JAX will broadcast the 1D arrays to 3D during the operation.
    k3d_sq = (
        k1d.reshape(Ndim, 1, 1) ** 2
        + k1d.reshape(1, Ndim, 1) ** 2
        + k1d.reshape(1, 1, Ndim) ** 2
    )

    # Shard the first large array created. Subsequent element-wise ops
    # will inherit the sharding.
    if sharding is not None:
        k3d_sq = jax.device_put(k3d_sq, sharding)

    k3d = jnp.sqrt(k3d_sq)

    # Optional roll-off: start rolling at kroll_frac * kmax and finish at kmax
    k_roll_start = kroll_frac * kmax
    taper = _cosine_taper(k3d, k_roll_start, kmax)

    # Mask inside k-band where we want power
    band_mask = (k3d >= kmin) & (k3d <= kmax)

    # Set amplitude, avoiding division by zero at k=0
    k_safe = jnp.where(k3d == 0.0, 1.0, k3d)
    amplitude = A0 * (k_safe**slope)
    if zero_mean:
        amplitude = amplitude.at[0, 0, 0].set(0.0)

    # Apply band + taper
    amplitude = amplitude * band_mask * taper

    # Sample complex Fourier coefficients with variance ~ amplitude^2
    sigma = amplitude / jnp.sqrt(2.0)
    subkeys = jax.random.split(key, 2)
    re = jax.random.normal(subkeys[0], shape=k3d.shape) * sigma
    im = jax.random.normal(subkeys[1], shape=k3d.shape) * sigma
    F = re + 1j * im

    # --- Memory Optimization 2: Enforce Hermitian symmetry without indices ---
    # The operation F_conj_flipped[-k] is equivalent to flipping all axes and
    # rolling by one element due to the fftfreq convention. This is much
    # cheaper than a gather operation with explicit index arrays.
    F_conj_flipped = jnp.roll(
        jnp.flip(jnp.conj(F), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2)
    )
    F_sym = 0.5 * (F + F_conj_flipped)

    # --- Memory Optimization 3: Find self-conjugate modes without indices ---
    # Self-conjugate modes are where k_i = -k_i (mod N), which occurs at
    # indices 0 and N/2 (for even N). We can build this mask via broadcasting.
    idx_1d = jnp.arange(Ndim)
    sc_1d = idx_1d == ((-idx_1d) % Ndim)
    self_conj_mask = (
        sc_1d.reshape(Ndim, 1, 1)
        & sc_1d.reshape(1, Ndim, 1)
        & sc_1d.reshape(1, 1, Ndim)
    )

    # Force imaginary part to be zero at self-conjugate locations
    F_sym = jnp.where(self_conj_mask, jnp.real(F_sym).astype(F_sym.dtype), F_sym)

    # Optional: explicitly ensure DC is real zero (if zero_mean True)
    if zero_mean:
        F_sym = F_sym.at[0, 0, 0].set(0.0 + 0.0j)

    # Inverse transform back to real space
    rfield_complex = jnp.fft.ifftn(F_sym)
    rfield = jnp.real(rfield_complex)

    return rfield
