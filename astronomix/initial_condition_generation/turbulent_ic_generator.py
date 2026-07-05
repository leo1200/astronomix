"""
Generate Gaussian random fields with a prescribed power-law spectrum.

Builds a real-valued 3D turbulent field by sampling complex Fourier
coefficients with a target amplitude scaling ``A0 * k^slope`` inside a
wavenumber band, enforcing Hermitian symmetry so the inverse transform is
real, and transforming back to real space. The implementation deliberately
avoids explicit meshgrid / index arrays (relying on broadcasting and roll /
flip tricks instead) to keep the memory footprint low at large resolution.
"""

# jax
import jax
import jax.numpy as jnp


def _cosine_taper(k, k0, k1):
    """Cosine roll-off factor in [0, 1]: 1 for ``k <= k0``, smoothly down to 0
    at ``k >= k1``.

    Args:
        k: The wavenumber magnitude (array) to taper.
        k0: The wavenumber at which the taper starts (factor still 1).
        k1: The wavenumber at which the taper reaches 0.

    Returns:
        The taper factor, same shape as ``k``.
    """
    # Guard against a degenerate band (k1 == k0): fall back to a hard cutoff.
    small = 1e-12
    if k1 <= k0 + small:
        return jnp.where(k <= k0, 1.0, 0.0)
    fraction = (k - k0) / (k1 - k0)
    fraction = jnp.clip(fraction, 0.0, 1.0)
    return jnp.where(k <= k0, 1.0, 0.5 * (1.0 + jnp.cos(jnp.pi * fraction)))


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
    Generate a real Gaussian random field with a target amplitude scaling.

    This version is optimized for memory by avoiding explicit meshgrid / index
    arrays and using broadcasting and efficient array manipulations instead.

    Args:
        Ndim: The number of grid points along each of the three axes.
        A0: The amplitude prefactor of the power-law spectrum.
        slope: The spectral slope (``amplitude ~ A0 * k^slope``).
        kmin: The lower edge of the wavenumber band carrying power.
        kmax: The upper edge of the wavenumber band carrying power.
        key: The PRNG key used to sample the Fourier coefficients.
        sharding: An optional sharding applied to the large k-space arrays.
        kroll_frac: The fraction of ``kmax`` at which the cosine roll-off begins.
        zero_mean: Whether to remove the DC component so the field has zero mean.

    Returns:
        The real-valued turbulent field of shape ``(Ndim, Ndim, Ndim)``.
    """
    # Integer wavenumbers laid out in FFT order [-N/2 .. N/2-1].
    wavenumbers_1d = jnp.fft.fftfreq(Ndim, d=1.0) * Ndim

    # --- Memory Optimization 1: Broadcasting instead of meshgrid ---
    # Compute the k-space magnitude without materialising full kx, ky, kz
    # arrays; JAX broadcasts the 1D arrays to 3D during the operation.
    wavenumber_magnitude_squared = (
        wavenumbers_1d.reshape(Ndim, 1, 1) ** 2
        + wavenumbers_1d.reshape(1, Ndim, 1) ** 2
        + wavenumbers_1d.reshape(1, 1, Ndim) ** 2
    )

    # Shard the first large array created; subsequent element-wise operations
    # inherit the sharding.
    if sharding is not None:
        wavenumber_magnitude_squared = jax.device_put(
            wavenumber_magnitude_squared, sharding
        )

    wavenumber_magnitude = jnp.sqrt(wavenumber_magnitude_squared)

    # Optional roll-off: start tapering at kroll_frac * kmax and finish at kmax.
    k_roll_start = kroll_frac * kmax
    taper = _cosine_taper(wavenumber_magnitude, k_roll_start, kmax)

    # Restrict power to the requested k-band.
    band_mask = (wavenumber_magnitude >= kmin) & (wavenumber_magnitude <= kmax)

    # Set the target amplitude, avoiding division by zero at k = 0.
    wavenumber_magnitude_safe = jnp.where(
        wavenumber_magnitude == 0.0, 1.0, wavenumber_magnitude
    )
    amplitude = A0 * (wavenumber_magnitude_safe**slope)
    if zero_mean:
        amplitude = amplitude.at[0, 0, 0].set(0.0)

    # Apply the band restriction and the taper.
    amplitude = amplitude * band_mask * taper

    # Sample complex Fourier coefficients with variance ~ amplitude^2.
    sigma = amplitude / jnp.sqrt(2.0)
    subkeys = jax.random.split(key, 2)
    real_part = jax.random.normal(subkeys[0], shape=wavenumber_magnitude.shape) * sigma
    imag_part = jax.random.normal(subkeys[1], shape=wavenumber_magnitude.shape) * sigma
    fourier_coeffs = real_part + 1j * imag_part

    # --- Memory Optimization 2: Enforce Hermitian symmetry without indices ---
    # The map F[k] -> conj(F[-k]) is, under the fftfreq convention, equivalent
    # to flipping all axes and rolling by one element. This is much cheaper than
    # a gather with explicit index arrays.
    fourier_conj_flipped = jnp.roll(
        jnp.flip(jnp.conj(fourier_coeffs), axis=(0, 1, 2)),
        shift=1,
        axis=(0, 1, 2),
    )
    fourier_symmetric = 0.5 * (fourier_coeffs + fourier_conj_flipped)

    # --- Memory Optimization 3: Find self-conjugate modes without indices ---
    # Self-conjugate modes satisfy k_i = -k_i (mod N), occurring at indices 0
    # and N/2 (for even N). Build the 3D mask by broadcasting a 1D mask.
    index_1d = jnp.arange(Ndim)
    self_conj_1d = index_1d == ((-index_1d) % Ndim)
    self_conj_mask = (
        self_conj_1d.reshape(Ndim, 1, 1)
        & self_conj_1d.reshape(1, Ndim, 1)
        & self_conj_1d.reshape(1, 1, Ndim)
    )

    # The self-conjugate modes must be purely real, so drop their imaginary part.
    fourier_symmetric = jnp.where(
        self_conj_mask,
        jnp.real(fourier_symmetric).astype(fourier_symmetric.dtype),
        fourier_symmetric,
    )

    # Explicitly pin the DC component to a real zero when a zero mean is wanted.
    if zero_mean:
        fourier_symmetric = fourier_symmetric.at[0, 0, 0].set(0.0 + 0.0j)

    # Inverse transform back to real space; the imaginary part is zero by
    # construction up to round-off, so take the real part.
    real_space_field = jnp.real(jnp.fft.ifftn(fourier_symmetric))

    return real_space_field
