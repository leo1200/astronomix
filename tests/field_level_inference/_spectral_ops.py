"""Spectral prolongation / restriction for the multiresolution inverse, plus a
self-test.  prolong P: low-res field -> high-res by zero-padding the spectrum
(sinc interpolation).  restrict R: high-res -> low-res by spectral truncation.
Both are linear; with the chosen amplitude scaling a constant is preserved and
R is (up to a real scalar) the adjoint Pᵀ -- which is what makes jax.grad of a
high-res loss w.r.t. the coarse control equal the restricted high-res gradient.
"""
import jax.numpy as jnp


def _zero_nyquist(U):
    """Zero the -N/2 Nyquist plane along each of the last 3 (fftshifted) axes.

    For even N the retained band [-N/2, N/2) is asymmetric: it includes the
    Nyquist mode -N/2 but not +N/2.  For a *real* field those are the same mode,
    so keeping only -N/2 makes restrict/prolong inconsistent (R(P(x)) != x in
    that plane).  Zeroing it makes both operators clean projectors onto the OPEN
    band (-N/2, N/2)^3 -- band-limited, alias-free, and R(P) = identity there.
    The discarded plane is the single highest-frequency shell (negligible energy
    for smooth turbulent fields)."""
    U = U.at[..., 0, :, :].set(0)
    U = U.at[..., :, 0, :].set(0)
    U = U.at[..., :, :, 0].set(0)
    return U


def restrict(u, n_lo):
    """(..., N, N, N) high-res real field -> (..., n_lo, n_lo, n_lo).
    Ideal sharp spectral low-pass (FFT truncation), open-band / Nyquist-zeroed."""
    n_hi = u.shape[-1]
    U = jnp.fft.fftshift(jnp.fft.fftn(u, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    c, h = n_hi // 2, n_lo // 2
    U = U[..., c - h:c + h, c - h:c + h, c - h:c + h]
    U = _zero_nyquist(U)
    u_lo = jnp.fft.ifftn(jnp.fft.ifftshift(U, axes=(-3, -2, -1)),
                         axes=(-3, -2, -1)).real
    return u_lo * (n_lo / n_hi) ** 3


def prolong(u, n_hi):
    """(..., n, n, n) low-res real field -> (..., n_hi, n_hi, n_hi).
    Spectral zero-pad (sinc interpolation); the exact adjoint of restrict."""
    n_lo = u.shape[-1]
    U = jnp.fft.fftshift(jnp.fft.fftn(u, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    U = _zero_nyquist(U)
    pad = (n_hi - n_lo) // 2
    pads = [(0, 0)] * (U.ndim - 3) + [(pad, pad)] * 3
    U = jnp.pad(U, pads)
    u_hi = jnp.fft.ifftn(jnp.fft.ifftshift(U, axes=(-3, -2, -1)),
                         axes=(-3, -2, -1)).real
    return u_hi * (n_hi / n_lo) ** 3


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    Nlo, Nhi = 16, 32
    rng = np.random.default_rng(0)

    # 1) constant preserved both ways
    a = jnp.ones((Nhi, Nhi, Nhi)) * 3.7
    print("restrict(const) ~ const:", float(jnp.abs(restrict(a, Nlo) - 3.7).max()))
    b = jnp.ones((Nlo, Nlo, Nlo)) * 2.1
    print("prolong(const)  ~ const:", float(jnp.abs(prolong(b, Nhi) - 2.1).max()))

    # 2) idempotency on the open band: project first, then R(P(.)) is identity
    x = jnp.asarray(rng.normal(size=(3, Nlo, Nlo, Nlo)))
    xb = restrict(prolong(x, Nhi), Nlo)              # band-limited (Nyquist-free)
    print("R(P(.)) idempotent on band:", float(jnp.abs(restrict(prolong(xb, Nhi), Nlo) - xb).max()))

    # 3) adjoint: <P x, y>_hi == <x, R' y>_lo where R' = (Nhi/Nlo)^3 * restrict
    #    (P and restrict share scaling; the adjoint of P is restrict up to (Nhi/Nlo)^3)
    y = jnp.asarray(rng.normal(size=(3, Nhi, Nhi, Nhi)))
    lhs = float(jnp.sum(prolong(x, Nhi) * y))
    rhs = float(jnp.sum(x * restrict(y, Nlo)) * (Nhi / Nlo) ** 3)
    print(f"adjoint <Px,y>={lhs:.6f} vs <x,Pty>={rhs:.6f} reldiff="
          f"{abs(lhs - rhs) / max(abs(lhs), 1e-30):.2e}")

    # 4) jax.grad through prolong gives the restricted gradient (the crux)
    def f(theta):
        return jnp.sum(prolong(theta, Nhi) * y)
    g = jax.grad(f)(x)
    g_expected = restrict(y, Nlo) * (Nhi / Nlo) ** 3
    print("grad(prolong) == Pt y:", float(jnp.abs(g - g_expected).max()))
