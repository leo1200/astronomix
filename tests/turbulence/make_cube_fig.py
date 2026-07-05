#!/usr/bin/env python
"""Render a 3D density cube with a line-of-sight projection shown on a
"screen" floating in front of the viewer-facing face.

The data cube is drawn as its three camera-facing boundary faces (so it
reads as a solid volume); a separate plane in front shows the column
density integrated along the line of sight, with faint rays connecting
the cube's front face to the screen.

Usage:
    python make_cube_fig.py [--field rho3d.npy] [--out cube.png]

If --field is omitted a synthetic turbulence-like Gaussian random field is
generated so the rendering can be developed without real data.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm


def synthetic_field(N=128, slope=-5.0 / 3.0, seed=0):
    """Gaussian random field with a power-law spectrum -> turbulence-like."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kmag[0, 0, 0] = 1.0
    amp = kmag ** (slope)
    amp[0, 0, 0] = 0.0
    phase = rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))
    f = np.fft.ifftn(amp * phase).real
    f = (f - f.mean()) / f.std()
    # log-normal density (positive, intermittent like compressible turbulence)
    rho = np.exp(0.9 * f)
    return rho.astype(np.float32)


def downsample(F, target):
    N = F.shape[0]
    if N <= target:
        return F
    s = N // target
    return F[::s, ::s, ::s]


def face_grid(n):
    g = np.linspace(0.0, 1.0, n + 1)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", type=str, default=None)
    ap.add_argument("--out", type=str, default="cube.png")
    ap.add_argument("--render-n", type=int, default=128)
    ap.add_argument("--cmap", type=str, default="magma")
    ap.add_argument("--screen-cmap", type=str, default="bone")
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-58.0)
    args = ap.parse_args()

    if args.field:
        F = np.load(args.field).astype(np.float32)
    else:
        F = synthetic_field(N=128)
    F = downsample(F, args.render_n)
    N = F.shape[0]

    # color normalisation on the cube faces (clip tails for contrast)
    lo, hi = np.percentile(F, [2, 99.0])
    norm = Normalize(vmin=lo, vmax=hi)
    cmap = plt.get_cmap(args.cmap)

    # line-of-sight projection along y (depth). proj indexed [x, z]
    proj = F.sum(axis=1)
    pn = Normalize(vmin=np.percentile(proj, 2), vmax=np.percentile(proj, 99.5))
    scmap = plt.get_cmap(args.screen_cmap)

    g = face_grid(N)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))

    # ---- three visible cube faces -------------------------------------
    # TOP  z = 1, varies over (x, y), data F[:, :, -1]
    X, Y = np.meshgrid(g, g, indexing="ij")
    Z = np.ones_like(X)
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(F[:, :, -1])),
                    rstride=1, cstride=1, shade=False, antialiased=False)

    # RIGHT  x = 1, varies over (y, z), data F[-1, :, :]
    Yr, Zr = np.meshgrid(g, g, indexing="ij")
    Xr = np.ones_like(Yr)
    ax.plot_surface(Xr, Yr, Zr, facecolors=cmap(norm(F[-1, :, :])),
                    rstride=1, cstride=1, shade=False, antialiased=False)

    # FRONT  y = 0 (viewer-facing), varies over (x, z), data F[:, 0, :]
    Xf, Zf = np.meshgrid(g, g, indexing="ij")
    Yf = np.zeros_like(Xf)
    ax.plot_surface(Xf, Yf, Zf, facecolors=cmap(norm(F[:, 0, :])),
                    rstride=1, cstride=1, shade=False, antialiased=False)

    # ---- projection screen floating in front (y = -offset) ------------
    off = 0.55
    Xs, Zs = np.meshgrid(g, g, indexing="ij")
    Ys = np.full_like(Xs, -off)
    ax.plot_surface(Xs, Ys, Zs, facecolors=scmap(pn(proj)),
                    rstride=1, cstride=1, shade=False, antialiased=False)

    # ---- rays connecting front face corners to screen corners ---------
    corners = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (x, z)
    for (cx, cz) in corners:
        ax.plot([cx, cx], [0.0, -off], [cz, cz], color="0.4",
                lw=0.8, ls=(0, (4, 4)), alpha=0.7)

    # frame the screen
    ax.plot([0, 1, 1, 0, 0], [-off] * 5, [0, 0, 1, 1, 0],
            color="0.3", lw=1.0)

    ax.set_xlim(0, 1)
    ax.set_ylim(-off, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  (N={N}, proj along y)")


if __name__ == "__main__":
    main()
