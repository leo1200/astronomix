#!/usr/bin/env python
"""Volumetric rendering of a 3D density cube (pyvista add_volume, in the
same style as field_level_inference/image_optim.py) with the line-of-sight
projection shown as a "screen" plane on the side of the cube, in the same
3D scene.

Run with the `jf1uids` env (has pyvista + vtk):
    .../envs/jf1uids/bin/python make_cube_vol.py [--field rho3d.npy] [--out cube_vol.png]
"""
import argparse
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


def synthetic_field(N=160, slope=-5.0 / 3.0, seed=0):
    """Gaussian random field with power-law spectrum -> turbulence-like
    log-normal density."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kmag[0, 0, 0] = 1.0
    amp = kmag ** (slope)
    amp[0, 0, 0] = 0.0
    ph = rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))
    f = np.fft.ifftn(amp * ph).real
    f = (f - f.mean()) / f.std()
    return np.exp(1.0 * f).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", type=str, default=None)
    ap.add_argument("--out", type=str, default="cube_vol.png")
    ap.add_argument("--proj-axis", type=int, default=0,
                    help="axis to integrate the projection over (0=x,1=y,2=z)")
    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument("--screen-cmap", type=str, default="magma")
    ap.add_argument("--gap", type=float, default=0.6,
                    help="gap between cube and screen, in units of cube size")
    args = ap.parse_args()

    if args.field:
        F = np.load(args.field).astype(np.float32)
    else:
        F = synthetic_field()
    N = F.shape
    L = max(N)  # cube spans [0,N) per axis

    # ---- volume -------------------------------------------------------
    grid = pv.ImageData()
    grid.dimensions = np.array(F.shape) + 1
    grid.cell_data["density"] = F.ravel(order="F")
    lo, hi = np.percentile(F, [50, 99.5])

    pl = pv.Plotter(off_screen=True, window_size=(1100, 1000))
    pl.add_volume(
        grid, scalars="density", cmap=args.cmap,
        clim=(float(lo), float(hi)),
        opacity="sigmoid", blending="composite", shade=True,
        show_scalar_bar=False,
    )

    # ---- projection onto a side screen --------------------------------
    ax = args.proj_axis
    proj = F.sum(axis=ax)            # 2D image in the two remaining axes
    # place a screen plane perpendicular to `ax`, offset to the negative side
    gap = args.gap * N[ax]
    # build a structured plane sized to the cube, positioned at ax = -gap
    others = [i for i in range(3) if i != ax]
    na, nb = N[others[0]], N[others[1]]
    # grid of plane corners in the two in-plane axes
    a = np.linspace(0, N[others[0]], na + 1)
    b = np.linspace(0, N[others[1]], nb + 1)
    A, B = np.meshgrid(a, b, indexing="ij")
    coords = {others[0]: A, others[1]: B, ax: np.full_like(A, -gap)}
    sg = pv.StructuredGrid(coords[0], coords[1], coords[2])
    sg.cell_data["proj"] = proj.ravel(order="F")
    pl.add_mesh(sg, scalars="proj", cmap=args.screen_cmap,
                show_scalar_bar=False, lighting=False)

    # faint frame around the screen
    pl.add_mesh(sg.extract_feature_edges(), color=[0.3, 0.3, 0.3], line_width=1.5)

    # ---- rays from cube face to screen --------------------------------
    cube_face = 0.0  # the ax=0 face of the cube
    for ca in (0, N[others[0]]):
        for cb in (0, N[others[1]]):
            p_cube = {others[0]: ca, others[1]: cb, ax: cube_face}
            p_scr = {others[0]: ca, others[1]: cb, ax: -gap}
            line = pv.Line([p_cube[0], p_cube[1], p_cube[2]],
                           [p_scr[0], p_scr[1], p_scr[2]])
            pl.add_mesh(line, color=[0.45, 0.45, 0.45], line_width=1.0)

    pl.set_background("white")
    pl.camera_position = "iso"
    pl.camera.azimuth = 25
    pl.camera.elevation = 10
    img = pl.screenshot(args.out, return_img=True)
    pl.close()
    print(f"wrote {args.out}  field={F.shape} proj-axis={ax}  img={img.shape}")


if __name__ == "__main__":
    main()
