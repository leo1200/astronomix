#!/usr/bin/env python
"""2x2 field-level-inference figure.

Four states (initial optimized / shortly before / at / shortly after the loss
time) in a 2x2 grid.  Each panel is a volumetric rendering of the 3D density
cube (pyvista add_volume) with the line-of-sight projection shown as a "screen"
plane brought to the FRONT of the cube (toward the viewer) so it is clearly
visible, with faint rays cube -> screen.  Volume and screen share one modern
white-background -> blue colormap that prints cleanly.

Run in the jf1uids env (pyvista + vtk).  Reads panel_snaps.npz.
"""
import argparse
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

pv.OFF_SCREEN = True

# Shared modern white -> blue colormap (sequential, perceptually clean, prints
# well on white paper): background white, increasing density -> deeper blue.
WHITEBLUE = LinearSegmentedColormap.from_list(
    "whiteblue",
    ["#ffffff", "#dbe9f6", "#9ecae1", "#4495c8", "#1f6fb2", "#08306b"])
try:
    matplotlib.colormaps.register(WHITEBLUE, name="whiteblue", force=True)
except Exception:
    pass


def render_panel(F, proj_axis, vclim, pclim, cmap, scmap, gap, azimuth, elevation,
                 screen_scale=1.35, window=(950, 950)):
    """Render one density cube + a FRONT projection screen, return an RGB image."""
    N = F.shape
    grid = pv.ImageData()
    grid.dimensions = np.array(F.shape) + 1
    grid.cell_data["density"] = F.ravel(order="F")

    pl = pv.Plotter(off_screen=True, window_size=window)
    pl.set_background("white")
    # Opacity ramp: low density stays transparent (white paper shows through),
    # high density becomes opaque blue -> clean blue structure on white.
    pl.add_volume(grid, scalars="density", cmap=cmap, clim=vclim,
                  opacity=[0.0, 0.04, 0.12, 0.30, 0.6, 0.92],
                  blending="composite", shade=True, show_scalar_bar=False)

    ax = proj_axis
    proj = F.sum(axis=ax)
    gpix = gap * N[ax]
    others = [i for i in range(3) if i != ax]
    # Screen enlarged + centred on the cube footprint, placed in FRONT of the
    # near (+ax) face so it faces the viewer and is not occluded by the volume.
    n0, n1 = N[others[0]], N[others[1]]
    pad0, pad1 = (screen_scale - 1) * n0 / 2, (screen_scale - 1) * n1 / 2
    a = np.linspace(-pad0, n0 + pad0, n0 + 1)
    b = np.linspace(-pad1, n1 + pad1, n1 + 1)
    A, B = np.meshgrid(a, b, indexing="ij")
    front = N[ax] + gpix
    coords = {others[0]: A, others[1]: B, ax: np.full_like(A, front)}
    sg = pv.StructuredGrid(coords[0], coords[1], coords[2])
    sg.cell_data["proj"] = proj.ravel(order="F")
    pl.add_mesh(sg, scalars="proj", cmap=scmap, clim=pclim,
                show_scalar_bar=False, lighting=False)
    pl.add_mesh(sg.extract_feature_edges(), color=[0.45, 0.45, 0.45], line_width=1.5)

    # faint rays from the near cube face to the screen corners
    for ca in (0, n0):
        for cb in (0, n1):
            p0 = {others[0]: ca, others[1]: cb, ax: float(N[ax])}
            p1 = {others[0]: ca, others[1]: cb, ax: front}
            pl.add_mesh(pv.Line([p0[0], p0[1], p0[2]], [p1[0], p1[1], p1[2]]),
                        color=[0.6, 0.6, 0.6], line_width=1.0)

    pl.camera_position = "iso"
    pl.camera.azimuth = azimuth
    pl.camera.elevation = elevation
    pl.reset_camera()          # fit all actors so the enlarged screen isn't cut
    pl.camera.zoom(1.05)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def autocrop(img, bg=248, pad=6):
    """Trim the white border so the rendered content fills the panel."""
    mask = (np.asarray(img)[..., :3] < bg).any(axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", type=str, default="panel_snaps.npz")
    ap.add_argument("--out", type=str, default="panels.png")
    ap.add_argument("--proj-axis", type=int, default=2, help="loss projection axis (z=2)")
    ap.add_argument("--cmap", type=str, default="whiteblue")
    ap.add_argument("--screen-cmap", type=str, default="whiteblue")
    ap.add_argument("--gap", type=float, default=0.6)
    ap.add_argument("--screen-scale", type=float, default=1.35)
    ap.add_argument("--azimuth", type=float, default=-28.0)
    ap.add_argument("--elevation", type=float, default=7.0)
    ap.add_argument("--rot90", type=int, default=0,
                    help="rotate each cube k*90° CLOCKWISE about the projection axis.")
    ap.add_argument("--rot-image", type=int, default=1,
                    help="rotate the whole rendered panel image k*90° CLOCKWISE.")
    args = ap.parse_args()

    d = np.load(args.snaps)
    order = ["init", "pre", "target", "post"]
    cubes = [d[k].astype(np.float32) for k in order]
    titles = [
        "initial optimized state",
        f"shortly before  (t = {d['f_pre']:.3f} $t_{{loss}}$)",
        "target time  ($t_{loss}$)",
        f"shortly after  (t = {d['f_post']:.3f} $t_{{loss}}$)",
    ]

    if args.rot90:
        ai = [i for i in range(3) if i != args.proj_axis]
        cubes = [np.rot90(c, k=-args.rot90, axes=(ai[0], ai[1])) for c in cubes]
    cubes = [np.moveaxis(c, args.proj_axis, 0).copy() for c in cubes]
    render_axis = 0

    allv = np.concatenate([c.ravel() for c in cubes])
    vclim = (float(np.percentile(allv, 50)), float(np.percentile(allv, 99.5)))
    # Per-panel screen colour scale (the logo's projection contrast is ~16x
    # smaller than the turbulent initial state's, so a global scale would flatten
    # the logo screen to a single colour).

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax_, F, title in zip(axes.flat, cubes, titles):
        proj_vals = F.sum(axis=render_axis)
        pclim = (float(np.percentile(proj_vals, 2)), float(np.percentile(proj_vals, 98)))
        img = render_panel(F, render_axis, vclim, pclim, args.cmap,
                           args.screen_cmap, args.gap, args.azimuth, args.elevation,
                           screen_scale=args.screen_scale)
        if args.rot_image:
            img = np.rot90(img, k=-args.rot_image)
        ax_.imshow(autocrop(img))
        ax_.set_title(title, fontsize=15, pad=4)
        ax_.set_xticks([]); ax_.set_yticks([])
        for s in ax_.spines.values():
            s.set_visible(False)

    # Tight layout: no figure title (goes in the caption) and minimal gaps.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01,
                        wspace=0.02, hspace=0.08)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
