"""GKS-style 2D slice comparison: astronomix vs AthenaK, before and after the SN.

Density and temperature slices through the box midplane, with colour scales
SHARED across codes so the panels are directly comparable (a per-panel
normalisation would hide exactly the differences we care about).

Usage:
  compare_slices.py --astx-pre A.npz --astx-post B.npz \
                    --atk-pre DIR --atk-post DIR [--out NAME]
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, "/export/home/lstorcks/athena/athenak/vis/python")
import bin_convert  # noqa: E402

TEMP_UNIT = 71.06
BOX = 64.0
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def load_athenak(dirname):
    files = sorted(glob.glob(os.path.join(dirname, "bin", "*.bin")))
    d = bin_convert.read_binary(files[-1])
    a = np.array(d["mb_data"]["dens"])
    nblk = int(round(len(d["mb_logical"]) ** (1.0 / 3.0)))
    n = nblk * a.shape[3]

    def cube(v):
        arr = np.array(d["mb_data"][v])
        out = np.zeros((n, n, n), dtype=arr.dtype)
        for m, lb in enumerate(d["mb_logical"]):
            nz, ny, nx = arr.shape[1], arr.shape[2], arr.shape[3]
            i, j, k = lb[0] * nx, lb[1] * ny, lb[2] * nz
            out[k:k + nz, j:j + ny, i:i + nx] = arr[m]
        return np.ascontiguousarray(out.transpose(2, 1, 0))

    rho = cube("dens")
    return rho, cube("eint") * (2.0 / 3.0) / rho * TEMP_UNIT, d["time"]


def load_astx(path):
    d = np.load(path)
    rho = np.asarray(d["rho"])
    return rho, np.asarray(d["press"]) / rho * TEMP_UNIT, float(d["age"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--astx-pre", required=True)
    ap.add_argument("--astx-post", required=True)
    ap.add_argument("--atk-pre", required=True)
    ap.add_argument("--atk-post", required=True)
    ap.add_argument("--out", default="gks_slices")
    args = ap.parse_args()

    panels = [
        ("astronomix\n(WENO5)", "pre-SN", *load_astx(args.astx_pre)),
        ("astronomix\n(WENO5)", "post-SN", *load_astx(args.astx_post)),
        ("AthenaK\n(PPM4+HLLC)", "pre-SN", *load_athenak(args.atk_pre)),
        ("AthenaK\n(PPM4+HLLC)", "post-SN", *load_athenak(args.atk_post)),
    ]

    # shared colour scales across ALL panels
    rmin = min(np.percentile(p[2], 0.1) for p in panels)
    rmax = max(np.percentile(p[2], 99.99) for p in panels)
    tmin = max(1.0, min(np.percentile(p[3], 0.1) for p in panels))
    tmax = max(np.percentile(p[3], 99.99) for p in panels)

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.2))
    ext = [-BOX / 2, BOX / 2, -BOX / 2, BOX / 2]
    for c, (code, phase, rho, T, t) in enumerate(panels):
        mid = rho.shape[2] // 2
        im0 = axes[0, c].imshow(rho[:, :, mid].T, origin="lower", extent=ext,
                                norm=LogNorm(vmin=rmin, vmax=rmax), cmap="viridis")
        axes[0, c].set_title(f"{code}  {phase}\nt = {t:.2f} Myr", fontsize=11)
        im1 = axes[1, c].imshow(np.maximum(T[:, :, mid], 1.0).T, origin="lower",
                                extent=ext, norm=LogNorm(vmin=tmin, vmax=tmax),
                                cmap="inferno")
        for r in (0, 1):
            axes[r, c].set_xlabel("x [pc]")
            if c == 0:
                axes[r, c].set_ylabel("y [pc]")
        if c == 3:
            plt.colorbar(im0, ax=axes[0, c], label=r"$n$ [cm$^{-3}$]")
            plt.colorbar(im1, ax=axes[1, c], label=r"$T$ [K]")

    fig.suptitle("Guo-Kim-Stone setup: two-phase ISM before and after a supernova — "
                 "astronomix vs AthenaK (shared colour scales)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, args.out + ".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
