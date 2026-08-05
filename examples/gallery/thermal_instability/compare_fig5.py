"""Reproduce the panel layout of Guo, Kim & Stone (2025) Figure 5.

Slices through the z = 0 plane of (left to right): volumetric cooling rate
rho*L, number density n, and tangential velocity v_phi about the SN centre --
astronomix on the top row, AthenaK on the bottom, with SHARED colour scales.

The paper's fourth panel (contribution to hot gas from four passive scalars)
is NOT reproduced: neither run carries passive scalars.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

sys.path.insert(0, "/export/home/lstorcks/athena/athenak/vis/python")
import bin_convert  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEMP_UNIT = 71.06
BOX = 64.0
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# AthenaK's ISMCoolFn (KI2002 below log T = 4.2, Schure SPEX above, CGOLS tail)
_LHD = np.array([
    -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
    -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
    -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
    -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
    -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
    -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
    -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
    -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
    -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
    -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
    -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
    -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
    -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928])
_LHD_T = 4.12 + 0.04 * np.arange(len(_LHD))


def cool_fn(T):
    """Lambda(T) in erg cm^3 / s, vectorised."""
    T = np.maximum(T, 1e-3)
    lt = np.log10(T)
    ki = 2.0e-19 * np.exp(-1.184e5 / (T + 1.0e3)) + 2.8e-28 * np.sqrt(T) * np.exp(-92.0 / T)
    spex = 10.0 ** np.interp(lt, _LHD_T, _LHD)
    cgols = 10.0 ** (0.45 * lt - 26.065)
    return np.where(lt <= 4.2, ki, np.where(lt > 8.15, cgols, spex))


def diagnostics(rho, p, vx, vy, vz):
    n = rho                                   # code density = n_p in GKS units
    T = p / np.maximum(rho, 1e-30) * TEMP_UNIT
    rhoL = n ** 2 * cool_fn(T)                # erg s^-1 cm^-3
    N = rho.shape[0]
    x = (np.arange(N) + 0.5) * (BOX / N) - BOX / 2
    X = x[:, None, None]; Y = x[None, :, None]
    R = np.sqrt(X ** 2 + Y ** 2)
    # azimuthal velocity in the z=0 plane, about the SN centre
    vphi = (-Y * vx + X * vy) / np.maximum(R, 1e-9) * 0.978    # km/s
    return rhoL, n, T, vphi


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
    return (*diagnostics(rho, cube("eint") * (2.0 / 3.0), cube("velx"),
                         cube("vely"), cube("velz")), d["time"], rho.shape[0])


def load_astx(path):
    d = np.load(path)
    rho = np.asarray(d["rho"])
    return (*diagnostics(rho, np.asarray(d["press"]), np.asarray(d["vx"]),
                         np.asarray(d["vy"]), np.asarray(d["vz"])),
            float(d["age"]), rho.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--astx", required=True)
    ap.add_argument("--atk", required=True)
    ap.add_argument("--zoom", type=float, default=16.0, help="half-width shown [pc]")
    ap.add_argument("--out", default="gks_fig5_style")
    args = ap.parse_args()

    rows = [("astronomix (WENO5)", *load_astx(args.astx)),
            ("AthenaK (PPM4+HLLC)", *load_athenak(args.atk))]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.2))
    for r, (label, rhoL, n, T, vphi, t, N) in enumerate(rows):
        dx = BOX / N
        mid = N // 2
        h = int(args.zoom / dx)
        sl = slice(max(0, mid - h), min(N, mid + h))
        ext = [-args.zoom, args.zoom, -args.zoom, args.zoom]
        im0 = axes[r, 0].imshow(rhoL[sl, sl, mid].T, origin="lower", extent=ext,
                                norm=LogNorm(1e-27, 1e-20), cmap="afmhot")
        im1 = axes[r, 1].imshow(n[sl, sl, mid].T, origin="lower", extent=ext,
                                norm=LogNorm(1e-1, 1e3), cmap="magma")
        im2 = axes[r, 2].imshow(vphi[sl, sl, mid].T, origin="lower", extent=ext,
                                norm=TwoSlopeNorm(0.0, -500, 500), cmap="RdBu_r")
        axes[r, 0].set_ylabel(f"{label}\n$\\Delta x$ = {dx:.3f} pc\n\nY [pc]", fontsize=10)
        for c, (im, lab) in enumerate(((im0, r"$\rho\mathcal{L}$ [erg s$^{-1}$ cm$^{-3}$]"),
                                       (im1, r"$n$ [cm$^{-3}$]"),
                                       (im2, r"$v_\phi$ [km s$^{-1}$]"))):
            axes[r, c].set_xlabel("X [pc]")
            plt.colorbar(im, ax=axes[r, c], label=lab, fraction=0.046)
        axes[r, 1].set_title(f"t = {t:.2f} Myr", fontsize=10)

    fig.suptitle("Guo, Kim & Stone (2025) Figure 5 layout — z = 0 slices, 1 Myr after the SN\n"
                 "(paper: $\\Delta x$ = 1/64 pc = 0.016 pc; passive-scalar panel omitted, no tracers carried)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG, args.out + ".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
