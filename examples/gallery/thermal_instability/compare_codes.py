"""Side-by-side astronomix vs AthenaK comparison plots for the ISM/TI box.

Both codes are compared on the SAME physical state and time: the AthenaK dump
is stitched from its meshblocks and the astronomix state is read from its
save-state npz, then identical diagnostics are computed for each.

Because the two drivers draw independent random realisations, the comparison is
necessarily STATISTICAL (PDFs, phase fractions, phase diagrams) rather than
cell-by-cell.

Usage:
  compare_codes.py <astronomix.npz> <athenak_dir> [--time T] [--out NAME]
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

TEMP_UNIT = 71.06          # K per code temperature (mu = 0.618, pc/Myr units)
T_COLD, T_WARM = 184.0, 5050.0   # ti.athinput phase cuts
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def load_athenak(dirname, want_time=None):
    files = sorted(glob.glob(os.path.join(dirname, "bin", "*.bin")))
    if not files:
        raise SystemExit(f"no .bin dumps in {dirname}/bin")
    if want_time is None:
        pick = files[-1]
    else:
        times = []
        for f in files:
            d = bin_convert.read_binary(f)
            times.append(d["time"])
        pick = files[int(np.argmin(np.abs(np.array(times) - want_time)))]
    d = bin_convert.read_binary(pick)
    nb = np.array(d["mb_data"]["dens"]).shape
    # infer the global cube size from the logical locations
    nblocks_per_dim = int(round(len(d["mb_logical"]) ** (1.0 / 3.0)))
    n = nblocks_per_dim * nb[3]

    def cube(v):
        a = np.array(d["mb_data"][v])
        out = np.zeros((n, n, n), dtype=a.dtype)
        for m, lb in enumerate(d["mb_logical"]):
            nz, ny, nx = a.shape[1], a.shape[2], a.shape[3]
            i, j, k = lb[0] * nx, lb[1] * ny, lb[2] * nz
            out[k:k + nz, j:j + ny, i:i + nx] = a[m]
        return np.ascontiguousarray(out.transpose(2, 1, 0))

    rho = cube("dens")
    p = cube("eint") * (2.0 / 3.0)
    v2 = cube("velx") ** 2 + cube("vely") ** 2 + cube("velz") ** 2
    return dict(rho=rho, p=p, v2=v2, time=d["time"], label="AthenaK (PPM4+HLLC)")


def load_astronomix(path):
    d = np.load(path)
    return dict(rho=np.asarray(d["rho"]), p=np.asarray(d["press"]),
                v2=np.asarray(d["vx"]) ** 2 + np.asarray(d["vy"]) ** 2
                + np.asarray(d["vz"]) ** 2,
                time=float(d["age"]), label="astronomix (WENO5+char.LF)")


def stats(s):
    rho, p = s["rho"], s["p"]
    T = (p / np.maximum(rho, 1e-30)) * TEMP_UNIT
    m = rho.sum()
    cold = T < T_COLD
    unst = (T >= T_COLD) & (T < T_WARM)
    return dict(
        T=T, Pk=rho * T,
        vrms=np.sqrt(s["v2"].mean()) * 0.978,
        sig=np.log(np.maximum(rho, 1e-30) / rho.mean()).std(),
        f_cold=rho[cold].sum() / m,
        f_unst=rho[unst].sum() / m,
        rho_max=rho.max(), T_min=np.nanmin(T),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("astronomix")
    ap.add_argument("athenak_dir")
    ap.add_argument("--time", type=float, default=None,
                    help="pick the AthenaK dump nearest this time")
    ap.add_argument("--out", default="code_comparison")
    args = ap.parse_args()

    A = load_astronomix(args.astronomix)
    B = load_athenak(args.athenak_dir, args.time)
    sA, sB = stats(A), stats(B)

    print(f"{'quantity':22s} {'astronomix':>14} {'AthenaK':>14}")
    for k, f in (("time [Myr]", "{:.2f}"), ("v_rms [km/s]", "{:.1f}"),
                 ("sigma_ln_rho", "{:.3f}"), ("f_cold", "{:.4f}"),
                 ("f_unstable", "{:.4f}"), ("rho_max", "{:.2f}"),
                 ("T_min [K]", "{:.1f}")):
        key = {"time [Myr]": None, "v_rms [km/s]": "vrms", "sigma_ln_rho": "sig",
               "f_cold": "f_cold", "f_unstable": "f_unst", "rho_max": "rho_max",
               "T_min [K]": "T_min"}[k]
        va = A["time"] if key is None else sA[key]
        vb = B["time"] if key is None else sB[key]
        print(f"{k:22s} {f.format(va):>14} {f.format(vb):>14}")

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    cols = {"astronomix": "tab:blue", "AthenaK": "tab:red"}
    for s, raw, name in ((sA, A, "astronomix"), (sB, B, "AthenaK")):
        c = cols[name]
        lr = np.log10(raw["rho"] / raw["rho"].mean()).ravel()
        ax[0, 0].hist(lr, bins=120, histtype="step", density=True, color=c,
                      label=f"{raw['label']}  ($\\sigma$={s['sig']:.2f})")
        lt = np.log10(np.maximum(s["T"], 1.0)).ravel()
        ax[0, 1].hist(lt, bins=120, histtype="step", density=True, color=c,
                      label=raw["label"])
        ax[0, 2].hist(np.log10(np.maximum(s["Pk"], 1e-3)).ravel(), bins=120,
                      histtype="step", density=True, color=c, label=raw["label"])
    ax[0, 0].set_xlabel(r"$\log_{10}(\rho/\langle\rho\rangle)$"); ax[0, 0].set_ylabel("PDF")
    ax[0, 0].set_title("density PDF"); ax[0, 0].legend(fontsize=8); ax[0, 0].set_yscale("log")
    for cut in (T_COLD, T_WARM):
        ax[0, 1].axvline(np.log10(cut), color="k", ls=":", lw=0.8)
    ax[0, 1].set_xlabel(r"$\log_{10} T$ [K]"); ax[0, 1].set_title("temperature PDF (dotted = phase cuts)")
    ax[0, 1].set_yscale("log"); ax[0, 1].legend(fontsize=8)
    ax[0, 2].set_xlabel(r"$\log_{10} (P/k)$ [K cm$^{-3}$]"); ax[0, 2].set_title("pressure PDF")
    ax[0, 2].set_yscale("log"); ax[0, 2].legend(fontsize=8)

    for j, (s, raw) in enumerate(((sA, A), (sB, B))):
        h = ax[1, j].hist2d(np.log10(raw["rho"]).ravel(),
                            np.log10(np.maximum(s["T"], 1.0)).ravel(),
                            bins=140, norm=LogNorm(), cmap="viridis")
        ax[1, j].set_xlabel(r"$\log_{10} n$"); ax[1, j].set_ylabel(r"$\log_{10} T$ [K]")
        ax[1, j].set_title(f"{raw['label']}  t={raw['time']:.2f} Myr")
        plt.colorbar(h[3], ax=ax[1, j])

    labels = ["cold\n(T<184K)", "unstable\n(184-5050K)", "warm/hot\n(>5050K)"]
    xs = np.arange(3); w = 0.36
    for off, s, name in ((-w / 2, sA, "astronomix"), (w / 2, sB, "AthenaK")):
        vals = [s["f_cold"], s["f_unst"], 1.0 - s["f_cold"] - s["f_unst"]]
        ax[1, 2].bar(xs + off, vals, w, color=cols[name], label=name)
    ax[1, 2].set_xticks(xs); ax[1, 2].set_xticklabels(labels, fontsize=8)
    ax[1, 2].set_ylabel("mass fraction"); ax[1, 2].set_title("phase mass fractions")
    ax[1, 2].legend(fontsize=8)

    fig.suptitle("astronomix vs AthenaK — driven ISM box with radiative cooling + heating "
                 f"(same initial state, t={A['time']:.1f} vs {B['time']:.1f} Myr)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, args.out + ".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
