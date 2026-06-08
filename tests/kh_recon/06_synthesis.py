"""
KH reconstruction -- Stage 5: synthesis figures.

Figure A (headline): per-kx recovery error for single / hard-MS / soft-MS,
overlaid on the information frontier (recoverable band); plus a truth / SS /
soft-MS seed-field triptych.
Figure E: segment-size sweep -- recovery error and iteration count vs M.

Reads frontier.npz and ms_*.npz from data/. Robust to missing files.
"""

import glob, re, os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_ms():
    runs = {}
    for f in glob.glob(str(DATA / "ms_*.npz")):
        d = np.load(f, allow_pickle=True)
        runs[Path(f).stem] = d
    return runs


def fig_A(frontier, runs, Trec):
    # pick the single/hard/soft runs at horizon Trec (first seed found)
    def pick(mode):
        for k, d in runs.items():
            if str(d["mode"]) == mode and float(d["t_g"]) == Trec:
                return d
        return None
    ss, hard, soft = pick("single"), pick("hard"), pick("soft")
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 4)
    ax = fig.add_subplot(gs[0, :2])
    if frontier is not None:
        kx = frontier["kx"]
        sg = frontier.get(f"sigkx_{Trec}", None)
        if sg is None:  # nearest available horizon
            avail = [float(t) for t in frontier["T_g"]]
            near = min(avail, key=lambda t: abs(t - Trec))
            sg = frontier[f"sigkx_{near}"]
        recoverable = sg / sg.max()
        ax.fill_between(kx, 0, 1, where=recoverable > 1e-2, color="green", alpha=0.08,
                        transform=ax.get_xaxis_transform(), label="recoverable band")
        axr = ax.twinx()
        axr.loglog(kx, recoverable, "k:", alpha=0.6, label=r"$\sigma(k_x)/\sigma_{max}$")
        axr.set_ylabel("recoverability gain"); axr.set_ylim(1e-4, 2)
    for d, c, lab in [(ss, "C3", "single shooting"), (hard, "C1", "hard MS"),
                      (soft, "C0", "soft MS")]:
        if d is not None:
            ax.loglog(d["kx"][1:], d["errk"][1:], "o-", color=c, ms=3, label=lab)
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_xlabel("kx"); ax.set_ylabel("per-kx recovery error")
    ax.set_title(f"(A) recovery vs k (T={Trec:.0f} t_g)"); ax.legend(fontsize=8, loc="lower right")

    # seed triptych
    for j, (d, lab) in enumerate([(soft or ss, "truth"), (ss, "single shooting"),
                                  (soft, "soft MS")]):
        axp = fig.add_subplot(gs[0, 2 + j]) if j < 2 else None
    # render truth / SS / soft fields if available
    panel = [(ss, "single shooting"), (soft, "soft MS")]
    base = ss if ss is not None else soft
    axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
    if base is not None:
        truth = base["truth"][1]
        vmax = np.percentile(np.abs(truth), 99)
        for axp, (d, lab) in zip(axes, panel):
            if d is not None:
                axp.imshow(np.asarray(d["rec"][1]).T, origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            axp.set_title(lab); axp.set_xticks([]); axp.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "fig_A_recovery_vs_k.png", dpi=160); plt.close(fig)
    print(f"-> {OUT / 'fig_A_recovery_vs_k.png'}")


def fig_E(runs, Trec):
    # soft-MS segment sweep: error + iters(proxy via runtime) vs M
    pts = []
    for k, d in runs.items():
        if str(d["mode"]) == "soft" and float(d["t_g"]) == Trec:
            pts.append((int(d["M"]), float(d["lowk_err"]), float(d["runtime"])))
    if len(pts) < 2:
        print("not enough soft-MS segment points for fig E"); return
    pts.sort()
    Ms = [p[0] for p in pts]; err = [p[1] for p in pts]; rt = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ms, err, "o-", color="C0", label="low-k recovery error")
    ax.set_xlabel("segments M"); ax.set_ylabel("recovery error", color="C0")
    ax2 = ax.twinx(); ax2.plot(Ms, rt, "s--", color="C2", label="wall time (s)")
    ax2.set_ylabel("wall time (s)", color="C2")
    ax.set_title(f"(E) segment sweep (T={Trec:.0f} t_g, soft MS)")
    fig.tight_layout(); fig.savefig(OUT / "fig_E_segment_sweep.png", dpi=160); plt.close(fig)
    print(f"-> {OUT / 'fig_E_segment_sweep.png'}")


def main():
    Trec = float(os.environ.get("KH_TREC", 80))
    frontier = np.load(DATA / "frontier.npz") if (DATA / "frontier.npz").exists() else None
    runs = load_ms()
    print(f"loaded {len(runs)} MS runs; frontier={'yes' if frontier is not None else 'no'}")
    if runs:
        fig_A(frontier, runs, Trec)
        fig_E(runs, Trec)


if __name__ == "__main__":
    main()
