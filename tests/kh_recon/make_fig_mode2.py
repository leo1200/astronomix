"""Mode-2 advantage figure. Across cold restarts, single shooting should SCATTER
across basins (high variance, some restarts fit the data with the wrong IC = the
mode-2 fingerprint), while Path-A reduced-order multiple shooting collapses to the
truth basin. (a) ic_err per restart, single vs redms; (b) ic_err vs final data
misfit (loss-coincides-but-IC-diverges = multimodality)."""
from pathlib import Path
import glob, re
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"


def load(mode):
    rows = []
    for f in sorted(glob.glob(str(D / f"mode2_{mode}_s*.npz"))):
        m = re.search(rf"mode2_{mode}_s(\d+)\.npz", f)
        if not m:
            continue
        d = np.load(f, allow_pickle=True)
        rows.append((int(m.group(1)), float(d["ic_err"]), float(d["final_data"])))
    return np.array(sorted(rows)) if rows else np.empty((0, 3))


ss, rm = load("single"), load("redms")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

for x, (data, c, lab) in enumerate([(ss, "C3", "single shooting"), (rm, "C0", "reduced-order MS (Path A)")]):
    if not len(data):
        continue
    jit = 0.08 * np.random.default_rng(0).standard_normal(len(data))
    ax[0].scatter(np.full(len(data), x) + jit, data[:, 1], color=c, s=45, alpha=0.8, zorder=3)
    med = np.median(data[:, 1]); sd = np.std(data[:, 1])
    ax[0].hlines(med, x - 0.25, x + 0.25, color=c, lw=2)
    ax[0].text(x, 1.55, f"med={med:.2f}\nstd={sd:.2f}", ha="center", fontsize=9, color=c)
    print(f"{lab}: n={len(data)} ic_err median={med:.3f} std={sd:.3f} "
          f"min={data[:,1].min():.3f} max={data[:,1].max():.3f}")
ax[0].axhline(0.1, color="gray", ls=":", label="recovered (ic_err<0.1)")
ax[0].axhline(1.0, color="k", lw=0.5)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["single", "reduced-order MS"])
ax[0].set_ylabel("IC recovery error (per cold restart)"); ax[0].set_ylim(0, 1.75)
ax[0].set_title("(a) basin scatter: single is ~unimodal (recovers); reduced-MS biased high")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")

for data, c, lab in [(ss, "C3", "single"), (rm, "C0", "reduced-order MS")]:
    if len(data):
        ax[1].scatter(data[:, 2], data[:, 1], color=c, s=45, alpha=0.8, label=lab)
ax[1].set_xscale("log")
ax[1].axhline(0.1, color="gray", ls=":")
ax[1].set_xlabel("final data misfit J_data"); ax[1].set_ylabel("IC recovery error")
ax[1].set_title("(b) single fits to ~1e-6 & recovers; reduced-MS floored by POD truncation")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
fig.suptitle("KH vortex-pairing (T*=65 t_g, 4-DOF seed): NO mode-2 advantage -- "
             "single is unimodal; reduced-order MS biased by POD truncation", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "fig_mode2.png", dpi=160); plt.close(fig)
print(f"-> {OUT/'fig_mode2.png'}")
