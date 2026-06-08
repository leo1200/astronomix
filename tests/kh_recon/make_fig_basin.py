"""Terminal-only mode-2 basin test (T=45, kx2-6, fixed truth, 8 cold inits).
(a) single shooting SCATTERS across basins (clean mode-2: low data misfit but
ic_err 0.05->1.0); soft full-field MS shrinks variance but RAISES the floor (bias),
never reaching the truth basin. (b) ic_err vs data misfit: single's wrong basins
fit the data (multimodality); MS clusters at biased-mediocre."""
from pathlib import Path
import glob, re
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"


def load(tag):
    rows = []
    for f in sorted(glob.glob(str(D / f"basin_{tag}_i*.npz"))):
        m = re.search(rf"basin_{tag}_i(\d+)\.npz", f)
        d = np.load(f, allow_pickle=True)
        rows.append((int(m.group(1)), float(d["lowk_err"]), float(d["final_data"])))
    return np.array(sorted(rows))


ss, ms = load("single"), load("softM2")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
rng = np.random.default_rng(0)
for x, (data, c, lab) in enumerate([(ss, "C3", "single shooting"), (ms, "C0", "soft MS M=2 (full-field interior)")]):
    jit = 0.07 * rng.standard_normal(len(data))
    ax[0].scatter(np.full(len(data), x) + jit, data[:, 1], color=c, s=55, alpha=0.85, zorder=3)
    ax[0].hlines(np.median(data[:, 1]), x - 0.25, x + 0.25, color=c, lw=2)
    ax[0].text(x, 1.08, f"min={data[:,1].min():.2f}\nmed={np.median(data[:,1]):.2f}\nstd={data[:,1].std():.2f}",
               ha="center", fontsize=9, color=c)
    print(f"{lab}: min={data[:,1].min():.3f} med={np.median(data[:,1]):.3f} "
          f"max={data[:,1].max():.3f} std={data[:,1].std():.3f}")
ax[0].axhline(0.1, color="gray", ls=":", label="recovered (ic_err<0.1)")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["single", "soft MS M=2"])
ax[0].set_ylabel("IC recovery error (per cold init)"); ax[0].set_ylim(0, 1.25)
ax[0].set_title("(a) single is multimodal (some restarts -> truth);\nMS shrinks variance but biases the floor up")
ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3, axis="y")

for data, c, lab in [(ss, "C3", "single"), (ms, "C0", "soft MS M=2")]:
    ax[1].scatter(data[:, 2], data[:, 1], color=c, s=55, alpha=0.85, label=lab)
ax[1].set_xscale("log"); ax[1].axhline(0.1, color="gray", ls=":")
ax[1].set_xlabel("final data misfit J_data"); ax[1].set_ylabel("IC recovery error")
ax[1].set_title("(b) loss DISCRIMINATES: truth basin = global min at noise floor;\n"
                "wrong basins 7-17x higher J; MS stuck 2-8x above floor")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
fig.suptitle("Terminal-only KH mode-2 (T=45, kx2-6): truth = unique global min (at noise floor); "
             "wrong basins higher-loss; soft+Adam MS never reaches the min", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "fig_basin_mode2.png", dpi=160); plt.close(fig)
print(f"-> {OUT/'fig_basin_mode2.png'}")
