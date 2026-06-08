"""Adam vs Gauss-Newton single shooting, same 8 cold inits (T=45, kx2-6). The
"mode-2 basins" that trapped first-order Adam (ic_err scatter 0.05->1.0) DISSOLVE
under second-order GN (all inits -> noise floor, ic_err<0.05): they were an
optimizer artifact, not genuine multimodality."""
from pathlib import Path
import glob, re
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"


def load(pat, key="lowk_err"):
    out = {}
    for f in sorted(glob.glob(str(D / pat))):
        m = re.search(r"_i(\d+)\.npz", f)
        d = np.load(f, allow_pickle=True)
        out[int(m.group(1))] = float(d[key]) if key in d else float(d["ic_err"])
    return out


adam = load("basin_single_i*.npz", "lowk_err")
gn = load("gn_single_i*.npz", "ic_err")
inits = sorted(set(adam) & set(gn))
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(inits))
ax.bar(x - 0.2, [adam[i] for i in inits], 0.4, label="Adam (first-order)", color="C3")
ax.bar(x + 0.2, [gn[i] for i in inits], 0.4, label="Gauss-Newton (second-order)", color="C0")
ax.axhline(0.1, color="gray", ls=":", label="recovered (ic_err<0.1)")
ax.axhline(1.0, color="k", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([f"init {i}" for i in inits], fontsize=8)
ax.set_ylabel("IC recovery error"); ax.set_xlabel("cold restart")
ax.set_title("Single shooting, same inits (T=45, kx2-6):\n"
             "Adam's 'mode-2 basins' DISSOLVE under Gauss-Newton (all -> floor)")
ax.legend(fontsize=9)
print("Adam:", {i: round(adam[i], 3) for i in inits})
print("GN:  ", {i: round(gn[i], 3) for i in inits})
print(f"Adam: {sum(v<0.1 for v in adam.values())}/{len(adam)} recovered; "
      f"GN: {sum(v<0.1 for v in gn.values())}/{len(gn)} recovered")
fig.tight_layout(); fig.savefig(OUT / "fig_gn_vs_adam.png", dpi=160); plt.close(fig)
print(f"-> {OUT/'fig_gn_vs_adam.png'}")
