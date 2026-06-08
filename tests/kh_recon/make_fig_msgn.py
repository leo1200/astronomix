"""THE multiple-shooting money figure. Long-horizon genuine mode-2 (T=80, kx2-6,
identical cold inits): single-shooting Gauss-Newton recovers 0/12; constrained
GN multiple shooting (M=4, feasible interiors) recovers 3/12 and is uniformly
closer to truth. A real (partial) basin enlargement that single shooting
structurally cannot achieve."""
from pathlib import Path
import glob, re
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"


def load(pat):
    out = {}
    for f in glob.glob(str(D / pat)):
        out[int(re.search(r"_i(\d+)\.npz", f).group(1))] = float(np.load(f, allow_pickle=True)["ic_err"])
    return out


fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k, (T, panel) in enumerate([(80, ax[0]), (120, ax[1])]):
    sg = load(f"gnh_T{T}_i*.npz")
    ms = load(f"msgn_feasible_M4_T{T}_i*.npz")
    inits = sorted(set(sg) & set(ms))
    x = np.arange(len(inits))
    panel.bar(x - 0.2, [min(sg[i], 22) for i in inits], 0.4, color="C3", label="single shooting (GN)")
    panel.bar(x + 0.2, [min(ms[i], 22) for i in inits], 0.4, color="C0", label="constrained GN-MS (M=4)")
    panel.axhline(0.1, color="gray", ls=":", label="recovered (<0.1)")
    panel.set_yscale("log")
    panel.set_xticks(x); panel.set_xticklabels([str(i) for i in inits], fontsize=7)
    panel.set_xlabel("cold restart"); panel.set_ylabel("IC recovery error (log)")
    sr = sum(sg[i] < 0.1 for i in inits); mr = sum(ms[i] < 0.1 for i in inits)
    panel.set_title(f"T={T} t_g: single {sr}/{len(inits)} vs GN-MS {mr}/{len(inits)} recovered\n"
                    f"best: single {min(sg.values()):.2f}, GN-MS {min(ms.values()):.3f}")
    panel.legend(fontsize=8)
    print(f"T={T}: single {sr}/{len(inits)} (best {min(sg.values()):.3f}) | "
          f"GN-MS {mr}/{len(inits)} (best {min(ms.values()):.3f})")
fig.suptitle("Constrained Gauss-Newton MULTIPLE SHOOTING vs SINGLE SHOOTING (genuine long-T mode-2): "
             "MS recovers cold starts single shooting cannot", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig_msgn_vs_single.png", dpi=160); plt.close(fig)
print(f"-> {OUT/'fig_msgn_vs_single.png'}")
