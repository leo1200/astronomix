"""HEADLINE: connect claims 1+2+3. The observable-subspace ("mode-space")
reconstruction recovers the seed while the SVD information frontier k_rec(T) is
above the control band kmax, and fails once the horizon pushes k_rec below it --
a good method reaches the frontier; none crosses it.

Left:  mode-space lowk recovery error vs horizon T, for Re=2000 and a low Re,
       with the SVD frontier crossing (k_rec(T)=kmax) marked per Re.
Right: method comparison at the short horizon (T=5 t_g) -- full-field single
       shooting (cold / warm) and Tikhonov fail; mode-space recovers."""
from pathlib import Path
import glob, re
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"
KMAX = 6


def load_modes(re_tag):
    rows = []
    for f in sorted(glob.glob(str(D / f"modes_T*_Re{re_tag}.npz"))):
        m = re.search(r"modes_T(\d+)_Re", f)
        if not m:
            continue
        d = np.load(f, allow_pickle=True)
        rows.append((int(m.group(1)), float(d["lowk_err"]), float(d["full_err"]),
                     int(d["stopped_at"]), float(d["runtime"])))
    rows.sort()
    return np.array(rows) if rows else np.empty((0, 5))


def krec_cross(svd_npz):
    """horizon (t_g) where SVD k_rec drops below KMAX, by linear interp."""
    if not Path(svd_npz).exists():
        return None
    d = np.load(svd_npz)
    Tg = np.asarray(d["T_g"], float); kr = np.asarray(d["krec"], float)
    below = np.where(kr < KMAX)[0]
    if not len(below):
        return None
    i = below[0]
    if i == 0:
        return float(Tg[0])
    # interp between i-1 (>=KMAX) and i (<KMAX)
    t0, t1, k0, k1 = Tg[i - 1], Tg[i], kr[i - 1], kr[i]
    return float(t0 + (KMAX - k0) * (t1 - t0) / (k1 - k0))


fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

for re_tag, c, svd in [("2000", "C0", D / "frontier_svd.npz"),
                       ("250", "C3", D / "frontier_svd_Re250.npz")]:
    r = load_modes(re_tag)
    if not len(r):
        continue
    ax[0].semilogy(r[:, 0], r[:, 1], "o-", color=c, label=f"mode-space recovery, Re={re_tag}")
    tc = krec_cross(svd)
    if tc:
        ax[0].axvline(tc, color=c, ls="--", alpha=0.7,
                      label=f"SVD frontier k_rec=kmax (Re={re_tag}, T~{tc:.0f})")
    print(f"Re={re_tag}: " + "  ".join(f"T{int(x[0])}:lk={x[1]:.3f}" for x in r) +
          (f"  frontier_cross~{tc:.0f}" if tc else ""))
ax[0].axhline(0.1, color="gray", ls=":", label="recovered (lowk<0.1)")
ax[0].axhline(1.0, color="k", lw=0.5)
ax[0].set_xlabel("horizon T / t_g"); ax[0].set_ylabel("low-k recovery error")
ax[0].set_title("(a) optimization frontier (recovery fails) is tighter than the\n"
                "information frontier; lower Re pushes it out toward k_rec=kmax")
ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3, which="both")

# method comparison at T=5
methods = []
mm = D / "modes_T5_Re2000.npz"
if mm.exists():
    methods.append(("mode-space\n(observable subspace)", float(np.load(mm)["lowk_err"]), "C0"))
ff = D / "hz_single_T5_k6.npz"
if ff.exists():
    methods.append(("full-field\ncold start", float(np.load(ff, allow_pickle=True)["lowk_err"]), "C3"))
fw = D / "diag_warm03_T5.npz"
if fw.exists():
    methods.append(("full-field\nwarm (0.3)", float(np.load(fw, allow_pickle=True)["lowk_err"]), "C1"))
# best Tikhonov at T=5
tik = []
for f in glob.glob(str(D / "alpha_T5_a*.npz")):
    tik.append(float(np.load(f, allow_pickle=True)["lowk_err"]))
if tik:
    methods.append(("full-field\nTikhonov (best)", min(tik), "C2"))

if methods:
    labels = [m[0] for m in methods]; vals = [m[1] for m in methods]; cols = [m[2] for m in methods]
    ax[1].bar(range(len(vals)), vals, color=cols)
    ax[1].axhline(1.0, color="k", lw=0.5, label="no recovery")
    ax[1].axhline(0.1, color="gray", ls=":", label="recovered")
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("low-k recovery error"); ax[1].set_ylim(0, 1.8)
    ax[1].set_title("(b) T=5 t_g: only observable-subspace control recovers")
    ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_recovery.png", dpi=160); plt.close(fig)
print(f"-> {OUT/'fig_recovery.png'}")
