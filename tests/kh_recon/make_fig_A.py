"""Figure A: per-kx recovery error, single vs soft-MS (warm2, T=50), overlaid
with the information context (KH marginal-stability frontier k_rec ~ 1/h)."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"
ss = np.load(D / "ms_single_M1_T50_warm2.npz", allow_pickle=True)
sf = np.load(D / "ms_soft_M4_T50_warm2.npz", allow_pickle=True)
fr = np.load(D / "frontier.npz")

# frontier at T=60 t_g (nearest); h there -> KH marginal k_rec ~ 1/(2 pi h)
h60 = float(fr["h"][list(fr["T_g"]).index(60.0)]) if 60.0 in list(fr["T_g"]) else 0.018
k_rec = 1.0 / (2 * np.pi * h60)

fig, ax = plt.subplots(figsize=(8, 5.5))
for d, c, lab in [(ss, "C3", f"single shooting (lowk_err={float(ss['lowk_err']):.2f})"),
                  (sf, "C0", f"soft MS M=4 (lowk_err={float(sf['lowk_err']):.2f})")]:
    kx = d["kx"]; e = d["errk"]
    ax.semilogx(kx[1:], e[1:], "o-", color=c, ms=3, label=lab)
ax.axhline(0.15, color="gray", ls=":", label="warm-start init error")
ax.axhline(1.0, color="k", lw=0.5)
ax.axvspan(0.8, k_rec, color="green", alpha=0.08)
ax.axvline(k_rec, color="green", ls="--", label=f"KH frontier $k_{{rec}}\\sim1/h$ ({k_rec:.0f})")
ax.set_xlabel("streamwise wavenumber kx"); ax.set_ylabel("per-kx recovery error")
ax.set_title("(A) Recovery vs k (warm start, T=50 t_g)\n"
             "both methods pushed out of the truth basin (mode 2); soft MS untuned does not rescue")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "fig_A_recovery_vs_k.png", dpi=170); plt.close(fig)
print(f"-> {OUT/'fig_A_recovery_vs_k.png'}  k_rec~{k_rec:.1f}")
print(f"  single lowk_err={float(ss['lowk_err']):.3f}, soft lowk_err={float(sf['lowk_err']):.3f}")
