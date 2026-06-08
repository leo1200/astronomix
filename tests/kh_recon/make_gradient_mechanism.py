"""
Mode-1 segmentation fix, from the Stage-2 tangent-gain data (no extra compute).

Single shooting back-propagates across the whole window, so its gradient grows
as sigma_max(T) ~ e^{lambda T}. Multiple shooting with M segments only ever
back-propagates across one segment of length T/M, so its per-segment gradient is
sigma_max(T/M) ~ e^{lambda T/M} -- exponentially smaller. We read sigma_max(T)
from frontier.npz (log-linear in T) and plot single vs M=2,4,8.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = np.load(DATA / "frontier.npz")
    Tg = np.asarray(d["T_g"], float)
    smax = np.asarray(d["smax"], float)
    # log-linear fit sigma_max ~ exp(lam*Tg)
    lam, c = np.polyfit(Tg, np.log(smax), 1)
    sig = lambda t: np.exp(lam * t + c)

    Tplot = np.linspace(Tg.min(), Tg.max(), 100)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.semilogy(Tg, smax, "ko", label="measured $\\sigma_{max}(T)$")
    ax.semilogy(Tplot, sig(Tplot), "C3-", lw=2, label="single shooting $\\sim e^{\\lambda T}$")
    for M, c_ in [(2, "C0"), (4, "C2"), (8, "C1")]:
        ax.semilogy(Tplot, sig(Tplot / M), c_ + "--", lw=1.8,
                    label=f"multiple shooting M={M} (per-seg $\\sim e^{{\\lambda T/{M}}}$)")
    ax.set_xlabel("total window T / t_g")
    ax.set_ylabel("back-prop gradient gain")
    ax.set_title(f"Mode-1 fix by segmentation ($\\lambda={lam:.3f}/t_g$)\n"
                 "single shooting rides $e^{\\lambda T}$; multiple shooting caps at $e^{\\lambda T/M}$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "fig_C_gradient_mechanism.png", dpi=170)
    plt.close(fig)
    print(f"-> {OUT / 'fig_C_gradient_mechanism.png'}  (lambda={lam:.3f}/t_g)")
    for M in (1, 2, 4, 8):
        print(f"  M={M}: per-seg gain at T=160 t_g = {sig(160/M):.3e}")


if __name__ == "__main__":
    main()
