"""
Assemble the two publication figures for the TGV single- vs multiple-shooting
study. PURE PLOTTING -- reads the cached .npz arrays, no GPU, no solver.

  fig1_gradient_mechanism.png
      Single panel. The scale-resolved adjoint sensitivity vs time-within-
      window: single shooting blows up monotonically toward t=0; multiple
      shooting is a sawtooth that resets at every segment boundary. Overlaid:
      the tangent-linear FORWARD growth G of a large-scale perturbation, which
      sets the envelope -- the SS curve rides G(remaining horizon) and each MS
      segment is capped at G(T/m). (De-singularized: the window starts from an
      evolved state, not the pristine single-mode IC, so the teeth follow one
      clean envelope.)

  fig2_money_plot.png
      Final large-scale recovery error vs observation horizon, single shooting
      (m=1) vs multiple shooting (m=4).

Run: python make_publication_figures.py
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "figures"


def gradient_mechanism_panel():
    f = DATA / "gradient_mechanism_adjoint.npz"
    if not f.exists():
        print(f"missing {f} -- run gradient_mechanism.py first")
        return
    d = np.load(f)
    t = np.asarray(d["t_grid"])
    ss = np.asarray(d["ss"])
    T_win = float(t[-1])
    ms_keys = sorted(k for k in d.files if re.match(r"ms_m\d+$", k))
    ms = {int(re.match(r"ms_m(\d+)", k).group(1)): np.asarray(d[k]) for k in ms_keys}
    has_tlm = "tlm_highk" in d.files
    tlm = np.asarray(d["tlm_highk"]) if has_tlm else None

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # --- adjoint curves ---
    ax.semilogy(t, ss, "o-", ms=4, color="C3", lw=1.8, label="single shooting (m=1)")
    markers = {4: "s-", 8: "^-"}
    colors = {4: "C0", 8: "C1"}
    for m in sorted(ms):
        ax.semilogy(t, ms[m], markers.get(m, "d-"), ms=4, color=colors.get(m, "C2"),
                    lw=1.4, label=f"multiple shooting (m={m})")

    # segment-boundary guides for the largest m
    m_max = max(ms) if ms else 0
    for i in range(1, m_max):
        ax.axvline(i * T_win / m_max, color="gray", ls=":", lw=0.5, alpha=0.4)

    ax.set_xlabel(r"time within window  $t / t_c$")
    ax.set_ylabel("adjoint high-$k$ band energy")
    ax.set_title("Adjoint high-$k$ sensitivity vs time within the window")
    ax.set_xlim(0, T_win)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = OUT / "fig1_gradient_mechanism.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"-> {out}")


def money_plot():
    files = sorted(DATA.glob("ss_vs_ms_Tobs*.npz"))
    if not files:
        print("no ss_vs_ms_Tobs*.npz -- run the recovery sweep first")
        return
    curves = {}
    for fp in files:
        T = float(re.search(r"Tobs([0-9.]+)\.npz", fp.name).group(1))
        d = np.load(fp)
        for key in d.files:
            mo = re.match(r"err_m(\d+)$", key)
            if mo:
                curves.setdefault(int(mo.group(1)), []).append((T, float(d[key][-1])))

    fig, ax = plt.subplots(figsize=(8, 6))
    style = {1: ("o-", "C3", "single shooting (m=1)"),
             4: ("s-", "C0", "multiple shooting (m=4)")}
    for m in sorted(curves):
        pts = sorted(curves[m])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        mk, c, lab = style.get(m, ("d-", "C2", f"m={m}"))
        ax.plot(xs, ys, mk, color=c, ms=6, lw=1.8, label=lab)
    ax.set_xlabel(r"observation horizon  $T_{\rm obs} / t_c$")
    ax.set_ylabel("final large-scale recovery error")
    ax.set_title("Recovery of the TGV initial condition: single vs multiple shooting")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = OUT / "fig2_money_plot.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gradient_mechanism_panel()
    money_plot()
