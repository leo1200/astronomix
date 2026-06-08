"""
Basin / mode-2 figure from the warm-start reconstruction log: low-k recovery
error vs iteration for single shooting vs soft multiple shooting, both started
near the truth. Single shooting is pushed out of the truth basin; soft MS (if
the claim holds) stays.
"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
LOG = HERE / (sys.argv[1] if len(sys.argv) > 1 else "run_stage4_warm.log")
OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)

pat = re.compile(r"(single|soft)\s+M=\d+.*it=\s*(\d+):.*lowk_err=([\d.]+)")
series = {"single": ([], []), "soft": ([], [])}
for line in LOG.read_text().splitlines():
    m = pat.search(line)
    if m:
        meth, it, lk = m.group(1), int(m.group(2)), float(m.group(3))
        series[meth][0].append(it); series[meth][1].append(lk)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
style = {"single": ("C3", "single shooting"), "soft": ("C0", "soft multiple shooting")}
for meth, (its, lks) in series.items():
    if its:
        c, lab = style[meth]
        ax.plot(its, lks, "o-", color=c, ms=3, label=lab)
if series["single"][1]:
    ax.axhline(series["single"][1][0], color="k", ls=":", lw=1, label="warm-start init")
ax.set_xlabel("optimization iteration")
ax.set_ylabel("low-k recovery error  ||lowpass(rec-truth)||/||lowpass(truth)||")
_s = series["single"][1]; _f = series["soft"][1]
_verdict = ("soft MS holds" if (_f and _s and _f[-1] < 0.6 * _s[-1] and _f[-1] < 0.7)
            else "both pushed out (soft MS untuned does not rescue)")
ax.set_title("Warm-start basin test near truth (mode 2):\n"
             f"single shooting pushed out of the truth basin; {_verdict}")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_D_basin.png", dpi=170); plt.close(fig)
print(f"-> {OUT/'fig_D_basin.png'}")
for meth, (its, lks) in series.items():
    if lks:
        print(f"  {meth}: lowk_err {lks[0]:.2f} -> {lks[-1]:.2f}")
