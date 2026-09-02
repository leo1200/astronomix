"""Side-by-side animation of the dynamo: two codes' fields and their spectra.

Three panels sharing one clock:

* mid-plane magnetic energy of AthenaPK PLM+VL2 (2nd order, GLM cleaning),
* the same for astronomix WENO5+CT (5th order, constrained transport), and
* both magnetic spectra, with the kinetic spectrum behind them for scale.

The two slices share a single colour scale at every frame, so the panels are
directly comparable, and that scale follows the CT run's instantaneous maximum
-- the point of the animation is the *structure* and the relative amplitude, and
a fixed scale over six decades of dynamo growth would show a black frame
followed by a white one. The current normalisation is printed on each frame so
the growth is not hidden by it.

    python make_dynamo_movie.py --data data/anim
"""

# general
import argparse
import sys
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_metrics import E_MAG, E_V, load_runs, spectra_of

HERE = Path(__file__).resolve().parent

#: Decades of magnetic energy shown in the slice panels below the frame maximum.
SLICE_DECADES = 3.0


def _resample(series, t_src, t_dst):
    """Nearest-in-time frame of ``series`` for each time in ``t_dst``.

    The two codes are dumped on their own cadences and neither is guaranteed to
    hit the other's times, so the animation runs on a common clock and each
    panel shows its own nearest snapshot. Nearest-neighbour rather than
    interpolation: averaging two turbulent fields half a crossing time apart
    would produce a structure neither code ever had.
    """
    idx = np.abs(np.asarray(t_dst)[:, None] - np.asarray(t_src)[None, :]).argmin(1)
    return series[idx], np.asarray(t_src)[idx]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "data" / "anim"))
    p.add_argument("--figures", default=str(HERE / "figures"))
    p.add_argument("--out", default="dynamo_side_by_side.gif")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--stride", type=int, default=2,
                   help="keep every Nth frame. The saturated phase is visually "
                        "repetitive and a 121-frame 256^3 gif is 31 MB, so the "
                        "default halves it at no cost to the story.")
    p.add_argument("--colors", type=int, default=96,
                   help="palette size for the final quantisation (0 disables)")
    p.add_argument("--n", type=int, default=256)
    args = p.parse_args()

    runs = {("astronomix" if str(r["code"]) == "astronomix"
             else str(r["scheme_key"])): r
            for r in load_runs(args.data, skip=("smoke",))
            if int(r["N"]) == args.n and "EB_slice_series" in r}
    missing = {"astronomix", "plm"} - set(runs)
    if missing:
        raise SystemExit(f"no {args.n}^3 run with a slice series for: "
                         f"{sorted(missing)} (run with --slice-series)")

    left, right = runs["plm"], runs["astronomix"]
    # Common clock: the coarser of the two cadences, over the overlap.
    t_lo = max(float(r["t_over_tc"][0]) for r in (left, right))
    t_hi = min(float(r["t_over_tc"][-1]) for r in (left, right))
    n_frames = min(len(left["t_over_tc"]), len(right["t_over_tc"]))
    clock = np.linspace(t_lo, t_hi, n_frames)[::max(1, args.stride)]

    panels = []
    for run in (left, right):
        sl, t_act = _resample(np.asarray(run["EB_slice_series"]),
                              np.asarray(run["t_over_tc"]), clock)
        spec = spectra_of(run, deconvolve=False)
        sp, _ = _resample(spec, np.asarray(run["t_over_tc"]), clock)
        ratio, _ = _resample(np.asarray(run["E_B"])
                             / np.maximum(np.asarray(run["E_K"]), 1e-30),
                             np.asarray(run["t_over_tc"]), clock)
        panels.append(dict(slices=sl, spectra=sp, t=t_act, ratio=ratio,
                           label=str(run["label"]),
                           n_shell=np.asarray(run["n_shell"], dtype=float)))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    ims, curves_B, curves_v = [], [], []
    colours = ("#d62728", "#1f77b4")

    for ax, panel, colour in zip(axes[:2], panels, colours):
        im = ax.imshow(np.log10(np.maximum(panel["slices"][0], 1e-300)).T,
                       origin="lower", cmap="inferno",
                       extent=(0, 1, 0, 1), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel["label"], fontsize=11, color=colour)
        ims.append(im)

    ax = axes[2]
    for panel, colour in zip(panels, colours):
        n = panel["n_shell"]
        curves_B.append(ax.loglog(n[1:], np.maximum(panel["spectra"][0][E_MAG][1:],
                                                    1e-300),
                                  color=colour, lw=2.0,
                                  label=panel["label"])[0])
        curves_v.append(ax.loglog(n[1:], np.maximum(panel["spectra"][0][E_V][1:],
                                                    1e-300),
                                  color=colour, lw=1.0, ls=":", alpha=0.55)[0])
    ax.set_xlabel(r"mode number $n = kL/2\pi$")
    ax.set_ylabel(r"$E(n)$")
    ax.set_title("magnetic (solid) and kinetic (dotted) spectra", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(1, args.n / 2)
    # Fixed over the whole animation: the growth is the story, so the spectrum
    # panel must not rescale under it. Nine decades is enough to hold the whole
    # dynamo -- the seed itself is a single shell and would otherwise stretch the
    # axis over five decades of empty space.
    all_B = np.concatenate([p["spectra"][:, E_MAG, 1:].ravel() for p in panels])
    all_v = np.concatenate([p["spectra"][:, E_V, 1:].ravel() for p in panels])
    top = max(all_B.max(), all_v.max())
    ax.set_ylim(top * 1e-9, top * 3.0)

    caption = fig.text(0.5, 0.965, "", ha="center", fontsize=11)
    scale_note = fig.text(0.5, 0.03, "", ha="center", fontsize=8, color="0.35")

    def update(i):
        # One colour scale for both panels, following the CT run, so the two
        # slices can be compared at a glance rather than each auto-scaling.
        vmax = np.log10(max(panels[1]["slices"][i].max(), 1e-300))
        for im, panel in zip(ims, panels):
            im.set_data(np.log10(np.maximum(panel["slices"][i], 1e-300)).T)
            im.set_clim(vmax - SLICE_DECADES, vmax)
        for cB, cv, panel in zip(curves_B, curves_v, panels):
            cB.set_ydata(np.maximum(panel["spectra"][i][E_MAG][1:], 1e-300))
            cv.set_ydata(np.maximum(panel["spectra"][i][E_V][1:], 1e-300))
        caption.set_text(
            f"$t / t_{{\\rm cross}} = {clock[i]:5.1f}$      mid-plane magnetic "
            f"energy, ${args.n}^3$      $E_B/E_K$ = "
            f"{panels[0]['ratio'][i]:.1e} (PLM), "
            f"{panels[1]['ratio'][i]:.1e} (CT)")
        scale_note.set_text(
            f"slice colour scale: $\\log_{{10}} E_B$ over "
            f"[{vmax - SLICE_DECADES:.1f}, {vmax:.1f}], shared by both panels "
            f"and following the CT run's frame maximum")
        return ims + curves_B + curves_v + [caption, scale_note]

    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    anim = FuncAnimation(fig, update, frames=len(clock), blit=False)
    out = Path(args.figures) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    if args.colors:
        # Matplotlib writes full-colour frames; the slices only ever use one
        # colormap, so a small palette is lossless in practice and much smaller.
        from PIL import Image
        src = Image.open(out)
        frames = []
        for i in range(src.n_frames):
            src.seek(i)
            frames.append(src.convert("RGB").quantize(colors=args.colors,
                                                      dither=Image.NONE))
        src.close()
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out}  ({len(clock)} frames, "
          f"t/t_cross {clock[0]:.2f} to {clock[-1]:.2f}, "
          f"{out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
