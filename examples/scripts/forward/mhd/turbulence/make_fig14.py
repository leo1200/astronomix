"""
The paper's ``turbulence_slices`` figure (the analogue of HOW-MHD, Seo & Ryu
2023, Fig. 14): mid-plane slices of the magnetic energy log10 E_B of the
driven isothermal-MHD turbulence runs of ``paper_turbulence.py``, 2x2:

  rows    : ISM (M_turb ~ 10, beta_p = 0.1) / ICM (M_turb ~ 0.5, beta_p = 1e6)
  columns : the lowest and the highest resolution present in the
            script-relative data/ directory (paper: 256^3 and 512^3)

The slice is the one the run kept, from its last alive snapshot (normally
the final one); the title gives the mass-weighted M_S = sqrt(2 E_K) / a of that
snapshot. E_B from the default v_rms ~ 1 normalisation is converted to
HOW-MHD's a = 1 normalisation by the exact similarity scaling
E_B^paper = M_turb^2 E_B, so the colour ranges match theirs (ISM 0..3,
ICM -6..0).

Writes figures/turbulence_slices.{svg,pdf,png}. ``--suffix S`` reads the runs
tagged ``<case>_N<N>S`` instead (e.g. ``_uct0.03``, the runs with the upwind EMF
dissipation).
"""

# general
import sys
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix.plotting_helpers.paper_style import (
    SEQUENTIAL_COLORMAP,
    TEXT_WIDTH,
)

# astronomix functions
from astronomix.plotting_helpers.paper_style import (
    attach_colorbar,
    reserve_colorbar_space,
    use_paper_style,
)


#: Script-relative directories (data/ is shared with paper_turbulence.py).
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

#: Suffix of the run tags to draw (``--suffix``; empty: the plain runs).
RUN_SUFFIX = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
FIGURE_DIR = SCRIPT_DIR / "figures"

#: Resolutions looked for, low to high (paper: 256 and 512).
RESOLUTIONS = (128, 256, 512)

#: Rows: case, row label, colour range of log10 E_B (HOW-MHD's).
ROWS = (
    ("ISM", "WENO ISM", (0.0, 3.0)),
    ("ICM", "WENO ICM", (-6.0, 0.0)),
)

#: The unit box, centred, as the image extent.
EXTENT = [-0.5, 0.5, -0.5, 0.5]


def load_run(case, resolution):
    """The reduced data of one run, or ``None`` if its npz is absent."""
    path = DATA_DIR / f"paper_{case}_N{resolution}{RUN_SUFFIX}.npz"
    return dict(np.load(path, allow_pickle=True)) if path.exists() else None


def lowest_and_highest_resolution(case):
    """The runs for the two columns: the lowest and highest resolution present."""
    runs = [run for run in (load_run(case, resolution) for resolution in RESOLUTIONS) if run is not None]
    if not runs:
        return None, None
    if len(runs) == 1:
        return runs[0], None
    return runs[0], runs[-1]


def log_magnetic_energy_paper_units(run):
    """
    log10 E_B of the kept slice in HOW-MHD's a = 1 normalisation.

    A run in the v_rms ~ 1 normalisation has velocities and fields smaller by
    M_turb than the same flow at a = 1, so E_B^paper = M_turb^2 E_B (the aimed
    M_turb, which sets the normalisation); a run made with ``--norm paper`` is
    already in those units.

    Args:
        run: The reduced data of a run.

    Returns:
        The slice of log10 E_B (NaN where E_B <= 0).
    """
    magnetic_energy = np.asarray(run["EB_slice"], dtype=np.float64)
    normalisation = str(run.get("norm", "vrms1"))
    scale = float(run["mturb_aim"]) ** 2 if normalisation == "vrms1" else 1.0
    magnetic_energy = np.where(magnetic_energy > 0, magnetic_energy, np.nan) * scale
    return np.log10(magnetic_energy)


def mach_numbers(run, mass_weighted=True):
    """
    The turbulent Mach number per snapshot: mass-weighted, sqrt(2 E_K / <rho>) / a
    (<rho> = 1), or volume-weighted, v_rms / a. The volume-weighted value of the
    supersonic runs is dominated by the fast, near-vacuum cells; the
    mass-weighted one is what the driving sets. Older reduced files carry no
    ``Ms_mass_t``; it then follows from ``EK_t`` and ``sound_t`` (or ``a``).
    """
    if not mass_weighted:
        return np.asarray(run["Ms_t"], dtype=float)
    if "Ms_mass_t" in run:
        return np.asarray(run["Ms_mass_t"], dtype=float)
    sound_speed = np.asarray(run["sound_t"], dtype=float) if "sound_t" in run else float(run["a"])
    return np.sqrt(2.0 * np.asarray(run["EK_t"], dtype=float)) / sound_speed


def slice_mach_number(run, mass_weighted=True):
    """
    M_turb (see ``mach_numbers``) of the snapshot the slice was taken from.

    Older reduced files carry no ``slice_index``; for them the slice is the
    last snapshot before the first non-finite one.
    """
    mach = mach_numbers(run, mass_weighted)
    if "slice_index" in run:
        return float(mach[int(run["slice_index"])])
    first_bad = int(run["first_bad"])
    return float(mach[first_bad - 1 if first_bad >= 0 else -1])


def beta_label(beta):
    """Plasma beta as a compact LaTeX string (10^6 for large values, 0.1 for small)."""
    beta = float(beta)
    if beta >= 100:
        return rf"10^{{{int(round(np.log10(beta)))}}}"
    return f"{beta:g}"


def main():
    """Draw and write the 2x2 slice figure."""
    FIGURE_DIR.mkdir(exist_ok=True)
    rows = [(label, *lowest_and_highest_resolution(case), color_range) for case, label, color_range in ROWS]

    use_paper_style()
    figure, axes = plt.subplots(2, 2, figsize=(0.85 * TEXT_WIDTH, 5.6))
    # One representative image per row: both columns share the colour range.
    row_images = {}
    for row, (label, low_resolution_run, high_resolution_run, (vmin, vmax)) in enumerate(rows):
        for column, run in enumerate((low_resolution_run, high_resolution_run)):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            if run is None:
                axis.text(
                    0.5,
                    0.5,
                    "pending",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="0.4",
                )
                continue
            row_images[row] = axis.imshow(
                log_magnetic_energy_paper_units(run).T,
                origin="lower",
                extent=EXTENT,
                cmap=SEQUENTIAL_COLORMAP,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_title(
                rf"$N={int(run['N'])}^3$,  $M_S={slice_mach_number(run):.2f}$,  "
                rf"$\beta_p={beta_label(run['beta'])}$",
            )

    # One colour bar per row, of the height of the panels, on the far right
    # (the columns share the range); the left panels keep the same size.
    for row, image in row_images.items():
        reserve_colorbar_space(axes[row, 0])
        attach_colorbar(figure, image, axes[row, 1], label=r"$\log_{10} E_B$")
    figure.tight_layout()

    out = FIGURE_DIR / "turbulence_slices"
    figure.savefig(str(out) + ".svg")
    figure.savefig(str(out) + ".pdf")
    figure.savefig(str(out) + ".png")
    print("wrote", str(out) + ".svg/.pdf/.png")

    for label, low_resolution_run, high_resolution_run, _ in rows:
        for run in (low_resolution_run, high_resolution_run):
            if run is None:
                continue
            image = log_magnetic_energy_paper_units(run)
            print(f"{label} N={int(run['N'])}: M_turb(slice) mass-weighted={slice_mach_number(run):.2f} "
                  f"volume-weighted={slice_mach_number(run, mass_weighted=False):.2f}, "
                  f"log E_B range [{np.nanmin(image):.2f}, {np.nanmax(image):.2f}]")


if __name__ == "__main__":
    main()
