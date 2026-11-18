"""
The paper's ``turbulence_spectra`` figure (the analogue of HOW-MHD, Seo & Ryu
2023, Fig. 15): time-averaged power spectra of the driven isothermal-MHD
turbulence runs of ``paper_turbulence.py``, in a 2x3 grid

  rows    : ISM (M_turb ~ 10) / ICM (M_turb ~ 0.5)
  columns : density P_rho, kinetic E_K, magnetic E_B

Each panel overlays every resolution present in the script-relative data/
directory (paper: 256^3 and 512^3). The spectra are averaged over the
stationary window, the last third of the snapshots (only snapshots before a
non-finite one, should a run have failed). The paper version smooths the
time-averaged shell spectra by averaging the occupied shells within
logarithmic wavenumber bins, which removes the scatter of the sparsely
populated low-k shells and the shell-to-shell noise at high k. A k^-5/3 guide
is drawn in each panel.

Writes figures/turbulence_spectra_smooth.{svg,pdf,png} (the paper figure) and
figures/turbulence_spectra.{svg,png} (the unsmoothed shells). ``--suffix S``
reads the runs tagged ``<case>_N<N>S`` instead (e.g. ``_uct0.03``, the runs with
the upwind EMF dissipation).
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
    GUIDE_COLOR,
    TEXT_WIDTH,
)

# astronomix functions
from astronomix.plotting_helpers.paper_style import (
    figure_legend,
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

#: The two cases (rows).
CASES = ("ISM", "ICM")

#: The spectra (columns): npz key, axis label, panel title.
COLUMNS = (
    ("spec_rho", r"$P_\rho(k)$", "density spectrum"),
    ("spec_EK", r"$E_K(k)$", "kinetic spectrum"),
    ("spec_EB", r"$E_B(k)$", "magnetic spectrum"),
)

#: Line colour per resolution: shades of the finite-difference blue of the
#: paper, darker with resolution.
COLORS = {resolution: plt.get_cmap("Blues")(shade) for resolution, shade in ((128, 0.35), (256, 0.55), (512, 0.9))}

#: Box fundamental wavenumber (box size 1): the x-axis is k / k_0.
FUNDAMENTAL_WAVENUMBER = 2.0 * np.pi

#: The plotted wavenumber range in k / k_0: from just below the box scale to
#: ~85 % of the Nyquist mode, which drops the grid-noise tail.
LOWEST_MODE = 0.8
HIGHEST_MODE_PER_CELL = 0.42

#: Logarithmic bins per decade of the smoothed spectra.
BINS_PER_DECADE = 12

#: The stationary window: the last third of the snapshots.
STATIONARY_FRACTION = 1.0 / 3.0


def load_run(case, resolution):
    """The reduced data of one run, or ``None`` if its npz is absent."""
    path = DATA_DIR / f"paper_{case}_N{resolution}{RUN_SUFFIX}.npz"
    return dict(np.load(path, allow_pickle=True)) if path.exists() else None


def runs_of_case(case):
    """All resolutions present for a case, low to high, as (N, data) pairs."""
    runs = []
    for resolution in RESOLUTIONS:
        run = load_run(case, resolution)
        if run is not None:
            runs.append((int(run["N"]), run))
    return runs


def stationary_spectrum(run, key):
    """
    Average one spectrum over the stationary window: the last third of the
    snapshots, among those before the first non-finite one.

    Args:
        run: The reduced data of a run.
        key: The spectrum's npz key.

    Returns:
        The time-averaged shell spectrum.
    """
    spectra = np.asarray(run[key], dtype=np.float64)
    first_bad = int(run["first_bad"])
    usable = spectra[:first_bad] if first_bad >= 0 else spectra
    window = max(1, int(round(STATIONARY_FRACTION * (len(usable) - 1))))
    return np.nanmean(usable[-window:], axis=0)


def plotted_shells(run, spectrum):
    """
    The shells inside the plotted range that hold modes.

    Args:
        run: The reduced data of a run.
        spectrum: A shell spectrum of that run.

    Returns:
        ``(modes, values)``: the shell wavenumbers in units of k_0 and their
        spectrum values.
    """
    modes = np.asarray(run["k"], dtype=np.float64) / FUNDAMENTAL_WAVENUMBER
    resolution = int(run["N"])
    # The physical-wavenumber shells (width dk = 1) are narrower than the
    # spacing of the integer modes at low k, so many of them are empty.
    occupied = (
        (spectrum > 0.0)
        & np.isfinite(spectrum)
        & (modes >= LOWEST_MODE)
        & (modes <= HIGHEST_MODE_PER_CELL * resolution)
    )
    return modes[occupied], spectrum[occupied]


def log_binned(modes, values):
    """
    Smooth a shell spectrum by averaging the occupied shells within
    logarithmic bins (``BINS_PER_DECADE`` per decade, edges at powers of
    10^(1/BINS_PER_DECADE)); empty bins are skipped.

    Args:
        modes: The occupied shells' wavenumbers (units of k_0).
        values: Their spectrum values.

    Returns:
        ``(bin_centres, bin_means)``, the centres geometric.
    """
    bin_numbers = np.floor(BINS_PER_DECADE * np.log10(modes)).astype(int)
    bin_centres = []
    bin_means = []
    for bin_number in np.unique(bin_numbers):
        in_bin = bin_numbers == bin_number
        bin_centres.append(10.0 ** ((bin_number + 0.5) / BINS_PER_DECADE))
        bin_means.append(np.mean(values[in_bin]))
    return np.asarray(bin_centres), np.asarray(bin_means)


def plot_spectra(smooth):
    """
    Draw the 2x3 spectra figure.

    Args:
        smooth: Log-bin the spectra (the paper figure) or plot every shell.

    Returns:
        The matplotlib figure.
    """
    use_paper_style()
    figure, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 4.6), layout="constrained")
    for row, case in enumerate(CASES):
        runs = runs_of_case(case)
        for column, (key, label, title) in enumerate(COLUMNS):
            axis = axes[row, column]
            largest_value = -np.inf
            for resolution, run in runs:
                modes, values = plotted_shells(run, stationary_spectrum(run, key))
                if smooth:
                    modes, values = log_binned(modes, values)
                axis.loglog(
                    modes,
                    values,
                    color=COLORS.get(resolution),
                    label=f"$N={resolution}^3$",
                )
                largest_value = max(largest_value, np.nanmax(values))
            # k^-5/3 guide anchored in the inertial range, lifted above the data.
            if np.isfinite(largest_value):
                guide_modes = np.array([2.0, 20.0])
                guide_values = 1.6 * largest_value * (guide_modes / guide_modes[0]) ** (-5.0 / 3.0)
                axis.plot(
                    guide_modes,
                    guide_values,
                    color=GUIDE_COLOR,
                    linestyle="--",
                    linewidth=1.0,
                    label=r"$\propto k^{-5/3}$",
                )
                # About five decades: clips the noise floor.
                axis.set_ylim(largest_value / 3e4, largest_value * 4)
            axis.set_xlim(LOWEST_MODE, None)
            axis.set_xlabel(r"$k/k_0$")
            axis.set_ylabel(label)
            if row == 0:
                axis.set_title(title)
            axis.text(
                0.05,
                0.06,
                case,
                transform=axis.transAxes,
                fontweight="bold",
                va="bottom",
            )
    figure_legend(figure)
    return figure


def main():
    """Write the smoothed (paper) and the unsmoothed spectra figures."""
    FIGURE_DIR.mkdir(exist_ok=True)

    figure = plot_spectra(smooth=True)
    smooth_path = FIGURE_DIR / "turbulence_spectra_smooth"
    for suffix in (".svg", ".pdf", ".png"):
        figure.savefig(str(smooth_path) + suffix)
    plt.close(figure)
    print("wrote", str(smooth_path) + ".svg/.pdf/.png")

    figure = plot_spectra(smooth=False)
    raw_path = FIGURE_DIR / "turbulence_spectra"
    figure.savefig(str(raw_path) + ".svg")
    figure.savefig(str(raw_path) + ".png")
    plt.close(figure)
    print("wrote", str(raw_path) + ".svg/.png")

    for case in CASES:
        for resolution, run in runs_of_case(case):
            sound_speed = float(run["sound_t"][-1]) if "sound_t" in run else float(run["a"])
            mass_weighted = np.sqrt(2.0 * float(run["EK_t"][-1])) / sound_speed
            print(f"{case} N={resolution}^3: M_turb(last) mass-weighted={mass_weighted:.2f} "
                  f"volume-weighted={float(run['Ms_t'][-1]):.2f}, "
                  f"snapshots={run['spec_EK'].shape[0]}, first_bad={int(run['first_bad'])}")


if __name__ == "__main__":
    main()
