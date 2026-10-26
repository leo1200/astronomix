"""The shell-wise numerical diffusivities behind the FD and FV Reynolds numbers.

``make_fd_fv_reynolds_figure.py`` shows ``Re``, ``Rm`` and ``Pm`` of astronomix's
two MHD schemes; this figure shows what they are computed from. For each shell,

    dE(n)/dt = T(n) - D(n),

with ``T(n)`` the ideal transfer (``_mhd_spectral.transfer_spectra``), ``dE/dt``
differenced between snapshots and ``D(n)`` the dissipation the scheme adds, and
the effective diffusivities are ``nu_eff(n) = D_v / (2 k^2 E_v)`` and
``eta_eff(n) = D_B / (2 k^2 E_B)``.

Two layouts:

    fd_fv_transfer_spectra       nu_eff(n), eta_eff(n) and Pm(n) at 128^3 and
                                 256^3, the averaging band and the table's band
                                 means
    fd_fv_transfer_spectra_full  nu_eff(n) and eta_eff(n) for every grid, next
                                 to the full kinetic and magnetic budgets at 256^3

The full layout and every band mean are taken exactly as the table takes them:
raw (not deconvolved) spectra, the five-snapshot window centred where
``E_B/E_K`` first reaches 0.01 (``make_mechanism_table.matched_snapshots``),
``D`` and ``E`` averaged over the window before they are divided.

The curves of the three-panel layout use a lower-noise estimator of the same
quantities, over the whole passive kinematic phase: every snapshot with
``0.003 <= E_B/E_K <= 0.03`` after two crossing times, ``T`` and ``E``
time-averaged over that window, ``dE/dt`` from its end points (the exact window
average of the derivative, instead of differences of consecutive snapshots),
and ``D`` and ``2 k^2 E`` summed over logarithmic shell bins (16 per decade)
before they are divided. Its standard error comes from four contiguous
sub-windows. It reads the densely sampled kinematic-phase reruns in
``data/fd_fv/kinematic/`` (same setup and seed, spectra every 1/16 crossing
time up to ``E_B/E_K ~ 0.05``) where they exist. For the VL2 + GLM scheme
``D_B`` contains the divergence-cleaning sink ``-T_psi``. The script also
prints where in ``n`` each scheme dissipates and how much the kinematic
estimator smooths ``Pm(n)``.

    python make_fd_fv_transfer_figure.py --data data/fd_fv
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
from matplotlib.lines import Line2D

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

# the dynamo study's measurement
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_metrics import E_MAG, E_V, load_runs, spectra_of
from make_dissipation_figure import T_MAG, T_V, dissipation_series
from make_fd_fv_reynolds_figure import SERIES
from make_mechanism_table import BAND, matched_snapshots

HERE = Path(__file__).resolve().parent

#: Line width and opacity per resolution in the diffusivity panels: the finest
#: grid is the boldest, the coarser ones recede.
RESOLUTION_STYLE = {64: dict(lw=0.7, alpha=0.45), 128: dict(lw=1.0, alpha=0.7),
                    256: dict(lw=1.6, alpha=1.0)}

#: The resolutions of the simplified (diffusivity-only) figure.
SIMPLE_RESOLUTIONS = (128, 256)

#: Fraction of Nyquist above which dissipation counts as "near the grid scale"
#: in the printed dissipation-location numbers.
GRID_SCALE_FRACTION = 0.5

#: First shell drawn in the diffusivity panels. Below it the forcing acts and
#: the budget measures the injection, not a dissipation.
FIRST_DISSIPATION_SHELL = 4

#: Smallest ``n / n_Nyq`` drawn in the diffusivity panels. Further out of the
#: band, at the large scales, ``T`` and ``dE/dt`` nearly cancel, so their
#: difference -- and its sign -- is dominated by the snapshot noise.
MINIMUM_NYQUIST_FRACTION = 0.1

#: Shared vertical range of the two diffusivity panels, so nu and eta read on
#: one scale: with every grid, and with the 128^3 and 256^3 grids only.
DIFFUSIVITY_LIMITS = (4e-5, 4e-3)
SIMPLE_DIFFUSIVITY_LIMITS = (4e-5, 1.6e-3)

#: Linear vertical range of the Prandtl-number panel.
PRANDTL_LIMITS = (0.3, 1.6)

#: Vertical ranges of the kinetic and magnetic budget panels.
BUDGET_LIMITS = {"v": (1e-5, 3.0), "B": (1e-7, 3e-2)}

#: Band over which the slopes of nu_eff(n) and eta_eff(n) are fitted, as in
#: ``make_dissipation_figure.py``.
SLOPE_BAND = (0.15, 0.6)


def matched_budget(run):
    """The shell budget of one run, averaged over the matched window.

    Args:
        run: A run file of ``dynamo_convergence.py --transfer``.

    Returns:
        A dict with the shells ``n``, the window means of ``T``, ``dE/dt``,
        ``D`` and ``E`` for both fields, ``-T_psi`` (or ``None``), the
        diffusivities per shell, their snapshot-to-snapshot standard errors
        and the band means.
    """
    series = dissipation_series(run, deconvolve=False)
    window = matched_snapshots(series)
    spectra = spectra_of(run, deconvolve=False)[window]
    shells = series["n"]
    wavenumber = 2.0 * np.pi * shells
    number_of_cells = int(run["N"])
    inband = ((shells / (number_of_cells / 2) >= BAND[0])
              & (shells / (number_of_cells / 2) <= BAND[1]))

    budget = dict(n=shells, N=number_of_cells, inband=inband,
                  t_over_tc=float(series["t_over_tc"][window].mean()))
    for field, transfer_row, dissipation_key, spectrum_key in (
            ("v", T_V, "D_v", "E_v"), ("B", T_MAG, "D_B", "E_B")):
        transfer = spectra[:, transfer_row].mean(0)
        dissipation = series[dissipation_key][window].mean(0)
        spectrum = series[spectrum_key][window].mean(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            diffusivity = dissipation / (2.0 * wavenumber ** 2 * spectrum)
            per_snapshot = (series[dissipation_key][window]
                            / (2.0 * wavenumber ** 2 * series[spectrum_key][window]))
            # The n = 0 shell has k = 0 and no diffusivity; it stays NaN.
            diffusivity_error = per_snapshot.std(0, ddof=1) / np.sqrt(len(window))
        budget[f"diffusivity_per_snapshot_{field}"] = per_snapshot
        budget[f"T_{field}"] = transfer
        budget[f"D_{field}"] = dissipation
        budget[f"dEdt_{field}"] = transfer - dissipation
        budget[f"E_{field}"] = spectrum
        budget[f"diffusivity_{field}"] = diffusivity
        budget[f"diffusivity_error_{field}"] = diffusivity_error
        # The table's number: the band mean of the window-averaged ratio.
        budget[f"band_mean_{field}"] = float(np.mean(diffusivity[inband]))
    # The magnetic Prandtl number shell by shell, and its standard error from
    # the per-snapshot ratios (nu and eta are correlated through the shared
    # flow, so the ratio's scatter is measured rather than propagated).
    with np.errstate(divide="ignore", invalid="ignore"):
        budget["prandtl"] = budget["diffusivity_v"] / budget["diffusivity_B"]
        prandtl_per_snapshot = (budget.pop("diffusivity_per_snapshot_v")
                                / budget.pop("diffusivity_per_snapshot_B"))
        budget["prandtl_error"] = prandtl_per_snapshot.std(0, ddof=1) / np.sqrt(len(window))
    # The table's number: the ratio of the two band means.
    budget["band_mean_prandtl"] = budget["band_mean_v"] / budget["band_mean_B"]
    cleaning = series.get("T_psi")
    budget["cleaning_sink"] = None if cleaning is None else -cleaning[window].mean(0)
    return budget


#: Passive-field window of the kinematic-phase estimator: every snapshot after
#: the spin-up with E_B/E_K in this range. The decade the growth rates are
#: fitted over; centred (in log) on the table's matched state 0.01, and with
#: the field at most 3 % of the kinetic energy it is still passive.
KINEMATIC_RATIO_RANGE = (0.003, 0.03)
KINEMATIC_T_MIN = 2.0

#: Logarithmic shell bins of the kinematic-phase estimator, per decade of n.
BINS_PER_DECADE = 16

#: Number of contiguous sub-windows the kinematic window is split into for
#: its standard error.
SUB_WINDOWS = 4


def kinematic_window(run, ratio_range=KINEMATIC_RATIO_RANGE, t_min=KINEMATIC_T_MIN):
    """Indices of the contiguous snapshots from the first with ``E_B/E_K``
    above ``ratio_range[0]`` to the last below ``ratio_range[1]`` (after
    ``t_min`` crossing times), or ``None`` if fewer than two."""
    t_over_tc = np.asarray(run["t_over_tc"])
    ratio = np.asarray(run["E_B"]) / np.maximum(np.asarray(run["E_K"]), 1e-30)
    inside = np.where((t_over_tc >= t_min) & (ratio >= ratio_range[0])
                      & (ratio <= ratio_range[1]))[0]
    if len(inside) < 2:
        return None
    return np.arange(inside[0], inside[-1] + 1)


def window_budget(spectra, times, window):
    """Time-averaged ``T``, ``E`` and ``D`` of both fields over one window.

    ``dE/dt`` is the endpoint difference ``(E(t1) - E(t0)) / (t1 - t0)``, which
    is the exact window average of the derivative, and ``T`` and ``E`` are
    trapezoidal time averages over the same window, so ``D = <T> - <dE/dt>``
    is the window-averaged dissipation without differencing consecutive
    snapshots.

    Args:
        spectra: Raw spectra of a run, ``(snapshots, rows, shells)``.
        times: Snapshot times.
        window: Snapshot indices of the window (contiguous).

    Returns:
        ``{"D_v", "E_v", "D_B", "E_B"}``, each per shell.
    """
    window_times = times[window]
    duration = window_times[-1] - window_times[0]
    averages = {}
    for field, energy_row, transfer_row in (("v", E_V, T_V), ("B", E_MAG, T_MAG)):
        energy = spectra[window, energy_row]
        transfer_average = np.trapezoid(spectra[window, transfer_row], window_times, axis=0) / duration
        energy_rate = (energy[-1] - energy[0]) / duration
        averages[f"D_{field}"] = transfer_average - energy_rate
        averages[f"E_{field}"] = np.trapezoid(energy, window_times, axis=0) / duration
    return averages


def log_bin_edges(number_of_cells):
    """Shell-bin edges in ``n``, ``BINS_PER_DECADE`` per decade from the first
    dissipation shell to Nyquist."""
    nyquist = number_of_cells / 2
    decades = np.log10((nyquist + 0.5) / (FIRST_DISSIPATION_SHELL - 0.5))
    count = int(np.ceil(decades * BINS_PER_DECADE))
    return np.geomspace(FIRST_DISSIPATION_SHELL - 0.5, nyquist + 0.5, count + 1)


def binned_diffusivities(averages, shells, number_of_cells):
    """``nu_eff``, ``eta_eff`` and ``Pm`` in logarithmic shell bins.

    ``D`` and ``2 k^2 E`` are summed over the shells of a bin separately and
    then divided, the same ratio estimator as per shell, so a bin is the
    diffusivity of its shells together. Bins without a shell are dropped.

    Returns:
        ``(n / n_Nyq of each bin, nu, eta, Pm)``.
    """
    wavenumber = 2.0 * np.pi * shells
    edges = log_bin_edges(number_of_cells)
    positions, viscosities, resistivities = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        members = (shells >= low) & (shells < high)
        if not members.any():
            continue
        viscosity = (averages["D_v"][members].sum()
                     / (2.0 * wavenumber[members] ** 2 * averages["E_v"][members]).sum())
        resistivity = (averages["D_B"][members].sum()
                       / (2.0 * wavenumber[members] ** 2 * averages["E_B"][members]).sum())
        positions.append(shells[members].mean() / (number_of_cells / 2))
        viscosities.append(viscosity)
        resistivities.append(resistivity)
    viscosities = np.array(viscosities)
    resistivities = np.array(resistivities)
    return np.array(positions), viscosities, resistivities, viscosities / resistivities


def kinematic_budget(run):
    """The kinematic-phase estimator of one run: binned diffusivities and Pm
    over :func:`kinematic_window`, with sub-window standard errors.

    Args:
        run: A run file of ``dynamo_convergence.py --transfer``.

    Returns:
        A dict with the bin positions, ``nu``, ``eta``, ``Pm`` and their
        standard errors, the per-shell band means (the table's definition)
        over the same window, and the window itself; ``None`` if the run has
        fewer than two snapshots in the window.
    """
    window = kinematic_window(run)
    if window is None:
        return None
    spectra = spectra_of(run, deconvolve=False)
    times = np.asarray(run["t_over_tc"]) * float(run["t_cross"])
    shells = np.asarray(run["n_shell"], dtype=float)
    number_of_cells = int(run["N"])
    averages = window_budget(spectra, times, window)
    positions, viscosity, resistivity, prandtl = binned_diffusivities(
        averages, shells, number_of_cells)

    # Standard error from contiguous sub-windows sharing their end points.
    errors = {"nu": np.full_like(viscosity, np.nan), "eta": np.full_like(viscosity, np.nan),
              "Pm": np.full_like(viscosity, np.nan)}
    if len(window) > 2 * SUB_WINDOWS:
        boundaries = np.linspace(0, len(window) - 1, SUB_WINDOWS + 1).round().astype(int)
        estimates = [binned_diffusivities(
            window_budget(spectra, times, window[start:stop + 1]), shells, number_of_cells)[1:]
            for start, stop in zip(boundaries[:-1], boundaries[1:])]
        for index, key in enumerate(("nu", "eta", "Pm")):
            errors[key] = (np.std([estimate[index] for estimate in estimates], axis=0, ddof=1)
                           / np.sqrt(SUB_WINDOWS))

    # The table's band mean (mean over in-band shells of the per-shell ratio)
    # over this window, for comparison with the published numbers.
    wavenumber = 2.0 * np.pi * shells
    inband = ((shells / (number_of_cells / 2) >= BAND[0])
              & (shells / (number_of_cells / 2) <= BAND[1]))
    band_viscosity = float(np.mean(averages["D_v"][inband]
                                   / (2.0 * wavenumber[inband] ** 2 * averages["E_v"][inband])))
    band_resistivity = float(np.mean(averages["D_B"][inband]
                                     / (2.0 * wavenumber[inband] ** 2 * averages["E_B"][inband])))
    v_rms = float(np.asarray(run["v_rms"])[window].mean())
    return dict(N=number_of_cells, position=positions, nu=viscosity, eta=resistivity,
                Pm=prandtl, nu_error=errors["nu"], eta_error=errors["eta"],
                Pm_error=errors["Pm"], band_nu=band_viscosity, band_eta=band_resistivity,
                band_Pm=band_viscosity / band_resistivity,
                band_Re=v_rms * 0.5 / band_viscosity, band_Rm=v_rms * 0.5 / band_resistivity,
                snapshots=len(window),
                t_window=(float(run["t_over_tc"][window[0]]),
                          float(run["t_over_tc"][window[-1]])))


def prandtl_roughness(positions, prandtl):
    """Shell-to-shell scatter of ``log Pm`` in the band: the rms residual about
    a straight line in ``log(n / n_Nyq)``, so a smooth trend does not count."""
    inband = (positions >= BAND[0]) & (positions <= BAND[1]) & (prandtl > 0)
    x, y = np.log(positions[inband]), np.log(prandtl[inband])
    return float(np.std(y - np.polyval(np.polyfit(x, y, 1), x)))


def draw_budget(ax, budget, colour, field):
    """``T``, ``|dE/dt|`` and ``D`` of one field of one run, against ``n``.

    Magnitudes on a logarithmic axis: below ``n = 4`` the kinetic ``T`` and ``D``
    are negative (the omitted forcing appears there as a source), and the
    kinetic ``dE/dt`` scatters about zero; the shading marks that range.
    """
    shells = budget["n"]
    visible = shells >= 1
    ax.loglog(shells[visible], np.abs(budget[f"T_{field}"])[visible], "-", color=colour, lw=1.6)
    ax.loglog(shells[visible], np.abs(budget[f"dEdt_{field}"])[visible], ":", color=colour, lw=1.3)
    ax.loglog(shells[visible], np.abs(budget[f"D_{field}"])[visible], "--", color=colour, lw=1.4)
    if field == "B" and budget["cleaning_sink"] is not None:
        ax.loglog(shells[visible], np.abs(budget["cleaning_sink"])[visible], "-.",
                  color=colour, lw=0.9, alpha=0.8)


def draw_diffusivity(ax, budget, colour, field, resolutions):
    """``nu_eff(n)`` or ``eta_eff(n)`` against ``n / n_Nyq`` with its band mean.

    Args:
        ax: The axes.
        budget: The output of :func:`matched_budget`.
        colour: The scheme colour.
        field: ``"v"`` or ``"B"``.
        resolutions: The resolutions drawn in this panel; the finest of them
            carries the standard-error band.
    """
    shells = budget["n"]
    nyquist_fraction = shells / (budget["N"] / 2)
    visible = (shells >= FIRST_DISSIPATION_SHELL) & (nyquist_fraction >= MINIMUM_NYQUIST_FRACTION)
    style = RESOLUTION_STYLE[budget["N"]]
    # A non-positive value cannot be read as a diffusivity (and has no
    # logarithm); it is left as a gap. Inside the band there is none.
    diffusivity = np.where(budget[f"diffusivity_{field}"] > 0.0,
                           budget[f"diffusivity_{field}"], np.nan)
    ax.loglog(nyquist_fraction[visible], diffusivity[visible], "-", color=colour, **style)
    if budget["N"] == max(resolutions):
        error = budget[f"diffusivity_error_{field}"]
        lower = np.where(diffusivity - error > 0.0, diffusivity - error, DIFFUSIVITY_LIMITS[0])
        ax.fill_between(nyquist_fraction[visible], lower[visible],
                        (diffusivity + error)[visible],
                        color=colour, alpha=0.18, lw=0)
    ax.plot(BAND, [budget[f"band_mean_{field}"]] * 2, color=colour, lw=style["lw"] + 0.6,
            ls=(0, (4, 2)), alpha=style["alpha"], solid_capstyle="butt")


def draw_prandtl(ax, budget, colour, resolutions):
    """``Pm(n) = nu_eff(n) / eta_eff(n)`` against ``n / n_Nyq`` with the table's
    band-mean ``Pm``.

    Args:
        ax: The axes.
        budget: The output of :func:`matched_budget`.
        colour: The scheme colour.
        resolutions: The resolutions drawn; the finest carries the
            standard-error band.
    """
    shells = budget["n"]
    nyquist_fraction = shells / (budget["N"] / 2)
    visible = (shells >= FIRST_DISSIPATION_SHELL) & (nyquist_fraction >= MINIMUM_NYQUIST_FRACTION)
    style = RESOLUTION_STYLE[budget["N"]]
    # Only shells where both diffusivities are positive give a Prandtl number.
    positive = (budget["diffusivity_v"] > 0.0) & (budget["diffusivity_B"] > 0.0)
    prandtl = np.where(positive, budget["prandtl"], np.nan)
    ax.semilogx(nyquist_fraction[visible], prandtl[visible], "-", color=colour, **style)
    if budget["N"] == max(resolutions):
        error = budget["prandtl_error"]
        ax.fill_between(nyquist_fraction[visible], (prandtl - error)[visible],
                        (prandtl + error)[visible], color=colour, alpha=0.18, lw=0)
    ax.plot(BAND, [budget["band_mean_prandtl"]] * 2, color=colour, lw=style["lw"] + 0.6,
            ls=(0, (4, 2)), alpha=style["alpha"], solid_capstyle="butt")


def style_diffusivity_axes(ax, field, limits):
    """Logarithmic axes, band shading, limits, labels and title of a
    diffusivity panel."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvspan(*BAND, color="0.92", zorder=0)
    ax.set_xlim(MINIMUM_NYQUIST_FRACTION / 1.08, 1.08)
    ax.set_ylim(*limits)
    ax.set_xlabel(r"$n / n_{\rm Nyq}$")
    if field == "v":
        ax.set_title(r"$\nu_{\rm eff}(n) = D_v / 2k^2 E_v$")
        ax.set_ylabel(r"$\nu_{\rm eff}$")
    else:
        ax.set_title(r"$\eta_{\rm eff}(n) = D_B / 2k^2 E_B$")
        ax.set_ylabel(r"$\eta_{\rm eff}$")


def fitted_slope(budget, field):
    """Log-log slope of the diffusivity over :data:`SLOPE_BAND` of Nyquist."""
    shells = budget["n"]
    nyquist_fraction = shells / (budget["N"] / 2)
    diffusivity = budget[f"diffusivity_{field}"]
    fit = ((nyquist_fraction >= SLOPE_BAND[0]) & (nyquist_fraction <= SLOPE_BAND[1])
           & np.isfinite(diffusivity) & (diffusivity > 0))
    return float(np.polyfit(np.log(shells[fit]), np.log(diffusivity[fit]), 1)[0])


def dissipation_location(budget, field):
    """Where in ``n`` a scheme dissipates, from the measured ``D(n)``.

    Only shells above the forcing (``n >= FIRST_DISSIPATION_SHELL``) count, and
    negative ``D`` is clipped to zero, the convention of
    ``make_mechanism_table.dissipation_shells``.

    Args:
        budget: The output of :func:`matched_budget`.
        field: ``"v"`` or ``"B"``.

    Returns:
        ``(peak shell, fraction of the dissipation above GRID_SCALE_FRACTION of
        Nyquist, dissipation-weighted mean shell)``, the shells in units of
        ``n``.
    """
    shells = budget["n"]
    above_forcing = shells >= FIRST_DISSIPATION_SHELL
    dissipation = np.where(above_forcing, np.maximum(budget[f"D_{field}"], 0.0), 0.0)
    nyquist = budget["N"] / 2
    peak_shell = float(shells[np.argmax(dissipation)])
    grid_scale_fraction = float(dissipation[shells > GRID_SCALE_FRACTION * nyquist].sum()
                                / dissipation.sum())
    mean_shell = float((shells * dissipation).sum() / dissipation.sum())
    return peak_shell, grid_scale_fraction, mean_shell


def save(fig, figures, stem):
    """Write ``fig`` as SVG, PDF and PNG."""
    figures.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        path = figures / f"{stem}.{extension}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


def draw_binned(ax, kinematic, colour, quantity, band_mean, resolutions):
    """One binned kinematic-phase curve (``nu``, ``eta`` or ``Pm``) with the
    table's band mean as a dash.

    Args:
        ax: The axes.
        kinematic: The output of :func:`kinematic_budget`.
        colour: The scheme colour.
        quantity: ``"nu"``, ``"eta"`` or ``"Pm"``.
        band_mean: The table's band-mean value of ``quantity``.
        resolutions: The resolutions drawn; the finest carries the
            standard-error band.
    """
    positions = kinematic["position"]
    visible = positions >= MINIMUM_NYQUIST_FRACTION
    style = RESOLUTION_STYLE[kinematic["N"]]
    # A non-positive bin cannot be read as a diffusivity; it is left as a gap.
    values = np.where(kinematic[quantity] > 0.0, kinematic[quantity], np.nan)
    ax.plot(positions[visible], values[visible], "-", color=colour, **style)
    error = kinematic[f"{quantity}_error"]
    if kinematic["N"] == max(resolutions) and np.isfinite(error).any():
        lower = values - error
        if quantity != "Pm":
            lower = np.where(lower > 0.0, lower, SIMPLE_DIFFUSIVITY_LIMITS[0])
        ax.fill_between(positions[visible], lower[visible], (values + error)[visible],
                        color=colour, alpha=0.2, lw=0)
    ax.plot(BAND, [band_mean] * 2, color=colour, lw=style["lw"] + 0.6,
            ls=(0, (4, 2)), alpha=style["alpha"], solid_capstyle="butt")


def draw_diffusivity_figure(budgets, kinematic_budgets, figures, stem):
    """The 1 x 3 figure: ``nu_eff(n)``, ``eta_eff(n)`` and ``Pm(n)`` at 128^3
    and 256^3, from the kinematic-phase estimator in logarithmic shell bins,
    with the table's band means (from the matched window) as dashes."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.75), layout="constrained")
    for scheme in ("wenoz", "vl2"):
        for N in SIMPLE_RESOLUTIONS:
            if (scheme, N) not in kinematic_budgets:
                continue
            table = budgets[(scheme, N)]
            band_means = {"nu": table["band_mean_v"], "eta": table["band_mean_B"],
                          "Pm": table["band_mean_prandtl"]}
            for ax, quantity in zip(axes, ("nu", "eta", "Pm")):
                draw_binned(ax, kinematic_budgets[(scheme, N)], SERIES[scheme][0], quantity,
                            band_means[quantity], SIMPLE_RESOLUTIONS)
    for ax, field in zip(axes[:2], ("v", "B")):
        style_diffusivity_axes(ax, field, SIMPLE_DIFFUSIVITY_LIMITS)
    prandtl_axes = axes[2]
    prandtl_axes.set_xscale("log")
    prandtl_axes.axvspan(*BAND, color="0.92", zorder=0)
    prandtl_axes.axhline(1.0, color=GUIDE_COLOR, lw=0.9, ls=":", zorder=1)
    prandtl_axes.set_xlim(MINIMUM_NYQUIST_FRACTION / 1.08, 1.08)
    prandtl_axes.set_ylim(*PRANDTL_LIMITS)
    prandtl_axes.set_xlabel(r"$n / n_{\rm Nyq}$")
    prandtl_axes.set_title(r"$\mathrm{Pm}(n) = \nu_{\rm eff}(n) / \eta_{\rm eff}(n)$")
    prandtl_axes.set_ylabel(r"$\mathrm{Pm}$")

    scheme_handles = [Line2D([], [], color=SERIES[key][0], lw=1.6, label=SERIES[key][3])
                      for key in ("wenoz", "vl2")]
    resolution_handles = [Line2D([], [], color="0.25", label=f"${N}^3$", **RESOLUTION_STYLE[N])
                          for N in SIMPLE_RESOLUTIONS]
    resolution_handles.append(Line2D([], [], color="0.25", ls=(0, (4, 2)), lw=1.5,
                                     label="band mean"))
    figure_legend(fig, scheme_handles + resolution_handles, ncols=3, handlelength=2.4)
    save(fig, figures, stem)


def print_noise_report(budgets, kinematic_budgets):
    """How much the kinematic-phase estimator smooths ``Pm(n)``, and whether its
    band means agree with the table's."""
    print(f"\n{'scheme':6s} {'N':>4s} | log-Pm roughness: {'per shell':>9s} {'kinematic':>9s} | "
          f"kinematic window: {'t/tc':>9s} {'snaps':>5s} {'nu':>9s} {'eta':>9s} {'Pm':>6s} "
          f"{'Re':>6s} {'Rm':>6s} | table: {'nu':>9s} {'eta':>9s} {'Pm':>6s}")
    for (scheme, N), kinematic in sorted(kinematic_budgets.items()):
        table = budgets[(scheme, N)]
        shells = table["n"]
        above_forcing = shells >= FIRST_DISSIPATION_SHELL
        per_shell = prandtl_roughness(shells[above_forcing] / (N / 2),
                                      table["prandtl"][above_forcing])
        binned = prandtl_roughness(kinematic["position"], kinematic["Pm"])
        window = f"{kinematic['t_window'][0]:.1f}-{kinematic['t_window'][1]:.1f}"
        print(f"{scheme:6s} {N:4d} | {'':17s} {per_shell:9.3f} {binned:9.3f} | {'':17s}"
              f"{window:>9s} {kinematic['snapshots']:5d} {kinematic['band_nu']:9.2e} "
              f"{kinematic['band_eta']:9.2e} {kinematic['band_Pm']:6.3f} {kinematic['band_Re']:6.0f} "
              f"{kinematic['band_Rm']:6.0f} | {'':6s} {table['band_mean_v']:9.2e} "
              f"{table['band_mean_B']:9.2e} {table['band_mean_prandtl']:6.3f}")


def draw_full_figure(budgets, figures, stem):
    """The 2 x 2 figure: the 256^3 budgets and the diffusivities of every grid."""
    finest = max(N for _, N in budgets)
    resolutions = tuple(sorted(RESOLUTION_STYLE))
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 5.4), layout="constrained")
    for scheme in ("wenoz", "vl2"):
        colour = SERIES[scheme][0]
        draw_budget(axes[0, 0], budgets[(scheme, finest)], colour, "v")
        draw_budget(axes[1, 0], budgets[(scheme, finest)], colour, "B")
        for N in resolutions:
            if (scheme, N) in budgets:
                draw_diffusivity(axes[0, 1], budgets[(scheme, N)], colour, "v", resolutions)
                draw_diffusivity(axes[1, 1], budgets[(scheme, N)], colour, "B", resolutions)

    for ax, field in zip(axes[:, 0], ("v", "B")):
        ax.axvspan(0.8, FIRST_DISSIPATION_SHELL - 0.5, color="0.92", zorder=0)
        ax.set_xlim(0.8, finest / 2 * 1.1)
        ax.set_ylim(*BUDGET_LIMITS[field])
        ax.set_xlabel(r"mode number $n = kL/2\pi$")
        ax.set_ylabel("per-shell rate")
    axes[0, 0].set_title(rf"kinetic budget, ${finest}^3$")
    axes[1, 0].set_title(rf"magnetic budget, ${finest}^3$")
    for ax, field in zip(axes[:, 1], ("v", "B")):
        style_diffusivity_axes(ax, field, DIFFUSIVITY_LIMITS)

    # One legend below the panels, two rows: what the colour, the line weight
    # and the line style mean. Keeping it out of the panels keeps it off the
    # curves.
    scheme_handles = [Line2D([], [], color=SERIES[key][0], lw=1.6, label=SERIES[key][3])
                      for key in ("wenoz", "vl2")]
    resolution_handles = [Line2D([], [], color="0.25", label=f"${N}^3$", **RESOLUTION_STYLE[N])
                          for N in resolutions]
    term_handles = [
        Line2D([], [], color="0.25", ls=(0, (4, 2)), lw=1.5, label="band mean"),
        Line2D([], [], color="0.25", ls="-", lw=1.6, label=r"$|T(n)|$, ideal transfer"),
        Line2D([], [], color="0.25", ls=":", lw=1.3, label=r"$|dE(n)/dt|$"),
        Line2D([], [], color="0.25", ls="--", lw=1.4, label=r"$D(n) = T - dE/dt$"),
        Line2D([], [], color="0.25", ls="-.", lw=0.9, label=r"$-T_\psi(n)$ (FV)"),
    ]
    first_row = scheme_handles + resolution_handles
    # matplotlib fills legend columns first, so interleave the two rows.
    handles = [handle for pair in zip(first_row, term_handles) for handle in pair]
    figure_legend(fig, handles, ncols=len(first_row), handlelength=2.4)
    save(fig, figures, stem)


def print_caption_numbers(budgets):
    """Slopes, band means, sign of ``D`` in the band, the cleaning share, where
    the FD/FV ratios sit in ``n``, and where each scheme dissipates."""
    print(f"\n{'scheme':8s} {'N':>4s} {'t/tc':>5s} {'nu band':>9s} {'eta band':>9s} "
          f"{'nu slope':>8s} {'eta slope':>9s} {'D_v<0':>5s} {'D_B<0':>5s} {'psi/D_B':>7s}")
    for (scheme, N), budget in sorted(budgets.items()):
        inband = budget["inband"]
        cleaning_share = (np.sum(budget["cleaning_sink"][inband]) / np.sum(budget["D_B"][inband])
                          if budget["cleaning_sink"] is not None else np.nan)
        print(f"{scheme:8s} {N:4d} {budget['t_over_tc']:5.1f} {budget['band_mean_v']:9.2e} "
              f"{budget['band_mean_B']:9.2e} {fitted_slope(budget, 'v'):8.2f} "
              f"{fitted_slope(budget, 'B'):9.2f} {int(np.sum(budget['D_v'][inband] <= 0)):5d} "
              f"{int(np.sum(budget['D_B'][inband] <= 0)):5d} {cleaning_share:7.4f}")

    for N in sorted(RESOLUTION_STYLE):
        if ("wenoz", N) not in budgets or ("vl2", N) not in budgets:
            continue
        fd, fv = budgets[("wenoz", N)], budgets[("vl2", N)]
        fraction = fd["n"] / (N / 2)
        ratios = []
        for low, high in ((0.1, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.0)):
            shells = (fraction >= low) & (fraction < high) & (fd["n"] >= FIRST_DISSIPATION_SHELL)
            nu_ratio = np.mean(fd["diffusivity_v"][shells] / fv["diffusivity_v"][shells])
            eta_ratio = np.mean(fd["diffusivity_B"][shells] / fv["diffusivity_B"][shells])
            ratios.append(f"{low:.1f}-{high:.1f}: nu {nu_ratio:.2f} eta {eta_ratio:.2f}")
        print(f"FD/FV N={N}: " + " | ".join(ratios))

    print(f"\nWhere the dissipation sits (shells n >= {FIRST_DISSIPATION_SHELL}, D clipped at 0): "
          f"peak shell, fraction above {GRID_SCALE_FRACTION:g} n_Nyq, <n>_D / n_Nyq")
    for (scheme, N), budget in sorted(budgets.items()):
        cells = []
        for field, name in (("v", "kinetic"), ("B", "magnetic")):
            peak, fraction, mean = dissipation_location(budget, field)
            cells.append(f"{name}: peak n = {peak:3.0f} ({peak / (N / 2):.2f}), "
                         f"f(>{GRID_SCALE_FRACTION:g}) = {fraction:.2f}, "
                         f"<n>_D/n_Nyq = {mean / (N / 2):.2f}")
        print(f"{scheme:6s} {N:4d}  " + " | ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(HERE / "data" / "fd_fv"),
                        help="directory of dynamo_convergence.py --transfer runs")
    parser.add_argument("--figures", default=str(HERE / "figures"))
    parser.add_argument("--kinematic-data", default=str(HERE / "data" / "fd_fv" / "kinematic"),
                        help="directory of the densely sampled kinematic-phase reruns; "
                             "a scheme and grid without one falls back to its "
                             "production run")
    parser.add_argument("--stem", default="fd_fv_transfer_spectra")
    args = parser.parse_args()

    budgets, kinematic_budgets = {}, {}
    for run in load_runs(args.data):
        scheme = str(run.get("scheme_key", ""))
        if scheme in ("wenoz", "vl2"):
            budgets[(scheme, int(run["N"]))] = matched_budget(run)
            kinematic_budgets[(scheme, int(run["N"]))] = kinematic_budget(run)
    # The dense reruns replace the production runs' kinematic phase: same
    # seed and setup, so the same realisation, sampled eight times as often.
    if Path(args.kinematic_data).is_dir():
        for run in load_runs(args.kinematic_data):
            scheme = str(run.get("scheme_key", ""))
            if scheme in ("wenoz", "vl2"):
                kinematic_budgets[(scheme, int(run["N"]))] = kinematic_budget(run)
    kinematic_budgets = {key: value for key, value in kinematic_budgets.items()
                         if value is not None}

    use_paper_style()
    figures = Path(args.figures)
    draw_diffusivity_figure(budgets, kinematic_budgets, figures, args.stem)
    draw_full_figure(budgets, figures, f"{args.stem}_full")
    print_caption_numbers(budgets)
    print_noise_report(budgets, kinematic_budgets)


if __name__ == "__main__":
    main()
