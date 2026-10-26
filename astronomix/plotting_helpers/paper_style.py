"""
Common matplotlib style of the figures of the astronomix methods paper.

The figure scripts call ``use_paper_style`` before they create a figure and take
the colors and line styles of the schemes from here, so that a scheme looks the
same in every figure: finite differences blue, finite volumes orange (with the
HLLE solver green), AthenaPK red (its third-order setup purple), reference
solutions black and guide lines (slopes, fits, ideal values) grey. A variant of
a scheme (another backend, precision or option) keeps the color of the scheme
and changes its line style and marker fill (``variant_style``).

The figures are drawn at the size at which they appear in the paper, so that
their fonts match from figure to figure, and a legend that applies to several
panels is placed once, outside the panels, in the standard framed box
(``figure_legend``). A colorbar is attached to its panel with
``make_axes_locatable`` (``attach_colorbar``), so that it has exactly the height
of the panel; figures with such colorbars use the tight layout, since the
constrained layout does not account for the divided axes. Apart from the font
sizes and a light grid, the style is matplotlib's default.

The paper exists in two layouts, single column (cas-sc) and two columns
(cas-dc). The environment variable ``ASTRONOMIX_PAPER_LAYOUT=twocolumn``
switches the figure scripts to the two-column sizes (``TWO_COLUMN``,
``TEXT_WIDTH``, ``SINGLE_PANEL_WIDTH``); they then write to a ``twocolumn``
directory next to their usual output (``layout_path``), and figures with two
panels side by side stack them in one column.
"""

# general
import os
from pathlib import Path

# plotting
import matplotlib as mpl
from matplotlib.layout_engine import ConstrainedLayoutEngine
from mpl_toolkits.axes_grid1 import make_axes_locatable


#: Whether the figures are drawn for the two-column layout of the paper.
TWO_COLUMN = os.environ.get("ASTRONOMIX_PAPER_LAYOUT", "single") == "twocolumn"

#: The text width of the paper (468 pt single column, 494.5 pt two columns);
#: full-width figures are drawn at this width so that their 9 pt labels appear
#: at 9 pt in the paper.
TEXT_WIDTH = 6.84 if TWO_COLUMN else 6.5

#: The column width of the two-column layout (238.25 pt).
COLUMN_WIDTH = 3.30

#: The width of a single-panel figure: 0.6 of the text width in the single-column
#: layout, one column in the two-column layout.
SINGLE_PANEL_WIDTH = COLUMN_WIDTH if TWO_COLUMN else 0.6 * TEXT_WIDTH

FD_COLOR = "tab:blue"
FV_COLOR = "tab:orange"
FV_HLLE_COLOR = "tab:green"
ATHENAPK_COLOR = "tab:red"
ATHENAPK_THIRD_ORDER_COLOR = "tab:purple"
REFERENCE_COLOR = "black"
GUIDE_COLOR = "0.45"

#: Width of a colorbar and its gap to the panel, as fractions of the panel
#: width (``attach_colorbar``).
COLORBAR_SIZE = "5%"
COLORBAR_PAD = 0.05

#: Colors of the self-gravity couplings: the fourth-order coupling of the
#: finite difference scheme carries the finite difference color and the
#: Athena++ coupling of the finite volume scheme the finite volume color. The
#: work corrections of both get colors of their own, since their curves and
#: point clouds lie on those of the coupling they correct.
SIMPLE_SOURCE_COLOR = "tab:brown"
FLUX_BASED_COLOR = "tab:green"
FLUX_BASED_CORRECTED_COLOR = FD_COLOR
FLUX_CORRECTED_WORK_COLOR = "tab:purple"
ATHENA_COUPLING_COLOR = FV_COLOR
TIME_CENTRED_WORK_COLOR = "tab:pink"

SEQUENTIAL_COLORMAP = "viridis"
DIVERGING_COLORMAP = "RdBu_r"

PAPER_RC_PARAMETERS = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "figure.titlesize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "image.cmap": SEQUENTIAL_COLORMAP,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "svg.fonttype": "path",
}


def layout_path(path):
    """
    The output path of a figure in the current layout.

    Args:
        path: The path of the figure in the single-column layout.

    Returns:
        ``path`` itself, or in the two-column layout the same file name in a
        ``twocolumn`` directory next to it (created if needed), as the same type
        (``str`` or ``Path``) as ``path``.
    """
    if not TWO_COLUMN:
        return path
    two_column_path = Path(path).parent / "twocolumn" / Path(path).name
    two_column_path.parent.mkdir(parents=True, exist_ok=True)
    return two_column_path if isinstance(path, Path) else str(two_column_path)


def use_paper_style():
    """Reset matplotlib to its defaults and apply the font sizes and grid of the paper."""
    mpl.rcdefaults()
    mpl.rcParams.update(PAPER_RC_PARAMETERS)


def variant_style(color, linestyle="--"):
    """
    The line style of a variant of a scheme: the color of the scheme, a
    different line style and hollow markers.

    Args:
        color: The color of the scheme.
        linestyle: The line style of the variant.

    Returns:
        Keyword arguments for ``Axes.plot``.
    """
    return dict(
        color=color,
        linestyle=linestyle,
        markerfacecolor="white",
    )


def attach_colorbar(figure, mappable, axis, label=None, **colorbar_options):
    """
    Draw the colorbar of a panel in an axes split off its right side with
    ``make_axes_locatable``, so that the colorbar has exactly the height of the
    panel (also for an image of fixed aspect ratio).

    Args:
        figure: The figure.
        mappable: The image (or other mappable) the colorbar describes.
        axis: The panel.
        label: The colorbar label.
        **colorbar_options: Further keyword arguments for ``Figure.colorbar``.

    Returns:
        The colorbar.
    """
    colorbar_axis = make_axes_locatable(axis).append_axes("right", size=COLORBAR_SIZE, pad=COLORBAR_PAD)
    colorbar = figure.colorbar(mappable, cax=colorbar_axis, label=label, **colorbar_options)
    # The grid of the paper style would draw lines across the color scale.
    colorbar_axis.grid(False)
    return colorbar


def reserve_colorbar_space(axis):
    """
    Split off the same, invisible axes as ``attach_colorbar`` from a panel
    without a colorbar, so that it keeps the size of the panels next to it
    whose colorbar it shares (e.g. a row of images with one colorbar).

    Args:
        axis: The panel.
    """
    spacer = make_axes_locatable(axis).append_axes("right", size=COLORBAR_SIZE, pad=COLORBAR_PAD)
    spacer.set_visible(False)


def figure_legend(figure, handles=None, labels=None, ncols=None, location="outside lower center", **legend_options):
    """
    Place one framed legend for all panels of a figure outside the panels.

    With the constrained layout, the legend is placed ``outside`` and the
    layout makes room for it. Otherwise (figures with colorbars from
    ``attach_colorbar`` use the tight layout) it is placed below the panels and
    the tight layout is applied around it, so this has to be the last layout
    step of the figure. Without explicit handles, the handles of all axes are
    collected and duplicate labels are dropped.

    Args:
        figure: The figure.
        handles: The legend handles; by default those of all axes.
        labels: The legend labels; by default those of the handles.
        ncols: The number of legend columns; by default up to four.
        location: The legend location with the constrained layout, an
            ``outside`` location of matplotlib.
        **legend_options: Further keyword arguments for ``Figure.legend``.

    Returns:
        The legend.
    """
    if handles is None:
        handles_by_label = {}
        for axis in figure.axes:
            for handle, label in zip(*axis.get_legend_handles_labels()):
                handles_by_label.setdefault(label, handle)
        handles = list(handles_by_label.values())
        labels = list(handles_by_label)
    elif labels is None:
        labels = [handle.get_label() for handle in handles]
    if ncols is None:
        ncols = min(len(labels), 4)
    if isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine):
        return figure.legend(
            handles,
            labels,
            loc=location,
            ncols=ncols,
            frameon=True,
            **legend_options,
        )

    # Below the panels: the tight layout keeps the panels above the height of
    # the legend (in figure coordinates, plus a small gap).
    legend = figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=ncols,
        frameon=True,
        **legend_options,
    )
    legend_height = legend.get_window_extent(figure.canvas.get_renderer()).height / figure.bbox.height
    figure.tight_layout(rect=(0.0, legend_height + 0.01, 1.0, 1.0))
    return legend
