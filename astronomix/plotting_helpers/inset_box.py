"""
Add a zoomed-in inset axes to an existing matplotlib axes.

Replicates the line and scatter artists of a parent axes inside a magnified
inset box (preserving styling such as colors, markers, sizes and colormaps)
and draws connector lines between the inset and the highlighted region. Useful
for emphasizing a small feature of a plot without a second figure.
"""

# numerics
import numpy as np

# plotting (optional dependency — this helper only runs when matplotlib is
# installed, so guard the import to keep the module importable without it)
try:
    from matplotlib.collections import PathCollection
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
except ModuleNotFoundError:  # pragma: no cover - exercised only without matplotlib
    PathCollection = None
    inset_axes = mark_inset = None


def add_inset_box(
    ax,
    x1,
    x2,
    y1,
    y2,
    loc="lower left",
    connect_loc1=2,
    connect_loc2=4,
    width="40%",
    height="40%",
):
    """Add a magnified inset axes mirroring the artists of ``ax``.

    Args:
        ax: The parent axes whose line and scatter artists are replicated.
        x1: The left edge of the zoomed-in x-range.
        x2: The right edge of the zoomed-in x-range.
        y1: The lower edge of the zoomed-in y-range.
        y2: The upper edge of the zoomed-in y-range.
        loc: The location of the inset within the parent axes.
        connect_loc1: The first corner used for the inset connector lines.
        connect_loc2: The second corner used for the inset connector lines.
        width: The inset width (as a fraction string or absolute size).
        height: The inset height (as a fraction string or absolute size).

    Returns:
        The created inset axes.
    """
    if inset_axes is None:
        raise ModuleNotFoundError(
            "matplotlib is required for add_inset_box but is not installed. "
            "Install it with `pip install matplotlib`."
        )

    axins = inset_axes(ax, width=width, height=height, loc=loc)

    for line in ax.get_lines():
        axins.plot(
            line.get_xdata(),
            line.get_ydata(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            color=line.get_color(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            alpha=line.get_alpha(),
        )

    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue

        offsets = collection.get_offsets()
        if offsets is None or len(offsets) == 0:
            continue

        offsets = np.asarray(offsets)
        if np.ma.isMaskedArray(offsets):
            offsets = offsets.filled(np.nan)

        facecolors = collection.get_facecolors()
        edgecolors = collection.get_edgecolors()

        has_no_facecolor = facecolors is None or len(facecolors) == 0

        scatter_kwargs = {
            "alpha": collection.get_alpha(),
            "linewidths": collection.get_linewidths(),
        }

        paths = collection.get_paths()
        if len(paths) > 0:
            scatter_kwargs["marker"] = paths[0]

        sizes = collection.get_sizes()
        if sizes is not None and len(sizes) > 0:
            scatter_kwargs["s"] = sizes

        if has_no_facecolor:
            scatter_kwargs["facecolors"] = "none"

            if edgecolors is not None and len(edgecolors) > 0:
                scatter_kwargs["edgecolors"] = edgecolors
            else:
                scatter_kwargs["edgecolors"] = collection.get_edgecolor()

        else:
            array = collection.get_array()

            if array is not None and len(array) == len(offsets):
                scatter_kwargs["c"] = np.asarray(array)
                scatter_kwargs["cmap"] = collection.cmap
                scatter_kwargs["norm"] = collection.norm
            else:
                scatter_kwargs["facecolors"] = facecolors

            if edgecolors is not None and len(edgecolors) > 0:
                scatter_kwargs["edgecolors"] = edgecolors

        axins.scatter(
            offsets[:, 0],
            offsets[:, 1],
            **scatter_kwargs,
        )

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.tick_params(labelleft=False, labelbottom=False)
    axins.set_xticks([])
    axins.set_yticks([])

    mark_inset(
        ax,
        axins,
        loc1=connect_loc1,
        loc2=connect_loc2,
        fc="none",
        ec="0.5",
    )

    return axins