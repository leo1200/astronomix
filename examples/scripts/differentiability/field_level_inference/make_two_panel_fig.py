"""
One-row version of the field-level-inference figure for the paper.

The graphical abstract keeps the 2x2 figure of ``make_panels_fig.py`` (initial
optimized state, shortly before, at and shortly after the loss time). The paper
figure shows the initial optimized state and the state at the loss time side by
side. The two panels are cut from the rendered 2x2 image (volume rendering needs
an EGL context, see ``make_panels_fig.py``) without their titles and set in one
row at the size at which the figure appears in the paper, with titles in the
paper style; ``ASTRONOMIX_PAPER_LAYOUT=twocolumn`` draws it for one column.

    python make_two_panel_fig.py --panels field_level_inference.png --out field_level_inference_row.png
"""

# general
import argparse
from pathlib import Path

# numerics
import numpy as np
from PIL import Image

# plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# astronomix plotting style
from astronomix.plotting_helpers.paper_style import (  # noqa: E402
    COLUMN_WIDTH,
    TEXT_WIDTH,
    TWO_COLUMN,
    layout_path,
    use_paper_style,
)

#: Gray level above which a pixel counts as background.
BACKGROUND = 245


def bands(mask):
    """The ``(first, last)`` index ranges of the runs of ``True`` in a 1D mask."""
    indices = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]]))
    return list(zip(starts, ends))


def panels_of(image):
    """
    Cut the four panels out of the rendered 2x2 figure.

    The figure consists of two rows, each a line of titles above a row of two
    panels, separated by background. The panels are the taller row bands
    (the titles are the thin ones), split at the background column band.

    Args:
        image: The RGB image of the 2x2 figure.

    Returns:
        The four panels (without titles) in reading order: top left, top
        right, bottom left, bottom right.
    """
    content = np.asarray(image).astype(int).sum(axis=2) < 3 * BACKGROUND
    row_bands = [band for band in bands(content.any(axis=1)) if band[1] - band[0] > 100]
    column_bands = bands(content.any(axis=0))
    panels = []
    for top, bottom in row_bands:
        for left, right in column_bands:
            region = content[top : bottom + 1, left : right + 1]
            rows = np.flatnonzero(region.any(axis=1))
            columns = np.flatnonzero(region.any(axis=0))
            panels.append(
                image.crop((left + columns[0], top + rows[0], left + columns[-1] + 1, top + rows[-1] + 1))
            )
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panels", type=Path, required=True, help="the rendered 2x2 figure")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    initial, _, target, _ = panels_of(Image.open(arguments.panels).convert("RGB"))
    use_paper_style()
    width = COLUMN_WIDTH if TWO_COLUMN else TEXT_WIDTH
    figure, axes = plt.subplots(1, 2, figsize=(width, 0.53 * width))
    for axis, panel, title in zip(
        axes,
        (initial, target),
        ("initial optimized state", r"target time ($t_\mathrm{loss}$)"),
    ):
        axis.imshow(np.asarray(panel))
        axis.set_title(title)
        axis.set_axis_off()
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9, wspace=0.03)
    path = layout_path(arguments.out)
    figure.savefig(path, dpi=600 if TWO_COLUMN else 300)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
