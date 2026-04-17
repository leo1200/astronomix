import numpy as np
import matplotlib.pyplot as plt


def add_power_law_indicators(
    ax,
    anchor,
    exponents,
    x_span=3.0,
    scales=None,
    labels=None,
    x_label='x',
    line_kwargs=None,
    text_kwargs=None,
):
    """
    Add power-law indicator lines of the form y ~ x^{p} to a log-log plot.
    """

    x0, y0 = anchor
    exponents = np.atleast_1d(exponents)

    if scales is None:
        scales = np.ones_like(exponents, dtype=float)
    if labels is None:
        labels = [rf' ${x_label}^{{{p}}}$' for p in exponents]

    if line_kwargs is None:
        line_kwargs = dict(color='k', linestyle='--', linewidth=1)
    if text_kwargs is None:
        text_kwargs = dict(fontsize=10, ha='left', va='center')

    # x range for indicator
    x_ref = np.array([x0, x0 * x_span])

    for p, s, label in zip(exponents, scales, labels):
        y_ref = s * y0 * (x_ref / x0) ** (p)

        ax.loglog(x_ref, y_ref, **line_kwargs)

        # place label at end of line
        ax.text(x_ref[-1], y_ref[-1], label, **text_kwargs)
