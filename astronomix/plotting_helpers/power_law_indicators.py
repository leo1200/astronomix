"""
Draw power-law slope indicators on log-log plots.

Adds short reference lines of the form ``y ~ x^p`` (with optional vertical
offsets and labels) anchored at a chosen point, to make the slope of a spectrum
or convergence curve easy to read off by eye.
"""

# numerics
import numpy as np


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
    Add power-law indicator lines of the form ``y ~ x^p`` to a log-log plot.

    Args:
        ax: The (log-log) axes to draw the indicator lines on.
        anchor: The ``(x0, y0)`` point at which the indicators start.
        exponents: One or more power-law exponents ``p`` to draw.
        x_span: The multiplicative x-extent of each indicator line.
        scales: Optional per-exponent vertical scale factors (default 1).
        labels: Optional per-exponent label strings (default ``x^p``).
        x_label: The symbol used in the default labels.
        line_kwargs: Keyword arguments forwarded to the line plot.
        text_kwargs: Keyword arguments forwarded to the label text.
    """

    x0, y0 = anchor
    exponents = np.atleast_1d(exponents)

    if scales is None:
        scales = np.ones_like(exponents, dtype=float)
    if labels is None:
        labels = [rf' ${x_label}^{{{exponent}}}$' for exponent in exponents]

    if line_kwargs is None:
        line_kwargs = dict(color='k', linestyle='--', linewidth=1)
    if text_kwargs is None:
        text_kwargs = dict(fontsize=10, ha='left', va='center')

    # The indicator spans one multiplicative decade-fraction in x from the
    # anchor, set by ``x_span``.
    x_ref = np.array([x0, x0 * x_span])

    for exponent, scale, label in zip(exponents, scales, labels):
        y_ref = scale * y0 * (x_ref / x0) ** exponent

        ax.loglog(x_ref, y_ref, **line_kwargs)

        # Place the label at the far end of the indicator line.
        ax.text(x_ref[-1], y_ref[-1], label, **text_kwargs)
