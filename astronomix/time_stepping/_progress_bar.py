"""
Host-side progress bar for the time-integration loop.

Renders a single-line, terminal-width-aware progress bar that is driven from
inside the jitted loop via ``jax.debug.callback``. The "iteration" it is fed is
the simulation time, so the bar tracks progress towards ``t_end``.
"""

# general
import math
import shutil


def _show_progress(
    iteration, total, prefix="", suffix="", decimals=1, fill="█", printEnd="\r"
) -> None:
    """
    Render one frame of the progress bar, sized to the current terminal width.

    Args:
        iteration: The current progress value (the simulation time).
        total: The value of ``iteration`` at which the bar is full (``t_end``).
        prefix: Text printed before the bar.
        suffix: Text printed after the percentage.
        decimals: Number of decimal places shown in the percentage.
        fill: Character used for the filled portion of the bar.
        printEnd: Line terminator; ``"\\r"`` keeps overwriting the same line.
    """
    # On a blow-up the simulation time goes non-finite, and ``int(NaN)`` would
    # raise and abort the whole run. Clamp to ``total`` so the bar finishes
    # cleanly instead of crashing; the diagnostics elsewhere report the NaN.
    try:
        if not math.isfinite(float(iteration)):
            iteration = total
    except (TypeError, ValueError):
        iteration = total

    # Recompute the terminal width every frame so the bar keeps filling the
    # line correctly even if the terminal is resized mid-run.
    terminal_width = shutil.get_terminal_size((80, 20)).columns

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))

    # Size the bar so the whole line fits the terminal: subtract the fixed
    # decorations (prefix, suffix, percentage, separators) from the width, and
    # never shrink below a readable minimum.
    fixed_part = f"{prefix} | | {percent}% {suffix}"
    fixed_length = len(fixed_part)
    bar_length = max(10, terminal_width - fixed_length)

    filled_length = int(bar_length * iteration // total)
    bar = fill * filled_length + "-" * (bar_length - filled_length)

    progress_line = f"{prefix} |{bar}| {percent}% {suffix}"

    # Pad the line out to the full terminal width so a shorter line never leaves
    # leftover characters from the previous, longer frame.
    padded_line = progress_line.ljust(terminal_width)

    print(f"\r{padded_line}", end=printEnd, flush=True)

    # Drop to a fresh line once the bar is full so subsequent output is clean.
    if iteration == total:
        print()
