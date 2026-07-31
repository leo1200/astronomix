"""
Host-side progress bar for the time-integration loop.

Renders a single-line, terminal-width-aware progress bar that is driven from
inside the jitted loop via ``jax.debug.callback``. The "iteration" it is fed is
the simulation time, so the bar tracks progress towards ``t_end``.
"""

# general
import math
import shutil
import sys
import time

# Non-interactive (redirected-log) mode state: last progress step (in 0.5%
# increments), the wall time of the last emitted line, and the previous
# callback's simulation time. The callback fires once per solver step, so the
# difference between consecutive simulation times IS the current ``dt`` — a
# collapsing ``dt`` (the classic blast-run failure mode) is visible directly
# in the log, and the wall-time heartbeat keeps emitting lines even when the
# percentage stalls, so a stalled run can never look identical to a slow one.
_last_logged_step = None
_last_logged_wall = None
_last_sim_time = None
_HEARTBEAT_SECONDS = 60.0


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

    # When stdout is not a terminal (queued/redirected runs) a carriage-return
    # bar would flood the log with full-width frames. Emit a plain, throttled
    # progress line instead: one line per 0.5% of progress, plus a heartbeat
    # line at least every _HEARTBEAT_SECONDS that includes the current per-step
    # dt (inferred from consecutive callback times).
    if not sys.stdout.isatty():
        global _last_logged_step, _last_logged_wall, _last_sim_time
        t_now = float(iteration)
        dt = None if _last_sim_time is None else t_now - _last_sim_time
        _last_sim_time = t_now
        percent = 100.0 * t_now / float(total)
        step = int(percent * 2)
        wall = time.monotonic()
        due = (_last_logged_wall is None
               or wall - _last_logged_wall >= _HEARTBEAT_SECONDS)
        if step == _last_logged_step and not due:
            return
        _last_logged_step = step
        _last_logged_wall = wall
        dt_note = "" if dt is None else f"  dt = {dt:.3e}"
        print(
            f"{prefix}progress {percent:5.1f}%  "
            f"t = {t_now:.6g} / {float(total):.6g}{dt_note} {suffix}".rstrip(),
            flush=True,
        )
        return

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
