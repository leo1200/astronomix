import shutil


def _show_progress(
    iteration, total, prefix="", suffix="", decimals=1, fill="█", printEnd="\r"
) -> None:
    """
    Progress bar that adapts to terminal width and handles resizing.
    """
    # Get terminal width
    terminal_width = shutil.get_terminal_size((80, 20)).columns

    # Format percentage string
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))

    # Fixed parts (prefix + suffix + percent + " |" + "| " + spaces)
    fixed_part = f"{prefix} | | {percent}% {suffix}"
    fixed_length = len(fixed_part)

    # Compute bar length dynamically
    bar_length = max(10, terminal_width - fixed_length)

    # Compute filled length of the bar
    filledLength = int(bar_length * iteration // total)
    bar = fill * filledLength + "-" * (bar_length - filledLength)

    # Assemble full line
    progress_line = f"{prefix} |{bar}| {percent}% {suffix}"

    # Pad with spaces to ensure full overwrite (avoids leftovers)
    padded_line = progress_line.ljust(terminal_width)

    # Print progress line with carriage return
    print(f"\r{padded_line}", end=printEnd, flush=True)

    # Print newline when complete
    if iteration == total:
        print()
