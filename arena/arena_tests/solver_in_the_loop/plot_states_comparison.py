from pathlib import Path
import os
from jax import Array
import matplotlib.pyplot as plt
from matplotlib import animation, legend
from matplotlib.colors import Normalize
from IPython.display import HTML
import jax.numpy as jnp
from typing import Optional
import logging
from arena.arena_tests.solver_in_the_loop.utils import downaverage

logger = logging.Logger(__name__)


def plot_and_animate_states(
    states_list,
    z_levels,
    timepoints,
    slice_axis: str = "z",
    titles=None,
    vmin=0,
    vmax=1,
    save_path: Optional[str | Path] = None,
):
    """
    Plot and animate multiple simulation states (e.g. LR, HR, etc.)

    Args:
        states_list: list of 5D arrays [batch, channel, x, y, z]
        z_levels: list of indices corresponding to each state array along slice_axis
        timepoints: jax array with the time of each snapshot
        slice_axis: axis to slice along ("x", "y", or "z")
        titles: optional list of titles (e.g. ["LR", "HR"])
        vmin, vmax: normalization range for color scaling
        save_path: optional path to a folder where the animation will be saved
    """
    # assert all([len(states) == 5 for states in states_list])

    n_states = len(states_list)
    n_fields = 4  # density, velocity magnitude, pressure, magnetic field

    if titles is None:
        titles = [f"State {i + 1}" for i in range(n_states)]

    if slice_axis not in {"x", "y", "z"}:
        raise ValueError(
            f"Invalid slice_axis '{slice_axis}', must be 'x', 'y', or 'z'."
        )

    if slice_axis == "z":
        axis_labels = ("x", "y")
    elif slice_axis == "y":
        axis_labels = ("x", "z")
    else:
        axis_labels = ("y", "z")

    def _slice_field(field, level):
        if slice_axis == "z":
            return field[:, :, level].T
        if slice_axis == "y":
            return field[:, level, :].T
        return field[level, :, :].T

    fig, axs = plt.subplots(n_states, n_fields, figsize=(5 * n_fields, 5 * n_states))
    if n_states == 1:
        axs = axs[None, :]  # handle single row case

    # Store all imshow handles for animation
    caxs = []

    for row, (states, slice_level, title_prefix) in enumerate(
        zip(states_list, z_levels, titles, strict=True)
    ):
        # --- Field 1: Density ---
        c0 = axs[row][0].imshow(
            _slice_field(states[0, 0], slice_level),
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c0, ax=axs[row][0])
        axs[row][0].set_title(f"{title_prefix} - Density")
        axs[row][0].set_xlabel(axis_labels[0])
        axs[row][0].set_ylabel(axis_labels[1])

        # --- Field 2: Velocity magnitude ---
        c1 = axs[row][1].imshow(
            _slice_field(
                jnp.sqrt(states[0, 1] ** 2 + states[0, 2] ** 2 + states[0, 3] ** 2),
                slice_level,
            ),
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c1, ax=axs[row][1])
        axs[row][1].set_title(f"{title_prefix} - Velocity Magnitude")

        # --- Field 3: Pressure ---
        c2 = axs[row][2].imshow(
            _slice_field(states[0, 4], slice_level),
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c2, ax=axs[row][2])
        axs[row][2].set_title(f"{title_prefix} - Pressure")

        # --- Field 4: Magnetic field magnitude ---
        c3 = axs[row][3].imshow(
            _slice_field(
                jnp.sqrt(states[0, 5] ** 2 + states[0, 6] ** 2 + states[0, 7] ** 2),
                slice_level,
            ),
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c3, ax=axs[row][3])
        axs[row][3].set_title(f"{title_prefix} - Magnetic Field")

        # Store for animation
        caxs.append((c0, c1, c2, c3))

    plt.tight_layout()

    # --- Animation function ---
    n_snapshots = states_list[0].shape[0]

    def animate_all(i):
        fig.suptitle(
            f"Snapshot {i} / {n_snapshots - 1} | t={float(timepoints[i])}",
            fontsize=16,
        )
        updated = []
        for (states, slice_level), (c0, c1, c2, c3) in zip(
            zip(states_list, z_levels, strict=True), caxs, strict=True
        ):
            c0.set_array(_slice_field(states[i, 0], slice_level))
            c1.set_array(
                _slice_field(
                    jnp.sqrt(states[i, 1] ** 2 + states[i, 2] ** 2 + states[i, 3] ** 2),
                    slice_level,
                )
            )
            c2.set_array(_slice_field(states[i, 4], slice_level))
            c3.set_array(
                _slice_field(
                    jnp.sqrt(states[i, 5] ** 2 + states[i, 6] ** 2 + states[i, 7] ** 2),
                    slice_level,
                )
            )
            updated.extend([c0, c1, c2, c3])
        return updated

    # Use frame count from first state
    ani = animation.FuncAnimation(
        fig, animate_all, frames=n_snapshots, interval=1000 / 4
    )

    if save_path is not None:
        import os

        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, "animation.mp4")
        ani.save(filepath, writer="ffmpeg", dpi=150)
        logger.info(f"Animation saved to {filepath}")

    plt.show()
    html_video = HTML(ani.to_html5_video())
    return html_video


def animate_several_sims_states_with_diagonal(
    sim_states: list[Array],
    sim_legend: list[str],
    z_levels: list[int],
    timepoints: Array,
    slice_axis: str = "z",
    save_path: Optional[str] = None,
    file_name: Optional[str] = None,
    fps: float = 1000 / 150,
):
    """
    Plot and animate a single simulation state with diagonal magnetic field values.

    Args:
        sim_states: list of 5D array [batch, channel, x, y, z]
        sim_legend: list of string corresponding to the title of the simulations
        z_levels: list of indices corresponding to the slice along slice_axis,
            one per simulation
        timepoints: jax array with the time of each snapshot
        slice_axis: axis to slice along ("x", "y", or "z")
        save_path: optional path to a folder where the animation will be saved
        file_name: optional file name for the animation, if not animation.mp4 will be used
        fps: frames per second of the animation
    """
    if slice_axis not in {"x", "y", "z"}:
        raise ValueError(
            f"Invalid slice_axis '{slice_axis}', must be 'x', 'y', or 'z'."
        )

    if slice_axis == "z":
        axis_labels = ("x", "y")
    elif slice_axis == "y":
        axis_labels = ("x", "z")
    else:
        axis_labels = ("y", "z")

    def _slice_field(field, level):
        if slice_axis == "z":
            return field[:, :, level].T
        if slice_axis == "y":
            return field[:, level, :].T
        return field[level, :, :].T

    n_sims = len(sim_states)
    resolutions = []
    for sim in sim_states:
        resolutions.append(sim.shape[-1])

    min_resolution = min(resolutions)
    downaverage_factors = [r // min_resolution for r in resolutions]
    min_z_level = min(z_levels)
    if len(sim_legend) != n_sims:
        raise ValueError("`sim_legend` and `sim_states` must have the same length.")
    if len(z_levels) != n_sims:
        raise ValueError("`z_levels` and `sim_states` must have the same length.")
    n_fields = 4  # density, velocity magnitude, pressure, magnetic field
    fig, axs = plt.subplots(
        n_sims + 1, n_fields, figsize=(5 * n_fields, 5 * (n_sims + 1)), squeeze=False
    )
    sim_axs = axs[:n_sims]
    axs_diag = axs[n_sims]

    cs = []
    diag_lines = []
    sims_diag_series = []
    diag_titles = [
        "Density along diagonal",
        "Velocity magnitude along diagonal",
        "Pressure along diagonal",
        "Magnetic field magnitude along diagonal",
    ]
    diag_ymins = [jnp.inf] * n_fields
    diag_ymaxs = [-jnp.inf] * n_fields
    diag_xmax = 0.0

    for sim_index in range(n_sims):
        axs = sim_axs[sim_index]
        states = sim_states[sim_index]
        z_level = z_levels[sim_index]
        sim_title = sim_legend[sim_index]
        downaverage_factor = downaverage_factors[sim_index]
        c0 = axs[0].imshow(
            _slice_field(states[0, 0], z_level),
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c0, ax=axs[0])
        axs[0].set_title(f"{sim_title} - Density")
        axs[0].set_xlabel(axis_labels[0])
        axs[0].set_ylabel(axis_labels[1])

        c1 = axs[1].imshow(
            _slice_field(
                jnp.sqrt(states[0, 1] ** 2 + states[0, 2] ** 2 + states[0, 3] ** 2),
                z_level,
            ),
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c1, ax=axs[1])
        axs[1].set_title(f"{sim_title} - Velocity Magnitude")

        c2 = axs[2].imshow(
            _slice_field(states[0, 4], z_level),
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c2, ax=axs[2])
        axs[2].set_title(f"{sim_title} - Pressure")

        c3 = axs[3].imshow(
            _slice_field(
                jnp.sqrt(states[0, 5] ** 2 + states[0, 6] ** 2 + states[0, 7] ** 2),
                z_level,
            ),
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c3, ax=axs[3])
        axs[3].set_title(f"{sim_title} - Magnetic Field")

        cs.append((c0, c1, c2, c3))

        if downaverage_factor == 1:
            rescaled_states = states
        else:
            rescaled_states = downaverage(states, downaverage_factor)

        density_series = rescaled_states[:, 0]
        velocity_series = jnp.sqrt(
            rescaled_states[:, 1] ** 2
            + rescaled_states[:, 2] ** 2
            + rescaled_states[:, 3] ** 2
        )
        pressure_series = rescaled_states[:, 4]
        magnetic_series = jnp.sqrt(
            rescaled_states[:, 5] ** 2
            + rescaled_states[:, 6] ** 2
            + rescaled_states[:, 7] ** 2
        )

        if slice_axis == "z":
            diag_size = min(rescaled_states.shape[2], rescaled_states.shape[3])
            diag_indices = jnp.arange(diag_size)
            density_diag = density_series[:, diag_indices, diag_indices, min_z_level]
            velocity_diag = velocity_series[:, diag_indices, diag_indices, min_z_level]
            pressure_diag = pressure_series[:, diag_indices, diag_indices, min_z_level]
            magnetic_diag = magnetic_series[:, diag_indices, diag_indices, min_z_level]
        elif slice_axis == "y":
            diag_size = min(rescaled_states.shape[2], rescaled_states.shape[4])
            diag_indices = jnp.arange(diag_size)
            density_diag = density_series[:, diag_indices, min_z_level, diag_indices]
            velocity_diag = velocity_series[:, diag_indices, min_z_level, diag_indices]
            pressure_diag = pressure_series[:, diag_indices, min_z_level, diag_indices]
            magnetic_diag = magnetic_series[:, diag_indices, min_z_level, diag_indices]
        else:
            diag_size = min(rescaled_states.shape[3], rescaled_states.shape[4])
            diag_indices = jnp.arange(diag_size)
            density_diag = density_series[:, min_z_level, diag_indices, diag_indices]
            velocity_diag = velocity_series[:, min_z_level, diag_indices, diag_indices]
            pressure_diag = pressure_series[:, min_z_level, diag_indices, diag_indices]
            magnetic_diag = magnetic_series[:, min_z_level, diag_indices, diag_indices]

        r_diag = jnp.sqrt(2.0) * diag_indices
        diag_xmax = max(diag_xmax, float(r_diag[-1]))
        diag_series = [density_diag, velocity_diag, pressure_diag, magnetic_diag]
        sims_diag_series.append(diag_series)
        sim_diag_lines = []
        for field_index, (ax, series, series_title) in enumerate(
            zip(axs_diag, diag_series, diag_titles, strict=True)
        ):
            (line,) = ax.plot(r_diag, series[0], label=sim_title)
            ax.set_xlabel("diagonal")
            ax.set_ylabel("value")
            ax.set_title(series_title)
            finite_mask = jnp.isfinite(series)
            finite_vals = series[finite_mask]
            if int(finite_vals.size) != 0:
                diag_ymins[field_index] = jnp.minimum(
                    diag_ymins[field_index], jnp.min(finite_vals)
                )
                diag_ymaxs[field_index] = jnp.maximum(
                    diag_ymaxs[field_index], jnp.max(finite_vals)
                )
            sim_diag_lines.append(line)
        diag_lines.append(sim_diag_lines)

    for field_index, ax in enumerate(axs_diag):
        ax.set_xlim(0.0, diag_xmax)
        ymin = diag_ymins[field_index]
        ymax = diag_ymaxs[field_index]
        if (not bool(jnp.isfinite(ymin))) or (not bool(jnp.isfinite(ymax))):
            logger.warning(
                "Diagonal series has no finite values; using default y-limits."
            )
            ymin, ymax = 0.0, 1.0
        else:
            ymin = float(ymin.astype(float))
            ymax = float(ymax.astype(float))
            if ymin == ymax:
                pad = abs(ymin) * 1e-6 if ymin != 0.0 else 1e-6
                ymin -= pad
                ymax += pad
        ax.set_ylim(ymin, ymax)
        ax.legend()

    fig.tight_layout()

    n_snapshots = sim_states[0].shape[0]

    def animate_all(i):
        fig.suptitle(
            f"Snapshot {i} / {n_snapshots - 1} | t={float(timepoints[i])}",
            fontsize=16,
        )
        all_plots = []
        for state_index in range(n_sims):
            c0, c1, c2, c3 = cs[state_index]
            diag_series = sims_diag_series[state_index]
            sim_diag_lines = diag_lines[state_index]
            states = sim_states[state_index]
            z_level = z_levels[state_index]
            c0.set_array(_slice_field(states[i, 0], z_level))
            c1.set_array(
                _slice_field(
                    jnp.sqrt(states[i, 1] ** 2 + states[i, 2] ** 2 + states[i, 3] ** 2),
                    z_level,
                )
            )
            c2.set_array(_slice_field(states[i, 4], z_level))
            c3.set_array(
                _slice_field(
                    jnp.sqrt(states[i, 5] ** 2 + states[i, 6] ** 2 + states[i, 7] ** 2),
                    z_level,
                )
            )
            for line, series in zip(sim_diag_lines, diag_series, strict=True):
                line.set_ydata(series[i])
            all_plots.extend([c0, c1, c2, c3, *sim_diag_lines])
        return all_plots

    ani = animation.FuncAnimation(
        fig, animate_all, frames=n_snapshots, interval=1000 / 4
    )

    if save_path is not None:
        import os

        os.makedirs(save_path, exist_ok=True)
        if file_name is None:
            file_name = "animation"
        filepath = os.path.join(save_path, f"{file_name}.mp4")
        ani.save(filepath, writer="ffmpeg", dpi=1000 / fps)
        logger.info(f"Animation saved to {filepath}")

    plt.show()
    html_video = HTML(ani.to_html5_video())
    return html_video


def plot_and_animate_states_with_diagonal(
    states: Array,
    z_level: int,
    timepoints: Array,
    slice_axis: str = "z",
    title: str = "State",
    vmin=0,
    vmax=1,
    save_path: Optional[str] = None,
    file_name: Optional[str] = None,
    fps: float = 1000 / 150,
):
    """
    Plot and animate a single simulation state with diagonal magnetic field values.

    Args:
        states: 5D array [batch, channel, x, y, z]
        z_level: index corresponding to the slice along slice_axis
        timepoints: jax array with the time of each snapshot
        slice_axis: axis to slice along ("x", "y", or "z")
        title: optional title prefix
        vmin, vmax: normalization range for color scaling
        save_path: optional path to a folder where the animation will be saved
        file_name: optional file name for the animation, if not animation.mp4 will be used
    """
    if slice_axis not in {"x", "y", "z"}:
        raise ValueError(
            f"Invalid slice_axis '{slice_axis}', must be 'x', 'y', or 'z'."
        )

    if slice_axis == "z":
        axis_labels = ("x", "y")
    elif slice_axis == "y":
        axis_labels = ("x", "z")
    else:
        axis_labels = ("y", "z")

    def _slice_field(field, level):
        if slice_axis == "z":
            return field[:, :, level].T
        if slice_axis == "y":
            return field[:, level, :].T
        return field[level, :, :].T

    n_fields = 4  # density, velocity magnitude, pressure, magnetic field
    fig = plt.figure(figsize=(5 * n_fields, 10))
    gs = fig.add_gridspec(2, n_fields)
    axs = [fig.add_subplot(gs[0, i]) for i in range(n_fields)]
    axs_diag = [fig.add_subplot(gs[1, i]) for i in range(n_fields)]

    c0 = axs[0].imshow(
        _slice_field(states[0, 0], z_level),
        origin="lower",
        # norm=Normalize(vmin=vmin, vmax=vmax),
    )
    fig.colorbar(c0, ax=axs[0])
    axs[0].set_title(f"{title} - Density")
    axs[0].set_xlabel(axis_labels[0])
    axs[0].set_ylabel(axis_labels[1])

    c1 = axs[1].imshow(
        _slice_field(
            jnp.sqrt(states[0, 1] ** 2 + states[0, 2] ** 2 + states[0, 3] ** 2),
            z_level,
        ),
        origin="lower",
        # norm=Normalize(vmin=vmin, vmax=vmax),
    )
    fig.colorbar(c1, ax=axs[1])
    axs[1].set_title(f"{title} - Velocity Magnitude")

    c2 = axs[2].imshow(
        _slice_field(states[0, 4], z_level),
        origin="lower",
        # norm=Normalize(vmin=vmin, vmax=vmax),
    )
    fig.colorbar(c2, ax=axs[2])
    axs[2].set_title(f"{title} - Pressure")

    c3 = axs[3].imshow(
        _slice_field(
            jnp.sqrt(states[0, 5] ** 2 + states[0, 6] ** 2 + states[0, 7] ** 2),
            z_level,
        ),
        origin="lower",
        # norm=Normalize(vmin=vmin, vmax=vmax),
    )
    fig.colorbar(c3, ax=axs[3])
    axs[3].set_title(f"{title} - Magnetic Field")

    density_series = states[:, 0]
    velocity_series = jnp.sqrt(
        states[:, 1] ** 2 + states[:, 2] ** 2 + states[:, 3] ** 2
    )
    pressure_series = states[:, 4]
    magnetic_series = jnp.sqrt(
        states[:, 5] ** 2 + states[:, 6] ** 2 + states[:, 7] ** 2
    )
    if slice_axis == "z":
        diag_size = min(states.shape[2], states.shape[3])
        diag_indices = jnp.arange(diag_size)
        density_diag = density_series[:, diag_indices, diag_indices, z_level]
        velocity_diag = velocity_series[:, diag_indices, diag_indices, z_level]
        pressure_diag = pressure_series[:, diag_indices, diag_indices, z_level]
        magnetic_diag = magnetic_series[:, diag_indices, diag_indices, z_level]
    elif slice_axis == "y":
        diag_size = min(states.shape[2], states.shape[4])
        diag_indices = jnp.arange(diag_size)
        density_diag = density_series[:, diag_indices, z_level, diag_indices]
        velocity_diag = velocity_series[:, diag_indices, z_level, diag_indices]
        pressure_diag = pressure_series[:, diag_indices, z_level, diag_indices]
        magnetic_diag = magnetic_series[:, diag_indices, z_level, diag_indices]
    else:
        diag_size = min(states.shape[3], states.shape[4])
        diag_indices = jnp.arange(diag_size)
        density_diag = density_series[:, z_level, diag_indices, diag_indices]
        velocity_diag = velocity_series[:, z_level, diag_indices, diag_indices]
        pressure_diag = pressure_series[:, z_level, diag_indices, diag_indices]
        magnetic_diag = magnetic_series[:, z_level, diag_indices, diag_indices]

    r_diag = jnp.sqrt(2.0) * diag_indices
    diag_series = [density_diag, velocity_diag, pressure_diag, magnetic_diag]
    diag_titles = [
        "Density along diagonal",
        "Velocity magnitude along diagonal",
        "Pressure along diagonal",
        "Magnetic field magnitude along diagonal",
    ]
    diag_lines = []
    for ax, series, series_title in zip(
        axs_diag, diag_series, diag_titles, strict=True
    ):
        (line,) = ax.plot(r_diag, series[0])
        ax.set_xlabel("diagonal")
        ax.set_ylabel("value")
        ax.set_title(f"{title} - {series_title}")
        ax.set_xlim(float(r_diag[0]), float(r_diag[-1]))
        finite_mask = jnp.isfinite(series)
        finite_vals = series[finite_mask]
        if int(finite_vals.size) == 0:
            logger.warning(
                "Diagonal series has no finite values; using default y-limits."
            )
            ymin, ymax = 0.0, 1.0
        else:
            ymin = float(jnp.min(finite_vals).astype(float))
            ymax = float(jnp.max(finite_vals).astype(float))
            if ymin == ymax:
                pad = abs(ymin) * 1e-6 if ymin != 0.0 else 1e-6
                ymin -= pad
                ymax += pad
        ax.set_ylim(ymin, ymax)
        diag_lines.append(line)

    fig.tight_layout()

    n_snapshots = states.shape[0]

    def animate_all(i):
        fig.suptitle(
            f"Snapshot {i} / {n_snapshots - 1} | t={float(timepoints[i])}",
            fontsize=16,
        )
        c0.set_array(_slice_field(states[i, 0], z_level))
        c1.set_array(
            _slice_field(
                jnp.sqrt(states[i, 1] ** 2 + states[i, 2] ** 2 + states[i, 3] ** 2),
                z_level,
            )
        )
        c2.set_array(_slice_field(states[i, 4], z_level))
        c3.set_array(
            _slice_field(
                jnp.sqrt(states[i, 5] ** 2 + states[i, 6] ** 2 + states[i, 7] ** 2),
                z_level,
            )
        )
        for line, series in zip(diag_lines, diag_series, strict=True):
            line.set_ydata(series[i])
        return [c0, c1, c2, c3, *diag_lines]

    ani = animation.FuncAnimation(
        fig, animate_all, frames=n_snapshots, interval=1000 / 4
    )

    if save_path is not None:
        import os

        os.makedirs(save_path, exist_ok=True)
        if file_name is None:
            file_name = "animation"
        filepath = os.path.join(save_path, f"{file_name}.mp4")
        ani.save(filepath, writer="ffmpeg", dpi=1000 / fps)
        logger.info(f"Animation saved to {filepath}")

    plt.show()
    html_video = HTML(ani.to_html5_video())
    return html_video


def plot_states(
    states_list: list[Array],
    z_levels: list[int],
    fig_name: str = "states_comparison",
    model_name: Optional[str] = None,
    folder: Optional[str] = None,
    titles: Optional[list[str]] = None,
    slice_axis: str = "z",
):
    """
    Plot and animate multiple simulation states (e.g. LR, HR, etc.)

    Args:
        states_list: list of 4D arrays [channel, x, y, z]
        z_levels: list of z-level indices corresponding to each state array
        titles: optional list of titles (e.g. ["LR", "HR"])
        vmin, vmax: normalization range for color scaling
    """

    n_states = len(states_list)
    n_fields = 4  # density, velocity magnitude, pressure, magnetic field

    # NOTE: extra float added due to linter
    vmax_density = max(jnp.max(states[0]).astype(float) for states in states_list)
    vmin_density = min(jnp.min(states[0]).astype(float) for states in states_list)
    vmax_pressure = max(jnp.max(states[4]).astype(float) for states in states_list)
    vmin_pressure = min(jnp.min(states[4]).astype(float) for states in states_list)
    vmin_speed = min(
        jnp.min(jnp.sqrt(states[1] ** 2 + states[2] ** 2 + states[3] ** 2)).astype(
            float
        )
        for states in states_list
    )
    vmax_speed = max(
        jnp.max(jnp.sqrt(states[1] ** 2 + states[2] ** 2 + states[3] ** 2)).astype(
            float
        )
        for states in states_list
    )
    vmin_mag = min(
        jnp.min(jnp.sqrt(states[5] ** 2 + states[6] ** 2 + states[7] ** 2)).astype(
            float
        )
        for states in states_list
    )
    vmax_mag = max(
        jnp.max(jnp.sqrt(states[5] ** 2 + states[6] ** 2 + states[7] ** 2)).astype(
            float
        )
        for states in states_list
    )

    if titles is None:
        titles = [f"State {i + 1}" for i in range(n_states)]

    if slice_axis == "z":
        axis_labels = ("x", "y")
    elif slice_axis == "y":
        axis_labels = ("x", "z")
    else:
        axis_labels = ("y", "z")

    def _slice_field(field, level):
        if slice_axis == "z":
            return field[:, :, level].T
        if slice_axis == "y":
            return field[:, level, :].T
        return field[level, :, :].T

    fig, axs = plt.subplots(n_states, n_fields, figsize=(5 * n_fields, 5 * n_states))
    if n_states == 1:
        axs = axs[None, :]  # handle single row case

    for row, (states, z_level, title_prefix) in enumerate(
        zip(states_list, z_levels, titles, strict=True)
    ):
        # --- Field 1: Density ---
        c0 = axs[row][0].imshow(
            _slice_field(states[0], z_level),
            origin="lower",
            norm=Normalize(vmin=vmin_density, vmax=vmax_density),
        )
        fig.colorbar(c0, ax=axs[row][0])
        axs[row][0].set_title(f"{title_prefix} - Density")
        axs[row][0].set_xlabel(axis_labels[0])
        axs[row][0].set_ylabel(axis_labels[1])

        # --- Field 2: Velocity magnitude ---
        c1 = axs[row][1].imshow(
            _slice_field(
                jnp.sqrt(states[1] ** 2 + states[2] ** 2 + states[3] ** 2), z_level
            ),
            origin="lower",
            norm=Normalize(vmin=vmin_speed, vmax=vmax_speed),
        )
        fig.colorbar(c1, ax=axs[row][1])
        axs[row][1].set_title(f"{title_prefix} - Velocity Magnitude")
        axs[row][1].set_xlabel(axis_labels[0])
        axs[row][1].set_ylabel(axis_labels[1])

        # --- Field 3: Pressure ---
        c2 = axs[row][2].imshow(
            _slice_field(states[4], z_level),
            origin="lower",
            norm=Normalize(vmin=vmin_pressure, vmax=vmax_pressure),
        )
        fig.colorbar(c2, ax=axs[row][2])
        axs[row][2].set_title(f"{title_prefix} - Pressure")
        axs[row][2].set_xlabel(axis_labels[0])
        axs[row][2].set_ylabel(axis_labels[1])

        # --- Field 4: Magnetic field magnitude ---
        c3 = axs[row][3].imshow(
            _slice_field(
                jnp.sqrt(states[5] ** 2 + states[6] ** 2 + states[7] ** 2), z_level
            ),
            origin="lower",
            norm=Normalize(vmin=vmin_mag, vmax=vmax_mag),
        )
        fig.colorbar(c3, ax=axs[row][3])
        axs[row][3].set_title(f"{title_prefix} - Magnetic Field")
        axs[row][3].set_xlabel(axis_labels[0])
        axs[row][3].set_ylabel(axis_labels[1])

    plt.tight_layout()
    if folder is None:
        folder = f"arena/data/models/{model_name}/plots"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(Path(folder) / f"{fig_name}.png", dpi=400)

    pass
