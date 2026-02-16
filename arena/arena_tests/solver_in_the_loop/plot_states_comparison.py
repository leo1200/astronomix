from jax import Array
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize
from IPython.display import HTML
import jax.numpy as jnp
from typing import Optional
import logging

logger = logging.Logger(__name__)


def plot_and_animate_states(
    states_list,
    z_levels,
    timepoints: Array,
    slice_axis: str = "z",
    titles=None,
    vmin=0,
    vmax=1,
    save_path: Optional[str] = None,
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
        raise ValueError(f"Invalid slice_axis '{slice_axis}', must be 'x', 'y', or 'z'.")

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
                jnp.sqrt(
                    states[0, 1] ** 2 + states[0, 2] ** 2 + states[0, 3] ** 2
                ),
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
                jnp.sqrt(
                    states[0, 5] ** 2 + states[0, 6] ** 2 + states[0, 7] ** 2
                ),
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
                    jnp.sqrt(
                        states[i, 1] ** 2 + states[i, 2] ** 2 + states[i, 3] ** 2
                    ),
                    slice_level,
                )
            )
            c2.set_array(_slice_field(states[i, 4], slice_level))
            c3.set_array(
                _slice_field(
                    jnp.sqrt(
                        states[i, 5] ** 2 + states[i, 6] ** 2 + states[i, 7] ** 2
                    ),
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


def plot_and_animate_states_with_diagonal(
    states: Array,
    z_level: int,
    timepoints: Array,
    slice_axis: str = "z",
    title: str = "State",
    vmin=0,
    vmax=1,
    save_path: Optional[str] = None,
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
    """
    if slice_axis not in {"x", "y", "z"}:
        raise ValueError(f"Invalid slice_axis '{slice_axis}', must be 'x', 'y', or 'z'.")

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
    for ax, series, series_title in zip(axs_diag, diag_series, diag_titles, strict=True):
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
        filepath = os.path.join(save_path, "animation.mp4")
        ani.save(filepath, writer="ffmpeg", dpi=150)
        logger.info(f"Animation saved to {filepath}")

    plt.show()
    html_video = HTML(ani.to_html5_video())
    return html_video


def plot_states(
    states_list: list[Array],
    z_levels: list[int],
    model_name: str,
    fig_name: str = "states_comparison",
    titles: Optional[list[str]] = None,
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
    vmax = max(jnp.max(states).astype(float) for states in states_list)
    vmin = min(jnp.min(states).astype(float) for states in states_list)
    if titles is None:
        titles = [f"State {i + 1}" for i in range(n_states)]

    fig, axs = plt.subplots(n_states, n_fields, figsize=(5 * n_fields, 5 * n_states))
    if n_states == 1:
        axs = axs[None, :]  # handle single row case

    for row, (states, z_level, title_prefix) in enumerate(
        zip(states_list, z_levels, titles, strict=True)
    ):
        # --- Field 1: Density ---
        c0 = axs[row][0].imshow(
            states[0, :, :, z_level].T,
            origin="lower",
            norm=Normalize(vmin=vmin_density, vmax=vmax_density),
        )
        fig.colorbar(c0, ax=axs[row][0])
        axs[row][0].set_title(f"{title_prefix} - Density")
        axs[row][0].set_xlabel("x")
        axs[row][0].set_ylabel("y")

        # --- Field 2: Velocity magnitude ---
        c1 = axs[row][1].imshow(
            jnp.sqrt(
                states[1, :, :, z_level] ** 2
                + states[2, :, :, z_level] ** 2
                + states[3, :, :, z_level] ** 2
            ).T,
            origin="lower",
            norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c1, ax=axs[row][1])
        axs[row][1].set_title(f"{title_prefix} - Velocity Magnitude")

        # --- Field 3: Pressure ---
        c2 = axs[row][2].imshow(
            states[4, :, :, z_level].T,
            origin="lower",
            norm=Normalize(vmin=vmin_pressure, vmax=vmax_pressure),
        )
        fig.colorbar(c2, ax=axs[row][2])
        axs[row][2].set_title(f"{title_prefix} - Pressure")

        # --- Field 4: Magnetic field magnitude ---
        c3 = axs[row][3].imshow(
            jnp.sqrt(
                states[5, :, :, z_level] ** 2
                + states[6, :, :, z_level] ** 2
                + states[7, :, :, z_level] ** 2
            ).T,
            origin="lower",
            norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c3, ax=axs[row][3])
        axs[row][3].set_title(f"{title_prefix} - Magnetic Field")

    plt.tight_layout()
    plt.savefig(f"arena/data/models/{model_name}/plots/{fig_name}.png", dpi=400)
    logger.info(f"Created fig arena/data/models/{model_name}/plots/{fig_name}.png")

    pass
