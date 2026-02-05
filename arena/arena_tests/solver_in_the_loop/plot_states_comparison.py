from jax import Array
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize
from IPython.display import HTML
import jax.numpy as jnp
from typing import Optional


def plot_and_animate_states(states_list, z_levels, titles=None, vmin=0, vmax=1):
    """
    Plot and animate multiple simulation states (e.g. LR, HR, etc.)

    Args:
        states_list: list of 5D arrays [batch, channel, x, y, z]
        z_levels: list of z-level indices corresponding to each state array
        titles: optional list of titles (e.g. ["LR", "HR"])
        vmin, vmax: normalization range for color scaling
    """
    assert all([len(states) == 5 for states in states_list])

    n_states = len(states_list)
    n_fields = 4  # density, velocity magnitude, pressure, magnetic field

    if titles is None:
        titles = [f"State {i + 1}" for i in range(n_states)]

    fig, axs = plt.subplots(n_states, n_fields, figsize=(5 * n_fields, 5 * n_states))
    if n_states == 1:
        axs = axs[None, :]  # handle single row case

    # Store all imshow handles for animation
    caxs = []

    for row, (states, z_level, title_prefix) in enumerate(
        zip(states_list, z_levels, titles, strict=True)
    ):
        # --- Field 1: Density ---
        c0 = axs[row][0].imshow(
            states[0, 0, :, :, z_level].T,
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c0, ax=axs[row][0])
        axs[row][0].set_title(f"{title_prefix} - Density")
        axs[row][0].set_xlabel("x")
        axs[row][0].set_ylabel("y")

        # --- Field 2: Velocity magnitude ---
        c1 = axs[row][1].imshow(
            jnp.sqrt(
                states[0, 1, :, :, z_level] ** 2
                + states[0, 2, :, :, z_level] ** 2
                + states[0, 3, :, :, z_level] ** 2
            ).T,
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c1, ax=axs[row][1])
        axs[row][1].set_title(f"{title_prefix} - Velocity Magnitude")

        # --- Field 3: Pressure ---
        c2 = axs[row][2].imshow(
            states[0, 4, :, :, z_level].T,
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c2, ax=axs[row][2])
        axs[row][2].set_title(f"{title_prefix} - Pressure")

        # --- Field 4: Magnetic field magnitude ---
        c3 = axs[row][3].imshow(
            jnp.sqrt(
                states[0, 5, :, :, z_level] ** 2
                + states[0, 6, :, :, z_level] ** 2
                + states[0, 7, :, :, z_level] ** 2
            ).T,
            origin="lower",
            # norm=Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(c3, ax=axs[row][3])
        axs[row][3].set_title(f"{title_prefix} - Magnetic Field")

        # Store for animation
        caxs.append((c0, c1, c2, c3))

    plt.tight_layout()

    # --- Animation function ---
    def animate_all(i):
        updated = []
        for (states, z_level), (c0, c1, c2, c3) in zip(
            zip(states_list, z_levels, strict=True), caxs, strict=True
        ):
            c0.set_array(states[i, 0, :, :, z_level].T)
            c1.set_array(
                jnp.sqrt(
                    states[i, 1, :, :, z_level] ** 2
                    + states[i, 2, :, :, z_level] ** 2
                    + states[i, 3, :, :, z_level] ** 2
                ).T
            )
            c2.set_array(states[i, 4, :, :, z_level].T)
            c3.set_array(
                jnp.sqrt(
                    states[i, 5, :, :, z_level] ** 2
                    + states[i, 6, :, :, z_level] ** 2
                    + states[i, 7, :, :, z_level] ** 2
                ).T
            )
            updated.extend([c0, c1, c2, c3])
        return updated

    # Use frame count from first state
    n_frames = states_list[0].shape[0]
    ani = animation.FuncAnimation(fig, animate_all, frames=n_frames, interval=50)

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
    vmax_density = min(jnp.min(states[0]).astype(float) for states in states_list)
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
            norm=Normalize(vmin=vmax_density, vmax=vmax_density),
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

    pass
