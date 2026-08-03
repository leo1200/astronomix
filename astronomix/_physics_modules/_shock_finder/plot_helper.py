import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

def plot_shock_diagnostics_2d(
    p_final, rho_final, result,
    geometry_x, geometry_y,
    box_size=1.0,
    mach_vmin=1.0,
    mach_vmax=100.0,
    suptitle=None,
):
    """
    Generic 2D shock-finder diagnostic plot — works for ANY 2D simulation
    result 
    Produces 5 panels:
        1. Pressure field
        2. Density field
        3. Shock zones + shock surface overlay on pressure
        4. Mach number field
        5. Shock direction arrows at surface cells, overlaid on pressure
           + shock zone/surface

    Args:
        p_final, rho_final: final pressure/density fields, shape (nx, ny)
        result:             ShockFinderResult from find_shocks_pfrommer
        geometry_x, geometry_y: coordinate grids, shape (nx, ny)
        box_size:           domain size, for axis limits
        mach_vmin, mach_vmax: color scale bounds for the Mach panel
        suptitle:           optional figure title

    Returns:
        (fig, axes) — figure and the 2x3 axes array (axes[1,2] left blank
        for caller to fill with a problem-specific 6th panel)
    """
    geometry_x_np = np.array(geometry_x)
    geometry_y_np = np.array(geometry_y)
    p_final_np    = np.array(p_final)
    rho_final_np  = np.array(rho_final)

    zones_np   = np.array(result.shock_zones).astype(float)
    surface_np = np.array(result.shock_surface_cells).astype(bool)
    mach_np    = np.array(result.mach_numbers)

    shock_dir_x_np = np.array(result.shock_direction[0])
    shock_dir_y_np = np.array(result.shock_direction[1])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)

    # ------------------------------------------------------------------
    # 1. Pressure
    # ------------------------------------------------------------------
    im0 = axes[0, 0].pcolormesh(
        geometry_x_np, geometry_y_np, p_final_np,
        cmap="viridis", shading="auto",
    )
    axes[0, 0].set_title("Pressure")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0, 0])

    # ------------------------------------------------------------------
    # 2. Density
    # ------------------------------------------------------------------
    im1 = axes[0, 1].pcolormesh(
        geometry_x_np, geometry_y_np, rho_final_np,
        cmap="plasma", shading="auto",
    )
    axes[0, 1].set_title("Density")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0, 1])

    # ------------------------------------------------------------------
    # 3. Shock surface + shock zone on pressure
    # ------------------------------------------------------------------
    axes[0, 2].pcolormesh(
        geometry_x_np, geometry_y_np, p_final_np,
        cmap="viridis", shading="auto", alpha=0.8,
    )
    axes[0, 2].contourf(
        geometry_x_np, geometry_y_np, zones_np,
        levels=[0.5, 1.5], colors=["green"], alpha=0.25,
    )
    axes[0, 2].contour(
        geometry_x_np, geometry_y_np, surface_np.astype(float),
        levels=[0.5], colors="red", linewidths=1.5,
    )
    axes[0, 2].set_title("Shock surfaces and shock zones")
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_ylabel("y")
    axes[0, 2].legend(
        handles=[
            Patch(facecolor="green", edgecolor="green", alpha=0.25, label="shock zone"),
            Line2D([0], [0], color="red", lw=1.5, label="shock surface"),
        ],
        loc="upper right", fontsize=8,
    )

    # ------------------------------------------------------------------
    # 4. Mach number
    # ------------------------------------------------------------------
    im3 = axes[1, 0].pcolormesh(
        geometry_x_np, geometry_y_np, mach_np,
        cmap="hot", vmin=mach_vmin, vmax=mach_vmax, shading="auto",
    )
    axes[1, 0].set_title("Shock Mach number at surface cells")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    plt.colorbar(im3, ax=axes[1, 0], label="Shock Mach number")

    # ------------------------------------------------------------------
    # 5. Shock direction at surface cells
    # ------------------------------------------------------------------
    axes[1, 1].pcolormesh(
        geometry_x_np, geometry_y_np, p_final_np,
        cmap="viridis", shading="auto", alpha=0.55,
    )
    axes[1, 1].contourf(
        geometry_x_np, geometry_y_np, zones_np,
        levels=[0.5, 1.5], colors=["green"], alpha=0.20,
    )
    axes[1, 1].contour(
        geometry_x_np, geometry_y_np, surface_np.astype(float),
        levels=[0.5], colors="red", linewidths=1.8,
    )

    gx_surf = geometry_x_np[surface_np]
    gy_surf = geometry_y_np[surface_np]
    dx_surf = shock_dir_x_np[surface_np]
    dy_surf = shock_dir_y_np[surface_np]

    mag = np.sqrt(dx_surf**2 + dy_surf**2)
    valid = mag > 0
    gx_surf, gy_surf = gx_surf[valid], gy_surf[valid]
    dx_surf, dy_surf = dx_surf[valid] / mag[valid], dy_surf[valid] / mag[valid]

    n_arrows = 100
    if len(gx_surf) > n_arrows:
        idx = np.linspace(0, len(gx_surf) - 1, n_arrows).astype(int)
        gx_plot, gy_plot = gx_surf[idx], gy_surf[idx]
        dx_plot, dy_plot = dx_surf[idx], dy_surf[idx]
    else:
        gx_plot, gy_plot = gx_surf, gy_surf
        dx_plot, dy_plot = dx_surf, dy_surf

    axes[1, 1].quiver(
        gx_plot, gy_plot, dx_plot, dy_plot,
        angles="xy", scale_units="xy", scale=20,
        color="white", width=0.004, headwidth=4, headlength=5,
        pivot="middle", zorder=20,
    )
    axes[1, 1].set_title("Shock direction at surface cells")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].legend(
        handles=[
            Patch(facecolor="green", edgecolor="green", alpha=0.20, label="shock zone"),
            Line2D([0], [0], color="red", lw=1.8, label="shock surface"),
            Line2D([0], [0], color="white", lw=0, marker=r"$\rightarrow$",
                   markersize=12, label="shock direction"),
        ],
        loc="upper right", fontsize=8,
    )

    # ------------------------------------------------------------------
    # Common formatting for the 5 spatial panels
    # ------------------------------------------------------------------
    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)

    # axes[1, 2] intentionally left for caller to fill in with a
    # problem-specific 6th panel (or leave blank / call fig.delaxes)

    return fig, axes

def plot_shock_surface_3d(
    xs, ys, zs, mach,
    center=(0.5, 0.5, 0.5),
    box_size=1.0,
    cmap="hot",
    title="3D Shock Surface (smoothed)",
    ax=None,
):
    """
    Render a smoothed 3D shock surface from a point cloud 
    colored by per-triangle mean Mach number.

    Args:
        xs, ys, zs: 1D arrays of surface-cell coordinates
        mach:       1D array of Mach numbers at those same points
        center:     explosion center, marked with a '+' 
        box_size:   domain size, used to set axis limits
        cmap:       matplotlib colormap name
        title:      plot title
        ax:         optional existing 3D axis to draw into; creates one if None

    Returns:
        (fig, ax) — the figure and axis used
    """

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)
    mach = np.asarray(mach)

    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if len(xs) < 4:
        # not enough points to triangulate a hull; fall back to plain scatter
        sc = ax.scatter(xs, ys, zs, c=mach, cmap=cmap, s=15, alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="Shock Mach number")
    else:
        hull = ConvexHull(np.column_stack([xs, ys, zs]))
        triangles = hull.simplices

        # per-triangle color = mean Mach of its 3 vertices
        tri_mach = mach[triangles].mean(axis=1)
        norm = plt.Normalize(vmin=tri_mach.min(), vmax=tri_mach.max())
        face_colors = plt.get_cmap(cmap)(norm(tri_mach))

        surf = ax.plot_trisurf(
            xs, ys, zs,
            triangles=triangles,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        surf.set_fc(face_colors)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(tri_mach)
        fig.colorbar(mappable, ax=ax, shrink=0.6, label="Shock Mach number")

    ax.scatter(
        [center[0]], [center[1]], [center[2]],
        color="cyan", marker="+", s=150, linewidths=2, label="explosion center",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper right")

    return fig, ax