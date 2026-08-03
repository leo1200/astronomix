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
    n_arrows=100
):
    """
    Generic 2D shock-finder diagnostic plot — works for 2D shock simulation

    Produces 5 panels:
        1. Pressure field
        2. Density field
        3. Shock zones + shock surface overlay on pressure
        4. Mach number field
        5. Shock direction arrows at surface cells, overlaid on pressure + shock zone/surface
    
    6th panel (axes[1,2]) is left blank for the caller to fill in with a problem-specific plot.

    Args:
        p_final, rho_final:         final pressure/density fields, shape (nx, ny)
        result:                     ShockFinderResult from find_shocks_pfrommer
        geometry_x, geometry_y:     coordinate grids, shape (nx, ny)
        box_size:                   domain size, for axis limits
        mach_vmin, mach_vmax:       color scale bounds for the Mach panel
        suptitle:                   optional figure title
        n_arrows:                   max number of shock direction arrows drawn in panel 5

    Returns:
        (fig, axes) = figure and the 2x3 axes array 
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


    # 1. Pressure
    im0 = axes[0, 0].pcolormesh(
        geometry_x_np, geometry_y_np, p_final_np,
        cmap="viridis", shading="auto",
    )
    axes[0, 0].set_title("Pressure")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0, 0])


    # 2. Density
    im1 = axes[0, 1].pcolormesh(
        geometry_x_np, geometry_y_np, rho_final_np,
        cmap="plasma", shading="auto",
    )
    axes[0, 1].set_title("Density")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0, 1])


    # 3. Shock surface + shock zone on pressure
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


    # 4. Mach number
    im3 = axes[1, 0].pcolormesh(
        geometry_x_np, geometry_y_np, mach_np,
        cmap="hot", vmin=mach_vmin, vmax=mach_vmax, shading="auto",
    )
    axes[1, 0].set_title("Shock Mach number at surface cells")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    plt.colorbar(im3, ax=axes[1, 0], label="Shock Mach number")


    # 5. Shock direction at surface cells
    # draw pressure field in background, shock zone + surface overlay
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

    # extract surface-cell coordinates and shock directions
    gx_surf = geometry_x_np[surface_np]
    gy_surf = geometry_y_np[surface_np]
    dx_surf = shock_dir_x_np[surface_np]
    dy_surf = shock_dir_y_np[surface_np]

    # normalize shock direction vectors for plotting
    mag = np.sqrt(dx_surf**2 + dy_surf**2)
    valid = mag > 0
    gx_surf, gy_surf = gx_surf[valid], gy_surf[valid]
    dx_surf, dy_surf = dx_surf[valid] / mag[valid], dy_surf[valid] / mag[valid]

    # pick a subset of arrows to plot -> gx_plot, gy_plot, dx_plot, dy_plot
    if len(gx_surf) > n_arrows:
        idx = np.linspace(0, len(gx_surf) - 1, n_arrows).astype(int)
        gx_plot, gy_plot = gx_surf[idx], gy_surf[idx]
        dx_plot, dy_plot = dx_surf[idx], dy_surf[idx]
    else:
        gx_plot, gy_plot = gx_surf, gy_surf
        dx_plot, dy_plot = dx_surf, dy_surf

    # plot the shock direction arrows
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


    # Format
    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)

    return fig, axes

def plot_shock_surface_3d(
    xs, ys, zs,
    dxs, dys, dzs,
    mach,
    center=(0.5, 0.5, 0.5),
    box_size=1.0,
    cmap="hot",
    title="3D Shock Surface (smoothed)",
    ax=None,
    n_arrows=150,
    mode="SMOOTH"
):
    """
    Render a smoothed 3D shock surface, colored by per-triangle mean Mach number
    
    We use scipy.spatial.ConvexHull to triangulate the surface
    then color each triangle by the mean Mach number of its three vertices.

    Args:
        xs, ys, zs:     1D arrays of surface-cell coordinates
        dxs, dys, dzs:  1D arrays of shock direction vectors at those same surface points

        mach:           1D array of Mach numbers at those same points
        box_size:       domain size, used to set axis limits
        cmap:           matplotlib colormap name
        title:          plot title
        ax:             optional existing 3D axis to draw into; creates one if None
        n_arrows:       number of shock direction displayed
        mode:           sketch mode, possible value is "SMOOTH" or "SCATTER"


        # optional arguments for Sedov case
        center:         optional explosion center

    Returns:
        (fig, ax)
    """

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)
    mach = np.asarray(mach)
    dxs = np.asarray(dxs)
    dys = np.asarray(dys)
    dzs = np.asarray(dzs)

    # create 3D base plot
    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    """
    Render convex hull of the surface points, colored by mean Mach number per triangle.
        * too few points -> just scatter plot
        * enough points -> ConvexHull triangulation + plot triangles surfaces
            * scipy.spatial.ConvexHull to get triangles
            * compute mean Mach number per triangle + color by mean Mach number
            * matplotlib.plot_trisurf to render the triangles
    """
    if len(xs) < 4:
        sc = ax.scatter(xs, ys, zs, c=mach, cmap=cmap, s=15, alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="Shock Mach number")
    else:
        if mode == "SMOOTH":
            # smooth
            # triangulate the surface points using ConvexHull
            hull = ConvexHull(np.column_stack([xs, ys, zs]))
            
            # each row is a triangle, given as 3 integer indices into (xs, ys, zs)
            triangles = hull.simplices # shape (n_triangles, 3)

            # get mean Mach number for each triangle
            tri_mach = mach[triangles].mean(axis=1)
            # instance of Normalize to map values to colors
            norm = plt.Normalize(vmin=tri_mach.min(), vmax=tri_mach.max())
            # get colors for each triangle based on mean Mach number and norm scale
            face_colors = plt.get_cmap(cmap)(norm(tri_mach))

            # use plot_trisurf to render the triangles with the computed face colors
            surf = ax.plot_trisurf(
                xs, ys, zs, # real coordinates of the surface points
                triangles=triangles,
                linewidth=0,
                antialiased=True,
                shade=False,
            )
            surf.set_fc(face_colors)

            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array(tri_mach)
            fig.colorbar(mappable, ax=ax, shrink=0.6, label="Shock Mach number")
        else:
            # scatter plot
            sc = ax.scatter(xs, ys, zs, c=mach, cmap=cmap, s=15, alpha=0.8)
            fig.colorbar(sc, ax=ax, shrink=0.6, label="Shock Mach number")

    mag = np.sqrt(dxs**2 + dys**2 + dzs**2)
    valid = mag > 0

    xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]
    dxs_v = dxs[valid] / mag[valid]
    dys_v = dys[valid] / mag[valid]
    dzs_v = dzs[valid] / mag[valid]

    if len(xs_v) > n_arrows:
        idx = np.random.choice(len(xs_v), n_arrows, replace=False)
        xs_v, ys_v, zs_v = xs_v[idx], ys_v[idx], zs_v[idx]
        dxs_v, dys_v, dzs_v = dxs_v[idx], dys_v[idx], dzs_v[idx]


    ax.quiver(
        xs_v, ys_v, zs_v,
        dxs_v, dys_v, dzs_v,
        length=0.04, color="cyan", normalize=True, linewidth=1.0,
    )

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

def plot_shock_projections_3d(
    field, 
    result,
    geometry_x, geometry_y, geometry_z,
    center,
    box_size=1.0,
    field_cmap="plasma",
    zone_alpha=0.22,
    n_arrows=60,
    suptitle=None,
):
    """
    Takes a 3D simulation volume and shows 3 middle slices through it:
    * XY plane: slice at the middle z
        z≈z_center
	* XZ plane: slice at the middle y
        y≈y_center
	* YZ plane: slice at the middle x
        x≈x_center

    Args:
        field:      background scalar field to plot, shape (nx, ny, nz)
                    (e.g. density or pressure)
        result:     ShockFinderResult from find_shocks_pfrommer
        geometry_x, geometry_y, geometry_z: coordinate grids, shape (nx, ny, nz)
        center:     (cx, cy, cz) — point through which the three mid-planes
                    are sliced, and where a marker is drawn
        box_size:   domain size
        field_cmap: colormap
        zone_alpha: alpha for the shock-zone overlay
        n_arrows:   max number of direction arrows drawn per panel
        suptitle:   optional figure title

    Returns:
        (fig, axes) — figure and the 1x3 axes array (xy, xz, yz)
    """
    cx, cy, cz = center

    geometry_x_np = np.array(geometry_x)
    geometry_y_np = np.array(geometry_y)
    geometry_z_np = np.array(geometry_z)
    field_np      = np.array(field)

    zones_np   = np.array(result.shock_zones).astype(bool)
    surface_np = np.array(result.shock_surface_cells).astype(bool)

    shock_dir_x_np = np.array(result.shock_direction[0])
    shock_dir_y_np = np.array(result.shock_direction[1])
    shock_dir_z_np = np.array(result.shock_direction[2])

    def mid_index(coord_1d, target):
        return int(np.argmin(np.abs(coord_1d - target)))

    x_1d = geometry_x_np[:, 0, 0]
    y_1d = geometry_y_np[0, :, 0]
    z_1d = geometry_z_np[0, 0, :]

    ix_mid = mid_index(x_1d, cx)
    iy_mid = mid_index(y_1d, cy)
    iz_mid = mid_index(z_1d, cz)

    def plot_projection(ax, plane, slice_idx, axis0_label, axis1_label):
        if plane == "xy":
            A = geometry_x_np[:, :, slice_idx]
            B = geometry_y_np[:, :, slice_idx]
            f = field_np[:, :, slice_idx]
            zone_slice = zones_np[:, :, slice_idx]
            surf_slice = surface_np[:, :, slice_idx]
            da = shock_dir_x_np[:, :, slice_idx]
            db = shock_dir_y_np[:, :, slice_idx]
            c0, c1 = cx, cy
        elif plane == "xz":
            A = geometry_x_np[:, slice_idx, :]
            B = geometry_z_np[:, slice_idx, :]
            f = field_np[:, slice_idx, :]
            zone_slice = zones_np[:, slice_idx, :]
            surf_slice = surface_np[:, slice_idx, :]
            da = shock_dir_x_np[:, slice_idx, :]
            db = shock_dir_z_np[:, slice_idx, :]
            c0, c1 = cx, cz
        elif plane == "yz":
            A = geometry_y_np[slice_idx, :, :]
            B = geometry_z_np[slice_idx, :, :]
            f = field_np[slice_idx, :, :]
            zone_slice = zones_np[slice_idx, :, :]
            surf_slice = surface_np[slice_idx, :, :]
            da = shock_dir_y_np[slice_idx, :, :]
            db = shock_dir_z_np[slice_idx, :, :]
            c0, c1 = cy, cz
        else:
            raise ValueError(plane)

        ax.pcolormesh(A, B, f, cmap=field_cmap, shading="auto", alpha=0.85)

        ax.contourf(
            A, B, zone_slice.astype(float),
            levels=[0.5, 1.5], colors=["green"], alpha=zone_alpha,
        )

        ax.contour(
            A, B, surf_slice.astype(float),
            levels=[0.5], colors="red", linewidths=1.5,
        )

        Af = A[surf_slice]
        Bf = B[surf_slice]
        daf = da[surf_slice]
        dbf = db[surf_slice]

        mag = np.sqrt(daf**2 + dbf**2)
        valid = mag > 0
        Af, Bf, daf, dbf = Af[valid], Bf[valid], daf[valid], dbf[valid]
        mag = mag[valid]

        if len(Af) > 0:
            daf_u = daf / mag
            dbf_u = dbf / mag

            if len(Af) > n_arrows:
                idx = np.linspace(0, len(Af) - 1, n_arrows).astype(int)
                Af, Bf, daf_u, dbf_u = Af[idx], Bf[idx], daf_u[idx], dbf_u[idx]

            ax.quiver(
                Af, Bf, daf_u, dbf_u,
                angles="xy", scale_units="xy", scale=20,
                color="white", width=0.004, headwidth=4, headlength=5,
                pivot="middle", zorder=20,
            )

        ax.plot(c0, c1, marker="+", color="cyan", markersize=10, mew=2, zorder=25)

        ax.set_xlabel(axis0_label)
        ax.set_ylabel(axis1_label)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    plot_projection(axes[0], "xy", iz_mid, "x", "y")
    axes[0].set_title(f"xy plane (z ≈ {z_1d[iz_mid]:.3f})")

    plot_projection(axes[1], "xz", iy_mid, "x", "z")
    axes[1].set_title(f"xz plane (y ≈ {y_1d[iy_mid]:.3f})")

    plot_projection(axes[2], "yz", ix_mid, "y", "z")
    axes[2].set_title(f"yz plane (x ≈ {x_1d[ix_mid]:.3f})")

    axes[0].legend(
        handles=[
            Patch(facecolor="green", edgecolor="green", alpha=zone_alpha, label="shock zone"),
            Line2D([0], [0], color="red", lw=1.5, label="shock surface"),
            Line2D([0], [0], color="white", lw=0, marker=r"$\rightarrow$",
                   markersize=12, label="shock direction"),
        ],
        loc="upper right", fontsize=8,
    )

    return fig, axes