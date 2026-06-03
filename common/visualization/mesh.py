"""
3D frame mesh plotting.
Equivalent to drawingMesh.m and drawInterpolatedFrame3D.m
"""

import numpy as np
import matplotlib.pyplot as plt


def draw_interpolated_frame(ax: plt.Axes, node_coordinates: np.ndarray,
                             element_nodes: np.ndarray,
                             displacements: np.ndarray,
                             scale: float = 1.0,
                             n_points: int = 30,
                             color: str = "r",
                             linewidth: float = 1.5) -> None:
    """
    Plot smooth deformed shape using Hermite cubic interpolation along each element.
    Equivalent to drawInterpolatedFrame3D.m

    Shape functions: N1=1-3s²+2s³, N2=s-2s²+s³, N3=3s²-2s³, N4=-s²+s³

    Args:
        ax: Matplotlib 3D axes
        node_coordinates: (nNodes, 3)
        element_nodes: (nElements, 2) 0-based
        displacements: (GDof,) full 6-DOF displacement vector
        scale: Amplification factor
        n_points: Interpolation points per element
        color: Line color
        linewidth: Line width
    """
    nc = node_coordinates
    UX = displacements[0::6];  UY = displacements[1::6];  UZ = displacements[2::6]
    RX = displacements[3::6];  RY = displacements[4::6];  RZ = displacements[5::6]

    s  = np.linspace(0, 1, n_points)
    N1 = 1 - 3*s**2 + 2*s**3
    N2 = s - 2*s**2 + s**3
    N3 = 3*s**2 - 2*s**3
    N4 = -s**2 + s**3

    for n1, n2 in element_nodes:
        p1 = nc[n1];  p2 = nc[n2]
        dx = p2 - p1
        L  = np.linalg.norm(dx)
        if L < 1e-14:
            continue

        ex = dx / L

        # Robust local basis (same logic as MATLAB)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ex, ref)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0])
        ey = np.cross(ref, ex);  ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)

        Rlg = np.column_stack([ex, ey, ez])   # local→global (3×3)

        # Rotate nodal translations and rotations to local frame
        t1l = Rlg.T @ np.array([UX[n1], UY[n1], UZ[n1]])
        t2l = Rlg.T @ np.array([UX[n2], UY[n2], UZ[n2]])
        r1l = Rlg.T @ np.array([RX[n1], RY[n1], RZ[n1]])
        r2l = Rlg.T @ np.array([RX[n2], RY[n2], RZ[n2]])

        u1, v1, w1 = t1l
        u2, v2, w2 = t2l
        thz1, thz2 = r1l[2], r2l[2]   # bending about local z → displacement in y
        thy1, thy2 = r1l[1], r2l[1]   # bending about local y → displacement in z

        # Local interpolation
        uloc = (1 - s)*u1 + s*u2                                      # axial (linear)
        vloc = N1*v1 + (L*N2)*thz1 + N3*v2 + (L*N4)*thz2            # transverse y
        wloc = N1*w1 + (L*N2)*(-thy1) + N3*w2 + (L*N4)*(-thy2)      # transverse z

        # Back to global frame
        P0    = p1[:, None] + np.outer(ex, s * L)                     # (3, n_points)
        dglob = Rlg @ np.vstack([uloc, vloc, wloc])                   # (3, n_points)
        Pd    = P0 + scale * dglob

        ax.plot(Pd[0], Pd[1], Pd[2], color=color, linewidth=linewidth)


def plot_deformed(ax, nc, en, u_raw, name,
                  draw_mesh_fn, set_axes_fn, target_frac=0.08):
    """Overlay undeformed (dashed) and deformed (solid) mesh with auto-scale."""
    UX = u_raw[0::6];  UY = u_raw[1::6];  UZ = u_raw[2::6]
    umax = np.sqrt(UX**2 + UY**2 + UZ**2).max()
    bbox_diag = np.linalg.norm(nc.max(axis=0) - nc.min(axis=0))
    scale = np.clip(target_frac * bbox_diag / max(umax, 1e-12), 0.1, 200)

    nc_def = nc + scale * np.column_stack([UX, UY, UZ])
    draw_mesh_fn(ax, nc,     en, linestyle="k--")
    draw_mesh_fn(ax, nc_def, en, linestyle="r-")
    set_axes_fn(ax, np.vstack([nc, nc_def]))
    ax.set_title(f"{name}\nscale={scale:.1f}  umax={umax:.2e}", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=135)
    ax.grid(True)
