"""
This provides many different plotters

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # registers 3-D projection

def plot_surface_3d_scalar(f_scalar,
                           x1_range=(-5, 5),
                           x2_range=(-5, 5),
                           num_points=200,
                           cmap='viridis',
                           elev=None, azim=None):
    """
    Plot a 3-D surface when f_scalar(x1, x2) only accepts scalar inputs.

    Parameters
    ----------
    f_scalar : callable
        Function that takes two floats (x1, x2) and returns a float.
    x1_range, x2_range : tuple (min, max)
        Bounds for each axis.
    num_points : int, default 200
        Grid resolution per axis.
    cmap : str, default 'viridis'
        Colormap name.
    elev, azim : float or None
        Elevation and azimuth angles for initial view.

    Returns
    -------
    (fig, ax) : Matplotlib objects so you can further customise.
    """
    # 1. Create 1-D grids
    x1_vals = np.linspace(*x1_range, num_points)
    x2_vals = np.linspace(*x2_range, num_points)

    # 2. Allocate Z array
    Z = np.empty((num_points, num_points), dtype=float)

    # 3. Evaluate f_scalar for every grid point
    for i, x2 in enumerate(x2_vals):
        for j, x1 in enumerate(x1_vals):
            Z[i, j] = f_scalar(np.array([x1,x2]))

    # 4. Build 2-D coordinate arrays for plotting
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # 5. Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap=cmap,
                           rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$f(x_1,x_2)$')
    ax.set_title('3-D Surface of $f(x_1,x_2)$')
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    fig.colorbar(surf, shrink=0.75, aspect=15, label='Function value')
    plt.tight_layout()
    plt.show()

    return fig, ax

