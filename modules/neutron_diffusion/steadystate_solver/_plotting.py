import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import ndarray

from pyPDEs.spatial_discretization import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def plot_solution(self: 'SteadyStateSolver', title: str = None) -> None:
    """
    Plot the solution, including the precursors, if used.

    Parameters
    ----------
    title : str
        A title for the figure.
    """
    fig: Figure = plt.figure()
    if self.use_precursors:
        fig.set_size_inches(8, 4)
        if title:
            fig.suptitle(title)

        ax: Axes = fig.add_subplot(1, 2, 1)
        self.plot_flux(ax)

        ax: Axes = fig.add_subplot(1, 2, 2)
        self.plot_precursors(ax)
    else:
        ax: Axes = fig.add_subplot(1, 1, 1)
        self.plot_flux(ax, title)
    plt.tight_layout()


def plot_flux(self: 'SteadyStateSolver',
              ax: Axes = None, title: str = None) -> None:
    """
    Plot the scalar flux on an Axes.

    Parameters
    ----------
    ax : Axes
        An Axes to plot on.
    title : str, default None
        A title for the Axes.
    """
    if ax is None:
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
    if title:
        ax.set_title(title)

    grid = self.discretization.grid

    if self.mesh.dim == 1:
        grid = [p.z for p in grid]
        ax.set_xlabel("r (cm)", fontsize=12)
        ax.set_ylabel(r"$\phi_g(r)$", fontsize=12)
        for g in range(self.n_groups):
            label = f"Group {g}"
            phi = self.phi[g::self.n_groups]
            ax.plot(grid, phi, label=label)
        ax.legend()
        ax.grid(True)
    elif self.mesh.dim == 2:
        x = np.unique([p.x for p in grid])
        y = np.unique([p.y for p in grid])
        xx, yy = np.meshgrid(x, y)
        phi: ndarray = self.phi[0::self.n_groups]
        phi = phi.reshape(xx.shape)
        im = ax.pcolor(xx, yy, phi, cmap='jet', shading='auto',
                       vmin=0.0, vmax=phi.max())
        plt.colorbar(im)
    plt.tight_layout()


def plot_precursors(self: 'SteadyStateSolver',
                    ax: Axes = None, title: str = None) -> None:
    """Plot the delayed neutron precursors on an Axes.

    Parameters
    ----------
    ax : Axes
        An Axes to plot on.
    title : str, default None
        A title for the Axes.
    """
    if ax is None:
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
    if title:
        ax.set_title(title)

    if self.mesh.dim == 1:
        grid = [cell.centroid.z for cell in self.mesh.cells]

        lambdas = [xs.precursor_lambda for xs in self.material_xs]
        lambdas = list(np.unique(np.ravel(lambdas)))

        c = np.zeros((len(lambdas), len(grid)))
        for cell in self.mesh.cells:
            xs_id = self.matid_to_xs_map[cell.material_id]
            xs = self.material_xs[xs_id]
            for p in range(xs.n_precursors):
                j = lambdas.index(xs.precursor_lambda[p])
                idx = self.max_precursors*cell.id + p
                c[j, cell.id] = self.precursors[idx]

        for j in range(len(lambdas)):
            label = f"Precursor {j}"
            ax.set_xlabel("r (cm)")
            ax.set_ylabel("C$_j$")
            ax.plot(grid, c[j], label=label)
        ax.legend()
        ax.grid(True)
    else:
        raise AssertionError(
            f'Only 1D precursor plotting is implemented.')
    plt.tight_layout()
