import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pyPDEs.spatial_discretization import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import HeatConductionSolver


def plot_solution(self: "HeatConductionSolver", title: str = None) -> None:
    """Plot the currently stored solution.

    Parameters
    ----------
    title : str, default None
        A title for the figure.s
    """
    grid = self.discretization.grid
    if title:
        plt.title(title)
    plt.xlabel("Location")
    plt.ylabel(r"T(r)")
    plt.plot(grid, self.u, '-ob', label='Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()