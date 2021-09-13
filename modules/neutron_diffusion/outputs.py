import os
import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from pyPDEs.spatial_discretization import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


class Outputs:
    def __init__(self):
        self.grid: List[List[float]] = []
        self.times: List[float] = []
        self.power: List[float] = []
        self.flux: List[List[ndarray]] = []
        self.precursors: List[List[ndarray]] = []

    def store_grid(self, sd: SpatialDiscretization):
        self.grid.clear()
        for point in sd.grid:
            self.grid.append([point.x, point.y, point.z])

    def store_outputs(self, solver: 'TransientSolver',
                      time: float) -> None:
        if time == 0.0:
            self.store_grid(solver.discretization)

        self.times.append(time)

        self.power.append(solver.power)

        n_grps, phi = solver.n_groups, np.copy(solver.phi)
        flux = [phi[g::n_grps] for g in range(n_grps)]
        self.flux.append(flux)

        if solver.use_precursors:
            n_dnps = solver.n_precursors
            precursors = np.copy(solver.precursors)
            precursors = [precursors[j::n_dnps] for j in range(n_dnps)]
            self.precursors.append(precursors)

    def write_outputs(self, path: str = ".") -> None:
        if not os.path.isdir(path):
            os.makedirs(path)

        time_path = os.path.join(path, "times.txt")
        np.savetxt(time_path, self.times)

        grid_path = os.path.join(path, "grid.txt")
        np.savetxt(grid_path, self.grid)

        power_path = os.path.join(path, "power.txt")
        np.savetxt(power_path, self.power)

        flux_dirpath = os.path.join(path, "flux")
        if not os.path.isdir(flux_dirpath):
            os.makedirs(flux_dirpath)
        if len(os.listdir(flux_dirpath)) > 0:
            os.system(f"rm -r {flux_dirpath}/*")

        for g in range(len(self.flux[0])):
            filepath = os.path.join(flux_dirpath, f"g{g}.txt")
            np.savetxt(filepath, np.array(self.flux)[:, g])

        if len(self.precursors) > 0:
            precursor_dirpath = os.path.join(path, "precursors")
            if not os.path.isdir(precursor_dirpath):
                os.makedirs(precursor_dirpath)
            if len(os.listdir(precursor_dirpath)) > 0:
                os.system(f"rm -r {precursor_dirpath}/*")

            for j in range(len(self.precursors[0])):
                filepath = os.path.join(precursor_dirpath, f"j{j}.txt")
                np.savetxt(filepath, np.array(self.precursors)[:, j])

    def plot_flux(self, g: int = 0, t: float = 0.0,
                  title: str = None) -> None:
        """Plot the scalar flux on an Axes.

        Parameters
        ----------
        g : int, default 0
            The group to plot.
        t : float, default 0.0
            The time to plot the solution at.
        title : str, default None
            A title for the Axes.
        """
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        if title:
            ax.set_title(title)

        grid = np.array(self.grid)
        times = np.array(self.times)
        ind = np.argmin(abs(times - t))

        # 1D plots
        if np.sum(grid[:, 0:2]) == 0.0:
            z = grid[:, 2]

            ax.set_xlabel("Location (cm)")
            ax.set_ylabel(r"$\phi(r)$")
            label = f"Group {g}"

            phi= self.flux[ind][g]
            ax.plot(z, phi, label=label)
            ax.legend()
            ax.grid(True)

        # 2D plots
        elif np.sum(grid[:, 2]) == 0.0:
            x = np.unique(grid[:, 0])
            y = np.unique(grid[:, 1])
            xx, yy = np.meshgrid(x, y)

            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")

            phi = self.flux[ind][g]
            phi = phi.reshape(xx.shape)
            im = ax.pcolor(xx, yy, phi, cmap="jet", shading="auto",
                           vmin=0.0, vmax=phi.max())
            plt.colorbar(im)
        plt.tight_layout()

    def plot_power(self, logscale: bool = False, title: str = None) -> None:
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        if title:
            ax.set_title(title)

        times = np.array(self.times)
        power = np.array(self.power)
        plotter = ax.semilogy if logscale else ax.plot
        plotter(times, power)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")

    def reset(self):
        self.grid.clear()
        self.times.clear()
        self.power.clear()
        self.flux.clear()
        self.precursors.clear()