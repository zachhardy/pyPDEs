import os
import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from pyPDEs.spatial_discretization import *

from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


class Outputs:
    def __init__(self):
        self.dim: int = 0
        self.core_volume: float = 0.0

        self.grid: ndarray = []
        self.times: ndarray = []
        self.system_power: ndarray = []

        self.scalar_flux: ndarray = []
        self.precursors: ndarray = []
        self.power_density: ndarray = []
        self.temperature: ndarray = []

    def store_outputs(self, solver: "TransientSolver",
                      time: float) -> None:

        # Store grid, if storing initial conditions
        if time == 0.0:
            self.grid.clear()
            for point in solver.discretization.grid:
                self.grid += [[point.x, point.y, point.z]]
            self.core_volume = solver.core_volume

        # Store the current simulation time
        self.times += [time]

        # Store the global fission power in W
        self.system_power += [solver.power]

        # Store the groupswise fluxes. This stores a list
        # of G arrays where each array contains the group `g`
        # scalar flux at cell centers
        n_grps, phi = solver.n_groups, np.copy(solver.phi)
        flux = [phi[g::n_grps] for g in range(n_grps)]
        self.scalar_flux += [flux]

        # NOTE: For multi-region problems with the same delayed
        # neutron data, this does not work very well because
        # there is currently no way to declared precursors
        # identical across materials.
        if solver.use_precursors:
            n_dnps = solver.max_precursors
            precursors = np.copy(solver.precursors)
            precursors = [precursors[j::n_dnps] for j in range(n_dnps)]
            self.precursors += [precursors]

        # Store the power density. This stores the cell-wise fission
        # rate multiplied by the energy released per fission event.
        Ef = solver.energy_per_fission
        self.power_density += [Ef * solver.fission_density]

        # Store the temperatures, if feedback is being used.
        if solver.use_feedback:
            self.temperature += [solver.temperature]

    def finalize_outputs(self) -> None:
        self.grid = np.array(self.grid)
        self.times = np.array(self.times)
        self.system_power = np.array(self.system_power)
        self.scalar_flux = np.array(self.scalar_flux)
        self.power_density = np.array(self.power_density)
        if len(self.precursors) > 0:
            self.precursors = np.array(self.precursors)
        if len(self.temperature) > 0:
            self.temperature = np.array(self.temperature)

        # Define dimension
        if np.sum(self.grid[:, :2]) == 0.0:
            self.dim = 1
        elif np.sum(self.grid[:, 2]) == 0.0:
            self.dim = 2
        else:
            self.dim = 3

    def write_outputs(self, path: str = ".") -> None:
        if not os.path.isdir(path):
            os.makedirs(path)

        # Write the spatial grid
        grid_path = os.path.join(path, "grid.txt")
        np.savetxt(grid_path, self.grid)

        # Write the time step times
        time_path = os.path.join(path, "times.txt")
        np.savetxt(time_path, self.times)

        # Write the system power at each time step
        system_power_path = os.path.join(path, "system_power.txt")
        np.savetxt(system_power_path, self.system_power)

        # Write the average power at each time step
        average_power_path = os.path.join(path, "average_power.txt")
        np.savetxt(average_power_path, self.system_power / self.core_volume)

        # Write the group-wise scalar fluxes at each time step
        flux_dirpath = os.path.join(path, "flux")
        if not os.path.isdir(flux_dirpath):
            os.makedirs(flux_dirpath)
        if len(os.listdir(flux_dirpath)) > 0:
            os.system(f"rm -r {flux_dirpath}/*")

        for g in range(len(self.scalar_flux[0])):
            filepath = os.path.join(flux_dirpath, f"g{g}.txt")
            np.savetxt(filepath, self.scalar_flux[:, g])

        # Write the precursors at each time step.
        # NOTE: See note above about poor functionality in multi-
        # material region problems.
        if len(self.precursors) > 0:
            precursor_dirpath = os.path.join(path, "precursors")
            if not os.path.isdir(precursor_dirpath):
                os.makedirs(precursor_dirpath)
            if len(os.listdir(precursor_dirpath)) > 0:
                os.system(f"rm -r {precursor_dirpath}/*")

            for j in range(len(self.precursors[0])):
                filepath = os.path.join(precursor_dirpath, f"j{j}.txt")
                np.savetxt(filepath, self.precursors[:, j])

        # Write the power densities at each time step.
        power_density_path = os.path.join(path, "power_density.txt")
        np.savetxt(power_density_path, self.power_density)

        # Write the temperatures at each time step.
        if len(self.temperature) > 0:
            temperature_path = os.path.join(path, "temperature.txt")
            np.savetxt(temperature_path, self.temperature)

    def plot_1d_scalar_flux(self, groups: List[int] = None,
                            times: List[float] = None) -> None:
        """Plot specific group scalar fluxs at specific times.

        Parameters
        ----------
        groups : List[int], default None
            The groups to plot. Default is all groups
        times : List[float], default None
            The times to plot the group `g` scalar flux. The
            default is the initial condition and final result.
        """
        if self.dim != 1:
            raise AssertionError("This routine is only for 1D grids.")

        # Get the groups to plot
        n_grps = self.scalar_flux.shape[1]
        if groups is None:
            groups = [g for g in range(n_grps)]
        if isinstance(groups, int):
            groups = [groups]

        # Get the times to plot
        if times is None:
            times = [self.times[0], self.times[-1]]
        if isinstance(times, float):
            times = [times]

        # Get the grid
        z = self.grid[:, 2]

        # Get the scalar flux at the specified times
        phi = []
        for t in times:
            phi += [self._interpolate(t, self.scalar_flux)]
        phi = np.array(phi)

        # Loop over groups
        for g in groups:
            fig: Figure = plt.figure()
            ax: Axes = fig.add_subplot(1, 1, 1)
            ax.set_xlabel("r (cm)")
            ax.set_ylabel(f"$\phi_{{{g}}}(r)$")
            ax.set_title(f"Group {g}")

            # Plot this groups scalar fluxes
            phi_g = phi[:, g] / np.max(phi[:, g])
            for i in range(len(times)):
                label = f"Time = {times[i]:.2f} sec"
                ax.plot(z, phi_g[i], label=label)
            ax.legend()
            ax.grid(True)
        fig.tight_layout()

    def plot_2d_scalar_flux(self, groups: List[int] = None,
                            times: List[float] = None) -> None:
        """Plot specific group scalar fluxs at specific times.

        Parameters
        ----------
        groups : List[int], default None
            The groups to plot. Default is all groups
        times : List[float], default None
            The times to plot the group `g` scalar flux. The
            default is the initial condition and final result.
        """
        if self.dim != 2:
            raise AssertionError("This routine is only for 2D grids.")

        # Get the groups to plot
        n_grps = self.scalar_flux.shape[1]
        if groups is None:
            groups = [g for g in range(n_grps)]
        if isinstance(groups, int):
            groups = [groups]

        # Get the times to plot
        if times is None:
            times = [self.times[0], self.times[-1]]
        if isinstance(times, float):
            times = [times]

        # Get the grid
        x, y = self.grid[:, 0], self.grid[:, 1]
        x, y = np.unique(x), np.unique(y)
        xx, yy = np.meshgrid(x, y)

        # Get the dimensions of the subplots
        n_rows, n_cols = self._format_subplots(len(times))

        # Get the scalar flux at the specified times
        phi = []
        for t in times:
            phi += [self._interpolate(t, self.scalar_flux)]
        phi = np.array(phi)

        # Loop over groups
        for g in groups:
            figsize = (4*n_cols, 4*n_rows)
            fig: Figure = plt.figure(figsize=figsize)
            fig.suptitle(f"Group {g}")

            # Plot this groups scalar fluxes
            phi_g = phi[:, g] / np.max(phi[:, g])
            for i in range(len(times)):
                phi_gt = phi_g[i].reshape(xx.shape)

                ax: Axes = fig.add_subplot(n_rows, n_cols, i + 1)
                ax.set_xlabel("X (cm)")
                ax.set_ylabel("Y (cm)")
                ax.set_title(f"Time = {times[i]:.2f} sec")
                im = ax.pcolor(xx, yy, phi_gt, cmap="jet", shading="auto",
                               vmin=0.0, vmax=phi_g.max())
                fig.colorbar(im)
            fig.tight_layout()

    def plot_power(self, logscale: bool = False,
                   average: bool = True) -> None:
        """Plot the system power.

        Parameters
        ----------
        logscale : bool, default False
            Flag to plot power on a log scale.
        normalize : bool, default True
            Flag to normalize to the initial power.
        """
        times = self.times
        power = self.system_power
        if average:
            power /= self.core_volume

        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("System Power")
        plotter = ax.semilogy if logscale else ax.plot
        plotter(times, power, "-*b")
        plt.grid(True)
        plt.tight_layout()

    def _interpolate(self, t: float,
                     data: ndarray) -> ndarray:
        """Interpolate the solution at the specified time.

        Parameters
        ----------
        t : float
            The time to get the value(s) at.
        data : ndarray, (n_steps, n_nodes)
            The data to interpolate.

        Returns
        -------
        ndarray
            The interpolated result.
        """
        if not self.times[0] <= t <= self.times[-1]:
            raise ValueError(
                "Provided time is outside of simulation bounds.")

        dt = np.diff(self.times)[0]
        i = [int(np.floor(t/dt)), int(np.ceil(t/dt))]
        w = [i[1] - t/dt, t/dt - i[0]]
        if i[0] == i[1]:
            w = [1.0, 0.0]

        return w[0]*data[i[0]] + w[1]*data[i[1]]

    def _normalize_vector(self, vector: ndarray, method: str) -> ndarray:
        """Normalize a vector based on `method`.

        Parameters
        ----------
        vector : ndarray
            The vector to normalize.
        method : str, {"inf", "L1", "L2"}
            The normalization method.

        Returns
        -------
        ndarray
            The normalized input vector.
        """
        if method is None:
            return vector

        if method not in [np.inf, 1, 2]:
            raise ValueError("Invalid normalization type.")
        return vector / np.linalg.norm(vector, ord=method)

    @staticmethod
    def _format_subplots(n_plots: int) -> Tuple[int, int]:
        """Determine the number of rows and columns for subplots.

        Parameters
        ----------
        n_plots : int
            The number of subplots that will be used.

        """
        n_rows, n_cols = 1, 1
        if n_plots < 4:
            n_rows, n_cols = 1, 3
        elif 4 <= n_plots < 9:
            ref = int(np.ceil(np.sqrt((n_plots))))
            n_rows = n_cols = ref
            for n in range(1, n_cols + 1):
                if n * n_cols >= n_plots:
                    n_rows = n
                    break
        else:
            raise AssertionError("Maximum number of plots is 9.")
        return n_rows, n_cols

    def reset(self):
        self.grid.clear()
        self.times.clear()
        self.system_power.clear()
        self.scalar_flux.clear()
        self.precursors.clear()
        self.power_density.clear()
        self.temperature.clear()