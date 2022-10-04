from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import NeutronicsSimulationReader

import numpy as np
import matplotlib.pyplot as plt

from os.path import splitext
from typing import Union


def plot_flux_moment(
        self: 'NeutronicsSimulationReader',
        moment: int = 0,
        groups: Union[int, list[int]] = None,
        times: Union[float, list[float]] = None,
        filename: str = None
) -> None:
    """
    Plot a group-wise flux moments at the specified times.

    Parameters
    ----------
    self : NeutronicsSimulationReader
    moment : int, default 0
        The moment index.
    groups : list[int] or int or None, default None
        The group indices. If None, only the first group is plotted.
        If -1, all groups are plotted. If an int, only that group
        is plotted. If a list of int, the listed groups are plotted.
    times : list[float] or float or None, default None
        If None, the last snapshot is plotted. If a float or a list
        of float is specified, the specified times are plotted. If
        the specified time does not lie on a snapshot, an interpolation
        is performed.
    filename : str, default None.
        A location to save the plot to, if specified.
    """

    # Parse the groups input
    if groups is None:
        groups = [0]
    if groups == -1:
        groups = [g for g in range(self.n_groups)]
    if isinstance(groups, int):
        groups = [groups]

    # Parse the times input
    if times is None:
        times = [self.times[-1]]
    if isinstance(times, float):
        times = [times]
    elif isinstance(times, np.ndarray):
        times = times.tolist()

    # Check the moment index
    if moment > self.n_moments - 1:
        msg = f"Invalid moment index {moment}."
        raise ValueError(msg)

    # Check the group indices
    if not isinstance(groups, list):
        msg = "The groups must be a list."
        raise TypeError(msg)

    # Check group indices
    for group in groups:
        if group < 0 or group >= self.n_groups:
            msg = f"Invalid group index {group}."
            raise ValueError(msg)

    # Check the times
    for time in times:
        if time < self.times[0] or time > self.times[-1]:
            msg = f"Invalid time ({time}) specified."
            raise ValueError(msg)

    # Get the flux moments at the specified times
    phi = self._interpolate(times, self.flux_moments)

    # Plot 1D flux moments
    if self.dimension == 1:
        for t, time in enumerate(times):
            plt.figure()
            plt.title(f"Time = {time:.3f}")
            plt.xlabel(f"r")
            plt.ylabel(r"$\phi_{m,g}(r)$")

            # Plot the group-wise moments
            for group in groups:
                plt.plot(self.nodes[:, 2],
                         phi[t][moment][group],
                         label=f"Group {group}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if filename is not None:
                base, ext = splitext(filename)
                plt.savefig(f"{base}_{t}.pdf")

    elif self.dimension == 2:
        x = [node[0] for node in self.nodes]
        y = [node[1] for node in self.nodes]
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        if len(times) * len(groups) > 6:
            msg = "Too many plots. " \
                  "Consider reducing the number of groups or times."
            raise AssertionError(msg)

        for t, time in enumerate(times):
            for g, group in enumerate(groups):
                phi_ = phi[t][moment][g].reshape(X.shape)

                plt.figure()
                plt.title(f"Time = {time:.3f}\n"
                          f"Moment {moment}, Group {group}")
                plt.xlabel("X (cm)")
                plt.ylabel("Y (cm)")

                im = plt.pcolor(X, Y, phi_,
                                cmap="jet", shading="auto",
                                vmin=0.0, vmax=phi_.max())
                plt.colorbar(im)
                plt.tight_layout()

                if filename is not None:
                    base, ext = splitext(filename)
                    plt.savefig(f"{base}_g{group}_{t}.pdf")
    plt.show()


def plot_power_profile(
        self: 'NeutronicsSimulationReader',
        times: Union[float, list[float]] = None,
        filename: str = None
) -> None:
    """
    Plot the power density at the specified times.

    Parameters
    ----------
    self : NeutronicsSimulationReader
    times : list[float] or float or None, default None
        If None, the last snapshot is plotted. If a float or a list
        of float is specified, the specified times are plotted. If
        the specified time does not lie on a snapshot, an interpolation
        is performed.
    filename : str, default None.
        A location to save the plot to, if specified.
    """

    # Parse the times input
    if times is None:
        times = [self.times[-1]]
    if isinstance(times, float):
        times = [times]
    elif isinstance(times, np.ndarray):
        times = times.tolist()

    # Check the times
    for time in times:
        if time < self.times[0] or time > self.times[-1]:
            msg = f"Invalid time ({time}) specified."
            raise ValueError(msg)

    # Energy per fission in J/fission
    Ef = TransientSolver.energy_per_fission

    # Get the fission rate at the specifief times
    Sf = self._interpolate(times, self.fission_rates)

    # Plot 1D power densities
    if self.dimension == 1:
        plt.figure()
        plt.xlabel("z (cm)")
        plt.ylabel("Power Density")

        # Plot the power density at each time
        for t, time in enumerate(times):
            plt.plot(self.nodes[:, 2], Ef * Sf[t],
                     label=f"Time = {time:.3f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if filename is not None:
            base, ext = splitext(filename)
            plt.savefig(f"{base}.pdf")

    # Plot 2D power densities
    elif self.dimension == 2:
        x = [node[0] for node in self.nodes]
        y = [node[1] for node in self.nodes]
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        if len(times) > 6:
            msg = "Too many plots. " \
                  "Consider reducing the number of times."
            raise AssertionError(msg)

        for t, time in enumerate(times):
            P_ = Ef * Sf[t].reshape(X.shape)

            plt.figure()
            plt.title(f"Time = {time:.3f}")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")

            im = plt.pcolor(X, Y, P_,
                            cmap="jet", shading="auto",
                            vmin=0.0, vmax=P_.max())
            plt.colorbar(im)
            plt.tight_layout()

            if filename is not None:
                base, ext = splitext(filename)
                plt.savefig(f"{base}_{t}.pdf")
    plt.show()


def plot_temperature_profile(
        self: 'NeutronicsSimulationReader',
        times: Union[float, list[float]] = None,
        filename: str = None
) -> None:
    """
    Plot the power density at the specified times.

    Parameters
    ----------
    self : NeutronicsSimulationReader
    times : list[float] or float or None, default None
        If None, the last snapshot is plotted. If a float or a list
        of float is specified, the specified times are plotted. If
        the specified time does not lie on a snapshot, an interpolation
        is performed.
    filename : str, default None.
        A location to save the plot to, if specified.
    """

    # Parse the times input
    if times is None:
        times = [self.times[-1]]
    if isinstance(times, float):
        times = [times]
    elif isinstance(times, np.ndarray):
        times = times.tolist()

    # Check the times
    for time in times:
        if time < self.times[0] or time > self.times[-1]:
            msg = f"Invalid time ({time}) specified."
            raise ValueError(msg)

    # Get the temperatures at the specified times
    T = self._interpolate(times, self.temperatures)

    # Plot 1D power densities
    if self.dimension == 1:
        plt.figure()
        plt.xlabel("z (cm)")
        plt.ylabel("Temperature (K)")

        # Plot the power density at each time
        for t, time in enumerate(times):
            plt.plot(self.nodes[:, 2], T[t],
                     label=f"Time = {time:.3f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if filename is not None:
            base, ext = splitext(filename)
            plt.savefig(f"{base}.pdf")

    # Plot 2D power densities
    elif self.dimension == 2:
        x = [node[0] for node in self.nodes]
        y = [node[1] for node in self.nodes]
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        if len(times) > 6:
            msg = "Too many plots. " \
                  "Consider reducing the number of times."
            raise AssertionError(msg)

        for t, time in enumerate(times):
            T_ = T[t].reshape(X.shape)

            plt.figure()
            plt.title(f"Temperature (K)\n"
                      f"Time = {time:.3f}")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")

            im = plt.pcolor(X, Y, T_,
                            cmap="jet", shading="auto",
                            vmin=0.0, vmax=T_.max())
            plt.colorbar(im)
            plt.tight_layout()

            if filename is not None:
                base, ext = splitext(filename)
                plt.savefig(f"{base}_{t}.pdf")
    plt.show()


def plot_power(
        self: 'NeutronicsSimulationReader',
        mode: str = "TOTAL",
        logscale: bool = False,
        filename: str = None
) -> None:
    """
    Plot the reactor power, peak power density, or average power density
    as a function of time.

    Parameters
    ----------
    self : NeutronicsSimulationReader
    mode : str {'TOTAL', 'PEAK', 'AVERAGE', 'BOTH'}, default 'TOTAL'
        The power quantity to plot.
    logscale : bool, default False
        A flag for plotting the y-axis on a logarithmic scale.
    filename : str, default None.
        A location to save the plot to, if specified.
    """
    if mode not in ["TOTAL", "PEAK", "AVERAGE", "BOTH"]:
        msg = "Invalid mode. Mode must be [TOTAL/PEAK/AVERAGE]."
        raise ValueError(msg)

    # Get the appropriate quantity to plot
    if mode == "TOTAL":
        p = [self.powers]
        ylabel = "Power"
    elif mode == "PEAK":
        p = [self.peak_power_densities]
        ylabel = "Peak Power Density"
    elif mode == "AVERAGE":
        p = [self.average_power_densities]
        ylabel = "Average Power Density"
    else:
        p = [self.peak_power_densities,
             self.average_power_densities]
        ylabel = "Power Density"

    # Plot
    plt.figure()
    plt.xlabel("Time")
    plt.ylabel(ylabel)

    plotter = plt.plot if not logscale else plt.semilogy
    if len(p) == 1:
        plotter(self.times, p[0], '-*b')
    else:
        plotter(self.times, p[0], '-*b', label="Peak")
        plotter(self.times, p[1], '-*r', label="Average")
        plt.legend()

    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        base, ext = splitext(filename)
        plt.savefig(f"{base}.pdf")


def plot_fuel_temperature(
        self: 'NeutronicsSimulationReader',
        mode: str = "PEAK",
        logscale: bool = False,
        filename: str = None
) -> None:
    """
    Plot the peak or average fuel temperatures as a function of time.

    Parameters
    ----------
    self : NeutronicsSimulationReader
    mode : str {'PEAK', 'AVERAGE', 'BOTH'}, default 'PEAK'
        The fuel temperature quantity to plot.
    logscale : bool, default False
        A flag for plotting the y-axis on a logarithmic scale.
    filename : str, default None.
        A location to save the plot to, if specified.
    """
    if mode not in ["PEAK", "AVERAGE", "BOTH"]:
        msg = "Invalid mode. Mode must be [PEAK/AVERAGE/BOTH]."
        raise ValueError(msg)

    # Get appropriate quantity to plot
    if mode == "PEAK":
        T = [self.peak_fuel_temperatures]
        ylabel = "Peak Fuel Temperature (K)"
    elif mode == "AVERAGE":
        T = [self.average_fuel_temperatures]
        ylabel = "Average Fuel Temperature (K)"
    else:
        T = [self.peak_fuel_temperatures,
             self.average_fuel_temperatures]
        ylabel = "Temperature (K)"

    # Plot
    plt.figure()
    plt.xlabel("Time")
    plt.ylabel(ylabel)

    plotter = plt.plot if not logscale else plt.semilogy
    if len(T) == 1:
        plotter(self.times, T[0], '-*b')
    else:
        plotter(self.times, T[0], '*-b', label="Peak")
        plotter(self.times, T[1], '*-r', label="Average")
        plt.legend()

    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        base, ext = splitext(filename)
        plt.savefig(f"{base}.pdf")
