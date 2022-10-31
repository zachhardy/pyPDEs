from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import NeutronicsSimulationReader

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import Animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ArtistAnimation
from matplotlib.collections import Collection

from typing import Union


def animate_flux_moment(
        self: 'NeutronicsSimulationReader',
        moment: int = 0,
        groups: Union[int, list[int]] = None,
) -> FuncAnimation:
    """
    Animate a group-wise flux moment.

    Parameters
    ----------
    self : NeutronicsSimulationReader
    moment : int, default 0
        The moment index.
    groups : list[int] or int or None, default None
        The group indices. If None, only the first group is plotted.
        If -1, all groups are plotted. If an int, only that group
        is plotted. If a list of int, the listed groups are plotted.
    """

    ############################################################
    # Input checks
    ############################################################

    # parse the groups input
    if groups is None:
        groups = [0]
    if groups == -1:
        groups = [g for g in range(self.n_groups)]
    if isinstance(groups, int):
        groups = [groups]

    # check the moment index
    if moment > self.n_moments - 1:
        msg = f"Invalid moment index {moment}."
        raise ValueError(msg)

    # check the group indices
    if not isinstance(groups, list):
        msg = "The groups must be a list."
        raise TypeError(msg)

    ############################################################
    # Plot 1D data
    ############################################################

    if self.dimension == 1:

        # get the grid
        x = self.nodes[:, 2]

        # setup the plot
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Position")
        ax.set_ylabel(r"$\phi_{m,g}(r)$")
        ax.grid(True)

        title = fig.suptitle("")
        lines: list[plt.Artist] = [
            ax.plot([], [], label=f"Group {group}")[0]
            for group in groups
        ]
        labels: list[str] = [line.get_label() for line in lines]
        ax.legend(lines, labels)

        def _animate(n):
            # update plots
            phi = self.flux_moments[n][moment]
            for g, group in enumerate(groups):
                lines[g].set_data(self.nodes[:, 2], phi[group])

            # formatting
            x_margin = 0.05 * np.max(np.abs(x))
            y_margin = 0.05 * np.max(np.abs(phi))
            ax.set_xlim(min(x) - x_margin, max(x) + x_margin)
            ax.set_ylim(np.min(phi) - y_margin, np.max(phi) + y_margin)
            title.set_text(f"Time = {self.times[n]:.3g} $\mu$s")
            plt.tight_layout()
            return lines

        return FuncAnimation(fig, _animate, frames=len(self.times),
                             interval=20, blit=False)

    elif self.dimension == 2:

        # get the grid
        x = [node[0] for node in self.nodes]
        y = [node[1] for node in self.nodes]
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        # setup group-wise animations
        for g, group in enumerate(groups):

            # setup figure
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.add_subplot(1, 1, 1)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            title = fig.suptitle("")
            phi = self.flux_moments[0][moment][group]
            im: Collection = ax.pcolor(
                X, Y, phi.reshape(X.shape), cmap='jet',
                shading='auto', vmin=phi.min(), vmax=phi.max(),
            )
            fig.colorbar(im)

            def _animate(n):
                phi = self.flux_moments[n][moment][group]
                title.set_text(f"Group {group}, "
                               f"Time = {self.times[n]:.3g} s")

                im.set_array(phi)
                vmin, vmax = im.get_clim()
                if any(phi > vmax) or phi.max() < 0.1 * vmax:
                    im.set_clim(vmin=phi.min(), vmax=phi.max())
                return im

            mov = FuncAnimation(fig, _animate, frames=len(self.times),
                                interval=20, blit=False)
            plt.show()
