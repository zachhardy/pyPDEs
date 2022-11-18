from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import NeutronicsSimulationReader

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.animation import Animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ArtistAnimation
from matplotlib.collections import Collection

from typing import Union


def animate_flux_moment(
        self: 'NeutronicsSimulationReader',
        moment: int = 0,
        groups: Union[int, list[int]] = None,
        filename: str = None
) -> Union[FuncAnimation, list[FuncAnimation]]:
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

        # set up the plot
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Position")
        ax.set_ylabel(r"$\phi_{m,g}(r)$")
        ax.grid(True)

        title = plt.title("")
        lines: list[plt.Line2D] = [
            ax.plot([], [], label=f"Group {group}")[0]
            for group in groups
        ]
        labels: list[str] = [line.get_label() for line in lines]
        ax.legend(lines, labels)

        def _animate(n):
            # update plots
            y = self.flux_moments[n][moment]
            for g, group in enumerate(groups):
                lines[g].set_data(x, y[group])

            # formatting
            x_margin = 0.05 * np.max(np.abs(x))
            ax.set_xlim(min(x) - x_margin, max(x) + x_margin)

            if not np.min(y) == np.max(y):
                y_margin = 0.05 * np.max(np.abs(y))
                ax.set_ylim(np.min(y) - y_margin, np.max(y) + y_margin)

            title.set_text(f"Time = {self.times[n]:.3g} $\mu$s")
            plt.tight_layout()
            return lines

        anim = FuncAnimation(
            fig, _animate, frames=len(self.times), blit=False
        )

        if filename:
            base, ext = os.path.splitext(filename)
            if ext == ".gif":
                raise AssertionError("Invalid movie type.")
            writer = animation.FFMpegWriter(fps=15)
            anim.save(filename, writer=writer)
        return anim

    elif self.dimension == 2:

        # get the grid
        x = [node[0] for node in self.nodes]
        y = [node[1] for node in self.nodes]
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        # setup group-wise animations
        anims = []
        for g, group in enumerate(groups):

            # setup figure
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.add_subplot(1, 1, 1)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            title = plt.title("")
            phi = self.flux_moments[0][moment][group]
            im: Collection = ax.pcolor(
                X, Y, phi.reshape(X.shape), cmap='jet',
                shading='auto', vmin=phi.min(), vmax=phi.max(),
            )
            fig.colorbar(im)

            def _animate(n):
                vals = self.flux_moments[n][moment][group]
                title.set_text(f"Group {group}, "
                               f"Time = {self.times[n]:.3g} s")

                im.set_array(vals)
                vmin, vmax = im.get_clim()
                if any(vals > vmax) or vals.max() < 0.1 * vmax:
                    im.set_clim(vmin=vals.min(), vmax=vals.max())
                return im

            anim = FuncAnimation(
                fig, _animate, frames=len(self.times), blit=False
            )
            anims.append(anim)

            if filename:
                base, ext = os.path.splitext(filename)
                if ext == ".gif":
                    raise AssertionError("Invalid movie type.")
                writer = animation.FFMpegWriter(fps=15)
                anim.save(f"{base}_g{group}{ext}", writer=writer)
        return anims


def animate_spectrum(
        self: 'NeutronicsSimulationReader',
        filename: str = None
) -> FuncAnimation:
    """
    Animate the energy spectrum.

    Parameters
    ----------
    self : NeutronicsSimulationReader
    filename : str

    Returns
    -------
    FuncAnimation
    """

    def compute_spectrum(x):
        y = np.linalg.norm(x, axis=1)[::-1]
        return y / y.sum() if y.sum() != 0.0 else y

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Energy Group")
    ax.set_ylabel(r"$\phi(E)$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    title = plt.title("")
    phi = compute_spectrum(self.flux_moments[0][0])
    line: plt.Line2D = ax.plot(list(range(self.n_groups)), phi, '-*b')[0]
    line.set_xdata(list(range(self.n_groups)))

    def _animate(n):
        y = compute_spectrum(self.flux_moments[n][0])
        line.set_data(list(range(self.n_groups)), y)
        title.set_text(f"Time = {self.times[n]:<.3f} sh")
        plt.tight_layout()
        return line,

    anim = FuncAnimation(
        fig, _animate, frames=len(self.times), blit=False
    )

    if filename:
        base, ext = os.path.splitext(filename)
        if ext == ".gif":
            raise AssertionError("Invalid movie type.")
        writer = animation.FFMpegWriter(fps=15)
        anim.save(filename, writer=writer)

    return anim
