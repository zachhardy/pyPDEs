import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Mesh


def plot_material_ids(self: 'Mesh') -> None:
    """
    Plot the material IDs cell-wise.

    This is a utility to ensure that materials are set correctly
    in accordance to the user's wishes.
    """
    matids = [cell.material_id for cell in self.cells]
    if self.dim == 1:
        plt.xlabel('z')

        z = [cell.centroid.z for cell in self.cells]
        plt.plot(z, matids, 'ob')
        plt.grid(True)

    elif self.dim == 2:
        plt.xlabel('x')
        plt.ylabel('y')

        x = [cell.centroid.x for cell in self.cells]
        y = [cell.centroid.y for cell in self.cells]
        xx, yy = np.meshgrid(np.unique(x), np.unique(y))
        matids = np.array(matids).reshape(xx.shape)
        plt.pcolormesh(xx, yy, matids, cmap='jet', shading='auto')
        plt.colorbar()
    plt.tight_layout()
