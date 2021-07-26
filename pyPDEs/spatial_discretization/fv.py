import numpy as np
from numpy import ndarray

from ..mesh import Mesh, Cell
from .spatial_discretization import SpatialDiscretization
from ..utilities import UnknownManager


class FiniteVolume(SpatialDiscretization):
    """
    Finite Volume spatial discreatization.
    """
    def __init__(self, mesh: Mesh) -> None:
        super().__init__(mesh, "FINITE_VOLUME")

    @property
    def num_nodes(self) -> int:
        """
        Get the number of nodes on the discretization.

        Returns
        -------
        int
        """
        return self.mesh.num_cells

    def map_dof(self, cell: Cell, node: int = 0,
                unknown_manager: UnknownManager = None,
                unknown_id: int = 0, component: int = 0) -> int:
        """
        Get a dof on a cell.

        Parameters
        ----------
        cell : Cell
        node : int, unused, default 0
        unknown_manager : UnknownManager, default None
        unknown_id : int, default 0
        component : int, default 0

        Returns
        -------
        int
        """
        if not unknown_manager:
            return cell.local_id
        num_unknowns = unknown_manager.total_num_components
        block_id = unknown_manager.map_unknown(unknown_id, component)
        if unknown_manager.storage_method == "NODAL":
            return cell.local_id * num_unknowns + block_id
        elif unknown_manager.storage_method == "BLOCK":
            return int(self.num_nodes * block_id + cell.local_id)

    @property
    def grid(self) -> ndarray:
        """
        Get the node locations for this discretization.

        Returns
        -------
        ndarray
        """
        return np.array([cell.centroid for cell in self.mesh.cells])
