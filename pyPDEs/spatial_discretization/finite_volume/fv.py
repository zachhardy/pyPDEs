import numpy as np
from numpy import ndarray

from pyPDEs.mesh import Mesh, Cell
from pyPDEs.spatial_discretization.spatial_discretization import SpatialDiscretization
from pyPDEs.utilities import UnknownManager


class FiniteVolume(SpatialDiscretization):
    """
    Finite volume spatial discreatization.
    """
    def __init__(self, mesh: Mesh) -> None:
        super().__init__(mesh)
        self.type = "FV"

    @property
    def n_nodes(self) -> int:
        return self.mesh.n_cells

    @property
    def grid(self) -> ndarray:
        return np.array([cell.centroid for cell in self.mesh.cells])

    def map_dof(self, cell: Cell, node: int = 0,
                unknown_manager: UnknownManager = None,
                unknown_id: int = 0, component: int = 0) -> int:
        """
        Maps a node on a cell to a dof. Unknown managers
        are used for multi-component problems and are used
        to map to specific unknowns and components.
        """
        if not unknown_manager:
            return cell.id
        num_unknowns = unknown_manager.total_components
        block_id = unknown_manager.map_unknown(unknown_id, component)
        if unknown_manager.storage_method == "NODAL":
            return cell.id * num_unknowns + block_id
        elif unknown_manager.storage_method == "BLOCK":
            return int(self.n_nodes * block_id + cell.id)
