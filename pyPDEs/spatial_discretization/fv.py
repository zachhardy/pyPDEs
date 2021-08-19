import numpy as np
from numpy import ndarray
from typing import List

from pyPDEs.mesh import Mesh, Cell
from pyPDEs.spatial_discretization import SpatialDiscretization
from pyPDEs.utilities import UnknownManager, Vector


class FiniteVolume(SpatialDiscretization):
    """Finite volume spatial discreatization.

    Attributes
    ----------
    type : str
        The discretization type.
    mesh : Mesh
        The mesh being discretized.
    dim : int
        The dimentsion of the mesh being discretized.
    coord_sys : {"CARTESIAN", "CYLINDER", "SPHERICAL"}
        The coordinate system of the mesh.
    """

    def __init__(self, mesh: Mesh) -> None:
        """Finite volume constructor.

        Parameters
        ----------
        mesh : Mesh
        """
        super().__init__(mesh)
        self.type = "FV"

    @property
    def n_nodes(self) -> int:
        """Get the number of nodes in the discretization.

        For finite volume discretizations, the number of nodes
        is equal to the number of cells.

        Returns
        -------
        int
        """
        return self.mesh.n_cells

    @property
    def grid(self) -> List[Vector]:
        """Get the list of nodes that define the discretization.

        For finite volume discretizations, the nodes are located
        at the centroid of the cells.

        Returns
        -------
        List[Vector]
        """
        return [cell.centroid for cell in self.mesh.cells]

    def map_dof(self, cell: Cell, node: int = 0,
                unknown_manager: UnknownManager = None,
                unknown_id: int = 0, component: int = 0) -> int:
        """Map a node on a cell to a global DoF index.

        Parameters
        ----------
        cell : Cell
            The cell that the node under consideration lives on.
        node : int, default 0
            The local index of the node on the cell. This is
            unused because finite volume always has one node
            per cell.
        unknown_manager : UnknownManager, default None
            The unknown manager is used as a mapping from node
            to global DoF index for multi-component problems.
            If no unknown manager is supplied, it is assumed
            that it is a one component problem.
        unknown_id : int, default 0
            The unknown ID of the DoF being mapped.
        component : int, default 0
            The component of the unknown of the DoF being mapped.

        Returns
        -------
        int
        """
        if not unknown_manager:
            return cell.id
        num_unknowns = unknown_manager.total_components
        block_id = unknown_manager.map_unknown(unknown_id, component)
        if unknown_manager.storage_method == "NODAL":
            return cell.id * num_unknowns + block_id
        elif unknown_manager.storage_method == "BLOCK":
            return int(self.n_nodes * block_id + cell.id)
