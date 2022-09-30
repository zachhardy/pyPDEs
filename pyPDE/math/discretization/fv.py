from pyPDE.mesh import Mesh
from pyPDE.mesh import Cell
from pyPDE.mesh import CartesianVector

from .discretization import SpatialDiscretization


class FiniteVolume(SpatialDiscretization):
    """
    Implementation of a finite volume discretization.
    """

    def __init__(self, mesh: Mesh) -> None:
        """
        Parameters
        ----------
        mesh : Mesh
        """
        super().__init__(mesh, "FV")

    def n_nodes(self) -> int:
        """
        Return the total number of nodes in the discretization.

        Returns
        -------
        int
        """
        return self.mesh.n_cells

    def nodes_per_cell(self, cell: Cell) -> int:
        """
        Return the number of nodes on the specified Cell.

        Parameters
        ----------
        cell : Cell

        Returns
        -------
        int
        """
        return 1

    def n_dofs(self, n_components: int = 1) -> int:
        """
        Return the total number of degrees of freedom in the discretization.

        Parameters
        ----------
        n_components : int

        Returns
        -------
        int
        """
        return n_components * self.n_nodes()

    def n_dofs_per_cell(self, cell: Cell, n_components: int = 1) -> int:
        """
        Return the number of degrees of freedom on the specified Cell.

        Parameters
        ----------
        cell : Cell
        n_components : int

        Returns
        -------
        int
        """
        return n_components

    def nodes(self, cell: Cell) -> list[CartesianVector]:
        """
        Return the node coordinates on the specified Cell.

        Parameters
        ----------
        cell : Cell

        Returns
        -------
        list[CartesianVector]
        """
        return [cell.centroid]
