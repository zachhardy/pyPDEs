from numpy import ndarray
from typing import List

from ..mesh import Mesh, Cell
from ..utilities import UnknownManager, Vector


class SpatialDiscretization:
    """
    Base class for spatial discretizations.

    Parameters
    ----------
    mesh : Mesh
    """

    def __init__(self, mesh: 'Mesh') -> None:
        self.type: str = None
        self.mesh: Mesh = mesh
        self.dim: int = mesh.dim
        self.coord_sys: str = mesh.coord_sys

    @property
    def n_nodes(self) -> int:
        """
        Get the number of nodes in the discretization.

        This is an abstract property and must be implemented
        in derived classes.

        Returns
        -------
        int
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'Subclasses must implement to {cls_name}.n_nodes property.')

    @property
    def grid(self) -> List[Vector]:
        """
        Get the list of nodes that define the discretization.

        This is an abstract property and must be implemented
        in derived classes.

        Returns
        -------
        List[Vector]
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement to {cls_name}.grid property.")

    def n_dofs(self, unknown_manager: UnknownManager = None) -> int:
        """
        Get the total number of dofs in the problem.

        Parameters
        ----------
        unknown_manager : UnknownManager
           The unknown manager is used to communicate how
           many unknwons per node exists.

        Returns
        -------
        int
            The number of nodes multiplied by the number of
            unknowns from the unknown manager.
        """
        num_unknowns = 1
        if unknown_manager:
            num_unknowns = unknown_manager.total_components
        return num_unknowns * self.n_nodes

    def map_dof(self, cell: Cell, node: int,
                unknown_manager: UnknownManager = None,
                unknown_id: int = 0, component: int = 0) -> int:
        """
        Map a node on a cell to a global DoF index.

        This is an abstract method that must be implemented
        in derived classes.

        Parameters
        ----------
        cell : Cell
            The cell that the node under consideration lives on.
        node : int
            The local index of the node on the cell.
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
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'Subclasses must implement to {cls_name}.map_dof method.')
