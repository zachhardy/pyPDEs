from numpy import ndarray

from ..mesh import Mesh, Cell
from ..utilities import UnknownManager


class SpatialDiscretization:
    """
    Base class for spatial discretizations.

    Attributes
    ----------
    type : str
        An identifier for a spatial discretization.
    mesh : Mesh
        The mesh this discretization lives on.
    dim : int
        The dimension of the mesh.
    coord_sys : str
        The coordinate system type defined on the mesh.
    """

    def __init__(self, mesh: 'Mesh',
                 discretization_type: str = None) -> None:
        self.type: str = discretization_type

        self.mesh: Mesh = mesh
        self.dim: int = mesh.dim
        self.coord_sys: str = mesh.coord_sys

    @property
    def num_nodes(self) -> int:
        """
        Get the number of nodes on the discretization.

        Returns
        -------
        int
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement {cls_name}.num_nodes.")

    def num_dofs(self, unknown_manager: UnknownManager = None) -> int:
        """
        Get the total number of dofs in the problem.

        Returns
        -------
        int
        """
        num_unknowns = 1
        if unknown_manager:
            num_unknowns = unknown_manager.total_num_components
        return num_unknowns * self.num_nodes

    def map_dof(self, cell: Cell, node: int,
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
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement to {cls_name}.map_dof method.")

    @property
    def grid(self) -> ndarray:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement to {cls_name}.grid property.")
