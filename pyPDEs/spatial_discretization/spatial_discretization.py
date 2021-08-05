from numpy import ndarray

from ..mesh import Mesh, Cell
from ..utilities import UnknownManager


class SpatialDiscretization:
    """
    Base class for spatial discretizations.
    """
    def __init__(self, mesh: 'Mesh') -> None:
        self.type: str = None
        self.mesh: Mesh = mesh
        self.dim: int = mesh.dim
        self.coord_sys: str = mesh.coord_sys

    @property
    def n_nodes(self) -> int:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement to {cls_name}.n_nodes property.")

    @property
    def grid(self) -> ndarray:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement to {cls_name}.grid property.")

    def n_dofs(self, unknown_manager: UnknownManager = None) -> int:
        """
        Get the total number of dofs in the problem.
        """
        num_unknowns = 1
        if unknown_manager:
            num_unknowns = unknown_manager.total_components
        return num_unknowns * self.n_nodes

    def map_dof(self, cell: Cell, node: int,
                unknown_manager: UnknownManager = None,
                unknown_id: int = 0, component: int = 0) -> int:
        """
        Maps a node on a cell to a dof. Unknown managers
        are used for multi-component problems and are used
        to map to specific unknowns and components.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement to {cls_name}.map_dof method.")
