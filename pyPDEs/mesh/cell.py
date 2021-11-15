from typing import List
from ..utilities import Vector

__all__ = ['Cell']


class Cell:
    """
    General cell.
    """
    def __init__(self):
        self.cell_type: str = None
        self.coord_sys: str = None

        self.id: int = -1
        self.material_id: int = -1
        self.vertex_ids: List[int] = []

        self.faces: List[Face] = []

        self.volume: float = 0.0
        self.centroid: Vector = Vector()
        self.width: Vector = Vector()

    @property
    def n_faces(self) -> int:
        """
        Get the number of faces on this cell.

        Returns
        -------
        int
        """
        return len(self.faces)

    @property
    def n_vertices(self) -> int:
        """
        Get the number of vertices on this cell.

        Returns
        -------
        int
        """
        return len(self.vertex_ids)

    @property
    def is_boundary(self) -> bool:
        """
        Get whether a cell is a boundary cell.

        Returns
        -------
        bool
        """
        return any([not f.has_neighbor for f in self.faces])

    def __eq__(self, other: 'Cell') -> bool:
        return self.id == other.id
