from typing import List

from ..utilities import Vector


class Cell:
    """
    Base class for a cell on a mesh.
    """
    def __init__(self):
        self.cell_type: str = None
        self.coord_sys: str = None

        self.id: int = -1
        self.material_id: int = -1
        self.vertex_ids: List[int] = []

        self.faces: List[Face] = []

        self.volume: float = 0.0
        self.centroid: Vector = 0.0
        self.width: Vector = 0.0

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
        Return whether or not any face on this cell is
        on a boundary of the mesh.

        Returns
        -------
        bool
        """
        return any([not f.has_neighbor for f in self.faces])

    def __eq__(self, other: 'Cell') -> bool:
        return self.id == other.id


class Face:
    """
    Base class for a face on a cell.
    """
    def __init__(self):
        self.vertex_ids: List[int] = []

        self.normal: Vector = 0.0
        self.area: float = 0.0
        self.centroid: Vector = 0.0

        self.has_neighbor: bool = False
        self.neighbor_id = 0

    @property
    def n_vertices(self) -> int:
        """
        Get the number of vertices on this face.

        Returns
        -------
        int
        """
        return len(self.vertex_ids)

    def __eq__(self, other: 'Face') -> bool:
        return set(self.vertex_ids) == set(other.vertex_ids)
