from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from .mesh import Mesh


class Cell:
    """
    Base class for a cell on a mesh.

    Attributes
    ----------
    local_id : int
        The id of the cell on the mesh.
    material_id : int
        An identifier for the material that belongs to
        this cell. Currently, cells may only have one
        material defined on them.
    vertex_ids : List[int]
        The vertex ids that live on this cell.
    faces : List[Face]
        The faces the belong to this cell.
    volume : float
    centroid : float
    """
    def __init__(self):
        self.local_id: int = -1
        self.material_id: int = -1

        self.vertex_ids: List[int] = []
        self.faces: List[Face] = []

        self.volume: float = 0.0
        self.centroid: float = 0.0
        self.width: float = 0.0

    @property
    def num_vertices(self) -> int:
        """
        Get the number of vertices on this cell.

        Returns
        -------
        int
        """
        return len(self.vertex_ids)

    @property
    def num_faces(self) -> int:
        """
        Get the number of faces on this cell.

        Returns
        -------
        int
        """
        return len(self.faces)

    @property
    def is_boundary(self) -> bool:
        """
        Check whether the cell is on a boundary.

        Returns
        -------
        bool
        """
        return any([face.is_boundary for face in self.faces])


class Face:
    """
    Base class for a face on a cell.

    Attributes
    ----------
    vertex_ids : List[int]
        The vertex ids that live on the face.
    normal : float
        The outward-pointing normal vector of the face.
    area : float
    centroid : float
    neighbor_id: int
        The id of the adjacent cell to this face.
    """
    def __init__(self):
        self.vertex_ids: List[int] = []
        self.neighbor_id: int = -1

        self.normal: float = 0.0
        self.area: float = 0.0
        self.centroid: float = 0.0

    @property
    def num_vertices(self) -> int:
        """
        Get the number of vertices on this face.

        Returns
        -------
        int
        """
        return len(self.vertex_ids)

    @property
    def is_boundary(self) -> bool:
        """
        Check whether this face is on a boundary.

        Returns
        -------
        bool
        """
        return self.neighbor_id < 0

    def get_neighbor_cell(self, mesh: 'Mesh') -> 'Cell':
        """
        Get the neighboring cell to this face.

        Parameters
        ----------
        mesh : Mesh

        Returns
        -------
        Cell
        """
        if self.neighbor_id >= 0:
            return mesh.cells[self.neighbor_id]
