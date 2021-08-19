from typing import List

from ..utilities import Vector


class Cell:
    """General cell.

    Attributes
    ----------
    cell_type : {"SLAB", "QUAD"}
        The geometrical type of the cell. Options are "SLAB"
        for one-dimensional cells. and "QUAD" for two-dimensional
        cells.
    coord_sys : {"CARTESIAN", "CYLINDRICAL", "SPHERICAL"}
        The coordinate system for the cell. "CYLINDRICAL" and
        "SPHERICAL" are only supported for one-dimensional line
        meshes.
    id : int
        A unique ID for the cell.
    material_id : int
        The material ID associated with this cell.
    vertex_ids : List[int]
        The IDs of the vertices that define this cell.
    faces : List[Face]
        The faces that enclose this cell. See `Face` documentation
        for more information.
    volume : float
        The volume of the cell.
    centroid : Vector
        The centroid of the cell.
    width : Vector
        The x, y, and z widths of slab cells, quadrilateral
        cells, and hexahedron cells. This attribute is not
        valid for triangular and tetrahedron cells.
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
        """Get the number of faces on this cell.

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
        """Get whether a cell is a boundary cell.

        Returns
        -------
        bool
        """
        return any([not f.has_neighbor for f in self.faces])

    def __eq__(self, other: "Cell") -> bool:
        return self.id == other.id


__all__ = ["Cell"]
