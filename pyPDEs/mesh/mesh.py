import numpy as np
from typing import List

from .cell import Cell, Face


class Mesh:
    """
    Base class for meshes.

    Attributes
    ----------
    dim : int
        The dimension of the mesh. This should be 1, 2, or 3.
    coord_sys : str
        The coordinate system type. This should be 'slab',
        'cylinder', or 'sphere'.
    cells : List[Cell]
        The list of cells belonging to the mesh.
    vertices : List[float]
        The list of vertices belonging to the mesh.
    boundary_cell_ids : List[int]
        The indices of the cells which live on boundaries.
    """
    def __init__(self):
        self.dim: int = 0
        self.coord_sys: str = 'slab'

        self.cells: List[Cell] = []
        self.vertices: List[float] = []
        self.boundary_cell_ids: List[int] = []

    @property
    def num_cells(self) -> int:
        """
        Get the number of cells in the mesh.

        Returns
        -------
        int
        """
        return len(self.cells)

    @property
    def num_vertices(self) -> int:
        """
        Get the number of vertices on the mesh.

        Returns
        -------
        int
        """
        return len(self.vertices)

    def compute_cell_volume(self, cell: Cell) -> float:
        """
        Compute the volume of a cell.

        Parameters
        ----------
        cell : Cell

        Returns
        -------
        float
        """
        # ============================== Compute 1D cell volumes
        if cell.num_vertices == 2:
            vl = self.vertices[cell.vertex_ids[0]]
            vr = self.vertices[cell.vertex_ids[1]]
            if self.coord_sys == 'slab':
                return vr - vl
            elif self.coord_sys == 'cylinder':
                return 2.0 * np.pi * (vr ** 2 - vl ** 2)
            elif self.coord_sys == 'sphere':
                return 4.0 / 3.0 * np.pi * (vr ** 3 - vl ** 3)

    def compute_face_area(self, face: Face) -> float:
        """
        Compute the area of a cell face.

        Parameters
        ----------
        face : Face

        Returns
        -------
        float
        """
        # ============================== Compute 1D face areas
        if face.num_vertices == 1:
            v = self.vertices[face.vertex_ids[0]]
            if self.coord_sys == 'slab':
                return 1.0
            elif self.coord_sys == 'cylinder':
                return 2.0 * np.pi * v
            elif self.coord_sys == 'sphere':
                return 4.0 * np.pi * v ** 2

    def compute_cell_centroid(self, cell: Cell) -> float:
        """
        Compute the centroid of a cell.

        Parameters
        ----------
        cell : Cell

        Returns
        -------
        float
        """
        centroid = 0.0
        for vid in cell.vertex_ids:
            centroid += self.vertices[vid]
        return centroid / cell.num_vertices

    def compute_face_centroid(self, face: Face) -> float:
        """
        Compute the centroid of a face.

        Parameters
        ----------
        face : Face

        Returns
        -------
        float
        """
        centroid = 0.0
        for vid in face.vertex_ids:
            centroid += self.vertices[vid]
        return centroid / face.num_vertices
