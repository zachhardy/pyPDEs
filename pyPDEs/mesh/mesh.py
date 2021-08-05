import numpy as np
from typing import List, Union

from .cell import Cell, Face


class Mesh:
    """
    Base class for meshes.
    """
    def __init__(self):
        self.dim: int = 0
        self.coord_sys: str = "CARTESIAN"

        self.cells: List[Cell] = []
        self.vertices: List[float] = []
        self.boundary_cell_ids: List[int] = []

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_faces(self) -> int:
        if self.dim == 1:
            return self.n_cells + 1
        else:
            raise NotImplementedError(
                f"Only 1D meshes have been implemented.")

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def compute_volume(self, cell: Cell) -> float:
        """
        Compute the volume of a cell.
        """
        # ======================================== 1D volumes
        if self.dim == 1:
            vl = self.vertices[cell.vertex_ids[0]]
            vr = self.vertices[cell.vertex_ids[1]]
            if self.coord_sys == "CARTESIAN":
                return vr - vl
            elif self.coord_sys == "CYLINDRICAL":
                return 2.0 * np.pi * (vr ** 2 - vl ** 2)
            elif self.coord_sys == "SPHERICAL":
                return 4.0 / 3.0 * np.pi * (vr ** 3 - vl ** 3)

    def compute_area(self, face: Face) -> List[float]:
        """
        Compute the area of a cell face.
        """
        # ======================================== 1D faces
        if self.dim == 1:
            v = self.vertices[face.vertex_ids[0]]
            if self.coord_sys == "CARTESIAN":
                return 1.0
            elif self.coord_sys == "CYLINDRICAL":
                return 2.0 * np.pi * v
            elif self.coord_sys == "SPHERICAL":
                return 4.0 * np.pi * v ** 2

    def compute_centroid(self, obj: Union[Cell, Face]) -> float:
        """
        Compute the centroid of a cell.
        """
        centroid = 0.0
        for vid in obj.vertex_ids:
            centroid += self.vertices[vid]
        return centroid / len(obj.vertex_ids)
