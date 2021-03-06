import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import List, Tuple

from ..cell import Cell
from ..face import Face
from ...utilities import Vector


class Mesh:
    """
    Generic mesh.

    Meshes in pyPDEs are defined by a collection of vertices, faces,
    and cells. A cell is defined by a collection of faces, and a face
    is defined as a collection of vertices. There are two copies of
    interior faces, one from the perspective of each cell the face
    belongs to.
    """

    from ._plotting import plot_material_ids
    from ._connectivity import establish_connectivity
    from ._geometry import compute_volume, compute_area, compute_centroid

    def __init__(self) -> None:
        self.dim: int = 0
        self.type: str = None
        self.coord_sys: str = 'cartesian'
        self.is_orthogonal: bool = True

        self.cell_to_ijk_mapping: List[List[int]] = []
        self.n_cells_ijk: tuple = []

        self.vertices: List[Vector] = []
        self.cell_id_to_ijk_map: List[List[int]] = []

        self.cells: List[Cell] = []
        self.boundary_cell_ids: List[int] = []

        self._n_faces: int = 0

    @property
    def n_cells(self) -> int:
        """
        Get the total number of cells in the mesh.

        Returns
        -------
        int
        """
        return len(self.cells)

    @property
    def n_faces(self) -> int:
        """
        Get the total number of faces in the mesh.

        Notes
        -----
        This is a bit tricky because there is no defined list of
        unique faces. To compute the total number of faces,
        we look over cells and faces, counting each interior
        face as half and each boundary face as one. This way,
        when all perspectives of an interior face are encountered,
        the face accounts for one unique face.

        Returns
        -------
        int
        """
        if self._n_faces == 0:
            for cell in self.cells:
                for face in cell.faces:
                    if face.has_neighbor:
                        self._n_faces += 0.5
                    else:
                        self._n_faces += 1.0
            self._n_faces = int(self._n_faces)
        return self._n_faces

    @property
    def n_vertices(self) -> int:
        """
        Get the total number of vertices on the mesh.

        Returns
        -------
        int
        """
        return len(self.vertices)

    @property
    def x_min(self) -> float:
        """
        Get the minimum x-coordinate over all vertices.

        Returns
        -------
        float
        """
        return np.min([v.x for v in self.vertices])

    @property
    def x_max(self) -> float:
        """
        Get the maximum x-coordinate over all vertices.

        Returns
        -------
        float
        """
        return np.max([v.x for v in self.vertices])

    @property
    def y_min(self) -> float:
        """
        Get the minimum y-coordinate over all vertices.

        Returns
        -------
        float
        """
        return np.min([v.y for v in self.vertices])

    @property
    def y_max(self) -> float:
        """
        Get the maximum y-coordinate over all vertices.

        Returns
        -------
        float
        """
        return np.max([v.y for v in self.vertices])

    @property
    def z_min(self) -> float:
        """
        Get the minimum z-coordinate over all vertices.

        Returns
        -------
        float
        """
        return np.min([v.z for v in self.vertices])

    @property
    def z_max(self) -> float:
        """
        Get the maximum z-coordinate over all vertices.

        Returns
        -------
        float
        """
        return np.max([v.z for v in self.vertices])

    def get_associated_face(self, face: Face) -> int:
        """
        Get the associated face index on the adjacent cell face.

        Parameters
        ----------
        face : Face

        Returns
        -------
        int
        """
        if not face.has_neighbor:
            raise ValueError(
                'The provided cell face cannot be a boundary.')

        associated_face = -1
        adj_cell: Cell = self.cells[face.neighbor_id]
        cfvids = set(face.vertex_ids)
        for af, adj_face in enumerate(adj_cell.faces):
            adj_face: Face = adj_face
            afvids = set(adj_face.vertex_ids)
            if cfvids == afvids:
                associated_face = af
                break

        if associated_face < 0:
            raise AssertionError('Associated face not found.')
        return associated_face

    def get_associated_vertices(self, face: Face) -> List[int]:
        """
        Get the associated vertex indices on the adjacent cell face.

        Parameters
        ----------
        face : Face

        Returns
        -------
        List[int]
        """
        if not face.has_neighbor:
            raise ValueError(
                'The provided cell face cannot be a boundary.')

        associated_vertices = []
        adj_cell: Cell = self.cells[face.neighbor_id]
        associated_face = self.get_associated_face(face)
        adj_face: Face = adj_cell.faces[associated_face]

        for i, cfvid in enumerate(face.vertex_ids):
            for j, afvid in enumerate(adj_face.vertex_ids):
                if cfvid == afvid:
                    associated_vertices.append(j)
                    break
            if len(associated_vertices) < i + 1:
                raise AssertionError('Associated vertex not found.')
        return associated_vertices







    def map_ijk_to_cell_id(self, i: int, j: int = 0, k: int = 0) -> int:
        """
        Map an `ijk` index to a cell ID on the mesh.

        Parameters
        ----------
        i : int
        j : int, default 0
        k : int, default 0

        Returns
        -------
        int
            The global ID of the cell.
        """
        if self.dim == 1:
            return self.cells[i]
        elif self.dim == 2:
            nx = self._n_cells_ijk[0]
            return self.cells[j*nx + i]
        elif self.dim == 3:
            nx = self.n_cells_ijk[0]
            ny = self.n_cells_ijk[1]
            return self.cells[k*nx*ny + j*nx + i]
