import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import List, Union

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

        self.vertices: List[Vector] = []

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

    def define_associated_face(self, face: Face) -> None:
        """
        Get the neighboring cell's face that coincides with the
        specified face.

        Parameters
        ----------
        face : Face
        """
        if not face.has_neighbor:
            raise AssertionError('The specified face is on a boundary.')

        # Current face vertex IDs
        cfvids = set(face.vertex_ids)

        # Loop over adjacent cell faces
        associated_face = -1
        adj_cell: Cell = self.cells[face.neighbor_id]
        for af, adj_face in enumerate(adj_cell.faces):
            adj_face: Face = adj_face

            # Adjacent face vertex IDs
            afvids = set(adj_face.vertex_ids)

            # If this is the associated face
            if afvids == cfvids:
                associated_face = af
                break

        if associated_face < 0:
            raise AssertionError(
                'No associated face found on neighbor. Check that '
                'the mesh was constructed correctly.')

        face.associated_face = associated_face

    def define_associated_vertices(self, face: Face) -> None:
        """
        Get the neighboring cell's vertex IDs that coincides with the
        specified face's vertex IDs.

        Parameters
        ----------
        face : Face
        """
        if not face.has_neighbor:
            raise AssertionError('The specified face is on a boundary.')

        # Clear associated vertices
        face.associated_vertices.clear()

        # Get adjacent cell and face
        adj_cell: Cell = self.cells[face.neighbor_id]
        adj_face: Face = adj_cell.faces[face.associated_face]

        # Loop over current face vertices
        for cfvid in face.vertex_ids:

            # Loop over adjacent face vertices
            found = False
            for n, afvid in enumerate(adj_face.vertex_ids):
                if cfvid == afvid:
                    face.associated_vertices.append(n)
                    found = True
                    break

            # There must be a matching vertex on associated faces
            if not found:
                raise AssertionError(
                    'Could not find a matching vertex on the '
                    'associated face.')



