import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union

from matplotlib.figure import Figure

from .cell import Cell, Face
from ..utilities import Vector


class Mesh:
    """
    Base class for meshes.
    """
    def __init__(self):
        self.dim: int = 0
        self.coord_sys: str = "CARTESIAN"

        self.cells: List[Cell] = []
        self.vertices: List[Vector] = []
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
        Get the total number of faces in the mesh. This is
        a bit tricky because there is no defined list of
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
            for cell in self.mesh.cells:
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
        This is a marker for the left boundary.

        Returns
        -------
        float
        """
        return np.min([v.x for v in self.vertices])

    @property
    def x_max(self) -> float:
        """
        Get the maximum x-coordinate over all vertices.
        This is a marker for the right boundary.

        Returns
        -------
        float
        """
        return np.max([v.x for v in self.vertices])

    @property
    def y_min(self) -> float:
        """
        Get the minimum y-coordinate over all vertices.
        This is a marker for the front boundary.

        Returns
        -------
        float
        """
        return np.min([v.y for v in self.vertices])

    @property
    def y_max(self) -> float:
        """
        Get the maximum y-coordinate over all vertices.
        This is a marker for the back boundary.

        Returns
        -------
        float
        """
        return np.max([v.y for v in self.vertices])

    @property
    def z_min(self) -> float:
        """
        Get the minimum z-coordinate over all vertices.
        This is a marker for the bottom boundary.

        Returns
        -------
        float
        """
        return np.min([v.z for v in self.vertices])

    @property
    def z_max(self) -> float:
        """
        Get the maximum z-coordinate over all vertices.
        This is a marker for the top boundary.

        Returns
        -------
        float
        """
        return np.max([v.z for v in self.vertices])

    def compute_volume(self, cell: Cell) -> float:
        """
        Compute the volume of a cell.

        Returns
        -------
        float
        """
        # ======================================== 1D volumes
        if self.dim == 1:
            vl = self.vertices[cell.vertex_ids[0]].z
            vr = self.vertices[cell.vertex_ids[1]].z
            if self.coord_sys == "CARTESIAN":
                return vr - vl
            elif self.coord_sys == "CYLINDRICAL":
                return 2.0 * np.pi * (vr ** 2 - vl ** 2)
            elif self.coord_sys == "SPHERICAL":
                return 4.0 / 3.0 * np.pi * (vr ** 3 - vl ** 3)

        # ======================================== Quad volumes
        elif self.dim == 2 and cell.cell_type == "QUAD":
            vbl = self.vertices[cell.vertex_ids[0]]
            vtr = self.vertices[cell.vertex_ids[2]]
            dr = vtr - vbl
            return dr.x * dr.y

    def compute_area(self, face: Face) -> float:
        """
        Compute the area of a cell face.

        Returns
        -------
        float
        """
        # ======================================== 0D faces
        if self.dim == 1:
            v = self.vertices[face.vertex_ids[0]].z
            if self.coord_sys == "CARTESIAN":
                return 1.0
            elif self.coord_sys == "CYLINDRICAL":
                return 2.0 * np.pi * v
            elif self.coord_sys == "SPHERICAL":
                return 4.0 * np.pi * v ** 2

        # ======================================== 1D faces
        elif self.dim == 2:
            # Get the 2 face vertices
            v0 = self.vertices[face.vertex_ids[0]]
            v1 = self.vertices[face.vertex_ids[1]]
            return (v1 - v0).norm()

    def compute_centroid(self, obj: Union[Cell, Face]) -> Vector:
        """
        Compute the centroid of a cell.

        Returns
        -------
        Vector
        """
        centroid = Vector()
        for vid in obj.vertex_ids:
            centroid += self.vertices[vid]
        return centroid / len(obj.vertex_ids)

    def establish_connectivity(self) -> None:
        """
        Establish the neighbor connectivity of the mesh. This
        routine is very slow and should only be used when the
        connectivity of a mesh cannot be determined a priori.
        """
        # ======================================== Vertex-cell mapping
        vc_map = [set()] * len(self.vertices)
        for cell in self.cells:
            for vid in cell.vertex_ids:
                vc_map[vid].add(cell.id)

        # ======================================== Loop over cells
        cells_to_search = set()
        for cell in self.cells:

            # ==================== Get neighbor cells
            cells_to_search.clear()
            for vid in cell.vertex_ids:
                for cid in vc_map[vid]:
                    if cid != cell.id:
                        cells_to_search.add(cid)

            # =================================== Loop over faces
            for face in cell.faces:
                if face.has_neighbor:
                    continue

                this_vids = set(face.vertex_ids)

                # ============================== Loop over neighbors
                nbr_found = False
                for nbr_cell_id in cells_to_search:
                    nbr_cell: Cell = self.cells[nbr_cell_id]

                    # ========================= Loop over neighbor faces
                    for nbr_face in nbr_cell.faces:
                        nbr_vids = set(nbr_face.vertex_ids)

                        if this_vids == nbr_vids:
                            face.neighbor_id = nbr_cell.id
                            nbr_face.neighbor_id = cell.id

                            face.has_neighbor = True
                            nbr_face.has_neighbor = True

                            nbr_found = True

                        # Break loop over neighbor faces
                        if nbr_found:
                            break

                    # Break loop over neighbor cells
                    if nbr_found:
                        break

            if any([f.has_neighbor for f in cell.faces]):
                self.boundary_cell_ids.append(cell.id)

    def plot_material_ids(self) -> None:
        """
        Plot the material IDs that live on each cell of the mesh.
        This is a utility to ensure that materials are set correctly
        in accordance to the user's wishes.
        """
        fig: Figure = plt.figure()

        matids = [cell.material_id for cell in self.cells]
        if self.dim == 1:
            plt.xlabel("z")

            z = [cell.centroid.z for cell in self.cells]
            plt.plot(z, matids, "ob")
            plt.grid(True)

        elif self.dim == 2:
            plt.xlabel("x")
            plt.ylabel("y")

            x = [cell.centroid.x for cell in self.cells]
            y = [cell.centroid.y for cell in self.cells]
            xx, yy = np.meshgrid(np.unique(x), np.unique(y))
            matids = np.array(matids).reshape(xx.shape)
            plt.pcolormesh(xx, yy, matids, cmap="jet", shading="auto")
            plt.colorbar()

        plt.tight_layout()
