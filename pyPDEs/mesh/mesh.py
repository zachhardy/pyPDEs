import numpy as np
import matplotlib.pyplot as plt

from .cell import Cell
from .cartesian_vector import CartesianVector


class Mesh:
    """
    Implementation of a general computational mesh.
    """

    def __init__(self, dimension: int, coordinate_system: str) -> None:
        """
        Parameters
        ----------
        dimension : int, The spatial dimension of the mesh.
        coordinate_system : str, The coordinate system type.
        """
        if dimension > 3:
            msg = "The mesh dimension cannot exceed three."
            raise AssertionError(msg)

        coordinate_system = coordinate_system.upper()
        coordinate_system_types = ["CARTESIAN", "CYLINDRICAL", "SPHERICAL"]
        if coordinate_system not in coordinate_system_types:
            msg = "Unrecognized coordinate system type."
            raise AssertionError(msg)

        self.dimension: int = dimension
        self.coordinate_system: str = coordinate_system

        self.vertices: list[CartesianVector] = []
        self.cells: list[Cell] = []

    @property
    def n_cells(self) -> int:
        """
        Return the number of cells in the mesh.

        Returns
        -------
        int
        """
        return len(self.cells)

    @property
    def n_vertices(self) -> int:
        """
        Return the number of vertices in the mesh.

        Returns
        -------
        int
        """
        return len(self.vertices)

    def compute_geometric_info(self) -> None:
        """
        Compute the geometric properties of the cells and faces.
        """
        print(f"Computing geometric information on cells and faces.")

        # Loop over cells
        for cell in self.cells:

            # ========================================
            # Cell centroids
            # ========================================

            cell.centroid *= 0.0
            for vid in cell.vertex_ids:
                cell.centroid += self.vertices[vid]
            cell.centroid /= len(cell.vertex_ids)

            # ========================================
            # Cell volumes
            # ========================================

            if self.dimension == 1:
                v1 = self.vertices[cell.vertex_ids[1]].x
                v0 = self.vertices[cell.vertex_ids[0]].x

                if cell.type == "SLAB":
                    cell.volume = v1 - v0
                elif cell.type == "ANNULUS":
                    cell.volume = np.pi * (v1 * v1 - v0 * v0)
                elif cell.type == "SHELL":
                    cell.volume = 4.0 / 3.0 * np.pi * (v1 ** 3 - v0 ** 3)
                else:
                    msg = f"Unrecognized 1D cell type {cell.type}"
                    raise AssertionError(msg)

            elif self.dimension == 2:

                if cell.type == "QUADRILATERAL":
                    vbl = self.vertices[cell.vertex_ids[0]]
                    vtr = self.vertices[cell.vertex_ids[2]]
                    dr = vtr - vbl
                    cell.volume = dr.x * dr.y
                else:
                    msg = f"Unrecognized 2D cell type {cell.type}"
                    raise AssertionError(msg)

            else:
                msg = f"3D cells not implemented."
                raise AssertionError(msg)

            # Loop over faces
            for face in cell.faces:

                # ========================================
                # Face centroids
                # ========================================

                face.centroid *= 0.0
                for vid in face.vertex_ids:
                    face.centroid += self.vertices[vid]
                face.centroid /= len(face.vertex_ids)

                # ========================================
                # Face areas
                # ========================================

                if self.dimension == 1:
                    v = self.vertices[face.vertex_ids[0]].x

                    if cell.type == "SLAB":
                        face.area = 1.0
                    elif cell.type == "ANNULUS":
                        face.area = 2.0 * np.pi * v
                    elif cell.type == "SHELL":
                        face.area = 4.0 * np.pi * v**2
                    else:
                        msg = f"Unrecognized 1D cell type {cell.type}"
                        raise AssertionError(msg)

                elif self.dimension == 2:

                    if cell.type == "QUADRILATERAL":
                        v1 = self.vertices[face.vertex_ids[1]]
                        v0 = self.vertices[face.vertex_ids[0]]
                        face.area = (v1 - v0).norm()
                    else:
                        msg = f"Unrecognized 2D cell type {cell.type}"
                        raise AssertionError(msg)

        print("Done computing geometric information on cells and faces.")

    def establish_connectivity(self) -> None:
        """
        Establish the cell-face connectivity throughout the mesh.

        Notes
        -----
        This routine is quite expensive and should only be used when there
        is no predefined rules to determine connectivity, such as with
        unstructured meshes.
        """
        # Vertex-cell mapping
        vc_map = [set()] * len(self.vertices)
        for cell in self.cells:
            for vid in cell.vertex_ids:
                vc_map[vid].add(cell.id)

        # Loop over cells
        cells_to_search = set()
        for cell in self.cells:

            # Get neighbor cells
            cells_to_search.clear()
            for vid in cell.vertex_ids:
                for cid in vc_map[vid]:
                    if cid != cell.id:
                        cells_to_search.add(cid)

            # Loop over faces
            for face in cell.faces:
                if face.has_neighbor:
                    continue

                this_vids = set(face.vertex_ids)

                # Loop over neighbors
                nbr_found = False
                for nbr_cell_id in cells_to_search:
                    nbr_cell: Cell = self.cells[nbr_cell_id]

                    # Loop over neighbor faces
                    for nbr_face in nbr_cell.faces:
                        nbr_vids = set(nbr_face.vertex_ids)

                        # If the faces are
                        if this_vids == nbr_vids:
                            face.neighbor_id = nbr_cell.id
                            nbr_face.neighbor_id = cell.id

                            face.has_neighbor = True
                            nbr_face.has_neighbor = True

                            nbr_found = True

                        if nbr_found:
                            break
                    if nbr_found:
                        break

    def centroids_as_ndarray(self) -> np.ndarray:
        """
        Return the centroids of the cells as a numpy ndarray.

        Returns
        -------
        numpy.ndarray
        """
        centroids = []
        for cell in self.cells:
            centroid = cell.centroid
            centroids.append([centroid.x, centroid.y, centroid.z])
        return np.array(centroids)

    def material_ids_as_ndarray(self) -> np.ndarray:
        """
        Return the material IDs of the cells as a numpy ndarray.

        Returns
        -------
        numpy.ndarray
        """
        material_ids = []
        for cell in self.cells:
            material_ids.append(cell.material_id)
        return np.array(material_ids)

    def plot_material_ids(self) -> None:
        """
        Plot the material IDS.
        """
        matids = self.material_ids_as_ndarray()
        xlabel = "z" if self.dimension == 1 else "X"
        ylabel = "" if self.dimension == 2 else "Y"

        plt.figure()
        if self.dimension == 1:
            z = [cell.centroid.z for cell in self.cells]

            plt.plot(z, matids, "ob")

        elif self.dimension == 2:
            x = [cell.centroid.x for cell in self.cells]
            y = [cell.centroid.y for cell in self.cells]
            X, Y = np.meshgrid(np.unique(x), np.unique(y))

            matids = np.reshape(matids).reshape(X.shape)
            plt.pcolormesh(X, Y, matids, cmap="jet", shading="auto")
            plt.colorbar()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
