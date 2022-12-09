import time
import numpy as np

from typing import Union

from .mesh import Mesh
from .cell import Cell
from .cell import Face
from .cartesian_vector import CartesianVector


def create_2d_orthomesh(
        x_vertices: Union[list[float], np.ndarray],
        y_vertices: Union[list[float], np.ndarray]
) -> Mesh:
    """
    Create a 2D orthogonal mesh from x and y vertices.

    Parameters
    ----------
    x_vertices : numpy.ndarray
    y_vertices : numpy.ndarray

    Returns
    -------
    Mesh
    """

    # Create the mesh
    mesh = Mesh(2, "CARTESIAN")

    # ========================================
    # Create the vertices
    # ========================================

    vertices = []
    n_x, n_y = len(x_vertices), len(y_vertices)
    vmap = np.zeros((n_y, n_x), dtype=int)
    for i in range(n_y):
        for j in range(n_x):
            vmap[i][j] = len(vertices)
            vertex = CartesianVector(x_vertices[j], y_vertices[i])
            vertices.append(vertex)
    mesh.vertices = vertices

    # ========================================
    # Create the cells
    # ========================================

    ihat = CartesianVector(1.0, 0.0, 0.0)
    jhat = CartesianVector(0.0, 1.0, 0.0)
    khat = CartesianVector(0.0, 0.0, 1.0)

    x_min, x_max = np.min(x_vertices), np.max(x_vertices)
    y_min, y_max = np.min(y_vertices), np.max(y_vertices)

    for i in range(n_y - 1):
        for j in range(n_x - 1):
            cell = Cell("QUADRILATERAL")

            # Define the cell info
            cell.id = i * (n_x - 1) + j

            # Vertex IDs counter-clockwise from the bottom left
            cell.vertex_ids = [vmap[i][j], vmap[i][j + 1],
                               vmap[i + 1][j + 1], vmap[i + 1][j]]

            # Faces counter-clockwise from the bottom
            for f in range(4):
                face = Face()

                # Define vertex IDs
                if f < 3:
                    face.vertex_ids = [cell.vertex_ids[f],
                                       cell.vertex_ids[f + 1]]
                else:
                    face.vertex_ids = [cell.vertex_ids[f],
                                       cell.vertex_ids[0]]

                # Compute the outward normal
                v0 = mesh.vertices[face.vertex_ids[0]]
                v1 = mesh.vertices[face.vertex_ids[1]]
                face.normal = khat.cross(v0 - v1)
                face.normal /= face.normal.norm()

                # Define neighbors
                if face.normal == -jhat:  # bottom face
                    face.neighbor_id = cell.id - (n_x - 1)
                elif face.normal == ihat:  # right face
                    face.neighbor_id = cell.id + 1
                elif face.normal == jhat:  # top face
                    face.neighbor_id = cell.id + (n_x - 1)
                elif face.normal == -ihat:  # left face
                    face.neighbor_id = cell.id - 1
                else:
                    msg = "Unexpected face normal encountered."
                    raise AssertionError(msg)

                # Define boundary IDs
                if v0.y == y_min and v1.y == y_min:  # bottom boundary
                    face.neighbor_id = 0
                elif v0.x == x_max and v1.x == x_max:  # right boundary
                    face.neighbor_id = 1
                elif v0.y == y_max and v1.y == y_max:  # top boundary
                    face.neighbor_id = 2
                elif v0.x == x_min and v1.x == x_min:  # left boundary
                    face.neighbor_id = 3
                else:
                    face.has_neighbor = True

                cell.faces.append(face)
            mesh.cells.append(cell)
    mesh.compute_geometric_info()
    return mesh
