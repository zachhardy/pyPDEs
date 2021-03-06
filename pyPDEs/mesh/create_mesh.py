import time
import numpy as np

from numpy import ndarray
from typing import List

from .mesh import Mesh, Cell, Face
from ..utilities import Vector

__all__ = ['create_1d_mesh', 'create_2d_mesh']


def create_1d_mesh(zone_edges: List[float], zone_subdivs: List[int],
                   material_ids: List[int] = None,
                   coord_sys: str = 'cartesian',
                   verbose: bool = False) -> Mesh:
    """
    Create a 1D non-uniform mesh.

    Parameters
    ----------
    zone_edges : List[float]
        The edges of each zone of the mesh.
    zone_subdivs : List[int]
        The number of subdivisions in each zone.
    material_ids : List[int]
        The material IDs for each zone.
    coord_sys : str, default "CARTESIAN"
        The coordinate system of the mesh.

    Returns
    -------
    Mesh object
    """
    # Input checks
    if not material_ids:
        material_ids = [0]
    if coord_sys not in ['cartesian', 'cylindrical', 'spherical']:
        raise ValueError('Invalid coordinate system type.')
    elif len(zone_subdivs) != len(zone_edges) - 1:
        raise ValueError('Ambiguous combination of zone_bndrys '
                         'and zone_subdivs.')
    t_start = time.time()
    mesh = Mesh()
    mesh.dim = 1
    mesh.type = 'line'
    mesh.coord_sys = coord_sys

    # Define vertices
    verts = []
    for i in range(len(zone_subdivs)):
        le, re = zone_edges[i], zone_edges[i + 1]
        v = np.linspace(le, re, zone_subdivs[i] + 1)
        v = [Vector(z=vi) for vi in v]
        verts.extend(v if not verts else v[1::])
    mesh.vertices = np.array(verts, dtype=Vector)

    # Cells per dimension
    n_cells = sum(zone_subdivs)
    mesh.n_cells_ijk = (n_cells,)

    # Shorthand
    khat = Vector(z=1.0)

    # Define cells
    count = 0
    for i in range(len(zone_subdivs)):
        for c in range(zone_subdivs[i]):
            mesh.cell_to_ijk_mapping.append([count])

            # Create cell
            cell = Cell()
            cell.id = count
            mesh.cell_id_to_ijk_map.append([count])
            cell.cell_type = 'slab'
            cell.coord_sys = coord_sys
            cell.material_id = material_ids[i]

            # Vertices numbered left-to-right
            cell.vertex_ids = [count, count + 1]

            # Cell geometric quantities
            cell.volume = mesh.compute_volume(cell)
            cell.centroid = mesh.compute_centroid(cell)
            cell.width = verts[count + 1] - verts[count]

            # Create faces
            for f in range(2):
                face = Face()

                # Left face
                if f == 0:
                    face.vertex_ids = [count]
                    face.normal = -khat
                    face.has_neighbor = True if count > 0 else False
                    face.neighbor_id = count - 1 if count > 0 else 0

                # Right face
                else:
                    face.vertex_ids = [count + 1]
                    face.normal = khat
                    face.has_neighbor = True if count < n_cells - 1 else False
                    face.neighbor_id = count + 1 if count < n_cells - 1 else 1

                # Face geometric quantities
                face.area = mesh.compute_area(face)
                face.centroid = mesh.compute_centroid(face)

                # Add face to cell
                cell.faces.append(face)

            # Define face vertex mapping
            cell.face_vertex_mapping = [[0], [1]]

            # Cell on boundary?
            if count == 0 or count == n_cells - 1:
                mesh.boundary_cell_ids.append(count)

            # Add cell to mesh
            mesh.cells.append(cell)
            count += 1

    # Define associated faces and vertices
    for cell in mesh.cells:
        for face in cell.faces:
            if face.has_neighbor:
                face.associated_face = mesh.get_associated_face(face)
                face.associated_vertices = mesh.get_associated_vertices(face)

    # Verbose printout
    t_elapsed = time.time() - t_start
    if verbose:
        print('\n***** Summary of the 1D mesh:\n')
        print(f'Number of Cells:\t{mesh.n_cells}')
        print(f'Number of Faces:\t{mesh.n_faces}')
        print(f'Number of Vertices:\t{mesh.n_vertices}')
        print(f'Mesh Creation Time:\t{t_elapsed:.4g} sec')
    return mesh


def create_2d_mesh(x_vertices: ndarray, y_vertices: ndarray,
                   verbose: bool = False) -> Mesh:
    """
    Create a 2D mesh from x and y vertex locations.

    Parameters
    ----------
    x_vertices : ndarray
    y_vertices : ndarray

    Returns
    -------
    Mesh
    """
    t_start = time.time()
    mesh = Mesh()
    mesh.dim = 2
    mesh.type = 'orhto_quad'
    mesh.coord_sys = 'cartesian'

    # Create vertices
    verts = []
    nx, ny = len(x_vertices), len(y_vertices)
    vmap = np.zeros((ny, nx), dtype=int)
    for i in range(ny):
        for j in range(nx):
            vmap[i][j] = len(verts)
            verts.append(Vector(x_vertices[j], y_vertices[i]))
    mesh.vertices = verts

    # Cells per dimension
    mesh.n_cells_ijk = (nx - 1, ny - 1)

    # Reference normals
    ihat = Vector(x=1.0)
    jhat = Vector(y=1.0)
    khat = Vector(z=1.0)

    # Create cells
    for i in range(ny - 1):
        for j in range(nx - 1):
            mesh.cell_to_ijk_mapping.append([j, i])

            # Initialize cell
            cell = Cell()
            cell.cell_type = 'quad'
            cell.id = i * (nx - 1) + j
            mesh.cell_id_to_ijk_map.append([j, i])

            # Vertices start at the bottom left and
            # are numbered counter-clockwise
            cell.vertex_ids = [vmap[i][j], vmap[i][j + 1],
                               vmap[i + 1][j + 1], vmap[i + 1][j]]

            # Bottom left and top right vertices
            vbl = mesh.vertices[vmap[i][j]]
            vtr = mesh.vertices[vmap[i + 1][j + 1]]

            # Cell geometric quantites
            cell.width = vtr - vbl
            cell.centroid = mesh.compute_centroid(cell)
            cell.volume = mesh.compute_volume(cell)

            # Create faces
            for f in range(4):
                face = Face()

                # Left, right, back, front face
                if f == 0:
                    face.vertex_ids = [cell.vertex_ids[3],
                                       cell.vertex_ids[0]]
                elif f == 1:
                    face.vertex_ids = [cell.vertex_ids[1],
                                       cell.vertex_ids[2]]
                elif f == 2:
                    face.vertex_ids = [cell.vertex_ids[0],
                                       cell.vertex_ids[1]]
                else:
                    face.vertex_ids = [cell.vertex_ids[2],
                                       cell.vertex_ids[3]]

                v0 = mesh.vertices[face.vertex_ids[0]]
                v1 = mesh.vertices[face.vertex_ids[1]]

                # Face geometric quantities
                face.centroid = mesh.compute_centroid(face)
                face.area = mesh.compute_area(face)
                normal = khat.cross(v0 - v1)
                face.normal = normal.normalize()

                # Define neighbors
                if face.normal == ihat:
                    face.neighbor_id = cell.id + 1
                elif face.normal == -ihat:
                    face.neighbor_id = cell.id - 1
                elif face.normal == jhat:
                    face.neighbor_id = cell.id + nx - 1
                elif face.normal == -jhat:
                    face.neighbor_id = cell.id - nx + 1

                # Left, right, back, front boundaries
                if v0.x == v1.x == mesh.x_min:
                    face.neighbor_id = 0
                elif v0.x == v1.x == mesh.x_max:
                    face.neighbor_id = 1
                elif v0.y == v1.y == mesh.y_min:
                    face.neighbor_id = 2
                elif v0.y == v1.y == mesh.y_max:
                    face.neighbor_id = 3
                else:
                    face.has_neighbor = True

                cell.faces.append(face)

            # Cell on boundary?
            if any([not face.has_neighbor for face in cell.faces]):
                mesh.boundary_cell_ids.append(cell.id)

            # Add cell to mesh
            mesh.cells.append(cell)

    # Define associated faces and vertices
    for cell in mesh.cells:
        for face in cell.faces:
            if face.has_neighbor:
                face.associated_face = mesh.get_associated_face(face)
                face.associated_vertices = mesh.get_associated_vertices(face)

    # Verbose printout
    t_elapsed = time.time() - t_start
    if verbose:
        print()
        print('***** Summary of the 2D mesh *****')
        print(f'Number of Cells:\t{mesh.n_cells}')
        print(f'Number of Faces:\t{mesh.n_faces}')
        print(f'Number of Vertices:\t{mesh.n_vertices}')
        print(f'Mesh Creation Time:\t{t_elapsed:.4g} sec')
        print()
    return mesh
