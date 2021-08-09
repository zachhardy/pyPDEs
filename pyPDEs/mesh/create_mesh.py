import numpy as np

from numpy import ndarray
from typing import List

from .mesh import Mesh, Cell, Face
from ..utilities import Vector


def create_1d_mesh(zone_edges: List[float], zone_subdivs: List[int],
                   material_ids: List[int] = None,
                   coord_sys: str = "CARTESIAN",
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
    # ======================================== Input checks
    if not material_ids:
        material_ids = [0]
    if coord_sys not in ["CARTESIAN", "CYLINDRICAL", "SPHERICAL"]:
        raise ValueError("Invalid coordinate system type.")
    elif len(zone_subdivs) != len(zone_edges) - 1:
        raise ValueError("Ambiguous combination of zone_bndrys "
                         "and zone_subdivs.")

    mesh = Mesh()
    mesh.dim = 1
    mesh.coord_sys = coord_sys

    # ======================================== Define vertices
    verts = []
    for i in range(len(zone_subdivs)):
        le, re = zone_edges[i], zone_edges[i + 1]
        v = np.linspace(le, re, zone_subdivs[i] + 1)
        v = [Vector(z=vi) for vi in v]
        verts.extend(v if not verts else v[1::])
    mesh.vertices = np.array(verts, dtype=Vector)

    # ======================================== Define cells
    count = 0
    for i in range(len(zone_subdivs)):
        for c in range(zone_subdivs[i]):

            # ============================== Create cell
            cell = Cell()
            cell.cell_type = "SLAB"
            cell.coord_sys = coord_sys
            cell.id = count
            cell.vertex_ids = [count, count + 1]
            cell.material_id = material_ids[i]
            cell.volume = mesh.compute_volume(cell)
            cell.centroid = mesh.compute_centroid(cell)
            cell.width = verts[count + 1] - verts[count]

            # ============================== Create left face
            left_face = Face()
            left_face.vertex_ids = [count]
            left_face.area = mesh.compute_area(left_face)
            left_face.centroid = mesh.compute_centroid(left_face)
            left_face.normal = Vector(z=-1.0)
            cell.faces.append(left_face)

            # ============================== Create right face
            max_ind = sum(zone_subdivs) - 1
            right_face = Face()
            right_face.vertex_ids = [count + 1]
            right_face.area = mesh.compute_area(right_face)
            right_face.centroid = mesh.compute_centroid(right_face)
            right_face.normal = Vector(z=1.0)
            cell.faces.append(right_face)

            # ============================== Add cell to mesh
            mesh.cells.append(cell)
            count += 1

        # ======================================== Connectivity
        mesh.establish_connectivity()

        # ======================================== Verbose printout
        if verbose:
            print("***** Summary of the 1D mesh:\n")
            vertices = [v.z for v in mesh.vertices]
            print(f"Vertices:\n{np.round(vertices, 6)}\n")

            centroids = np.array([c.centroid.z for c in mesh.cells])
            print(f"Centroids:\n{np.round(centroids, 6)}\n")

            widths = [c.width.z for c in mesh.cells]
            print(f"Widths:\n{np.round(widths, 6)}\n")
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
    mesh = Mesh()
    mesh.dim = 2
    mesh.coord_sys = "CARTESIAN"

    # ======================================== Create vertices
    verts = []
    nx, ny = len(x_vertices), len(y_vertices)
    vmap = np.zeros((ny, nx), dtype=int)
    for i in range(ny):
        for j in range(nx):
            vmap[i][j] = len(verts)
            verts.append(Vector(x_vertices[j], y_vertices[i]))
    mesh.vertices = verts

    # ======================================== Create cells
    for i in range(ny - 1):
        for j in range(nx - 1):
            cell = Cell()
            cell.cell_type = "QUAD"
            cell.id = i * (nx - 1) + j

            cell.vertex_ids = [vmap[i][j], vmap[i][j + 1],
                               vmap[i + 1][j + 1], vmap[i + 1][j]]

            # Bottom left and top right vertices
            vbl = mesh.vertices[vmap[i][j]]
            vtr = mesh.vertices[vmap[i + 1][j + 1]]

            # Compute volumetric quantites
            cell.width = vtr - vbl
            cell.centroid = mesh.compute_centroid(cell)
            cell.volume = mesh.compute_volume(cell)

            # ============================== Create faces
            for f in range(4):
                face = Face()
                if f < 3:
                    face.vertex_ids = [cell.vertex_ids[f],
                                       cell.vertex_ids[f + 1]]
                else:
                    face.vertex_ids = [cell.vertex_ids[f],
                                       cell.vertex_ids[0]]

                face.centroid = mesh.compute_centroid(face)
                face.area = mesh.compute_area(face)

                v0 = mesh.vertices[face.vertex_ids[0]]
                v1 = mesh.vertices[face.vertex_ids[1]]
                normal = Vector(z=1.0).cross(v0 - v1)
                face.normal = normal.normalize()

                cell.faces.append(face)

            # ============================== Add cell to mesh
            mesh.cells.append(cell)

    # ======================================== Cell connectivity
    mesh.establish_connectivity()

    # ======================================== Verbose printout
    if verbose:
        print("***** Summary of the 2D mesh:\n")
        vertices = [v for v in mesh.vertices]
        print(f"Vertices:\n{vertices}\n")

        centroids = [c.centroid for c in mesh.cells]
        print(f"Centroids:\n{centroids}\n")

        widths = [c.width for c in mesh.cells]
        print(f"Widths:\n{widths}\n")



    return mesh
