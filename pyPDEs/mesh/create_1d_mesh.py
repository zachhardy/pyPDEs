import numpy as np
from typing import List

from .mesh import Mesh, Cell, Face


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
        verts.extend(v if not verts else v[1::])
    mesh.vertices = verts

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
            left_face.normal = -1.0
            left_face.neighbor_id = count - 1 if count > 0 else -1
            left_face.has_neighbor = True if count > 0 else False
            cell.faces.append(left_face)

            # ============================== Create right face
            max_ind = sum(zone_subdivs) - 1
            right_face = Face()
            right_face.vertex_ids = [count + 1]
            right_face.area = mesh.compute_area(right_face)
            right_face.centroid = mesh.compute_centroid(right_face)
            right_face.normal = 1.0
            right_face.neighbor_id = count + 1 if count < max_ind else -2
            right_face.has_neighbor = True if count < max_ind else False
            cell.faces.append(right_face)

            # ============================== Add cell to mesh
            mesh.cells.append(cell)
            count += 1

        # ================================================== Verbose printout
        if verbose:
            print("***** Summary of the 1D mesh:\n")
            print(f"Vertices:\n{np.round(mesh.vertices, 6)}\n")

            centroids = np.array([c.centroid for c in mesh.cells])
            print(f"Centroids:\n{np.round(centroids, 6)}\n")

            widths = np.array([c.width for c in mesh.cells])
            print(f"Widths:\n{np.round(widths, 6)}\n")
    return mesh
