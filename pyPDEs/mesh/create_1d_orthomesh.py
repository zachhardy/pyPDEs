import numpy as np

from .mesh import Mesh
from .cell import Cell
from .cell import Face
from .cartesian_vector import CartesianVector


def create_1d_orthomesh(
        zone_edges: list[float],
        zone_subdivisions: list[int],
        material_ids: list[int],
        coordinate_system: str = "CARTESIAN"
) -> Mesh:
    """
    Create a zoned 1D orthogonal mesh.

    This is defined in terms of zones which are defined by two
    edges, a number of subdivisions, or cells, and a material ID.

    Parameters
    ----------
    zone_edges : list[float]
        The edges of the zones on the mesh. This list should be
        monotonically increasing. This should have one more entry
        than the number of zones.
    zone_subdivisions : list[int]
        The number of cells in each zone.
    material_ids : list[int]
        The material ID for a particular zone.
    coordinate_system : str ["CARTESIAN", "CYLINDRICAL", "SPHERICAL"]

    Returns
    -------
    Mesh
    """
    if len(zone_edges) == 0:
        raise ValueError("No zone edges specified.")
    if len(zone_edges) != len(zone_subdivisions) + 1:
        raise ValueError("Incompatible number of zone edges and subdivisions.")
    if len(zone_subdivisions) != len(material_ids):
        raise ValueError("Incompatible material IDs and zone subdivisions.")
    if coordinate_system not in ["CARTESIAN", "CYLINDRICAL", "SPHERICAL"]:
        raise ValueError("Unrecognized coordinate system.")

    # Create the mesh
    mesh = Mesh(1, coordinate_system)

    # Count the number of zones and cells
    n_zones = len(zone_subdivisions)
    n_cells = sum(zone_subdivisions)

    # ========================================
    # Create the vertices
    # ========================================

    current_pos = zone_edges[0]
    vertices = [current_pos]
    for z in range(n_zones):
        zone_width = zone_edges[z + 1] - zone_edges[z]
        n_zone_cells = zone_subdivisions[z]
        dz = zone_width / n_zone_cells

        for c in range(n_zone_cells):
            vertices.append(current_pos + dz)
            current_pos += dz

    mesh.vertices.clear()
    for vertex in vertices:
        mesh.vertices.append(CartesianVector(vertex))

    # ========================================
    # Create the cells
    # ========================================

    # Define cell types
    if coordinate_system == "CARTESIAN":
        cell_type = "SLAB"
    elif coordinate_system == "CYLINDRICAL":
        cell_type = "ANNULUS"
    else:
        cell_type = "SHELL"

    # Construct the cells
    n = 0
    for z in range(n_zones):
        for c in range(zone_subdivisions[z]):
            cell = Cell(cell_type)
            left_face, right_face = Face(), Face()

            # Define the cell info
            cell.id = n
            cell.vertex_ids = [n, n + 1]
            cell.material_id = material_ids[z]

            # Define the left face info
            left_face.vertex_ids = [n]
            left_face.has_neighbor = n > 0
            left_face.neighbor_id = n - 1 if n > 0 else 0
            left_face.normal = CartesianVector(-1.0)
            cell.faces.append(left_face)

            # Define the right face info
            right_face.vertex_ids = [n + 1]
            right_face.has_neighbor = n < n_cells - 1
            right_face.neighbor_id = n + 1 if n < n_cells - 1 else 1
            right_face.normal = CartesianVector(1.0)
            cell.faces.append(right_face)

            mesh.cells.append(cell)
            n += 1
    mesh.compute_geometric_info()
    return mesh
