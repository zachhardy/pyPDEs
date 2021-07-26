import numpy as np
from typing import List

from .mesh import Mesh
from .cell import Cell, Face


class LineMesh(Mesh):
    """
    A 1D, non-uniform mesh.
    """

    def __init__(self,
                 zone_edges: List[float],
                 zone_subdivs: List[int],
                 material_zones: List[int] = None,
                 coord_sys: str = 'slab') -> None:
        """
        Parameters
        ----------
        zone_edges : List[float]
            The edges of each zone on the mesh.
        zone_subdivs : List[int]
            The number of subdivisions, or cells, in each zone.
        material_zones : List[int], default [0]
            The material id that belongs to each zone.
        coord_sys : str, default 'slab'
            The coordinate system of the mesh. Options are
            'slab', 'cylinder', and 'sphere'.
        """
        super().__init__()

        # ================================================== Input checks
        if material_zones is None:
            material_zones = [0]
        if coord_sys not in ['slab', 'cylinder', 'sphere']:
            raise ValueError('Invalid coordinate system type specified.')
        if len(zone_subdivs) != len(zone_edges) - 1:
            raise ValueError('Incompatible zone subdivisions and edges.')
        if len(material_zones) != len(zone_subdivs):
            raise ValueError(
                'Incompatible material zones and zone subdivisions ')

        self.dim = 1
        self.coord_sys = coord_sys

        # ================================================== Define vertices
        self.vertices.clear()
        for i in range(len(zone_subdivs)):
            l_edge, r_edge = zone_edges[i], zone_edges[i + 1]
            verts = np.linspace(l_edge, r_edge, zone_subdivs[i] + 1)
            verts = verts if not self.vertices else verts[1::]
            for v in verts:
                self.vertices.append(v)

        # ================================================== Define cells
        cell_count = 0
        for i in range(len(zone_subdivs)):
            for c in range(zone_subdivs[i]):
                # ======================================== Create the cell
                cell = Cell()
                self.cells.append(cell)

                vids = [cell_count, cell_count + 1]
                cell.local_id = cell_count
                cell.material_id = material_zones[i]
                cell.vertex_ids = vids
                cell.volume = self.compute_cell_volume(cell)
                cell.centroid = self.compute_cell_centroid(cell)
                cell.width = self.vertices[vids[1]] - self.vertices[vids[0]]

                # ======================================== Create left face
                l_face = Face()
                cell.faces.append(l_face)

                l_face.vertex_ids = [cell_count]
                l_face.area = self.compute_face_area(l_face)
                l_face.centroid = self.compute_face_centroid(l_face)
                l_face.normal = -1.0
                if cell_count == 0:
                    self.boundary_cell_ids.append(0)
                    l_face.neighbor_id = -1
                else:
                    l_face.neighbor_id = cell_count - 1

                # ======================================== Create right face
                r_face = Face()
                cell.faces.append(r_face)

                r_face.vertex_ids = [cell_count + 1]
                r_face.area = self.compute_face_area(r_face)
                r_face.centroid = self.compute_face_centroid(r_face)
                r_face.normal = 1.0
                if cell_count == sum(zone_subdivs) - 1:
                    self.boundary_cell_ids.append(sum(zone_subdivs) - 1)
                    r_face.neighbor_id = -2
                else:
                    r_face.neighbor_id = cell_count + 1

                # Increment the counter
                cell_count += 1
