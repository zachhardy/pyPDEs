from typing import List

from ..utilities import Vector


class Face:
    """Generalized face.

    Attributes
    ----------
    vertex_ids : List[int]
        The IDs of the vertices that define this face.
    normal : Vector
        The outward pointing normal vector.
    area : float
        The area of the face.
    centroid : Vector
        The centroid of the face.
    has_neighbor : bool
        Flag for whether or not this face borders another
        cell or a boundary.
    neighbor_id : int
        The ID of the neighboring cell if `has_neighbor` is
        true, otherwise, this is mapped to a boundary ID via
        ..math:: boundary_id = -(neighbor_id + 1)
    """
    def __init__(self):
        self.vertex_ids: List[int] = []

        self.normal: Vector = Vector()
        self.area: float = 0.0
        self.centroid: Vector = Vector()

        self.has_neighbor: bool = False
        self.neighbor_id = 0

    @property
    def n_vertices(self) -> int:
        """Get the number of vertices on this face.

        Returns
        -------
        int
        """
        return len(self.vertex_ids)

    def __eq__(self, other: "Face") -> bool:
        return set(self.vertex_ids) == set(other.vertex_ids)


__all__ = ["Face"]
