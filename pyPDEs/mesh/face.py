from typing import List

from ..utilities import Vector

__all__ = ['Face']


class Face:
    """
    Generalized face.
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
        """
        Get the number of vertices on this face.

        Returns
        -------
        int
        """
        return len(self.vertex_ids)

    def __eq__(self, other: "Face") -> bool:
        return set(self.vertex_ids) == set(other.vertex_ids)
