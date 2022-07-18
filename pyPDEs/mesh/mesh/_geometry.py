"""Methods for computing geometric quantities."""

import numpy as np

from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Mesh

from ..cell import Cell
from ..face import Face
from ...utilities import Vector


def compute_volume(self: 'Mesh', cell: Cell) -> float:
    """
    Compute the volume of a cell.

    Parameters
    ----------
    cell : Cell

    Returns
    -------
    float
    """
    # ======================================== 1D volumes
    if self.dim == 1:
        vl = self.vertices[cell.vertex_ids[0]].z
        vr = self.vertices[cell.vertex_ids[1]].z
        if self.coord_sys == 'cartesian':
            return vr - vl
        elif self.coord_sys == 'cylindrical':
            return np.pi * (vr ** 2 - vl ** 2)
        elif self.coord_sys == 'spherical':
            return 4.0 / 3.0 * np.pi * (vr ** 3 - vl ** 3)

    # ======================================== Quad volumes
    elif self.dim == 2 and cell.cell_type == 'quad':
        vbl = self.vertices[cell.vertex_ids[0]]
        vtr = self.vertices[cell.vertex_ids[2]]
        dr = vtr - vbl
        return dr.x * dr.y

def compute_area(self: 'Mesh', face: Face) -> float:
    """
    Compute the area of a cell face.

    Parameters
    ----------
    face : Face

    Returns
    -------
    float
    """
    # ======================================== 0D faces
    if self.dim == 1:
        v = self.vertices[face.vertex_ids[0]].z
        if self.coord_sys == 'cartesian':
            return 1.0
        elif self.coord_sys == 'cylindrical':
            return 2.0 * np.pi * v
        elif self.coord_sys == 'spherical':
            return 4.0 * np.pi * v ** 2

    # 1D faces
    elif self.dim == 2:
        # Get the 2 face vertices
        v0 = self.vertices[face.vertex_ids[0]]
        v1 = self.vertices[face.vertex_ids[1]]
        return (v1 - v0).norm()


def compute_centroid(self: 'Mesh', obj: Union[Cell, Face]) -> Vector:
    """
    Compute the centroid of a cell, or a face.

    Parameters
    ----------
    obj : Cell or Face

    Returns
    -------
    Vector
    """
    centroid = Vector()
    for vid in obj.vertex_ids:
        centroid += self.vertices[vid]
    return centroid / len(obj.vertex_ids)
