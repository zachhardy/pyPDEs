import numpy as np
from numpy import ndarray

from typing import List
from pyPDEs.mesh import *
from pyPDEs.utilities import Vector


class AngleSet:
    """
    Implementation of a set of angles that share a sweep ordering.
    """

    def __init__(self) -> None:
        self.sweep_ordering: List[int] = []
        self.angles: List[int] = []
        self.fluds: FLUDS = None


class FLUDS:
    """
    Implementation of a flux data structure.

    This strucutre holds interface values for angular fluxes
    and mappings to be used for upwinding values.
    """
    def __init__(self) -> None:
        self.psi: ndarray = None

    def upwind_psi(self, cell: Cell, f: int, n: int, g: int) -> float:
        """
        Get the upwind value of psi on a given cell and face
        for a given angle angle and group. It should be noted
        that this routine does not check whether or not a face
        is upwind, but rather assumes this is determined before
        calling.

        Parameters
        ----------
        cell : Cell
        f : int
            The face index on the cell
        n : int
            The direction index.
        g : int
            The group index

        Returns
        -------
        float
            The upwinded value of psi.
        """
        face = cell.faces[f]
        if not face.has_neighbor:
            raise ValueError(
                'The specified cell face is a boundary, '
                'no upwind value is available.')

        adj_cell = face.neighbor_id
        adj_face = face.associated_face
        return self.psi[adj_cell][adj_face][n][g]


class HarmonicIndex:
    """
    Structure for spherical harmonic indices.
    """

    def __init__(self, ell: int, m: int) -> None:
        self.ell: int = ell
        self.m: int = m

    def __eq__(self, other: 'HarmonicIndex') -> bool:
        return self.ell == other.ell and self.m == other.m
