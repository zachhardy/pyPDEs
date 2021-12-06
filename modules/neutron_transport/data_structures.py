import numpy as np
from numpy import ndarray

from typing import List
from pyPDEs.mesh import *
from pyPDEs.utilities import Vector

__all__ = ['AngleSet', 'HarmonicIndex']


class AngleSet:
    """
    Implementation of a set of angles that share a sweep ordering.
    """

    def __init__(self) -> None:
        self.sweep_ordering: List[int] = []
        self.angles: List[int] = []
        self.executed: bool = False


class HarmonicIndex:
    """
    Structure for spherical harmonic indices.
    """

    def __init__(self, ell: int, m: int) -> None:
        self.ell: int = ell
        self.m: int = m

    def __eq__(self, other: 'HarmonicIndex') -> bool:
        return self.ell == other.ell and self.m == other.m
