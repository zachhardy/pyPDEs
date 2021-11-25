import numpy as np
from numpy import ndarray
from numpy.polynomial.legendre import leggauss

from .quadrature import Quadrature
from .. import Vector


class GaussLegendre(Quadrature):
    """
    Gauss-Legendre 1D quadrature.

    Parameters
    ----------
    n_qpoints : int, default 2
        The number of quadrature points. This will integrate
        polynomials of 2N-1 quadrature points exactly.
    """

    def __init__(self, n_qpoints: int = 2) -> None:
        super().__init__()

        # Get the quadrature points and weights
        pts, wts = leggauss(n_qpoints)

        self.qpoints = [Vector(z=pt) for pt in pts]
        self.weights = wts
        self._domain = (-1.0, 1.0)
