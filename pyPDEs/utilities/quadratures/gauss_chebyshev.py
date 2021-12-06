import numpy as np
from numpy import ndarray
from numpy.polynomial.chebyshev import chebgauss

from .quadrature import Quadrature
from .. import Vector


class GaussChebyshev(Quadrature):
    """
    Gauss-Chebyshev 1D quadrature.

    Parameters
    ----------
    n_qpoints : int, default 2
        The number of quadrature points. This will integrate
        polynomials of 2N-1 quadrature points exactly.
    """

    def __init__(self, n_qpoints: int = 2) -> None:
        super().__init__()

        # Get the quadrature points and weights
        pts, wts = chebgauss(n_qpoints)

        self.qpoints = [Vector(z=pt) for pt in pts]
        self.weights = wts
        self._domain = (-1.0, 1.0)