import numpy as np
from numpy import ndarray
from numpy.polynomial.legendre import leggauss

from .quadrature import Quadrature


class GaussLegendre(Quadrature):
    """
    Gauss-Legendre 1D quadrature.
    """
    def __init__(self, order: int = 2) -> None:
        super().__init__(order)

        # Get the quadrature points and weights
        n_pts = int(np.ceil((order + 1.0) / 2.0))
        pts, wts = leggauss(n_pts)

        self.qpoints = pts
        self.weights = wts
        self.domain = [-1.0, 1.0]
        self.width = 2.0
