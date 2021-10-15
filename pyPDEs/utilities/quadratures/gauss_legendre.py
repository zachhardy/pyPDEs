import numpy as np
from numpy import ndarray
from numpy.polynomial.legendre import leggauss

from .quadrature import Quadrature
from .. import Vector


class GaussLegendre(Quadrature):
    """ Gauss-Legendre 1D quadrature.
    """

    def __init__(self, n_qpoints: int = 2) -> None:
        """GaussLegendre constructor.

        Parameters
        ----------
        n_qpoints : int, default 2
            The number of quadrature points and weights to
            generate. A quadrature set with `n_qpoints` quadrature
            points can integrate polynomials of up to degree
            2*`n_qpoints`-1 exactly.
        """
        super().__init__()

        # Get the quadrature points and weights
        pts, wts = leggauss(n_qpoints)

        self.qpoints = [Vector(z=pt) for pt in pts]
        self.weights = list(wts)
        self._domain = (-1.0, 1.0)
