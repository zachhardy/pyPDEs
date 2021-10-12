import numpy as np
from numpy import ndarray
from numpy.polynomial.legendre import leggauss

from .quadrature import Quadrature
from .. import Vector


class GaussLegendre(Quadrature):
    """ Gauss-Legendre 1D quadrature.
    """

    def __init__(self, order: int = 2) -> None:
        """GaussLegendre constructor.

        Parameters
        ----------
        order : int, default 2
            The maximum monomial order the quadrature set
            can integrate exactly. For GaussLegendre, the
            number of points this will yield is
            `n_qpoints = ceil(0.5 * (order + 1)`.
        """
        super().__init__(order)

        # Get the quadrature points and weights
        n_pts = int(np.ceil((order + 1.0) / 2.0))
        pts, wts = leggauss(n_pts)

        self.qpoints = [Vector(z=pt) for pt in pts]
        self.weights = wts
        self._domain = (-1.0, 1.0)
