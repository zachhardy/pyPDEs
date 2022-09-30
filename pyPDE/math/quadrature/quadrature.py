from numpy.polynomial.legendre import leggauss
from numpy.polynomial.chebyshev import chebgauss

from ...mesh import CartesianVector


class Quadrature:
    """
    Base class for quadrature sets.
    """

    def __init__(self) -> None:
        self.weights: list[float] = []
        self.quadrature_points: list[CartesianVector] = []

        # A 1D quadrature domain
        self._domain: tuple[float, float] = (-1.0, 1.0)

    @property
    def n_quadrature_points(self) -> int:
        """
        Return the number of quadrature points in the set.

        Returns
        -------
        float
        """
        return len(self.quadrature_points)

    def get_domain(self) -> tuple[float, float]:
        """
        Return the 1D quadrature domain.

        Returns
        -------
        tuple[float, float]
        """
        return self._domain

    def set_domain(self, domain: tuple[float, float]) -> None:
        """
        Set the 1D quadrature domain.

        Parameters
        ----------
        domain : tuple[float, float]
        """
        domain_old = self.get_domain()
        h_new = domain[1] - domain[0]
        h_old = domain_old[1] - domain_old[0]

        if h_new <= 0.0:
            msg = f"Invalid quadrature domain."
            raise ValueError(msg)
        if len(self.quadrature_points) == 0:
            msg = f"Quadrature has not been initialized."
            raise AssertionError(msg)

        f = h_new / h_old
        it = zip(self.quadrature_points, self.weights)
        for q, (qpoint, weight) in enumerate(it):
            weight *= f
            qpoint.z = domain[0] + f * (qpoint.z - domain_old[0])
        self._domain = domain


class GaussLegendreQuadrature(Quadrature):
    """
    Implementation of a Gauss-Legendre 1D quadrature.
    """

    def __init__(self, n_quadrature_points: int) -> None:
        """
        Parameters
        ----------
        n_quadrature_points : int
        """
        super().__init__()

        pts, wgts = leggauss(n_quadrature_points)
        self.quadrature_points = [CartesianVector(z=pt) for pt in pts]
        self.weights = wgts


class GaussChebyshevQuadrature(Quadrature):
    """
    Implementation of a Gauss-Chebyshev 1D quadrature.
    """

    def __init__(self, n_quadrature_points: int) -> None:
        """
        Parameters
        ----------
        n_quadrature_points : int
        """
        super().__init__()

        pts, wgts = chebgauss(n_quadrature_points)
        self.quadrature_points = [CartesianVector(z=pt) for pt in pts]
        self.weights = wgts
