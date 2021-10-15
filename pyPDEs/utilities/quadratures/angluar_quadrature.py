from numpy import ndarray
from typing import List

from . import GaussLegendre


class AngularQuadrature:
    """Base class for angular quadratures.
    """

    def __init__(self, n: int) -> None:
        """Constructor.

        Parameters
        ----------
        n : int
            The number of angles to use.
        """
        if not n % 2 == 0 or n == 0:
            raise ValueError("n_polar must be a positive even number.")

        gl = GaussLegendre(n)
        self.mu: List[float] = [p.z for p in gl.qpoints]
        self.weights: List[float] = list(gl.weights)

        self.M: ndarray =  None
        self.D: ndarray = None


def legendre(n: int, x: float) -> float:
    """Legendre polynomials.

    Parameters
    ----------
    n : int
        The index of the Legendre polynomial.
    x : float
        The argument of the Legendre polynomial.

    Returns
    -------
    float
    """
    if n < 0:
        raise ValueError("Invalid Legendre polynomial index.")
    if x < -1.0 or x > 1.0:
        raise ValueError("Argument must lie within [-1.0, 1.0].")

    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
      rhs = (2.0 * n + 1) * x * legendre(n - 1, x)
      rhs -= n * legendre(n - 2, x)
      return rhs / (n + 1)
