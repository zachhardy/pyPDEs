import numpy as np
from numpy import ndarray

from math import factorial as fac
from math import sqrt

from typing import List, Tuple

from . import GaussLegendre
from .. import Vector


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

        self.mu: List[float] = []
        self.weights: List[float] = []

        self.M: ndarray =  None
        self.D: ndarray = None

        gl = GaussLegendre(n)
        self.mu = [qpoint.z for qpoint in gl.qpoints]
        self.weights = list(gl.weights)

        self._mu_half: List[float] = None
        self._alphas: List[float] = None
        self._betas: List[float] = None

    @property
    def mu_half(self) -> List[float]:
        """Get the edge values for the quadrature set.

        Returns
        -------
        List[float]
        """
        if self._mu_half is None:
            self._mu_half = [-1.0]
            for n in range(len(self.mu)):
                mu_half = self._mu_half[n] + self.weights[n]
                if abs(mu_half) < 1.0e-12:
                    mu_half = 0.0
                self._mu_half.append(mu_half)
        return self._mu_half

    @property
    def alphas(self) -> List[float]:
        """Coefficients to preserve the constant solution.

        These coefficients preseve the constant solution in
        spherical geometries and are defined on the angular
        cell edges.

        Returns
        -------
        List[float]
        """
        if self._alphas is None:
            self._alphas = [0.0]  # by definition
            for n in range(len(self.mu)):
                alpha = self._alphas[n] - 2.0*self.weights[n]*self.mu[n]
                self._alphas.append(alpha)
            self._alphas.append(0.0)  # by definition
        return self._alphas

    @property
    def beta(self) -> List[float]:
        """Weighted diamond difference coefficients.

        These coefficients are used for the modified diamond
        difference equation in spherical geometries.

        Returns
        -------
        List[float]
        """
        if self._betas is None:
            self._betas = []
            for n in range(len(self.mu)):
                mu = self.mu[n]
                mu_l = self.mu_half[n]
                mu_r = self.mu_half[n+1]
                beta = (mu - mu_l) / (mu_r - mu_l)
                self._betas.append(beta)
        return self._betas


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
