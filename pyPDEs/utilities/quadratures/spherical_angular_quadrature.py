import numpy as np
from numpy import ndarray

from typing import List

from . import AngularQuadrature


class SphericalAngularQuadrature(AngularQuadrature):

    def __init__(self, n: int) -> None:
        super().__init__(n)

        self.mu_half: List[float] = self._define_mu_half()
        self.alpha: List[float] = self._define_alpha()
        self.beta: List[float] = self._define_beta()

    def _define_mu_half(self) -> List[float]:
        """Compute the angular cell edges.

        Returns
        -------
        List[float]
            The n + 1 angular cell edges.
        """
        w = self.weights  # shorthand
        mu_half = [-1.0]  # by definition
        for n in range(self.n_angles):
            mu_half.append(mu_half[n] + w[n])
            if abs(mu_half[n+1]) < 1.0e-14:
                mu_half[n+1] = 0.0
        mu_half[-1] = 1.0
        return mu_half

    def _define_alpha(self) -> List[float]:
        """Compute the angular advection coefficients.

        This routine computes coefficients for the n + 1
        angular cell edges that preserve the constant solution.

        Returns
        -------
        List[float]
            The n + 1 angular advection coefficients.
        """
        w = self.weights  # shorthand

        alpha = []
        alpha.append(0.0)  # by definition
        for n in range(self.n_angles - 1):
            alpha.append(alpha[n] - 2.0*w[n]*self.mu[n])
        alpha.append(0.0)  # by definition

    def _define_beta(self) -> List[float]:
        """Compute the diamond difference weighting coefficients.

        Returns
        -------
        List[float]
        """
        mu_half = self.mu_half  # shorthand

        beta = []
        for n in range(self.n_angles):
            mu = self.mu[n]
            mu_l = mu_half[n]
            mu_r = mu_half[n+1]
            beta.append((mu - mu_l) / (mu_r - mu_l))
        return beta
