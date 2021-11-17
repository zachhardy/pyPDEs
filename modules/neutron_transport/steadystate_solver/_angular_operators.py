import numpy as np
from numpy import ndarray

from pyPDEs.utilities.quadratures import Angle

from ._harmonics import HarmonicIndex, spherical_harmonics

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def discrete_to_moment_matrix(self: 'SteadyStateSolver') -> ndarray:
    """
    Build the moment to discrete operator.

    Returns
    -------
    ndarray (n_moments, n_angles)
    """
    D = np.zeros((self.n_moments, self.n_angles))

    # Loop over moments
    for m in range(self.n_moments):
        idx = self.harmonic_index_map[m]

        # Loop over angles
        for n in range(self.n_angles):
            w = self.quadrature.weights[n]
            angle = self.quadrature.abscissae[n]
            Ylm = spherical_harmonics(idx.ell, idx.m,
                                      angle.varphi, angle.theta)

            D[m, n] = Ylm * w
    return D


def moment_to_discrete_matrix(self: 'SteadyStateSolver') -> None:
    """
    Build the discrete to moment operator.

    Returns
    -------
    ndarray (n_moments, n_angles)
    """
    M = np.zeros((self.n_moments, self.n_angles))

    # Loop over moments
    weight_sum = sum(self.quadrature.weights)
    for m in range(self.n_moments):
        idx = self.harmonic_index_map[m]

        # Loop over angles
        for n in range(self.n_angles):
            angle = self.quadrature.abscissae[n]
            Ylm = spherical_harmonics(idx.ell, idx.m,
                                      angle.varphi, angle.theta)
            M[m, n] = (2.0 * idx.ell + 1.0) / weight_sum * Ylm
    return M
