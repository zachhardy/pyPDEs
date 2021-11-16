import numpy as np
from typing import Dict, List

from ..vector import Vector
from . import GaussLegendre


class Angle:
    """
    Container for an angle.

    Parameters
    ----------
    phi : float
        The azimuthal angle.
    theta : float
        The polar angle.
    """
    def __init__(self, phi: float, theta: float) -> None:
        self.phi: float = phi
        self.theta: float = theta


class AngularQuadrature:
    """
    Base class for angular quadratures.
    """
    def __init__(self) -> None:
        self.abscissae: List[Angle] = []
        self.weights: List[float] = []
        self.omegas: List[Vector] = []


class ProductQuadrature(AngularQuadrature):
    """
    Product quadratures.

    Parameters
    ----------
    n_polar : int
        The number of polar angles.
    n_azimuthal : int, default 1
        The number of azimuthal angles.
    quadrature_type : {'gl', 'gll', 'glc'}, default 'gl'
        The type of quadrature to use. Options are 'gl' for
        Gauss-Legendre for 1D quadratures, 'gll' for Gauss-Legendre
        quadratures for both polar and azimuthal angles, or 'glc' for
        Gauss-Legendre polar quadrature and Gauss-Chebyshev azimuthal
        quadratures.
    """
    def __init__(self, n_polar: int, n_azimuthal: int = 1,
                 quadrature_type: str = 'gl') -> None:
        super().__init__()
        self.direction_mapping: Dict[int, List[int]] = {}
        self.initialize()

    def initialize(self, n_polar: int, n_azimuthal: int = 1,
                   quadrature_type: str = 'gl') -> None:
        """
        Initialize the quadrature.

        Parameters
        ----------
        n_polar : int
            The number of polar angles.
        n_azimuthal : int, default 1
            The number of azimuthal angles.
        quadrature_type : {'gl', 'gll', 'glc'}, default 'gl'
            The type of quadrature to use. Options are 'gl' for
            Gauss-Legendre for 1D quadratures, 'gll' for Gauss-Legendre
            quadratures for both polar and azimuthal angles, or 'glc' for
            Gauss-Legendre polar quadrature and Gauss-Chebyshev azimuthal
            quadratures.
        """
        azimuthal_angles: List[float] = []
        polar_angles: List[float] = []
        weights: List[float] = []

        # Initialize Gauss-Legendre
        if quadrature_type == 'gl':
            gl_polar = GaussLegendre(Np*2)

            # Create single azimuthal angle
            azimuthal_angles = [0.0]

            # Create polar angles
            polar_angles = []
            for q in range(n_polar*2):
                # Quadrature points are cosines, to get the angles
                # apply arccos. To sort from least to greatest, subtract
                # this result from pi
                theta = np.pi - np.arccos(gl_polar.qpoints[q].z)

            # Create weights
            weights = gl_polar.weights

        else:
            raise NotImplementedError('Quadrature type not implemented.')

        # Get counts of polar, azimuthal, and weights
        Np = len(polar_angles)
        Na = len(azimuthal_angles)
        Nw = len(weights)

        # Define the product quadrature
        if Nw != Np*Na:
            raise AssertionError(
                'Invalid number of weights for the number of polar and '
                'azimuthal angles.')

        # Initialize direction mapping
        self.direction_mapping = {}
        for j in range(Np):
            self.direction_mapping[j] = []

        # Create angles and assign direction mapping
        self.abscissae.clear()
        self.weights.clear()
        weight_sum = 0.0
        for i in range(Na):
            for j  in range(Np):
                self.direction_mapping[j].append(i*Np + j)

                abscissa = Angle(azimuthal_angles[i], polar_angles[j])
                self.abscissae.append(abscissa)

                weight = weights[i*Np + j]
                weight_sum += weight
                self.weights.append(weight)

        # Create omegas
        self.omegas.clear()
        for qpoint in self.abscissae:
            omega = Vector()
            omega.x = np.sin(qpoint.theta) * np.cos(qpoint.phi)
            omega.y = np.sin(qpoint.theta) * np.sin(qpoint.phi)
            omega.z = np.cos(qpoint.theta)
            self.omegas.append(omega)
