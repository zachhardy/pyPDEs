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

    def __init__(self, n_polar: int, n_azimuthal: int = 0,
                 quadrature_type: str = "GL") -> None:
        """Constructor.

        Parameters
        ----------
        n_polar : int
            This corresponds to the number of polar angles.
        n_azimuthal : int
            This corresponds to the number of azimuthal angles.
        quadrature_type : str, default "GL"
            The type of quadrature to use.
            Currently, only Gauss-Legendre ("GL" is supported.
        """
        if not n_polar % 2 == 0 or n_polar == 0:
            raise ValueError("n_polar must be a positive even number.")
        if quadrature_type not in ["GL"]:
            raise NotImplementedError("Only Gauss-Legendre is supported.")

        self.azimuthal: List[float] = []
        self.polar: List[float] = []

        self.abscissae: List[Angle] = []
        self.weights: List[float] = []
        self.omegas: List[float] = []

        self.ell_m_map: List["HarmonicIndex"] = []

        self.M: ndarray =  None
        self.D: ndarray = None

        if quadrature_type == "GL":
            self.initialize_gl(n_polar)

    def initialize_gl(self, n_polar: int) -> None:
        """Initialize with Gauss-Legendre quadratures.

        This is only valid for 1D problems.
        """
        gl = GaussLegendre(n_polar)

        # 1D, only azumuthal angle is 0.0
        self.azimuthal.append(0.0)

        # Compute the polar angles in radians
        # This orders the polar angles in increasing order
        for i in range(n_polar):
            theta = np.pi - np.arccos(gl.qpoints[i].z)
            self.polar.append(theta)

        # Define the abscissae and weights
        for i in range(len(self.polar)):
            abscissa = Angle(self.azimuthal[0], self.polar[i])
            self.abscissae.append(abscissa)
            self.weights.append(gl.weights[i])

        # Define the omegas
        for qpoint in self.abscissae:
            x = np.sin(qpoint.theta) * np.cos(qpoint.varphi)
            y = np.sin(qpoint.theta) * np.sin(qpoint.varphi)
            z = np.cos(qpoint.theta)
            self.omegas.append(Vector(x, y, z))

    def generate_harmonic_indices(self, scattering_order: int,
                                  dimension: int) -> None:
        """Generate the harmonic index list.

        Parameters
        ----------
        scattering_order : int
            The scattering order to use. This determines the
            maximum value of ell.
        dimension : int
            The dimension of the problem. This determines the
            values of m that are used, if any.
        """
        if self.ell_m_map == []:
            if dimension == 1:
                for ell in range(scattering_order + 1):
                    ind = HarmonicIndex(ell, 0)
                    self.ell_m_map.append(ind)
            elif dimension == 2:
                for ell in range(scattering_order + 1):
                    for m in range(-ell, ell, 2):
                        if ell == 0 or m != 0:
                            ind = HarmonicIndex(ell, m)
                            self.ell_m_map.append(ind)
            elif dimension == 3:
                for ell in range(scattering_order + 1):
                    for m in range(-ell, ell):
                        ind = HarmonicIndex(ell, m)
                        self.ell_m_map.append(ind)

    def assemble_discrete_to_moment_matrix(
            self, scattering_order: int, dimension: int) -> None:
        """Assembe the discrete to moment operator.

        Parameters
        ----------
        scattering_order : int
            The scattering order to use. This determines the
            maximum value of ell.
        dimension : int
            The dimension of the problem. This determines the
            values of m that are used, if any.
        """
        self.D = []
        self.generate_harmonic_indices(scattering_order, dimension)

        for ell_m in self.ell_m_map:
            moment_ell_m = []
            for n in range(len(self.abscissae)):
                w = self.weights[n]
                angle = self.abscissae[n]
                Ylm_val = Y_lm(ell_m.ell, ell_m.m, angle)
                moment_ell_m.append(Ylm_val * w)
            self.D.append(moment_ell_m)
        self.D = np.array(self.D)

    def assemble_moment_to_discrete(
            self, scattering_order: int, dimension: int) -> None:
        """Assembe the moment to discrete operator.

        Parameters
        ----------
        scattering_order : int
            The scattering order to use. This determines the
            maximum value of ell.
        dimension : int
            The dimension of the problem. This determines the
            values of m that are used, if any.
        """
        self.M = []
        self.generate_harmonic_indices(scattering_order, dimension)

        norm = sum(self.weights)
        for ell_m in self.ell_m_map:
            moment_ell_m = []
            for n in range(len(self.abscissae)):
                angle = self.abscissae[n]
                w = (2.0 * ell_m.ell + 1.0) / norm
                Ylm_val = Y_lm(ell_m.ell, ell_m.m, angle)
                moment_ell_m.append(Ylm_val * w)
            self.M.append(moment_ell_m)
        self.M = np.array(self.M)


class Angle:
    def __init__(self, varphi: float, theta: float) -> None:
        self.varphi: float = varphi
        self.theta: float = theta


class HarmonicIndex:
    def __init__(self, ell: int, m: int) -> None:
        self.ell = ell
        self.m = m

    def __eq__(self, other: "HarmonicIndex") -> bool:
        return self.ell == other.ell and self.m == other.m


def Y_lm(ell: int, m: int, angle: Angle) -> float:
    """Spherical harmonic functions.

    Parameters
    ----------
    ell : int
    m : int
    angle : Angle

    Returns
    -------
    float
        The ell, m'th spherical harmonic evaluated at
        the specified angle.
    """

    Plm = P_lm(ell, abs(m), np.cos(angle.theta))
    if m == 0:
        return Plm
    else:
        Clm = pow(-1, abs(m))
        Clm *= sqrt(2.0 * fac(ell - abs(m))/fac(ell + abs(m)))
        if m < 0:
            return Clm * Plm * np.sin(abs(m)*angle.varphi)
        else:
            return Clm * Plm * np.cos(m*angle.varphi)


def P_lm(ell: int, m: int, x: float) -> float:
    """Legendre polynomials.

    Parameters
    ----------
    ell : int
    m : int
    x : float

    Returns
    -------
    float
    """
    assert abs(m) <= ell, "abs(m) <= ell was violated."

    if ell == 0:
        return 1.0
    elif ell == 1:
        if m == 0:
            return x
        elif m == 1:
            return -sqrt(1.0 - x**2)
    else:
        if ell == m:
            c = -(2*ell -1) * sqrt(1.0 - x**2)
            return c * P_lm(ell-1, m-1, x)
        else:
            Pml = (2*ell - 1)*x * P_lm(ell-1, m, x)
            Pml -= (ell + m - 1) * P_lm(ell-2, m, x)
            return Pml / (ell - m)

