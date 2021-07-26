import numpy as np
from numpy import ndarray


class Boundary:
    """
    Base class for a diffusion boundary condition.

    Attributes
    ----------
    type : str
        The boundary condition identifier.

    Parameters
    ----------
    bndry_type : str, default None
    """
    def __init__(self, bndry_type: str = None) -> None:
        self.type: str = bndry_type


class ReflectiveBoudnary(Boundary):
    """
    Reflective diffusion boundary condition.

    Parameters
    ----------
    bndry_values : ndarray
    """
    def __init__(self) -> None:
        super().__init__("REFLECTIVE")


class NeumanBoundary(Boundary):
    """
    Neumann diffusion boundary condition.

    Parameters
    ----------
    bndry_values : ndarray
    """
    def __init__(self, bndry_values: ndarray = None) -> None:
        super().__init__("NEUMANN")
        self.values: ndarray = bndry_values


class DirichletBoundary(Boundary):
    """
    Dirichlet diffusion boundary condition.

    Parameters
    ----------
    bndry_values : ndarray
    """
    def __init__(self, bndry_values: ndarray) -> None:
        super().__init__("DIRICHLET")
        self.values: ndarray = bndry_values


class RobinBoundary(Boundary):
    """
    Robin diffusion boundary condition. This is intended
    to be used with finite element discretizations

    Attributes
    ----------
    a : ndarray
        Multiplier attached to the gradient term.
    b : ndarray
        Multiplier attached to the function term.
    f : ndarray
        Boundary source term.

    Parameters
    ----------
    a_values : ndarray
    b_values : ndarray
    f_values : ndarray
    """
    def __init__(self, a_values: ndarray,
                 b_values: ndarray, f_values: ndarray) -> None:
        super().__init__("ROBIN")
        self.a: ndarray = a_values
        self.b: ndarray = b_values
        self.f: ndarray = f_values


class MarshakBoundary(Boundary):
    """
    Marshak diffusion boundary condition. This is intended
    to be used with finite volume discretizations

    Parameters
    ----------
    bndry_values : ndarray
    """
    def __init__(self, bndry_values: ndarray) -> None:
        super().__init__("MARSHAK")
        self.values: ndarray = bndry_values
