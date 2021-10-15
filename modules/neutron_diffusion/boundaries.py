from pyPDEs.utilities.boundaries import *

from typing import List


class ReflectiveBoundary(NeumannBoundary):
    """Reflective boundary.

    This imposes a zero Neumann boundary.
    """

    def __init__(self, n_components: int = 1) -> None:
        """Constructor.

        Parameters
        ----------
        n_components : int, default 1
            The number of components.
        """
        super().__init__(0.0, n_components)


class MarshakBoundary(RobinBoundary):
    """Marshak boundary.

    This imposes a Robin boundary condition that is
    equivalent to an incident current at the boundary.
    The coefficients are: a = -0.5, b = 1.0, and
    f = 2.0 * f^m, where f^m is the incident current
    at the boundary.
    """
    def __init__(self, j_hat: List[float], n_components: int = 1) -> None:
        """
        Constructor.

        Parameters
        ----------
        j_hat : float
            The incident current. This gets multiplied
            by two as part of the defintion of a Marshak
            boundary condition.
        n_components : int, default 1
            The number of components.
        """
        if isinstance(j_hat, float):
            f = [j_hat] * n_components
        f = [2.0 * f[i] for i in range(len(f))]
        super().__init__(-0.5, 1.0, f, n_components)


class VacuumBoundary(MarshakBoundary):
    """Vacuum boundary.

    This imposes a Marshak boundary with a
    zero incident current.
    """

    def __init__(self, n_components: int = 1) -> None:
        """Constructor.

        Parameters
        ----------
        n_components : int, default 1
            The number of components.
        """
        super().__init__(0.0, n_components)


class ZeroFluxBoundary(DirichletBoundary):
    """Zero flux boundary.

    This imposes a zero Dirichlet boundary.
    """

    def __init__(self, n_components: int = 1) -> None:
        """Constructor.

        Parameters
        ----------
        n_components : int, default 1
            The number of components.
        """
        super().__init__(0.0, n_components)


__all__ = ["ReflectiveBoundary", "MarshakBoundary",
           "VacuumBoundary", "ZeroFluxBoundary"]
