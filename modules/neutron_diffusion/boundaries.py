from pyPDEs.utilities.boundaries import *

from typing import List


class ReflectiveBoundary(NeumannBoundary):
    """Reflective boundary.

    This imposes a zero Neumann boundary.
    """

    def __init__(self, n_groups: int = 1) -> None:
        """Constructor.

        Parameters
        ----------
        n_groups : int, default 1
            The number of groups.
        """
        super().__init__([0.0] * n_groups)


class MarshakBoundary(RobinBoundary):
    """Marshak boundary.

    This imposes a Robin boundary condition that is
    equivalent to an incident current at the boundary.
    The coefficients are: a = -0.5, b = 1.0, and
    f = 2.0 * f^m, where f^m is the incident current
    at the boundary.
    """
    def __init__(self, j_hat: List[float]) -> None:
        """
        Constructor.

        Parameters
        ----------
        j_hat : float
            The incident current. This gets multiplied
            by two as part of the defintion of a Marshak
            boundary condition.
        """
        if isinstance(j_hat, float):
            j_hat = [j_hat]

        a = [-0.5 for i in range(len(j_hat))]
        b = [1.0 for i in range(len(j_hat))]
        f = [2.0 * j_hat[i] for i in range(len(j_hat))]
        super().__init__(a, b, f)


class VacuumBoundary(MarshakBoundary):
    """Vacuum boundary.

    This imposes a Marshak boundary with a
    zero incident current.
    """

    def __init__(self, n_groups: int = 1) -> None:
        """Constructor.

        Parameters
        ----------
        n_groups : int, default 1
            The number of components.
        """
        super().__init__([0.0] * n_groups)


class ZeroFluxBoundary(DirichletBoundary):
    """Zero flux boundary.

    This imposes a zero Dirichlet boundary.
    """

    def __init__(self, n_groups: int = 1) -> None:
        """Constructor.

        Parameters
        ----------
        n_groups : int, default 1
            The number of components.
        """
        super().__init__([0.0] * n_components)


__all__ = ["ReflectiveBoundary", "MarshakBoundary",
           "VacuumBoundary", "ZeroFluxBoundary"]
