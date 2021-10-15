__all__ = ["Boundary", "DirichletBoundary", "NeumannBoundary",
           "RobinBoundary"]

from typing import List


class Boundary:
    """Generic boundary.
    """

    def __init__(self) -> None:
        self.type: str = None


class DirichletBoundary(Boundary):
    """Dirichlet boundary.

    This imposes that the solution take a
    fixed value at the boundary.

    ..math:: u(x_b) = f^d.
    """

    def __init__(self, values: List[float],
                 n_components: int = 1) -> None:
        """Constructor.

        Parameters
        ----------
        values : List[float]
            The boundary value, or values.
        n_components : int, default 1
            The number of components
        """
        super().__init__()
        self.type = "DIRICHLET"

        if isinstance(values, float):
            values = [values] * n_components
        self.values: List[float] = values


class NeumannBoundary(Boundary):
    """Neumann boundary.

    This imposes that the solution gradient takes
    a fixed value at the boundary.

    ..math:: \grad u(x_b) = f^n
    """

    def __init__(self, values: List[float],
                 n_components: int = 1) -> None:
        """Constructor.

        Parameters
        ----------
        values : List[float]
        n_components : int, default 1
            The number of components.
        """
        self.type = "NEUMANN"

        if isinstance(values, float):
            values = [values] * n_components
        self.values: List[float] = values


class RobinBoundary(Boundary):
    """Robin boundary.

    This is a mixed Dirichlet and Neumann condition
    that imposes that the solution times a constant
    plus the gradient times a constant is equal to a
    fixed value at the boundary.

    ..math:: a u(x_b) + b \grad u(x_b) = f^r
    """

    def __init__(self, a: List[float], b: List[float],
                 f: List[float], n_components: int = 1) -> None:
        """Constructor.

        Parameters
        ----------
        a : List[float]
        b : List[float]
        f : List[float]
        n_components : int, default 1
            The number of components.
        """
        super().__init__()
        self.type = "ROBIN"

        if isinstance(a, float):
            a = [a] * n_components
        self.a: List[float] = a

        if isinstance(b, float):
            b = [b] * n_components
        self.b: List[float] = b

        if isinstance(f, float):
            f = [f] * n_components
        self.f: List[float] = f
