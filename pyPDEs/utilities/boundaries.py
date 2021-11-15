from typing import List

__all__ = ['Boundary', 'DirichletBoundary',
           'NeumannBoundary', 'RobinBoundary']


class Boundary:
    """
    Generic boundary.
    """

    def __init__(self) -> None:
        self.type: str = None


class DirichletBoundary(Boundary):
    """
    Dirichlet boundary.

    This imposes that the solution take a
    fixed value at the boundary.

    ..math:: u(x_b) = f^d.

    Parameters
    ----------
    values : List[float]
        The boundary value, or values.
    """

    def __init__(self, values: List[float]) -> None:
        super().__init__()
        self.type = 'dirichlet'

        if isinstance(values, float):
            values = [values]
        self.values: List[float] = values


class NeumannBoundary(Boundary):
    """
    Neumann boundary.

    This imposes that the solution gradient takes
    a fixed value at the boundary.

    ..math:: \grad u(x_b) = f^n

    Parameters
    ----------
    values : List[float]
    """

    def __init__(self, values: List[float]) -> None:
        self.type = 'neumann'

        if isinstance(values, float):
            values = [values]
        self.values: List[float] = values


class RobinBoundary(Boundary):
    """
    Robin boundary.

    This is a mixed Dirichlet and Neumann condition
    that imposes that the solution times a constant
    plus the gradient times a constant is equal to a
    fixed value at the boundary.

    ..math:: a u(x_b) + b \grad u(x_b) = f^r

    Parameters
    ----------
    a : List[float]
    b : List[float]
    f : List[float]
    """

    def __init__(self, a: List[float], b: List[float],
                 f: List[float]) -> None:
        super().__init__()
        self.type = 'robin'

        if isinstance(a, float):
            a = [a]
        self.a: List[float] = a

        if isinstance(b, float):
            b = [b]
        self.b: List[float] = b

        if isinstance(f, float):
            f = [f]
        self.f: List[float] = f
