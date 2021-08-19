__all__ = ["Boundary", "DirichletBoundary", "NeumannBoundary",
           "RobinBoundary", "ReflectiveBoundary", "MarshakBoundary",
           "VacuumBoundary", "ZeroFluxBoundary"]

class Boundary:
    """Generic boundary.

    Attributes
    ----------
    type : str
        The type of boundary.
    """

    def __init__(self) -> None:
        self.type: str = None


class DirichletBoundary(Boundary):
    """Dirichlet boundary.

    This imposes that the solution take a
    fixed value at the boundary.

    ..math:: u(x_b) = f^d

    Attributes
    ----------
    type : str
        The type of boundary.
    value : float
        The boundary value.
    """

    def __init__(self, value: float) -> None:
        """Constructor.

        Parameters
        ----------
        value : float
            The boundary value.
        """
        super().__init__()
        self.type = "DIRICHLET"
        self.value: float = value


class NeumannBoundary(Boundary):
    """Neumann boundary.

    This imposes that the solution gradient takes
    a fixed value at the boundary.

    ..math:: \grad u(x_b) = f^n

    Attributes
    ----------
    type : str
        The type of boundary.
    value : float
        The boundary value.
    """

    def __init__(self, value: float) -> None:
        """Constructor.

        Parameters
        ----------
        value : float
        """
        self.type = "NEUMANN"
        self.value: float = value


class RobinBoundary(Boundary):
    """Robin boundary.

    This is a mixed Dirichlet and Neumann condition
    that imposes that the solution times a constant
    plus the gradient times a constant is equal to a
    fixed value at the boundary.

    ..math:: a u(x_b) + b \grad u(x_b) = f^r

    Attributes
    ----------
    type : str
        The type of boundary.
    a : float
        The coefficient for the solution term
    b : float
        The coefficient for the gradient term
    f : float
        The source term.
    """

    def __init__(self, a: float, b: float, f: float) -> None:
        """Constructor.

        Parameters
        ----------
        a : float
        b : float
        f : float
        """
        super().__init__()
        self.type = "ROBIN"
        self.a: float = a
        self.b: float = b
        self.f: float = f


class ReflectiveBoundary(NeumannBoundary):
    """Reflective boundary.

    This imposes a zero Neumann boundary.

    Attributes
    ----------
    type : str
        The type of boundary.
    value : float
        The boundary value.
    """

    def __init__(self) -> None:
        super().__init__(0.0)


class MarshakBoundary(RobinBoundary):
    """Marshak boundary.

    This imposes a Robin boundary condition that is
    equivalent to an incident current at the boundary.
    The coefficients are: a = -0.5, b = 1.0, and
    f = 2.0 * f^m, where f^m is the incident current
    at the boundary.

    Attributes
    ----------
    type : str
        The type of boundary.
    a : float
        The coefficient for the solution term
    b : float
        The coefficient for the gradient term
    f : float
        The source term.
    """
    def __init__(self, j_hat: float) -> None:
        """
        Constructor.

        Parameters
        ----------
        j_hat : float
            The incident current. This gets multiplied
            by two as part of the defintion of a Marshak
            boundary condition.
        """
        super().__init__(-0.5, 1.0, 2.0 * j_hat)


class VacuumBoundary(MarshakBoundary):
    """Vacuum boundary.

    This imposes a Marshak boundary with a
    zero incident current.

    Attributes
    ----------
    type : str
        The type of boundary.
    a : float
        The coefficient for the solution term
    b : float
        The coefficient for the gradient term
    f : float
        The source term. This is
    """

    def __init__(self) -> None:
        super().__init__(0.0)


class ZeroFluxBoundary(DirichletBoundary):
    """Zero flux boundary.

    This imposes a zero Dirichlet boundary.

    Attributes
    ----------
    type : str
        The type of boundary.
    value : float
        The boundary value.
    """

    def __init__(self) -> None:
        super().__init__(0.0)
