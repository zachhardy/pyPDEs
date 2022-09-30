class Boundary:
    """
    Abstract base class for neutron diffusion boundaries.
    """

    def __init__(self, boundary_type: str) -> None:
        self._bndry_type: str = boundary_type

    @property
    def boundary_type(self) -> str:
        return self._bndry_type


class DirichletBoundary(Boundary):
    """
    Dirichlet boundary given by \f$ u_b = f^d \f$.
    """

    def __init__(self, value: float = 0.0) -> None:
        if not isinstance(value, float):
            raise ValueError(f"Boundary value must be a float.")

        super().__init__("DIRICHLET")
        self.value: float = value


class NeumannBoundary(Boundary):
    """
    Neumann boundary given by \f$ \nabla u \cdot \hat{n}_b = f^n \f$.
    """

    def __init__(self, value: float = 0.0) -> None:
        if not isinstance(value, float):
            raise ValueError(f"Boundary value must be a float.")

        super().__init__("NEUMANN")
        self.value: float = value


class RobinBoundary(Boundary):
    """
    Robin boundary given by \f$ a u_b + b \nable u \cdot \hat{n}_b = f^r \f$.

    This constructs a vacuum boundary by default.
    """

    def __init__(
            self,
            a: float = 0.25,
            b: float = 0.5,
            f: float = 0.0
    ) -> None:
        if (not isinstance(a, float) or
                not isinstance(b, float) or
                not isinstance(f, float)):
            raise ValueError("Boundary values must be floats.")

        super().__init__("ROBIN")
        self.a: float = a
        self.b: float = b
        self.f: float = f
