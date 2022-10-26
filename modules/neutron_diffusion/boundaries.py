from typing import Union, Callable
from pyPDEs.mesh import CartesianVector

BCFunc = Callable[[CartesianVector, float], float]
BCType = Union[float, BCFunc]

class Boundary:
    """
    Abstract base class for neutron diffusion boundaries.
    """

    def __init__(
            self,
            boundary_type: str,
            boundary_value: BCType = 0.0
    ) -> None:
        """
        Parameters
        ----------
        boundary_type : str
            The boundary type. This is provided by sub-classes.
        boundary_value : float or callable
            The boundary value or boundary value function. If the
            latter, the function must be callable via position
            (CartesianVector) and optionally time (float) and return
            a float.
        """

        self._bndry_type: str = boundary_type
        self._bndry_val: BCType = boundary_value

    @property
    def boundary_type(self) -> str:
        return self._bndry_type

    def boundary_value(
            self,
            r: CartesianVector,
            t: float = 0.0
    ) -> float:
        """
        Evaluate the boundary value.

        Parameters
        ----------
        r : CartesianVector, The spatial position.
        t : float, The time. Default 0.0

        Returns
        -------
        float, The boundary value at the specified position and time.
        """
        if callable(self._bndry_val):
            return self._bndry_val(r, t)
        else:
            return self._bndry_val


class DirichletBoundary(Boundary):
    """
    Dirichlet boundary given by \f$ u_b = f^d \f$.
    """

    def __init__(self, boundary_value: BCType = 0.0) -> None:
        super().__init__("DIRICHLET", boundary_value)

class NeumannBoundary(Boundary):
    """
    Neumann boundary given by \f$ \nabla u \cdot \hat{n}_b = f^n \f$.
    """

    def __init__(self, boundary_value: BCType = 0.0) -> None:
        super().__init__("NEUMANN", boundary_value)


class RobinBoundary(Boundary):
    """
    Robin boundary given by \f$ a u_b + b \nable u \cdot \hat{n}_b = f^r \f$.

    This constructs a vacuum boundary by default.
    """

    def __init__(
            self,
            a: float = 0.25,
            b: float = 0.5,
            f: BCType = 0.0
    ) -> None:
        if not isinstance(a, float) or not isinstance(b, float):
            raise ValueError("Invalid specification of \"a\" or \"b\".")

        super().__init__("ROBIN", f)
        self.a: float = a
        self.b: float = b
