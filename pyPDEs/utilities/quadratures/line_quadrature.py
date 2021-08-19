from . import GaussLegendre


class LineQuadrature(GaussLegendre):
    """Quadrature used for integrating over a line.

    This is a passthrough for GaussLegendre quadrature.

    Attributes
    ----------
    order : int
        The maximum monomial order the quadrature set
        can integrate exactly.
    qpoints : List[Vector]
        The quadrature points in the set.
    weights : List[float]
        The quadrature weights.
    domain : Tuple[float]
        The minimum and maximum coordinate of the quadrature
        domain. This is only used for one-dimensional problems
        to compute the Jacobian.
    """

    def __init__(self, order: int = 2) -> None:
        """Quadrature constructor.

        Parameters
        ----------
        order : int, default 2
            The maximum monomial order the quadrature set
            can integrate exactly.
        """
        super().__init__(order)
