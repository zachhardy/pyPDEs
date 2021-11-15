from . import GaussLegendre


class LineQuadrature(GaussLegendre):
    """
    Quadrature used for integrating over a line.

    Parameters
    ----------
    order : int, default 2
        The maximum monomial order the quadrature set
        can integrate exactly.
    """

    def __init__(self, order: int = 2) -> None:
        super().__init__(order)
