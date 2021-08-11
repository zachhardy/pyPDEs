from . import GaussLegendre


class LineQuadrature(GaussLegendre):
    """
    Quadrature used for integrating over a line.
    This is a passthrough for GaussLegendre quadrature.

    Parameters
    ----------
    order : int, default 2
        The maximum monomial order the quadrature set
        can integrate exactly.
    """
    def __init__(self, order: int = 2) -> None:
       super().__init__(order)
