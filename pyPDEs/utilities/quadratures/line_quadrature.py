from . import GaussLegendre


class LineQuadrature(GaussLegendre):
    """Quadrature used for integrating over a line.
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
