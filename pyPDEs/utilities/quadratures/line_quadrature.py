from . import GaussLegendre


class LineQuadrature(GaussLegendre):
    """
    Quadrature used for integrating over a line.

    Parameters
    ----------
    n_qpoints : int, default 2
        The number of quadrature points. This will integrate
        polynomials of 2N-1 quadrature points exactly.
    """

    def __init__(self, n_qpoints: int = 2) -> None:
        super().__init__(n_qpoints)
