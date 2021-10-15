from . import GaussLegendre


class LineQuadrature(GaussLegendre):
    """Quadrature used for integrating over a line.
    """

    def __init__(self, n_qpoints: int = 2) -> None:
        """Quadrature constructor.

        Parameters
        ----------
        n_qpoints : int, default 2
            The number of quadrature points and weights to
            generate. A quadrature set with `n_qpoints` quadrature
            points can integrate polynomials of up to degree
            2*`n_qpoints`-1 exactly.
        """
        super().__init__(n_qpoints)
