from . import Quadrature, GaussLegendre
from .. import Vector


class QuadrilateralQuadrature(Quadrature):
    """Quadrature used for integrating over a quadrilateral.

    This set is, in effect, an outer product of two sets
    of GaussLegendre.

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
        """QuadrilateralQuadrature constructor.

        Parameters
        ----------
        order : int, default 2
            The maximum monomial order the quadrature set
            can integrate exactly. For GaussLegendre, the
            number of points this will yield is
            n_qpoints = ceil(0.5 * (order + 1).
        """
        gl_quad = GaussLegendre(order)
        n = gl_quad.n_qpoints

        self.qpoints = [Vector() for _ in range(n ** 2)]
        self.weights = [0.0 for _ in range(n ** 2)]

        q = 0
        for i in range(n):
            for j in range(n):
                self.qpoints[q].x = gl_quad.qpoints[i].z
                self.qpoints[q].y = gl_quad.qpoints[j].z
                self.weights[q] = \
                    gl_quad.weights[i] * gl_quad.weights[j]
                q += 1
