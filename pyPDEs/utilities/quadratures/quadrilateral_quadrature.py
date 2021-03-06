from . import Quadrature, GaussLegendre
from .. import Vector


class QuadrilateralQuadrature(Quadrature):
    """
    Quadrature used for integrating over a quadrilateral.

    Parameters
    ----------
    order : int, default 2
        The maximum monomial order the quadrature set
        can integrate exactly. For GaussLegendre, the
        number of points this will yield is
        n_qpoints = ceil(0.5 * (order + 1).
    """

    def __init__(self, order: int = 2) -> None:
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
