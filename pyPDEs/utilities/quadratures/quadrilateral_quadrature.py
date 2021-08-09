from . import Quadrature, GaussLegendre
from .. import Vector


class QuadrilateralQuadrature(Quadrature):
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
