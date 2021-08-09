from typing import Tuple, List

import numpy as np
from numpy import poly1d
from scipy.interpolate import lagrange
from typing import TYPE_CHECKING

from ...mesh import Cell
from ...utilities.quadratures import GaussLegendre
from ...utilities import Vector
from .fe_view import CellFEView

if TYPE_CHECKING:
    from ...spatial_discretization import PiecewiseContinuous

FEBasis = Tuple[List[poly1d], List[poly1d]]


class SlabFEView(CellFEView):
    """
    Class for a finite element slab view.
    """
    def __init__(self, fe: 'PiecewiseContinuous',
                 quadrature: GaussLegendre,
                 cell: Cell) -> None:
        super().__init__(fe, quadrature)
        self.coord_sys: str = cell.coord_sys
        self.v0: float = fe.mesh.vertices[cell.vertex_ids[0]]

        c, degree = cell.id, fe.degree
        self.face_node_mapping = [[0], [degree]]
        self.node_ids = [c * degree + i for i in range(degree + 1)]

        self.nodes = []
        for nid in self.node_ids:
            self.nodes.append(fe.nodes[nid])

        tmp = lagrange_elements(degree, quadrature.domain)
        self._shape = tmp[0]
        self._grad_shape = tmp[1]

        self.compute_quadrature_data(cell)
        self.compute_integral_data(cell)

    def map_reference_to_global(self, point: Vector) -> float:
        """
        Map a point from the reference cell to
        the real cell.
        """
        domain = self.quadrature.domain
        if not min(domain) < point.z < max(domain):
            raise ValueError(
                f"Provided point not in volume quadrature domain.")
        return self.jacobian * (point.z + 1.0) + self.v0.z

    def compute_quadrature_data(self, cell: Cell) -> None:
        # =================================== Mapping data
        domain = self.quadrature.domain
        self.jacobian = cell.width.z / (domain[1] - domain[0])
        self.inverse_jacobian = 1.0 / self.jacobian

        weights = self.quadrature.weights
        self.jxw.resize(self.n_qpoints)
        for qp in range(self.n_qpoints):
            x = self.map_reference_to_global(self.qpoints[qp])
            self.jxw[qp] = self.jacobian * weights[qp]
            if self.coord_sys == "CYLINDRICAL":
                self.jxw[qp] *= 2.0 * np.pi * x
            elif self.coord_sys == "SPHERICAL":
                self.jxw[qp] *= 4.0 * np.pi * x ** 2

        # =================================== Finite element data
        n_nodes, n_qpoints = self.n_nodes, self.n_qpoints
        self.shape_values.resize((n_nodes, n_qpoints))
        self.grad_shape_values.resize((n_nodes, n_qpoints))
        for qp in range(self.n_qpoints):
            point = self.qpoints[qp].z
            for i in range(self.n_nodes):
                self.shape_values[i, qp] = self._shape[i](point)
                self.grad_shape_values[i, qp] = \
                    self._grad_shape[i](point) * self.inverse_jacobian

    def compute_integral_data(self, cell: Cell) -> None:
        """
        Compute the finite element integrals.
        """
        # ======================================== Compute volume integrals
        self.intV_shapeI *= 0.0
        self.intV_shapeI_shapeJ *= 0.0
        self.intV_gradI_gradJ *= 0.0
        self.intV_shapeI_gradJ *= 0.0

        self.intV_shapeI.resize(self.n_nodes)
        self.intV_shapeI_shapeJ.resize((self.n_nodes,) * 2)
        self.intV_gradI_gradJ.resize((self.n_nodes,) * 2)
        self.intV_shapeI_gradJ.resize((self.n_nodes,) * 2)
        for qp in range(self.n_qpoints):
            jxw = self.jxw[qp]

            for i in range(self.n_nodes):
                shape_i = self.shape_values[i, qp]
                grad_shape_i = self.grad_shape_values[i, qp]

                self.intV_shapeI[i] += shape_i * jxw

                for j in range(self.n_nodes):
                    shape_j = self.shape_values[j, qp]
                    grad_shape_j = self.grad_shape_values[j, qp]

                    self.intV_shapeI_shapeJ[i, j] += \
                        shape_i * shape_j * jxw

                    self.intV_gradI_gradJ[i, j] += \
                        grad_shape_i * grad_shape_j * jxw

                    self.intV_shapeI_gradJ[i, j] += \
                        shape_i * grad_shape_j * jxw

        # ======================================== Compute surface integrals
        self.intS_shapeI *= 0.0
        self.intS_shapeI_shapeJ *= 0.0
        self.intS_shapeI_gradJ *= 0.0

        self.intS_shapeI.resize((2, self.n_nodes))
        self.intS_shapeI_shapeJ.resize((2, *(self.n_nodes,) * 2))
        self.intS_shapeI_gradJ.resize((2, *(self.n_nodes,) * 2))

        self.intS_shapeI[0][0] = cell.faces[0].area
        self.intS_shapeI[1][-1] = cell.faces[1].area

        self.intS_shapeI_shapeJ[0][0][0] = cell.faces[0].area
        self.intS_shapeI_shapeJ[1][-1][-1] = cell.faces[1].area

        for j in range(self.n_nodes):
            grad_left = self._grad_shape[j](self.nodes[0].z)
            self.intS_shapeI_gradJ[0][0][j] = \
                grad_left * cell.faces[0].area

            grad_right = self._grad_shape[j](self.nodes[-1].z)
            self.intS_shapeI_gradJ[1][-1][j] = \
                grad_right * cell.faces[1].area


def lagrange_elements(degree: int, domain: Tuple[Vector]) -> FEBasis:
    """
    Generate the Lagrange finite elements.

    Parameters
    ----------
    degree : int, default 1.
        The finite element polynomial degree.
    domain : Tuple[Vector]

    Returns
    -------
    List[poly1d]
        Basis functions on the reference element.
    List[poly1d]
        Basis function derivatives on the reference element.
    """
    shapes, grad_shapes = [], []
    z_min, z_max = domain[0], domain[1]
    x = np.linspace(z_min, z_max, degree + 1)
    for i in range(degree + 1):
        y = [1.0 if i == j else 0.0 for j in range(degree + 1)]
        shape: poly1d = lagrange(x, y)
        shapes.append(shape)
        grad_shapes.append(shape.deriv())
    return shapes, grad_shapes
