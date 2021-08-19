from typing import Tuple, List

import numpy as np
from numpy import poly1d
from scipy.interpolate import lagrange
from typing import TYPE_CHECKING

from ...mesh import Cell
from ...utilities.quadratures import LineQuadrature
from ...utilities import Vector
from .fe_view import CellFEView

if TYPE_CHECKING:
    from ...spatial_discretization import PiecewiseContinuous

FEBasis = Tuple[List[poly1d], List[poly1d]]


class SlabFEView(CellFEView):
    """Finite element slab view.

    Attributes
    ----------
    degree : int
        The finite element polynomial degree.
    face_node_mapping : List[List[int]]
        A map the maps each face node ID to its corresponding cell node.

        The outer list corresponds to the faces of the cells and
        the inner list corresponds to the vertices on each face.

        face_node_mapping[face_id][i] will return the cell node ID
        corresponding to node `i` on face `face_id`.
    node_ids : List[int]
        The IDs of the nodes that live on the cell used to construct
        this object.
    nodes : List[Vector]
        The coordinates of the nodes that live on the cell used to
        construct this object.
    quadrature : Quadrature
        A dim-dimensional quadrature formula.
    face_quadrature : Quadrature
        A (dim-1)-dimensional quadrature formula.
    shape_values : ndarray (n_nodes, n_qpoints)
        All shape functions evaluated at all quadrature points.
    grad_shape_values : ndarray (n_nodes, n_qpoints)
        All shape function gradients evaluated at all quadrature points.
        Note that this entries are of type Vector.
    intV_shapeI : ndarray (n_nodes,)
        Integrals of each shape function over the cell.
    intV_shapeI_shapeJ : ndarray (n_nodes, n_nodes)
        Integrals of shape function i times shape function j
        for i, j = 0, ..., n_nodes over the cell.
    intV_gradI_gradJ : ndarray (n_nodes, n_nodes)
        Integrals of shape function i gradient dotted with
        shape function j gradient for i, j = 0, ..., n_nodes
        over the cell.
    intV_shapeI_gradJ : ndarray (n_nodes, n_nodes), type Vector
        Integrals of shape function i times shape function j gradient
        for i, j = 0, ..., n_nodes over the cell.
    intS_shapeI : List[ndarray (n_nodes,)]
        Integrals of each shape function over each face.
    intS_shapeI_shapeJ : List[ndarray (n_nodes, n_nodes)]
        Integrals of shape function i times shape function j
        for i, j = 0, ..., n_nodes over each face.
    intS_shapeI_gradJ : List[ndarray (n_nodes, n_nodes)], type Vector
        Integrals of shape function i times shape function gradient j
        for i, j = 0, ..., n_nodes over each face.
    coord_sys : {"CARTESIAN", "CYLINDRICAL", "SPHERICAL"}
        The coordinate system of the cell.
    v0 : Vector
        The left vertex of the slab cell. This is used
        for mapping between the reference and real cell.
    h : float
        The cell Jacobian, which is simply the cell width
        divided by the quadrature width.
    """
    def __init__(self, fe: 'PiecewiseContinuous',
                 quadrature: LineQuadrature,
                 cell: Cell) -> None:
        """SlabFEView constructor.

        Parameters
        ----------
        fe : PiecewiseContinuous
            The discretization that the `CellFEView` is being
            attached to.
        quadrature : LineQuadrature
            A quadrature set for integrating on a line.
        cell : Cell
            The cell that this `CellFEView` is based off of.
        """
        super().__init__(fe, quadrature)
        self.coord_sys: str = cell.coord_sys

        v0 = fe.mesh.vertices[cell.vertex_ids[0]]
        v1 = fe.mesh.vertices[cell.vertex_ids[1]]
        self.v0: Vector = v0

        domain = quadrature.domain
        self.h: float = (v1 - v0).norm() / (domain[1] - domain[0])

        # Node IDs and face-to-node mapping
        c, degree = cell.id, fe.degree
        self.node_ids = [c * degree + i for i in range(degree + 1)]
        self.face_node_mapping = [[0], [degree]]

        # Get the node locations
        self.nodes = []
        for nid in self.node_ids:
            self.nodes.append(fe.nodes[nid])

        tmp = lagrange_elements(degree, quadrature.domain)
        self._shape = tmp[0]
        self._grad_shape = tmp[1]

        self.compute_quadrature_data(cell)
        self.compute_integral_data(cell)

    def map_reference_to_global(self, point: Vector) -> Vector:
        """Map a point from the reference cell to the real cell.

        Parameters
        ----------
        point : Vector
            A point in the reference cell.

        Returns
        -------
        Vector
            The mapped point in the real cell.
        """
        domain = self.quadrature.domain
        if not min(domain) < point.z < max(domain):
            raise ValueError(
                f"Provided point not in volume quadrature domain.")
        return self.v0 + self.h * (point - Vector(z=min(domain)))

    def shape_value(self, i: int, point: Vector) -> float:
        """Evaluate shape function `i` at `point`.

        Parameters
        ----------
        i : int
            The local node index.
        point : Vector
            A point in reference coordinates used to evaluate the
            shape function.

        Returns
        -------
        float
        """
        return self._shape[i](point.z)

    def grad_shape_value(self, i: int, point: Vector) -> Vector:
        """Evaluate shape function `i` gradient at `point`

        Parameters
        ----------
        i : int
            The local node index.
        point : Vector
            A point in reference coordinates used to evaluate the
            gradient of the shape function.

        Returns
        -------
        Vector
        """
        val = self._grad_shape[i](point.z) / self.h
        return Vector(z=val)

    def compute_quadrature_data(self, cell: Cell) -> None:
        """Compute the quadrature point related data.

        This includes quantities such as the quadrature weights
        multiplied by the coordinate transformation Jacobian and
        shape function and shape function gradient evaluations.

        Parameters
        ----------
        cell : Cell
            The cell this `SlabFEView` is based off of.
        """
        # =================================== Mapping data
        weights = self.quadrature.weights
        self.jxw = np.zeros(self.n_qpoints)
        for qp in range(self.n_qpoints):
            qpoint = self.quadrature.qpoints[qp]
            x = self.map_reference_to_global(qpoint).z
            self.jxw[qp] = self.h * weights[qp]
            if self.coord_sys == "CYLINDRICAL":
                self.jxw[qp] *= 2.0 * np.pi * x
            elif self.coord_sys == "SPHERICAL":
                self.jxw[qp] *= 4.0 * np.pi * x ** 2

        # =================================== Finite element data
        n_nodes, n_qpoints = self.n_nodes, self.n_qpoints
        shapes = [[0.0 for _ in range(n_qpoints)] for _ in range(n_nodes)]
        grads = [[Vector() for _ in range(n_qpoints)] for _ in range(n_nodes)]
        for qp in range(self.n_qpoints):
            qpoint = self.quadrature.qpoints[qp]
            for i in range(self.n_nodes):
                shapes[i][qp] = self.shape_value(i, qpoint)
                grads[i][qp] = self.grad_shape_value(i, qpoint)
        self.shape_values = shapes
        self.grad_shape_values = grads

    def compute_integral_data(self, cell: Cell) -> None:
        """Compute finite element integral data.

        Parameters
        ----------
        cell : Cell
            The cell this `CellFEView` is based off of.
        """
        # ======================================== Compute volume integrals
        n = self.n_nodes
        self.intV_shapeI = [0.0 for _ in range(n)]
        self.intV_shapeI_shapeJ = \
            [[0.0 for _ in range(n)] for _ in range(n)]
        self.intV_gradI_gradJ = \
            [[0.0 for _ in range(n)] for _ in range(n)]
        self.intV_shapeI_gradJ = \
            [[Vector() for _ in range(n)] for _ in range(n)]

        for qp in range(self.n_qpoints):
            jxw = self.jxw[qp]

            for i in range(self.n_nodes):
                shape_i = self.shape_values[i][qp]
                grad_shape_i = self.grad_shape_values[i][qp]

                self.intV_shapeI[i] += shape_i * jxw

                for j in range(self.n_nodes):
                    shape_j = self.shape_values[j][qp]
                    grad_shape_j = self.grad_shape_values[j][qp]

                    self.intV_shapeI_shapeJ[i][j] += \
                        shape_i * shape_j * jxw

                    self.intV_gradI_gradJ[i][j] += \
                        grad_shape_i.dot(grad_shape_j) * jxw

                    self.intV_shapeI_gradJ[i][j] += \
                        shape_i * grad_shape_j * jxw

        # ======================================== Compute surface integrals
        self.intS_shapeI = [np.zeros(n) for _ in range(2)]
        self.intS_shapeI_shapeJ = \
            [np.zeros((n, n)) for _ in range(2)]
        self.intS_shapeI_gradJ = \
            [[[Vector() for _ in range(n)] for _ in range(n)] for _ in range(2)]

        self.intS_shapeI[0][0] = cell.faces[0].area
        self.intS_shapeI[1][-1] = cell.faces[1].area

        self.intS_shapeI_shapeJ[0][0][0] = cell.faces[0].area
        self.intS_shapeI_shapeJ[1][-1][-1] = cell.faces[1].area

        for j in range(self.n_nodes):
            grad_left = Vector(z=self._grad_shape[j](self.nodes[0].z))
            self.intS_shapeI_gradJ[0][0][j] = \
               grad_left * cell.faces[0].area

            grad_right = Vector(z=self._grad_shape[j](self.nodes[-1].z))
            self.intS_shapeI_gradJ[1][-1][j] = \
                grad_right * cell.faces[1].area


def lagrange_elements(degree: int, domain: Tuple[Vector]) -> FEBasis:
    """Generate the Lagrange finite elements in one dimension.

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
