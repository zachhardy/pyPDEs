import numpy as np
from numpy import ndarray
from typing import List, Callable, TYPE_CHECKING

from ...mesh import Cell
from ...utilities import Vector
from ...utilities.quadratures import Quadrature

if TYPE_CHECKING:
    from ...spatial_discretization import PiecewiseContinuous

VecFlt = List[float]
MatFlt = List[VecFlt]
VecVec3 = List[Vector]
MatVec3 = List[VecVec3]


class CellFEView:
    """Base class for a finite element cell view.

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
    """


    def __init__(self, fe: 'PiecewiseContinuous',
                 quadrature: Quadrature,
                 face_quadrature: Quadrature = None) -> None:
        """CellFEView constructor.

        Parameters
        ----------
        fe : PiecewiseContinuous
            The discretization that the `CellFEView` is being
            attached to.
        quadrature : Quadrature
            A quadrature set for integrating over a cell.
        face_quadrature : Quadrature
            A quadrature set for integrating over a face.
        cell : Cell
            The cell that this `CellFEView` is based off of.
        """

        self.degree: int = fe.degree
        self.face_node_mapping: List[List[int]] = None

        self.node_ids: List[int] = None
        self.nodes: List[Vector] = None

        self.quadrature: Quadrature = quadrature
        self.face_quadrature: Quadrature = face_quadrature

        self.shape_values: List[List[float]] = None
        self.grad_shape_values: List[List[Vector]] = None

        self.intV_shapeI: VecFlt = None
        self.intV_shapeI_shapeJ: MatFlt = None
        self.intV_gradI_gradJ: MatFlt = None
        self.intV_shapeI_gradJ: MatVec = None

        self.intS_shapeI: List[VecFlt] = None
        self.intS_shapeI_shapeJ: List[MatFlt] = None
        self.intS_shapeI_gradJ: List[MatVec] = None

    @property
    def n_nodes(self) -> int:
        """Get the number of nodes the cell.

        Returns
        -------
        int
        """
        return len(self.node_ids)

    @property
    def n_qpoints(self) -> int:
        """Get the number of volumetric quadrature points

        Returns
        -------
        int
        """
        return self.quadrature.n_qpoints

    @property
    def n_face_qpoints(self) -> int:
        """Get the number of face quadrature points.

        Returns
        -------
        int
        """
        if self.face_quadrature:
            return self.face_quadrature.n_qpoints

    def get_function_values(self, u: List[float]) -> ndarray:
        """
        Get the function values at quadrature points from
        a solution vector at the nodes on this cell.

        Parameters
        ----------
        u : List[float] or ndarray
            A solution defined on the nodes of this cell.

        Returns
        -------
        ndarray (n_qpoints,)
        """
        if len(u) != self.n_nodes:
            raise ValueError("u must have exactly n_nodes entries.")

        vals = np.zeros(self.n_qpoints)
        for qp in range(self.n_qpoints):
            for i in range(self.n_nodes):
                shape_i = self.shape_values[i][qp]
                vals[qp] += shape_i * u[i]
        return vals

    def get_function_grads(self, u: ndarray) -> ndarray:
        """
        Get the function gradients at quadrature points from
        a solution vector at the nodes on this cell.

        Parameters
        ----------
        u : List[float] or ndarray
            A solution defined on the nodes of this cell.

        Returns
        -------
        ndarray (n_qpoints,), type Vector
        """
        if len(u) != self.n_nodes:
            raise ValueError("u must have exactly n_nodes entries.")

        vals = np.zeros(self.n_qpoints)
        for qp in range(self.n_qpoints):
            for i in range(self.n_nodes):
                grad_i = self.grad_shape_values[i][qp]
                vals[qp] += grad_i * u[i]
        return vals

    def map_reference_to_global(self, point: Vector) -> Vector:
        """Map a point from the reference cell to the real cell.

        This is an abstract method and must be implemented
        in derived classes.

        Parameters
        ----------
        point : Vector
            A point in the reference cell.

        Returns
        -------
        Vector
            The mapped point in the real cell.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement {cls_name}."
            f"map_reference_to_global.")

    def compute_quadrature_data(self, cell: Cell) -> None:
        """Compute the quadrature point related data.

        This includes quantities such as the quadrature weights
        multiplied by the coordinate transformation Jacobian and
        shape function and shape function gradient evaluations.

        This is an abstract method and must be implemented
        in derived classes.

        Parameters
        ----------
        cell : Cell
            The cell this `CellFEView` is based off of.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement {cls_name}."
            f"compute_quadrature_data.")

    def compute_integral_data(self, cell: Cell) -> None:
        """Compute finite element integral data.

        This is an abstract method and must be implemented in derived
        classes.

        Parameters
        ----------
        cell : Cell
            The cell this `CellFEView` is based off of.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement {cls_name}."
            f"compute_integral_data.")
