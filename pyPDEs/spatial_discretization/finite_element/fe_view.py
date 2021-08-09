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
    """
    Base class for a finite element cell view.
    """
    def __init__(self, fe: 'PiecewiseContinuous',
                 quadrature: Quadrature,
                 face_quadrature: Quadrature = None) -> None:
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
        return len(self.node_ids)

    @property
    def n_qpoints(self) -> int:
        return self.quadrature.n_qpoints

    @property
    def qpoints(self) -> List[Vector]:
        return self.quadrature.qpoints

    @property
    def n_face_qpoints(self) -> int:
        if self.face_quadrature:
            return self.face_quadrature.n_qpoints

    @property
    def face_qpoints(self) -> List[Vector]:
        if self.face_quadrature:
            return self.face_quadrature.qpoints

    def get_function_values(self, u: List[float]) -> ndarray:
        """
        Get the function values at quadrature points from
        a solution vector at the nodes on this cell.
        """
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
        """
        vals = np.zeros(self.n_qpoints)
        for qp in range(self.n_qpoints):
            for i in range(self.n_nodes):
                grad_i = self.grad_shape_values[i][qp]
                vals[qp] += grad_i * u[i]
        return vals

    def map_reference_to_global(self, point: float) -> float:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement {cls_name}."
            f"map_reference_to_global.")

    def compute_quadrature_data(self, cell: Cell) -> None:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement {cls_name}."
            f"compute_quadrature_data.")

    def compute_integral_data(self, cell: Cell) -> None:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"Subclasses must implement {cls_name}."
            f"compute_integral_data.")
