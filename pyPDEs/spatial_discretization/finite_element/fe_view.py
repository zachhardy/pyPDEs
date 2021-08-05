import numpy as np
from numpy import ndarray
from typing import List, Callable, TYPE_CHECKING

from ...mesh import Cell
from ...utilities.quadratures import Quadrature

if TYPE_CHECKING:
    from ...spatial_discretization import PiecewiseContinuous


class CellFEView:
    """
    Base class for a finite element cell view.
    """
    def __init__(self, fe: 'PiecewiseContinuous',
                 quadrature: Quadrature) -> None:
        self.degree: int = fe.degree
        self.face_node_mapping: List[List[int]] = None

        self.node_ids: List[int] = None
        self.nodes: List[float] = None

        self.quadrature: Quadrature = quadrature

        self.jacobian: ndarray = np.empty(0)
        self.inverse_jacobian: ndarray = np.empty(0)
        self.jxw: ndarray = np.empty(0)

        self._shape: List[Callable] = None
        self._grad_shape: List[Callable] = None

        self.shape_values: ndarray = np.empty(0)
        self.grad_shape_values: ndarray = np.empty(0)

        self.intV_shapeI: ndarray = np.empty(0)
        self.intV_shapeI_shapeJ: ndarray = np.empty(0)
        self.intV_gradI_gradJ: ndarray = np.empty(0)
        self.intV_shapeI_gradJ: ndarray = np.empty(0)

        self.intS_shapeI: ndarray = np.empty(0)
        self.intS_shapeI_shapeJ: ndarray = np.empty(0)
        self.intS_shapeI_gradJ: ndarray = np.empty(0)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_qpoints(self) -> int:
        return self.quadrature.n_qpoints

    @property
    def qpoints(self) -> ndarray:
        return self.quadrature.qpoints

    def shape_value(self, i: int, point: float) -> float:
        return self._shape[i](point)

    def shape_grad(self, i: int, point: float) -> float:
        return self._grad_shape[i](point)

    def get_function_values(self, u: ndarray) -> ndarray:
        """
        Get the function values at quadrature points from
        a solution vector at the nodes on this cell.
        """
        vals = np.zeros(self.n_qpoints)
        for qp in range(self.n_qpoints):
            for i in range(self.n_nodes):
                shape_i = self.shape_values[i, qp]
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
                grad_i = self.grad_shape_values[i, qp]
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
