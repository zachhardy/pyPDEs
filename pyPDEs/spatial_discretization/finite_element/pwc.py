import numpy as np
from numpy import ndarray
from typing import List

from .. import SpatialDiscretization
from ...mesh import Mesh, Cell
from ...utilities import UnknownManager
from ...utilities.quadratures import GaussLegendre
from .fe_view import CellFEView
from .fe_slab_view import SlabFEView


class PiecewiseContinuous(SpatialDiscretization):
    """
    Piecwise continuous finite element discretization.
    """
    def __init__(self, mesh: Mesh, degree: int = 1,
                 order: int = None) -> None:
        super().__init__(mesh)
        self.type = "PWC"
        self.degree: int = degree
        if not order:
            order = 2 * degree

        self.quadrature: GaussLegendre = GaussLegendre(order)

        self.nodes: ndarray = None
        self.fe_views: List[CellFEView] = None

        self.create_nodes()
        self.create_cell_views()

    @property
    def n_nodes(self) -> int:
        if self.dim == 1:
            return len(self.mesh.cells) * self.degree + 1
        else:
            raise NotImplementedError(
                f"Only 1D discretizations are implemented.")

    @property
    def grid(self) -> List[float]:
        return self.nodes

    def create_nodes(self) -> None:
        """
        Define the node locations .
        """
        self.nodes = []

        if self.dim == 1:
            for cell in self.mesh.cells:
                v0 = self.mesh.vertices[cell.vertex_ids[0]]
                v1 = self.mesh.vertices[cell.vertex_ids[1]]
                x = np.linspace(v0, v1, self.degree + 1)
                self.nodes.extend(x)
            self.nodes = np.unique(self.nodes)
        else:
            raise NotImplementedError(
                f"Only 1D discretizations are implemented.")

    def create_cell_views(self) -> None:
        """
        Create the finite element cell views.
        """
        self.fe_views = []
        for cell in self.mesh.cells:
            if cell.cell_type == "SLAB":
                view: CellFEView = SlabFEView(
                    self, self.quadrature, cell)
                self.fe_views.append(view)
            else:
                raise NotImplementedError(
                    f"Only slabs have been implemented.")

    def map_dof(self, cell: Cell, node: int,
                unknown_manager: UnknownManager = None,
                unknown_id: int = 0, component: int = 0) -> int:
        """
        Maps a node on a cell to a dof. Unknown managers
        are used for multi-component problems and are used
        to map to specific unknowns and components.
        """
        view = self.fe_views[cell.id]
        node_id = view.node_ids[node]
        if not unknown_manager:
            return node_id
        else:
            uk_man = unknown_manager
            n_unknowns = uk_man.total_components
            block_id = uk_man.map_unknown(unknown_id, component)
            if unknown_manager.storage_method == "NODAL":
                return node_id * n_unknowns + block_id
            else:
                return self.n_nodes * block_id + node_id

    def map_face_dof(self, cell: Cell, face_id: int, node: int = 0,
                     unknown_manager: UnknownManager = None,
                     unknown_id: int = 0, component: int = 0) -> int:
        """
        Map a node on a face on a cell to a dof. The face node is mapped
        to the equivalent cell node and then `map_dof` is called.
        """
        view = self.fe_views[cell.id]
        node = view.face_node_mapping[face_id][node]
        return self.map_dof(cell, node, unknown_manager,
                            unknown_id, component)

    @staticmethod
    def zero_dirichlet_row(row: int, rows: list, data: list):
        # Find indiced in lists for row ir entries
        inds = []
        for i in range(len(rows)):
            if rows[i] == row:
                inds.append(i)

        # Set entries in data to zero
        for ind in inds:
            data[ind] = 0.0
