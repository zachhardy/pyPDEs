import numpy as np
from numpy import ndarray
from typing import List

from .. import SpatialDiscretization
from ...mesh import Mesh, Cell
from ...utilities import UnknownManager, Vector
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

        self.nodes: List[Vector] = None
        self.fe_views: List[CellFEView] = None

        self.create_nodes()
        self.create_cell_views()

    @property
    def n_nodes(self) -> int:
        """
        Get the number of nodes in the piecewise continuous
        spatial discretization. The number of nodes is obtained
        by summing object_{d} * (degree - 1)^d for d = 0, ..., dim.
        Object is a reference to the various types of mesh structures.

        object_{0} = vertices,
        object_{1} = lines,
        object_{2} = quadrilaterals
        object_{3} = hexahedra

        Returns
        -------
        int
        """
        n = self.mesh.n_vertices

        # ======================================== Slab meshes
        if self.mesh.type == "LINE":
            return n + self.mesh.n_cells * (self.degree - 1)

        # ======================================== Othogonal Quad meshes
        elif self.mesh.type == "ORTHO_QUAD":
            n += self.mesh.n_faces * (self.degree - 1)
            n += self.mesh.n_cells * (self.degree - 1) ** 2
            return n

        else:
            raise NotImplementedError(
                "Only line and quad meshes are implemented.")

    @property
    def grid(self) -> List[float]:
        """
        Get the list of nodes that defines the piecewise
        continuous spatial discretization. These are precomputed
        in the `PiecewiseContinuous.create_nodes()` routine.

        Returns
        -------
        List[Vector]
        """
        return self.nodes

    def create_nodes(self) -> None:
        """
        Define the node locations. For line, quad, and hex
        meshes, the nodes are defined at the vertices and
        degree - 1 evenly spaced points between all vertex
        connections. This amounts to an outer product of
        degree - 1 points in each dimension.
        """
        nodes = []

        # ======================================== Line meshes
        if self.mesh.type == "LINE":
            for cell in self.mesh.cells:
                # ========== Get left and right vertices
                v0 = self.mesh.vertices[cell.vertex_ids[0]]
                v1 = self.mesh.vertices[cell.vertex_ids[1]]

                # ========== Evenly space between vertices
                x = np.linspace(v0.z, v1.z, self.degree + 1)
                nodes.extend(x)
            nodes = np.unique(nodes)
            self.nodes = [Vector(z=node) for node in nodes]

        # ======================================== Orthogonal Quad meshes
        elif self.mesh.type == "ORTHO_QUAD":
            for cell in self.mesh.cells:
                # ========== Get bottom-left and top-right vertices
                vbl = self.mesh.vertices[cell.vertex_ids[0]]
                vtr = self.mesh.vertices[cell.vertex_ids[2]]

                # ========== Get all x, y coords for cominations
                x = np.linspace(vbl.x, vtr.x, self.degree + 1)
                y = np.linspace(vbl.y, vtr.y, self.degree + 1)

                # ========== Construct nodes row-wise
                for i in range(self.degree + 1):
                    for j in range(self.degree + 1):
                        nodes.append(Vector(x=x[j], y=y[i]))
            self.nodes = list(np.unique(nodes))

        else:
            raise NotImplementedError(
                "Only line and quad meshes are available.")


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
        Maps a node on a cell to a global DoF index. This is an
        abstract method that must be implemented in derived classes.

        Parameters
        ----------
        cell : Cell
            The cell that the node under consideration lives on.
        node : int
            The local index of the node on the cell.
        unknown_manager : UnknownManager, default None
            The unknown manager is used as a mapping from node
            to global DoF index for multi-component problems.
            If no unknown manager is supplied, it is assumed
            that it is a one component problem.
        unknown_id : int, default 0
            The unknown ID of the DoF being mapped.
        component : int, default 0
            The component of the unknown of the DoF being mapped.

        Returns
        -------
        int
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
        Maps a node on a face of a cell to a global DoF index.
        This is an abstract method that must be implemented
        in derived classes.

        Parameters
        ----------
        cell : Cell
            The cell that the node under consideration lives on.
        face_id : int
            The local index of the face on the cell. This must be
            less than `2 * dim`.
        node : int
            The local node index on the face. This must be less
            than `(degree + 1)^(dim - 1)`, where `dim` is the
            dimension of the cell and `dim - 1` is the dimension
            of the face.
        unknown_manager : UnknownManager, default None
            The unknown manager is used as a mapping from node
            to global DoF index for multi-component problems.
            If no unknown manager is supplied, it is assumed
            that it is a one component problem.
        unknown_id : int, default 0
            The unknown ID of the DoF being mapped.
        component : int, default 0
            The component of the unknown of the DoF being mapped.

        Returns
        -------
        int
        """
        view = self.fe_views[cell.id]
        node = view.face_node_mapping[face_id][node]
        return self.map_dof(cell, node, unknown_manager,
                            unknown_id, component)

    @staticmethod
    def zero_dirichlet_row(row: int, rows: List[int],
                           data: List[float]) -> None:
        """
        Utility function for removing non-zero entries from a row
        of a yet-to-be constructed spare matrix.

        Parameters
        ----------
        row : int
            The row to zero-out
        rows : List[int]
            The current list of rows in the preconstructed
            sparse matrix.
        data : List[float]
            The current list of entries in the preconstructed
            spare matrix.
        """
        # Find indiced in lists for row ir entries
        inds = []
        for i in range(len(rows)):
            if rows[i] == row:
                inds.append(i)

        # Set entries in data to zero
        for ind in inds:
            data[ind] = 0.0
