import numpy as np

from numpy import ndarray
from scipy.sparse import csr_matrix, lil_matrix

from pyPDEs.spatial_discretization import PiecewiseContinuous
from pyPDEs.utilities.boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import HeatConductionSolver


def assemble_matrix(self: 'HeatConductionSolver') -> csr_matrix:
    """
    Assemble the heat conduction matrix.

    Returns
    -------
    csr_matrix
    """
    pwc: PiecewiseContinuous = self.discretization

    # Loop over cells
    A = lil_matrix((pwc.n_nodes,) * 2)
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]

        # Evaluate the conductivity
        k = self.k[cell.material_id]
        if callable(k):
            u = self.u[view.node_ids]
            uq = view.get_function_values(u)
            k = k(uq)
        else:
            k = np.array([k] * view.n_qpoints)

        # Loop over quadrature
        for qp in range(view.n_qpoints):
            jxw = view.jxw[qp]

            # Loop over test functions
            for i in range(view.n_nodes):
                ii = pwc.map_dof(cell, i)
                grad_i = view.grad_shape_values[i][qp]

                # Loop over trial functions
                for j in range(view.n_nodes):
                    jj = pwc.map_dof(cell, j)
                    grad_j = view.grad_shape_values[j][qp]
                    A[ii, jj] += k[qp] * grad_i.dot(grad_j) * jxw
    return self.apply_matrix_bcs(A).tocsr()


def assemble_rhs(self: 'HeatConductionSolver') -> ndarray:
    """
    Assemble the right-hand side.

    Returns
    -------
    ndarray
    """
    pwc: PiecewiseContinuous = self.discretization

    # Loop over cells
    b = np.zeros(pwc.n_nodes)
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        q = self.q[cell.material_id]

        # Loop test functions
        for i in range(view.n_nodes):
            ii = pwc.map_dof(cell, i)

            # Loop over quadrature
            for qp in range(view.n_qpoints):
                b[ii] += q * view.shape_values[i][qp] * view.jxw[qp]
    return self.apply_vector_bcs(b)


def apply_matrix_bcs(self: 'HeatConductionSolver',
                     A: csr_matrix) -> csr_matrix:
    """
    Apply the boundary conditions to a matrix.

    Parameters
    ----------
    A : csr_matrix (n_nodes,) * 2

    Returns
    -------
    csr_matrix (n_nodes,) * 2
        The input matrix with boundary conditions applied.
    """
    A = A.tolil()
    pwc: PiecewiseContinuous = self.discretization

    # Loop over boundary cells
    for bndry_cell_id in self.mesh.boundary_cell_ids:
        cell = self.mesh.cells[bndry_cell_id]
        view = pwc.fe_views[cell.id]

        # Loop over faces
        for f, face in enumerate(cell.faces):
            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)
                bc = self.boundaries[bndry_id]

                # Dirichlet boundary
                if issubclass(type(bc), DirichletBoundary):

                    # Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f])
                    for fi in range(n_face_nodes):
                        ii = pwc.map_face_dof(cell, f, fi)
                        A[ii, :] *= 0.0
                        A[ii, ii] = 1.0

                # Robin boundary
                elif issubclass(type(bc), RobinBoundary):
                    bc: RobinBoundary = bc
                    a, b = bc.a[0], bc.b[0]

                    # Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f])
                    for fi in range(n_face_nodes):
                        ni = view.face_node_mapping[f][fi]
                        ii = pwc.map_face_dof(cell, f, fi)

                        # Loop over face nodes
                        for fj in range(n_face_nodes):
                            nj = view.face_node_mapping[f][fj]
                            jj = pwc.map_face_dof(cell, f, fj)

                            mass_fij = view.intS_shapeI_shapeJ[f][ni][nj]
                            A[ii, jj] += a / b * mass_fij
    return A.tocsr()


def apply_vector_bcs(self: 'HeatConductionSolver',
                     b: ndarray) -> ndarray:
    """
    Apply the boundary conditions to the right-hand side.

    Parameters
    ----------
    b : ndarray (n_nodes,)
        The vector to apply boundary conditions to.

    Returns
    -------
    ndarray (n_nodes,)
        The input vector with boundary conditions applied.
    """
    pwc: PiecewiseContinuous = self.discretization

    # Loop over boundary cells
    for bndry_cell_id in self.mesh.boundary_cell_ids:
        cell = self.mesh.cells[bndry_cell_id]
        view = pwc.fe_views[cell.id]

        # Loop over faces
        for f, face in enumerate(cell.faces):
            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)
                bc = self.boundaries[bndry_id]

                # Dirichlet boundary
                if issubclass(type(bc), DirichletBoundary):
                    bc: DirichletBoundary = bc
                    value = bc.values[0]

                    # Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f])
                    for fi in range(n_face_nodes):
                        ii = pwc.map_face_dof(cell, f, fi)
                        b[ii] = value

                # Neumann boundary
                elif issubclass(type(bc), NeumannBoundary):
                    bc: NeumannBoundary = bc
                    value = bc.values[0]

                    # Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f])
                    for fi in range(n_face_nodes):
                        ni = view.face_node_mapping[f][fi]
                        ii = pwc.map_face_dof(cell, f, fi)
                        b[ii] += value * view.intS_shapeI[f][ni]

                # Robin boundary
                elif issubclass(type(bc), RobinBoundary):
                    bc: RobinBoundary = bc
                    f, b = bc.f[0], bc.b[0]

                    # Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f])
                    for fi in range(n_face_nodes):
                        ni = view.face_node_mapping[f][fi]
                        ii = pwc.map_face_dof(cell, f, fi)
                        b[ii] += f / b * view.intS_shapeI[f][ni]
    return b
