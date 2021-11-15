import numpy as np

from numpy import ndarray
from scipy.sparse import csr_matrix, lil_matrix

from pyPDEs.spatial_discretization import PiecewiseContinuous
from pyPDEs.utilities.boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def _pwc_diffusion_matrix(self: 'SteadyStateSolver') -> csr_matrix:
    """
    Assemble the multigroup diffusion matrix.

    This routine assembles the diffusion plus interaction matrix
    for all groups according to the DoF ordering of `phi_uk_man`.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    A = lil_matrix((pwc.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.cellwise_xs[cell.id]

        # Loop over groups
        for g in range(self.n_groups):
            D = xs.D[g]
            sig_t = xs.sigma_t[g]
            B_sq = xs.B_sq[g]

            # Loop over nodes
            for i in range(view.n_nodes):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)
                for j in range(view.n_nodes):
                    jg = pwc.map_dof(cell, j, uk_man, 0, g)

                    mass_ij = view.intV_shapeI_shapeJ[i][j]
                    stiff_ij = view.intV_gradI_gradJ[i][j]
                    A[ig, jg] += D * stiff_ij + \
                                 (sig_t + D*B_sq) * mass_ij
    return A.tocsr()


def _pwc_scattering_matrix(self: 'SteadyStateSolver') -> csr_matrix:
    """
    Assemble the multigroup scattering matrix.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    A = lil_matrix((pwc.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        # Loop over to/from groups
        for g in range(self.n_groups):
            for gp in range(self.n_groups):
                sig_s = xs.transfer_matrix[0][gp][g]

                # Loop over nodes
                for i in range(view.n_nodes):
                    ig = pwc.map_dof(cell, i, uk_man, 0, g)
                    for j in range(view.n_nodes):
                        jgp = pwc.map_dof(cell, j, uk_man, 0, gp)

                        mass_ij = view.intV_shapeI_shapeJ[i][j]
                        A[ig, jgp] += sig_s * mass_ij
    return A.tocsr()


def _pwc_prompt_fission_matrix(self: 'SteadyStateSolver') -> csr_matrix:
    """
    Assemble the prompt multigroup fission matrix.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man
    use_prompt = self.use_precursors

    # Loop over cells
    A = lil_matrix((pwc.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]
        if not xs.is_fissile:
            continue

        # Loop over groups
        for g in range(self.n_groups):
            chi = xs.chi[g]
            if self.use_precursors:
                chi = xs.chi_prompt[g]

            # Loop over groups
            for gp in range(self.n_groups):
                nu_sig_f = xs.nu_sigma_f[gp]
                if self.use_precursors:
                    nu_sig_f = xs.nu_prompt_sigma_f[gp]

                # Loop over nodes
                for i in range(view.n_nodes):
                    ig = pwc.map_dof(cell, i, uk_man, 0, g)
                    for j in range(view.n_nodes):
                        jgp = pwc.map_dof(cell, j, uk_man, 0, gp)
                        mass_ij = view.intV_shapeI_shapeJ[i][j]
                        A[ig, jgp] += chi * nu_sig_f * mass_ij
    return A.tocsr()


def _pwc_delayed_fission_matrix(self: 'SteadyStateSolver') -> csr_matrix:
    """
    Assemble the multigroup fission matrix.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    A = lil_matrix((pwc.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]
        if not xs.is_fissile:
            continue

        # Loop over precursors
        for p in range(xs.n_precursors):
            gamma = xs.precursor_yield[p]

            # Loop over groups
            for g in range(self.n_groups):
                chi_d = xs.chi_delayed[g][p]

                # Loop over groups
                for gp in range(self.n_groups):
                    nu_d_sig_f = xs.nu_delayed_sigma_f[gp]

                    # Loop over nodes
                    for i in range(view.n_nodes):
                        ig = pwc.map_dof(cell, i, uk_man, 0, g)
                        for j in range(view.n_nodes):
                            jgp = pwc.map_dof(cell, j, uk_man, 0, gp)

                            mass_ij = view.intV_shapeI_shapeJ[i][j]
                            A[ig, jgp] += chi_d * gamma * nu_d_sig_f * mass_ij
    return A.tocsr()


def _pwc_set_source(self: 'SteadyStateSolver') -> ndarray:
    """
    Assemble the right-hand side.

    Returns
    -------
    ndarray (n_cells * n_groups)
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    b = np.zeros(pwc.n_dofs(uk_man))
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]

        src_id = self.matid_to_src_map[cell.material_id]
        if src_id < 0:
            continue

        src = self.material_src[src_id]

        # Loop over groups
        for g in range(self.n_groups):
            q = src.values[g]

            # Loop over nodes
            for i in range(view.n_nodes):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)
                b[ig] += q * view.intV_shapeI[i]
    return b


def _pwc_compute_precursors(self: 'SteadyStateSolver') -> None:
    """
    Compute the delayed neutron precursor concentrations.
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    self.precursors *= 0.0
    for cell in self.mesh.cells:
        volume = cell.volume
        view = pwc.fe_views[cell.id]

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]
        if not xs.is_fissile:
            continue

        # Loop over precursors
        for p in range(xs.n_precursors):
            ip = self.max_precursors * cell.id + p
            lambda_p = xs.precursor_lambda[p]
            yield_p = xs.precursor_yield[p]

            # Loop over groups
            for g in range(self.n_groups):
                nu_d_sig_f = xs.nu_delayed_sigma_f[g]

                # Loop over nodes
                for i in range(view.n_nodes):
                    ig = pwc.map_dof(cell, i, uk_man, 0, g)
                    self.precursors[ip] += \
                        yield_p / lambda_p * nu_d_sig_f * self.phi[ig] * \
                        view.intV_shapeI[i] / volume


def _pwc_apply_matrix_bcs(self: 'SteadyStateSolver',
                          A: csr_matrix) -> csr_matrix:
    """
    Apply the boundary conditions to a matrix.

    Parameters
    ----------
    A : csr_matrix (n_cells * n_groups,) * 2

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
        The input matrix with boundary conditions applied.
    """
    A = A.tolil()
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

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

                        # Loop over groups
                        for g in range(self.n_groups):
                            ig = pwc.map_face_dof(cell, f, fi, uk_man, 0, g)
                            A[ig, :] *= 0.0
                            A[ig, ig] = 1.0

                # Robin boundary
                elif issubclass(type(bc), RobinBoundary):
                    bc: RobinBoundary = bc

                    # Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f])
                    for fi in range(n_face_nodes):
                        ni = view.face_node_mapping[f][fi]

                        # Loop over face nodes
                        for fj in range(n_face_nodes):
                            nj = view.face_node_mapping[f][fj]

                            # Loop over groups
                            for g in range(self.n_groups):
                                ig = pwc.map_face_dof(cell, f, fi, uk_man, 0, g)
                                jg = pwc.map_face_dof(cell, f, fj, uk_man, 0, g)
                                mass_fij = view.intS_shapeI_shapeJ[f][ni, nj]
                                A[ig, jg] += bc.a[g]/bc.b[g] * mass_fij
    return A.tocsr()


def _pwc_apply_vector_bcs(self: 'SteadyStateSolver', b: ndarray) -> ndarray:
    """
    Apply the boundary conditions to the right-hand side.

    Parameters
    ----------
    b : ndarray (n_cells * n_groups)
        The vector to apply boundary conditions to.

    Returns
    -------
    ndarray (n_cells * n_groups)
        The input vector with boundary conditions applied.
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

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

                    # Loopp over face nodes
                    n_face_nodes = len(view.face_node_mapping[f])
                    for fi in range(n_face_nodes):

                        # Loop over groups
                        for g in range(self.n_groups):
                            ig = pwc.map_face_dof(cell, f, fi, uk_man, 0, g)
                            b[ig] = bc.values[g]

                # Neumann boundary
                elif issubclass(type(bc), NeumannBoundary):
                    bc: NeumannBoundary = bc

                    # Loopp over face nodes
                    n_face_nodes = len(view.face_node_mapping[f])
                    for fi in range(n_face_nodes):
                        ni = view.face_node_mapping[f][fi]
                        intS_shapeI = view.intS_shapeI[f][ni]

                        # Loop over groups
                        for g in range(self.n_groups):
                            ig = pwc.map_face_dof(cell, f, fi, uk_man, 0, g)
                            b[ig] += bc.values[g] * intS_shapeI

                # Robin boundary
                elif issubclass(type(bc), RobinBoundary):
                    bc: RobinBoundary = bc

                    # Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f])
                    for fi in range(n_face_nodes):
                        ni = view.face_node_mapping[f][fi]
                        intS_shapeI = view.intS_shapeI[f][ni]

                        # Loop over groups
                        for g in range(self.n_groups):
                            ig = pwc.map_face_dof(cell, f, fi, uk_man, 0, g)
                            b[ig] += bc.f[g]/bc.b[g] * intS_shapeI
    return b
