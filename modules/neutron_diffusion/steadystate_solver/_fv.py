"""
All routines of SteadyStateSolver that perform computations
using a finite volume spatial discretization.
"""
import numpy as np

from numpy import ndarray
from scipy.sparse import csr_matrix, lil_matrix

from pyPDEs.spatial_discretization import FiniteVolume
from pyPDEs.utilities.boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def _fv_diffusion_matrix(self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the multigroup diffusion matrix.

    This routine assembles the diffusion plus interaction matrix
    for all groups according to the DoF ordering of `phi_uk_man`.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    A = lil_matrix((fv.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.cellwise_xs[cell.id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            A[ig, ig] += xs.sigma_t[g] * volume
            A[ig, ig] += xs.D[g] * xs.B_sq[g] * volume

        # Loop over faces
        for face in cell.faces:
            if face.has_neighbor:  # interior faces
                nbr_cell = self.mesh.cells[face.neighbor_id]
                nbr_xs = self.material_xs[nbr_cell.material_id]

                # Geometric quantities
                d_pn = (cell.centroid - nbr_cell.centroid).norm()
                d_pf = (cell.centroid - face.centroid).norm()
                w = d_pf / d_pn

                # Loop over groups
                for g in range(self.n_groups):
                    ig = fv.map_dof(cell, 0, uk_man, 0, g)
                    jg = fv.map_dof(nbr_cell, 0, uk_man, 0, g)

                    # Face diffusion coefficient
                    D_f = (w/xs.D[g] + (1.0 - w)/nbr_xs.D[g]) ** (-1)

                    A[ig, ig] += D_f/d_pn * face.area
                    A[ig, jg] -= D_f/d_pn * face.area
    return A.tocsr()


def _fv_scattering_matrix(self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the multigroup scattering matrix.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    A = lil_matrix((fv.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            for gp in range(self.n_groups):
                igp = fv.map_dof(cell, 0, uk_man, 0, gp)
                A[ig, igp] += xs.transfer_matrix[0][gp][g] * volume
    return A.tocsr()


def _fv_prompt_fission_matrix(self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the prompt multigroup fission matrix.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    A = lil_matrix((fv.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]
        if not xs.is_fissile:
            continue

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            for gp in range(self.n_groups):
                igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                if not self.use_precursors:  # total fission
                    A[ig, igp] += xs.chi[g] * \
                                  xs.nu_sigma_f[gp] * \
                                  volume

                else:  # prompt fission
                    A[ig, igp] += xs.chi_prompt[g] * \
                                  xs.nu_prompt_sigma_f[gp] * \
                                  volume
    return A.tocsr()


def _fv_delayed_fission_matrix(self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the multigroup fission matrix.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    A = lil_matrix((fv.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]
        if not xs.is_fissile:
            continue

        # Loop over to/from groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            for gp in range(self.n_groups):
                igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                # Loop over precursors
                for j in range(xs.n_precursors):
                    A[ig, igp] += xs.chi_delayed[g][j] * \
                                  xs.precursor_yield[j] * \
                                  xs.nu_delayed_sigma_f[gp] * \
                                  volume
    return A.tocsr()


def _fv_set_source(self: "SteadyStateSolver") -> ndarray:
    """Assemble the right-hand side.

    Returns
    -------
    ndarray (n_cells * n_groups)
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    b = np.zeros(fv.n_dofs(uk_man))
    for cell in self.mesh.cells:
        volume = cell.volume
        src = self.material_src[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            b[ig] += src.values[g] * volume
    return b


def _fv_compute_precursors(self: "SteadyStateSolver") -> None:
    """Compute the delayed neutron precursor concentrations.
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    self.precursors *= 0.0
    for cell in self.mesh.cells:
        xs = self.material_xs[cell.material_id]
        if not xs.is_fissile:
            continue

        # Loop over precursors
        for p in range(xs.n_precursors):
            ip = self.max_precursors * cell.id + p
            lambda_p = xs.precursor_lambda[p]
            yield_p = xs.precursor_yield[p]

            # Loop over groups
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, uk_man, 0, g)
                nu_d_sig_f = xs.nu_delayed_sigma_f[g]
                self.precursors[ip] += \
                    yield_p / lambda_p * nu_d_sig_f * self.phi[ig]


def _fv_apply_matrix_bcs(self: "SteadyStateSolver",
                         A: csr_matrix) -> csr_matrix:
    """Apply the boundary conditions to a matrix.

    Parameters
    ----------
    A : csr_matrix (n_cells * n_groups,) * 2

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
        The input matrix with boundary conditions applied.
    """
    A = A.tolil()
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over boundary cells
    for bndry_cell_id in self.mesh.boundary_cell_ids:
        cell = self.mesh.cells[bndry_cell_id]
        xs = self.material_xs[cell.material_id]

        # Loop over faces
        for face in cell.faces:
            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)
                bc = self.boundaries[bndry_id]

                # Geometric quantities
                d_pf = (cell.centroid - face.centroid).norm()

                # Loop over groups
                for g in range(self.n_groups):
                    ig = fv.map_dof(cell, 0, uk_man, 0, g)

                    # Dirichlet conditions
                    if issubclass(type(bc), DirichletBoundary):
                        A[ig, ig] += xs.D[g] / d_pf * face.area

                    # Robin conditions
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc
                        tmp = bc.a[g] * d_pf - bc.b[g] * xs.D[g]
                        A[ig, ig] += bc.a[g] * xs.D[g] / tmp * face.area
    return A.tocsr()


def _fv_apply_vector_bcs(self: "SteadyStateSolver", b: ndarray) -> ndarray:
    """Apply the boundary conditions to the right-hand side.

    Parameters
    ----------
    b : ndarray (n_cells * n_groups)
        The vector to apply boundary conditions to.

    Returns
    -------
    ndarray (n_cells * n_groups)
        The input vector with boundary conditions applied.
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over boundary cells
    for bndry_cell_id in self.mesh.boundary_cell_ids:
        cell = self.mesh.cells[bndry_cell_id]
        xs = self.material_xs[cell.material_id]

        # Loop over faces
        for face in cell.faces:
            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)
                bc = self.boundaries[bndry_id]

                # Geometric information
                d_pf = (cell.centroid - face.centroid).norm()

                # Loop over groups
                for g in range(self.n_groups):
                    ig = fv.map_dof(cell, 0, uk_man, 0, g)

                    # Dirichlet conditions
                    if issubclass(type(bc), DirichletBoundary):
                        bc: DirichletBoundary = bc
                        b[ig] += xs.D[g]/d_pf*bc.values[g] * face.area

                    # Neumann conditions
                    elif issubclass(type(bc), NeumannBoundary):
                        bc: NeumannBoundary = bc
                        b[ig] += bc.values[g] * face.area

                    # Robin conditions
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc
                        tmp = bc.a[g]*d_pf - bc.b[g]*xs.D[g]
                        b[ig] += -bc.b[g]*xs.D[g]/tmp * bc.f[g] * face.area
    return b
