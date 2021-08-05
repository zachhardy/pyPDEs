from scipy.sparse import csr_matrix
from numpy import ndarray
from typing import TYPE_CHECKING

from pyPDEs.spatial_discretization import FiniteVolume
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import (DirichletBoundary,
                                         NeumannBoundary,
                                         RobinBoundary)

if TYPE_CHECKING:
    from .steadystate_solver import SteadyStateSolver


def fv_assemble_matrix(self: 'SteadyStateSolver', g: int) -> csr_matrix:
    """
    Assemble the diffusion matrix across all groups.
    The structure of this matrix follows the ordering
    of the unknown manager.
    """
    # ======================================== Loop over cells
    fv: FiniteVolume = self.discretization
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        dr = cell.width
        volume = cell.volume
        xs = self.material_xs[cell.material_id]
        i = fv.map_dof(cell)

        # ==================== Reaction term
        value = xs.sigma_t[g] * volume
        rows.append(i)
        cols.append(i)
        data.append(value)

        # =================================== Loop over faces
        for face in cell.faces:
            # ============================== Interior faces
            if face.has_neighbor:
                nbr_cell = self.mesh.cells[face.neighbor_id]
                nbr_dr = nbr_cell.width
                nbr_xs = self.material_xs[nbr_cell.material_id]
                j = fv.map_dof(nbr_cell)

                # Median-mesh cell width
                eff_dr = 0.5 * (dr + nbr_dr)

                # Diffusion coefficients
                diff_coeff = xs.diffusion_coeff[g]
                nbr_diff_coeff = nbr_xs.diffusion_coeff[g]

                # Effective diffusion coefficient
                tmp = dr / diff_coeff + nbr_dr / nbr_diff_coeff
                eff_diff_coeff = 2.0 * eff_dr / tmp

                # ==================== Diffusion term
                value = eff_diff_coeff / eff_dr * face.area
                rows.extend([i, i])
                cols.extend([i, j])
                data.extend([value, -value])

            # ============================== Boundary faces
            else:
                bndry_id = -1 * (face.neighbor_id + 1)
                bc = self.boundaries[bndry_id * self.n_groups + g]
                diff_coeff = xs.diffusion_coeff[g]

                # ==================== Boundary conditions
                value = 0.0
                if issubclass(type(bc), DirichletBoundary):
                    value = 2.0 * diff_coeff / dr * face.area
                elif issubclass(type(bc), RobinBoundary):
                    bc: RobinBoundary = bc
                    tmp = 2.0 * bc.b * diff_coeff + bc.a * dr
                    value = 2.0 * bc.a * diff_coeff / tmp * face.area

                rows.append(i)
                cols.append(i)
                data.append(value)
    return csr_matrix((data, (rows, cols)), shape=(fv.n_nodes,) * 2)


def fv_set_source(self: 'SteadyStateSolver', g: int, phi: ndarray,
                  apply_material_source: bool = True,
                  apply_scattering: bool = True,
                  apply_fission: bool = True,
                  apply_boundaries: bool = True) -> None:
    """
    Assemble the right-hand side of the multi-group diffusion
    equation for a finite volume discretization. This includes
    material sources, scattering sources, and fission sources.

    Parameters
    ----------
    g : int
        The group under consideration
    phi : ndarray
        A flux vector to use to compute sources.
    apply_material_source : bool, default True
    apply_scattering : bool, default True
    apply_fission : bool, default True
    apply_boundaries : bool, default True
    """
    fv: FiniteVolume = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # ======================================== Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]
        src = self.material_src[cell.material_id]
        ig = fv.map_dof(cell, 0, uk_man, 0, g)

        # ==================== Material source
        if apply_material_source:
            self.b[ig] += src.values[g] * volume

        # =================================== Loop over groups
        for gp in range(self.n_groups):
            igp = fv.map_dof(cell, 0, uk_man, 0, gp)

            # ==================== Scattering source
            if apply_scattering:
                self.b[ig] += \
                    xs.sigma_tr[gp, g] * phi[igp] * volume

            # ==================== Fission source
            if apply_fission:
                if not self.use_precursors:
                    self.b[ig] += xs.chi[g] * \
                                  xs.nu_sigma_f[gp] * \
                                  phi[igp] * volume

                else:
                    # =============== Prompt fission
                    self.b[ig] += xs.chi_prompt[g] * \
                                  xs.nu_prompt_sigma_f[gp] * \
                                  phi[igp] * volume

                    # =============== Delayed fission
                    for j in range(xs.n_precursors):
                        self.b[ig] += xs.chi_delayed[g][j] * \
                                      xs.precursor_yield[j] * \
                                      xs.nu_delayed_sigma_f[gp] * \
                                      phi[igp] * volume

        # ======================================== Loop over faces
        for face in cell.faces:
            if not face.has_neighbor and apply_boundaries:
                bndry_id = -1 * (face.neighbor_id + 1)
                bc = self.boundaries[bndry_id * self.n_groups + g]
                diff_coeff = xs.diffusion_coeff[g]
                dr = cell.width

                # ==================== Boundary conditions
                value = 0.0
                if issubclass(type(bc), DirichletBoundary):
                    bc: DirichletBoundary = bc
                    value = 2.0 * diff_coeff / dr * bc.value
                elif issubclass(type(bc), NeumannBoundary):
                    bc: NeumannBoundary = bc
                    value = bc.value
                elif issubclass(type(bc), RobinBoundary):
                    bc: RobinBoundary = bc
                    tmp = 2.0 * bc.b * diff_coeff + bc.a * dr
                    value = 2.0 * diff_coeff / tmp * bc.f

                self.b[ig] += value * face.area


def fv_compute_precursors(self: 'SteadyStateSolver') -> None:
    """
    Compute the delayed neutron precursor concentration.
    """
    if self.use_precursors and self.n_precursors > 0:
        fv: FiniteVolume = self.discretization
        flux_uk_man = self.flux_uk_man
        prec_uk_man = self.precursor_uk_man
        self.precursors *= 0.0

        # ======================================== Loop over cells
        for cell in self.mesh.cells:
            xs = self.material_xs[cell.material_id]

            # =================================== Loop over precursors
            for j in range(xs.n_precursors):
                ij = cell.id * prec_uk_man.total_components + j
                coeff = \
                    xs.precursor_yield[j] / xs.precursor_lambda[j]

                # ============================== Loop over groups
                for g in range(self.n_groups):
                    ig = fv.map_dof(cell, 0, flux_uk_man, 0, g)
                    self.precursors[ij] += \
                        coeff * xs.nu_delayed_sigma_f[g] * self.phi[ig]
