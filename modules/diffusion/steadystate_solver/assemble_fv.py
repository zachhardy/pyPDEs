from scipy.sparse import csr_matrix
from numpy import ndarray
from typing import TYPE_CHECKING

from pyPDEs.spatial_discretization import FiniteVolume
from pyPDEs.utilities import UnknownManager
from ..boundaries import DirichletBoundary, MarshakBoundary

if TYPE_CHECKING:
    from .steadystate_solver import SteadyStateSolver


def fv_assemble_matrix(self: 'SteadyStateSolver', g: int) -> csr_matrix:
    """
    Assemble the diffusion matrix across all groups.
    The structure of this matrix follows the ordering
    of the unknown manager.

    Parameters
    ----------
    g : int
        The group under consideration.

    Returns
    -------
    csr_matrix
    """
    # ======================================== Loop over cells
    fv: FiniteVolume = self.discretization
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        dr = cell.width
        volume = cell.volume
        xs = self.material_xs[cell.material_id]
        ir = fv.map_dof(cell)

        # ==================== Reaction term
        value = xs.sigma_t[g] * volume
        rows.append(ir)
        cols.append(ir)
        data.append(value)

        # =================================== Loop over faces
        for face in cell.faces:
            # ============================== Interior faces
            if not face.is_boundary:
                nbr_cell = face.get_neighbor_cell(self.mesh)
                nbr_dr = nbr_cell.width
                nbr_xs = self.material_xs[nbr_cell.material_id]
                nbr_ir = fv.map_dof(nbr_cell)

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
                rows.extend([ir, ir])
                cols.extend([ir, nbr_ir])
                data.extend([value, -value])

            # ============================== Boundary faces
            else:
                bc = self.boundaries[-1 * (face.neighbor_id + 1)]
                diff_coeff = xs.diffusion_coeff[g]

                # ==================== Boundary conditions
                value = 0.0
                if isinstance(bc, DirichletBoundary):
                    value = 2.0 * diff_coeff / dr * face.area
                elif isinstance(bc, MarshakBoundary):
                    tmp = 4.0 * diff_coeff + dr
                    value = 2.0 * diff_coeff / tmp * face.area

                rows.append(ir)
                cols.append(ir)
                data.append(value)
    return csr_matrix((data, (rows, cols)), shape=(fv.num_nodes,) * 2)


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
        ir = fv.map_dof(cell, 0, uk_man, 0, 0)

        # ==================== Material source
        if apply_material_source:
            self.b[ir + g] += src.values[g] * volume

        # =================================== Loop over groups
        for gp in range(self.num_groups):

            # ==================== Scattering source
            if apply_scattering:
                self.b[ir + g] += \
                    xs.sigma_tr[gp, g] * phi[ir + gp] * volume

            # ==================== Fission source
            if apply_fission:
                if not self.use_precursors:
                    self.b[ir + g] += xs.chi[g] * \
                                     xs.nu_sigma_f[gp] * \
                                     phi[ir + gp] * volume

                else:
                    # =============== Prompt fission
                    self.b[ir + g] += xs.chi_prompt[g] * \
                                     xs.nu_prompt_sigma_f[gp] * \
                                     phi[ir + gp] * volume

                    # =============== Delayed fission
                    for j in range(xs.num_precursors):
                        self.b[ir + g] += xs.chi_delayed[g][j] * \
                                         xs.precursor_yield[j] * \
                                         xs.nu_delayed_sigma_f[gp] * \
                                         phi[ir + gp] * volume

    # ======================================== Loop over boundary cells
    if apply_boundaries:
        for cell_id in self.mesh.boundary_cell_ids:
            cell = self.mesh.cells[cell_id]
            dr = cell.width

            xs = self.material_xs[cell.material_id]

            ir = fv.map_dof(cell, 0, uk_man, 0, 0)

            # =================================== Loop over faces
            for face in cell.faces:
                if face.is_boundary:
                    bc = self.boundaries[-1 * (face.neighbor_id + 1)]
                    diff_coeff = xs.diffusion_coeff[g]

                    # ==================== Boundary conditions
                    value = 0.0
                    if isinstance(bc, DirichletBoundary):
                        value = 2.0 * diff_coeff / dr * bc.values[g]
                    elif isinstance(bc, MarshakBoundary):
                        tmp = 4.0 * diff_coeff + dr
                        value = 2.0 * diff_coeff / tmp * bc.values[g]
                    self.b[ir + g] += value * face.area


def fv_compute_precursors(self: 'SteadyStateSolver') -> None:
    """
    Compute the delayed neutron precursor concentration.
    """
    if self.use_precursors and self.num_precursors > 0:
        fv: FiniteVolume = self.discretization
        flux_uk_man = self.flux_uk_man
        prec_uk_man = self.precursor_uk_man
        self.precursors *= 0.0

        # ======================================== Loop over cells
        for cell in self.mesh.cells:
            xs = self.material_xs[cell.material_id]
            ir = fv.map_dof(cell, 0, flux_uk_man, 0, 0)
            jr = fv.map_dof(cell, 0, prec_uk_man, 0, 0)

            # =================================== Loop over precursors
            for j in range(xs.num_precursors):
                coeff = xs.precursor_yield[j] / \
                        xs.precursor_lambda[j]

                # ============================== Loop over groups
                for g in range(self.num_groups):
                    self.precursors[jr + j] += coeff * \
                                               xs.nu_delayed_sigma_f[g] * \
                                               self.phi[ir + g]
