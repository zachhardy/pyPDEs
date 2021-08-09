from numpy import ndarray
from scipy.sparse import csr_matrix

from pyPDEs.spatial_discretization import PiecewiseContinuous
from pyPDEs.utilities.boundaries import DirichletBoundary

from ..steadystate_solver import SteadyStateSolver

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import TransientSolver


def pwc_assemble_mass_matrix(self: 'TransientSolver', g: int) -> csr_matrix:
    """
    Assemble the mass matrix for time stepping.
    """
    pwc: PiecewiseContinuous = self.discretization

    # ================================================== Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # ======================================== Loop over test functions
        for i in range(view.n_nodes):
            ii = pwc.map_dof(cell, i)
            value = xs.inv_velocity[g] * view.intV_shapeI[i]
            rows.append(ii)
            cols.append(ii)
            data.append(value)

        # ======================================== Loop over faces
        #                                          Stop on boundaries
        for f_id, face in enumerate(cell.faces):
            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)
                bc = self.boundaries[bndry_id * self.n_groups + g]

                # ========================= Dirichlet boundary
                if issubclass(type(bc), DirichletBoundary):

                    # ==================== Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f_id])
                    for fi in range(n_face_nodes):
                        ii = pwc.map_face_dof(cell, f_id, fi)
                        pwc.zero_dirichlet_row(ii, rows, data)
    return csr_matrix((data, (rows, cols)), shape=(pwc.n_nodes,) * 2)


def pwc_set_transient_source(self: 'TransientSolver', g: int,
                             phi: ndarray, step: int = 0):
    """
    Set the transient source.

    Parameters
    ----------
    g : int
        The group under consideration.
    phi : ndarray
        The solution vector to compute sources from.
    step : int, default 0
        The step of the time step.
    """
    flags = (True, True, False, False)
    SteadyStateSolver.pwc_set_source(self, g, phi, *flags)

    pwc: PiecewiseContinuous = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man
    eff_dt = self.effective_dt(step)

    # ============================================= Loop over cells
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # ======================================== Loop over test functions
        for i in range(view.n_nodes):
            ig = pwc.map_dof(cell, i, flux_uk_man, 0, g)

            # =================================== Loop over trial functions
            for k in range(view.n_nodes):
                mass_ik = view.intV_shapeI_shapeJ[i][k]

                # ============================== Loop over groups
                for gp in range(self.n_groups):
                    kgp = pwc.map_dof(cell, k, flux_uk_man, 0, gp)

                    # ==================== Total/prompt fission source
                    # Without delayed neutrons
                    if not self.use_precursors:
                        self.b[ig] += xs.chi[g] * \
                                      xs.nu_sigma_f[gp] * \
                                      phi[kgp] * mass_ik

                    # With delayed neutrons
                    else:
                        self.b[ig] += xs.chi_prompt[g] * \
                                      xs.nu_prompt_sigma_f[gp] * \
                                      phi[kgp] * mass_ik

        # ==================== Delayed fission
        if self.use_precursors:
            for j in range(xs.n_precursors):
                ij = cell.id * prec_uk_man.total_components + j
                prec = self.precursors[ij]
                prec_old = self.precursors_old[ij]

                coeff = xs.chi_delayed[g][j] * xs.precursor_lambda[j]
                if not self.lag_precursors:
                    coeff /= 1.0 + xs.precursor_lambda[j] * eff_dt

                for i in range(view.n_nodes):
                    ig = pwc.map_dof(cell, i, flux_uk_man, 0, g)
                    intV_shapeI = view.intV_shapeI[i]

                    # Old precursor contributions
                    if step == 0:
                        self.b[ig] += coeff * prec_old * intV_shapeI
                    else:
                        tmp = (4.0 * prec - prec_old) / 3.0
                        self.b[ig] += coeff * tmp * intV_shapeI

                    # Delayed fission contributions
                    if not self.lag_precursors:
                        coeff *= eff_dt * xs.precursor_yield[j]

                        for k in range(view.n_nodes):
                            mass_ik = view.intV_shapeI_shapeJ[i][k]

                            for gp in range(self.n_groups):
                                kgp = pwc.map_dof(cell, k, flux_uk_man, 0, gp)
                                self.b[ig] += \
                                    coeff * xs.nu_delayed_sigma_f[gp] * \
                                    phi[kgp] * mass_ik / cell.volume

    flags = (False, False, False, True)
    SteadyStateSolver.pwc_set_source(self, g, phi, *flags)


def pwc_update_precursors(self: 'TransientSolver',
                          step: int = 0) -> None:
    """
    Solve a precursor time step for finite volume discretizations.

    Parameters
    ----------
    step : int, default 0
        The step of the time step.
    """
    pwc: PiecewiseContinuous = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man
    eff_dt = self.effective_dt(step)

    # ======================================== Loop over cells
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # =================================== Loop over precursors
        for j in range(xs.n_precursors):
            ij = cell.id * prec_uk_man.total_components + j
            prec = self.precursors[ij]
            prec_old = self.precursors_old[ij]

            # Contributions from previous time step
            coeff = 1.0 / (1.0 + xs.precursor_lambda[j] * eff_dt)
            if step == 0:
                self.precursors[ij] = coeff * prec_old
            else:
                tmp = (4.0 * prec - prec_old) / 3.0
                self.precursors[ij] = coeff * tmp

            # ============================== Loop over trial functions
            for i in range(view.n_nodes):
                intV_shapeI = view.intV_shapeI[i]

                # =================================== Loop over groups
                # Contributions from delayed fission
                for g in range(self.n_groups):
                    ig = pwc.map_dof(cell, 0, flux_uk_man, 0, g)
                    self.precursors[ij] += coeff * eff_dt * \
                                           xs.precursor_yield[j] * \
                                           xs.nu_delayed_sigma_f[g] * \
                                           self.phi[ig] * intV_shapeI / \
                                           cell.volume
