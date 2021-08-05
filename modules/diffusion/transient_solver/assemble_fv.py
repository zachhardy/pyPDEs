from scipy.sparse import csr_matrix
from numpy import ndarray
from typing import TYPE_CHECKING

from pyPDEs.spatial_discretization import FiniteVolume
from ..steadystate_solver import SteadyStateSolver

if TYPE_CHECKING:
    from .transient_solver import TransientSolver


def fv_assemble_mass_matrix(self: 'TransientSolver', g: int) -> csr_matrix:
    """
    Assemble the mass matrix for time stepping.

    Parameters
    ----------
    g : int
        The group under consideration.

    Returns
    -------
    csr_matrix
    """
    fv: FiniteVolume = self.discretization

    # ================================================== Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]
        i = fv.map_dof(cell)

        value = xs.inv_velocity[g] * volume
        rows.append(i)
        cols.append(i)
        data.append(value)
    return csr_matrix((data, (rows, cols)), shape=(fv.n_nodes,) * 2)


def fv_set_transient_source(self: 'TransientSolver', g: int,
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
    SteadyStateSolver.fv_set_source(self, g, phi, *flags)

    fv: FiniteVolume = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man
    eff_dt = self.effective_dt(step)

    # ============================================= Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]
        ig = fv.map_dof(cell, 0, flux_uk_man, 0, g)

        # ======================================== Loop over groups
        for gp in range(self.n_groups):
            igp = fv.map_dof(cell, 0, flux_uk_man, 0, gp)

            # ==================== Total/prompt fission source
            # Without delayed neutrons
            if not self.use_precursors:
                self.b[ig] += xs.chi[g] * \
                              xs.nu_sigma_f[gp] * \
                              phi[igp] * volume

            # With delayed neutrons
            else:
                self.b[ig] += xs.chi_prompt[g] * \
                              xs.nu_prompt_sigma_f[gp] * \
                              phi[igp] * volume

        # ==================== Delayed fission
        if self.use_precursors:
            # =================================== Loop over precursors
            for j in range(xs.n_precursors):
                ij = cell.id * prec_uk_man.total_components + j

                prec = self.precursors[ij]
                prec_old = self.precursors_old[ij]

                coeff = xs.chi_delayed[g][j] * xs.precursor_lambda[j]
                if not self.lag_precursors:
                    coeff /= 1.0 + xs.precursor_lambda[j] * eff_dt

                # Old precursor contributions
                if step == 0:
                    self.b[ig] += coeff * prec_old * volume
                else:
                    tmp = (4.0 * prec - prec_old) / 3.0
                    self.b[ig] += coeff * tmp * volume

                # Delayed fission contributions
                if not self.lag_precursors:
                    coeff *= eff_dt * xs.precursor_yield[j]

                    # ============================== Loop over groups
                    for gp in range(self.n_groups):
                        igp = fv.map_dof(cell, 0, flux_uk_man, 0, gp)
                        self.b[ig] += \
                            coeff * xs.nu_delayed_sigma_f[gp] * \
                            phi[igp] * volume

    flags = (False, False, False, True)
    SteadyStateSolver.fv_set_source(self, g, phi, *flags)


def fv_update_precursors(self: 'TransientSolver',
                         step: int = 0) -> None:
    """
    Solve a precursor time step for finite volume discretizations.

    Parameters
    ----------
    step : int, default 0
        The step of the time step.
    """
    fv: FiniteVolume = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man
    eff_dt = self.effective_dt(step)

    # ================================================== Loop over cells
    for cell in self.mesh.cells:
        xs = self.material_xs[cell.material_id]

        # ============================================= Loop over precursors
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

            # =================================== Loop over groups
            # Contributions from delayed fission
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, flux_uk_man, 0, g)
                self.precursors[ij] += \
                    coeff * eff_dt * xs.precursor_yield[j] * \
                    xs.nu_delayed_sigma_f[g] * self.phi[ig]
