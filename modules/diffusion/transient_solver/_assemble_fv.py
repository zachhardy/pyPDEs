from numpy import ndarray

from scipy.sparse import csr_matrix

from pyPDEs.spatial_discretization import FiniteVolume
from pyPDEs.utilities import UnknownManager

from ..steadystate_solver import SteadyStateSolver

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


def _fv_assemble_mass_matrix(self: "TransientSolver") -> csr_matrix:
    """Assemble the multi-group mass matrix.

    Returns
    -------
    csr_matrix
    """
    fv: FiniteVolume = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    rows, data = [], []
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)

            value = xs.inv_velocity[g] * volume
            rows.append(ig)
            data.append(value)

    n_dofs = fv.n_dofs(uk_man)
    return csr_matrix((data, (rows, rows)), shape=(n_dofs,) * 2)


def _fv_assemble_transient_fission_matrix(
        self: "TransientSolver", step: int = 0) -> csr_matrix:
    """Assemble the transient multi-group fission matrix.

    Returns
    -------
    csr_matrix
    """
    fv: FiniteVolume = self.discretization
    uk_man: UnknownManager = self.flux_uk_man
    eff_dt = self.effective_dt(step)

    # Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            for gp in range(self.n_groups):
                igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                # Total fission
                if not self.use_precursors:
                    coeff = xs.chi[g] * xs.nu_sigma_f[gp]
                # Prompt fission
                else:
                    coeff = xs.chi_prompt[g] * \
                            xs.nu_prompt_sigma_f[gp]

                value = coeff * volume
                rows.append(ig)
                cols.append(igp)
                data.append(value)

            # Delayed fission
            if self.use_precursors and not self.lag_precursors:
                # Loop over precursors
                for j in range(xs.n_precursors):
                    # Multiplier for delayed fission term
                    coeff = xs.chi_delayed[g][j] * xs.precursor_lambda[j]
                    coeff /= 1.0 + eff_dt * xs.precursor_lambda[j]
                    coeff *= eff_dt * xs.precursor_yield[j]

                    # Loop over groups
                    for gp in range(self.n_groups):
                        igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                        value = coeff * xs.nu_delayed_sigma_f[gp] * volume
                        rows.append(ig)
                        cols.append(igp)
                        data.append(value)

    n_dofs = fv.n_dofs(uk_man)
    return csr_matrix((data, (rows, cols)), shape=(n_dofs,) * 2)


def _fv_set_transient_source(
        self: "TransientSolver", step: int = 0,
        apply_material_source: bool = True,
        apply_boundary_source: bool = True,
        apply_scattering_source: bool = False,
        apply_fission_source: bool = False) -> None:
    """Assemble the right-hand side.

    This includes previous time step contributions as well
    as material, scattering, fission, and boundary sources.

    Parameters
    ----------
    step : int, default 0
        The section of the time step.
    apply_material_source : bool, default True
    apply_boundary_source : bool, default True
    apply_scattering_source : bool, default False
    apply_fission_source : bool, default False
    """
    fv: FiniteVolume = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man
    eff_dt = self.effective_dt(step)

    # Material + boundary + scattering sources
    flags = (apply_material_source, apply_boundary_source,
             apply_scattering_source, False)
    SteadyStateSolver._fv_set_source(self, *flags)

    # Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, flux_uk_man, 0, g)

            # Fission source
            if apply_fission_source:

                # Loop over groups
                for gp in range(self.n_groups):
                    igp = fv.map_dof(cell, 0, flux_uk_man, 0, gp)

                    # Total fission
                    if not self.use_precursors:
                        coeff = xs.chi[g] * xs.nu_sigma_f[gp]
                    # Prompt fission
                    else:
                        coeff = xs.chi_prompt[g] * \
                                xs.nu_prompt_sigma_f[gp]

                    self.b[ig] += coeff * self.phi[igp] * volume

            # Delayed fission and precursors
            if self.use_precursors:
                # Loop over precursors
                for j in range(xs.n_precursors):
                    ij = cell.id * prec_uk_man.total_components + j

                    # Get the precursors at this DoF
                    prec = self.precursors[ij]
                    prec_old = self.precursors_old[ij]

                    # Multiplier for precursor term
                    coeff = xs.chi_delayed[g][j] * xs.precursor_lambda[j]
                    if not self.lag_precursors:
                        coeff /= 1.0 + eff_dt * xs.precursor_lambda[j]

                    # Old precursor contributions
                    if step == 0 or self.lag_precursors:
                        self.b[ig] += coeff * prec_old * volume
                    else:
                        tmp = (4.0 * prec - prec_old) / 3.0
                        self.b[ig] += coeff * tmp * volume

                    # Delayed fission term
                    if not self.lag_precursors and apply_fission_source:
                        # Modified multiplier for delayed fission
                        coeff *= eff_dt * xs.precursor_yield[j]

                        # Loop over groups
                        f_d = 0.0  # delayed fission
                        for gp in range(self.n_groups):
                            igp = fv.map_dof(cell, 0, flux_uk_man, 0, gp)
                            f_d += xs.nu_delayed_sigma_f[gp] * self.phi[igp]
                        self.b[ig] += coeff * f_d * volume


def _fv_update_precursors(self: "TransientSolver",
                         step: int = 0) -> None:
    """Solve a precursor time step.

    Parameters
    ----------
    step : int, default 0
        The section of the time step.
    """
    fv: FiniteVolume = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man
    eff_dt = self.effective_dt(step)

    # Loop over cells
    for cell in self.mesh.cells:
        xs = self.material_xs[cell.material_id]

        # Loop over precursors
        for j in range(xs.n_precursors):
            ij = cell.id * prec_uk_man.total_components + j

            # Get the precursors at this DoF
            prec = self.precursors[ij]
            prec_old = self.precursors_old[ij]

            # Multiplier for RHS
            coeff = (1.0 + eff_dt * xs.precursor_lambda[j]) ** (-1)

            # Initialize with old precursor term
            if step == 0:
                self.precursors[ij] = coeff * prec_old
            else:
                tmp = (4.0 * prec - prec_old) / 3.0
                self.precursors[ij] = coeff * tmp

            # Delayed fission multiplier modification
            coeff *= eff_dt * xs.precursor_yield[j]

            # Loop over groups
            f_d = 0.0  # delayed fission
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, flux_uk_man, 0, g)
                f_d += xs.nu_delayed_sigma_f[g] * self.phi[ig]
            self.precursors[ij] += coeff * f_d


def _fv_compute_power(self: "TransientSolver") -> float:
    """Compute the fission power.

    Notes
    -----
    This method uses the most current scalar flux solution.

    Returns
    -------
    float
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.flux_uk_man

    # Loop over cells
    power = 0.0
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            power += xs.sigma_f[g] * self.phi[ig] * volume
    return power * self.energy_per_fission
