from numpy import ndarray

from scipy.sparse import csr_matrix

from pyPDEs.spatial_discretization import PiecewiseContinuous
from pyPDEs.utilities.boundaries import DirichletBoundary
from pyPDEs.utilities import UnknownManager

from ..steadystate_solver import SteadyStateSolver

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


def _pwc_assemble_mass_matrix(self: "TransientSolver") -> csr_matrix:
    """Assemble the multi-group mass matrix.

    Returns
    -------
    csr_matrix
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    rows, data = [], []
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over test functions
        for i in range(view.n_nodes):
            intV_shapeI = view.intV_shapeI[i]

            # Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)

                value = xs.inv_velocity[g] * intV_shapeI
                rows.append(ig)
                data.append(value)

        # Loop over faces
        for f_id, face in enumerate(cell.faces):

            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)

                # Loop over groups
                for g in range(self.n_groups):
                    bc = self.boundaries[bndry_id * self.n_groups + g]

                    # Dirichlet boundary
                    if issubclass(type(bc), DirichletBoundary):

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ii = pwc.map_face_dof(cell, f_id, fi)
                            pwc.zero_dirichlet_row(ii, rows, data)

    n_dofs = pwc.n_dofs(uk_man)
    return csr_matrix((data, (rows, rows)), shape=(n_dofs,) * 2)


def _pwc_assemble_transient_fission_matrix(
        self: "TransientSolver", step: int = 0) -> csr_matrix:
    """Assemble the transient multi-group fission matrix.

    Returns
    -------
    csr_matrix
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man: UnknownManager = self.flux_uk_man
    eff_dt = self.effective_dt(step)

    # Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        volume = cell.volume
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over test functions
        for i in range(view.n_nodes):

            # Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)

                # Loop over trial functions
                for j in range(view.n_nodes):
                    mass_ij = view.intV_shapeI_shapeJ[i][j]

                    # Loop over groups
                    for gp in range(self.n_groups):
                        jgp = pwc.map_dof(cell, j, uk_man, 0, gp)

                        # Total fission
                        if not self.use_precursors:
                            coeff = xs.chi[g] * xs.nu_sigma_f[gp]
                        # Prompt fission
                        else:
                            coeff = xs.chi_prompt[g] * \
                                    xs.nu_prompt_sigma_f[gp]

                        value = coeff * mass_ij
                        rows.append(ig)
                        cols.append(jgp)
                        data.append(value)

                # Delayed fission
                if self.use_precursors and not self.lag_precursors:

                    # Loop over precursors
                    for p in range(xs.n_precursors):
                        # Multiplier for delayed fission term
                        coeff = xs.chi_delayed[g][p] * xs.precursor_lambda[p]
                        coeff /= 1.0 + eff_dt * xs.precursor_lambda[p]
                        coeff *= eff_dt * xs.precursor_yield[p]

                        # Loop over trial functions
                        for j in range(view.n_nodes):
                            mass_ij = view.intV_shapeI_shapeJ[i][j]

                            # Loop over groups
                            for gp in range(self.n_groups):
                                igp = pwc.map_dof(cell, j, uk_man, 0, gp)
                                nud_sigf = xs.nu_delayed_sigma_f[gp]

                                value = coeff * nud_sigf * mass_ij / volume
                                rows.append(ig)
                                cols.append(igp)
                                data.append(value)

    n_dofs = pwc.n_dofs(uk_man)
    return csr_matrix((data, (rows, cols)), shape=(n_dofs,) * 2)


def _pwc_set_transient_source(
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
    pwc: PiecewiseContinuous = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man
    eff_dt = self.effective_dt(step)

    # Material + scattering sources
    flags = (apply_material_source, False,
             apply_scattering_source, False)
    SteadyStateSolver._pwc_set_source(self, *flags)

    # Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over test functions
        for i in range(view.n_nodes):
            intV_shapeI = view.intV_shapeI[i]

            # Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, flux_uk_man, 0, g)

                # Fission source
                if apply_fission_source:

                    # Loop over trial functions
                    for j in range(view.n_nodes):
                        mass_ij = view.intV_shapeI_shapeJ[i][j]

                        # Loop over groups
                        for gp in range(self.n_groups):
                            jgp = pwc.map_dof(cell, j, flux_uk_man, 0, gp)

                            # Total fission
                            if not self.use_precursors:
                                coeff = xs.chi[g] * xs.nu_sigma_f[gp]
                            # Prompt fission
                            else:
                                coeff = xs.chi_prompt[g] * \
                                        xs.nu_prompt_sigma_f[gp]

                            self.b[ig] += coeff * self.phi[jgp] * mass_ij

                # Delayed fission
                if self.use_precursors:
                    # Loop over precursors
                    for p in range(xs.n_precursors):
                        ip = cell.id * prec_uk_man.total_components + p

                        # Get the precursors at this DoF
                        prec = self.precursors[ip]
                        prec_old = self.precursors_old[ip]

                        # Multiplier for precursor term
                        coeff = xs.chi_delayed[g][p] * xs.precursor_lambda[p]
                        if not self.lag_precursors:
                            coeff /= 1.0 + eff_dt * xs.precursor_lambda[p]

                        # Old precursor contributions
                        if step == 0 or self.lag_precursors:
                            self.b[ig] += coeff * prec_old * intV_shapeI
                        else:
                            tmp = (4.0 * prec - prec_old) / 3.0
                            self.b[ig] += coeff * tmp * intV_shapeI

                        # Delayed fission contributions
                        if not self.lag_precursors and apply_fission_source:
                            # Modified multiplier for delayed fission
                            coeff *= eff_dt * xs.precursor_yield[p]

                            # Loop over trial functions
                            for j in range(view.n_nodes):
                                mass_ij = view.intV_shapeI_shapeJ[i][j]

                                # Loop over groups
                                f_d = 0.0  # delayed fission
                                for gp in range(self.n_groups):
                                    jgp = pwc.map_dof(cell, j, flux_uk_man, 0, gp)
                                    f_d += xs.nu_delayed_sigma_f[gp] * self.phi[jgp]
                                self.b[ig] += coeff * f_d * mass_ij / volume

    flags = (False, True, False, False)
    SteadyStateSolver._pwc_set_source(self, *flags)


def _pwc_update_precursors(self: "TransientSolver",
                          step: int = 0) -> None:
    """Solve a precursor time step.

    Parameters
    ----------
    step : int, default 0
        The section of the time step.
    """
    pwc: PiecewiseContinuous = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man
    eff_dt = self.effective_dt(step)

    # Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over precursors
        for j in range(xs.n_precursors):
            ij = cell.id * prec_uk_man.total_components + j

            # Get precursors at this DoF
            prec = self.precursors[ij]
            prec_old = self.precursors_old[ij]

            # Multiplier for RHS
            coeff = (1.0 + eff_dt * xs.precursor_lambda[j]) ** (-1)

            # Initialize with old precursors
            if step == 0:
                self.precursors[ij] = coeff * prec_old
            else:
                tmp = (4.0 * prec - prec_old) / 3.0
                self.precursors[ij] = coeff * tmp

            # Modified multiplier for delayed fission
            coeff *= eff_dt * xs.precursor_yield[j]

            # Loop over trial functions
            for i in range(view.n_nodes):
                intV_shapeI = view.intV_shapeI[i]

                # Loop over groups
                f_d = 0.0  # delayed fission
                for g in range(self.n_groups):
                    ig = pwc.map_dof(cell, i, flux_uk_man, 0, g)
                    f_d += xs.nu_delayed_sigma_f[g] * self.phi[ig]
                self.precursors[ij] += coeff * f_d * intV_shapeI / volume


def _pwc_compute_power(self: "TransientSolver") -> float:
    """Compute the fission power.

    Notes
    -----
    This method uses the most current scalar flux solution.

    Returns
    -------
    float
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.flux_uk_man

    # Loop over cells
    power = 0.0
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over nodes
        for i in range(view.n_nodes):
            intV_shapeI = view.intV_shapeI[i]

            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)
                power += xs.sigma_f[g] * self.phi[ig] * intV_shapeI
    return power * self.energy_per_fission
