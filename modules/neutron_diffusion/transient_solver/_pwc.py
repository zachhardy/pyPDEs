"""
All routines of TransientSolver that perform computations
using a piecewise continuous spatial discretization.
"""
import numpy as np

from numpy import ndarray
from scipy.sparse import csr_matrix, lil_matrix

from pyPDEs.spatial_discretization import PiecewiseContinuous
from pyPDEs.utilities.boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


def _pwc_feedback_matrix(self: "TransientSolver") -> csr_matrix:
    """Assemble the feedback matrix.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

    T = self.temperature
    T0 = self.initial_temperature

    # Loop over cells
    A = lil_matrix((fv.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Compute feedback coefficient
        f = self.feedback_coeff * (sqrt(T[cell.id]) - sqrt(T0))

        # Loop over groups
        for g in range(self.n_groups):
            sig_t = xs.sigma_t[g]

            # Loop over nodes
            for i in range(view.n_nodes):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)
                for j in range(view.n_nodes):
                    jg = pwc.map_dof(cell, j, uk_man, 0, g)
                    mass_ij = view.intV_shapeI_shapeJ[i][j]
                    A[ig, jg] += sig_t * f * mass_ij
    return A.tocsr()


def _pwc_mass_matrix(self: "TransientSolver",
                     lumped: bool = True) -> csr_matrix:
    """Assemble the multigroup mass matrix.

    Parameters
    ----------
    lumped : bool, default True
        Whether to construct a lumped mass matrix, or not.

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
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            v = xs.velocity[g]

            # Loop over nodes
            for i in range(view.n_nodes):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)

                if lumped:
                    A[ig, ig] += inv_vel * view.intV_shapeI[i]
                else:
                    # Loop over nodes
                    for j in range(view.n_nodes):
                        jg = pwc.map_dof(cell, j, uk_man, 0, g)

                        mass_ij = view.intV_shapeI_shapeJ[i][j]
                        A[ig, jg] +=  mass_ij / v
    return A.tocsr()


def _pwc_precursor_substitution_matrix(self: "TransientSolver",
                                       m: int = 0) -> csr_matrix:
    """Assemble the transient precursor substitution matrix for step `m`.

    Parameters
    ----------
    m : int, default 0
        The step in a multi-step method.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man
    eff_dt = self.effective_time_step(m)

    # Loop over cells
    A = lil_matrix((pwc.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        volume = cell.volume
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over precursors
        for p in range(xs.n_precursors):
            lambda_p = xs.precursor_lambda[p]
            gamma_p = xs.precursor_yield[p]

            # Loop over groups
            for g in range(self.n_groups):
                chi_d = xs.chi_delayed[g][p]
                coeff = chi_d*lambda_p / (1.0 + eff_dt*lambda_p)
                coeff *= eff_dt * gamma_p

                # Loop over groups
                for gp in range(self.n_groups):
                    nud_sigf = xs.nu_delayed_sigma_f[gp]

                    # Loop over node
                    for i in range(view.n_nodes):
                        ig = pwc.map_dof(cell, i, uk_man, 0, g)
                        for j in range(view.n_nodes):
                            jgp = pwc.map_dof(cell, j, uk_man, 0, gp)
                            mass_ij = view.intV_shapeI_shapeJ[i][j]
                            A[ig, jgp] += coeff * nud_sigf * mass_ij/volume
    return A.tocsr()


def _pwc_old_precursor_source(self: "TransientSolver", m: int = 0) -> ndarray:
    """Assemble the delayed terms from the last time step for step `m`.

    Parameters
    ----------
    m : int, default 0
        The step in a multi-step method.

    Returns
    -------
    ndarray (n_cells * n_groups)
    """
    pwc: PiecewiseContinuous = self.discretization
    phi_ukm = self.phi_uk_man
    c_ukm = self.precursor_uk_man
    eff_dt = self.effective_time_step(m)

    # Loop over cells
    b = np.zeros(pwc.n_dofs(phi_ukm))
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over precursors
        for p in range(xs.n_precursors):
            ip = cell.id * c_ukm.total_components + p
            lambda_p = xs.precursor_lambda[p]
            gamma_p = xs.precursor_yield[p]

            # Get the precursors
            c_old = self.precursors_old[ip]
            if m == 1 and not self.lag_precursors:
                c = self.precursors[ip]
                c_old = (4.0*c - c_old) / 3.0

            # Loop over groups
            for g in range(self.n_groups):
                chi_d = xs.chi_delayed[g][p]

                # Decay emission coefficient
                coeff = chi_d * lambda_p
                if not self.lag_precursors:
                    coeff /= 1.0 + eff_dt * lambda_p

                # Loop over nodes
                for i in range(view.n_nodes):
                    ig = pwc.map_dof(cell, i, phi_ukm, 0, g)
                    b[ig] += coeff * c_old * view.intV_shapeI[i]
    return b


def _pwc_update_precursors(self: "TransientSolver", m: int = 0) -> None:
    """Update the precursors after a time step.

    Parameters
    ----------
    m : int, default 0
        The step in a multi-step method.
    """
    pwc: PiecewiseContinuous = self.discretization
    phi_ukm = self.phi_uk_man
    c_ukm = self.precursor_uk_man
    eff_dt = self.effective_time_step(m)

    # Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Compute delayed fission rate
        f_d = 0.0
        for g in range(self.n_groups):
            nud_sigf = xs.nu_delayed_sigma_f[g]
            for i in range(view.n_nodes):
                ig = pwc.map_dof(cell, i, phi_ukm, 0, g)
                f_d += nud_sigf * self.phi[ig] * \
                       view.intV_shapeI[i] / volume

        # Loop pver precursors
        for p in range(xs.n_precursors):
            ip = cell.id * c_ukm.total_components + p
            lambda_p = xs.precursor_lambda[p]
            gamma_p = xs.precursor_yield[p]

            # Get the precursors
            c_old = self.precursors_old[ip]
            if m == 1:
                c = self.precursors[ip]
                c_old = (4.0*c - c_old) / 3.0

            coeff = 1.0 / (1.0 + eff_dt*lambda_p)
            self.precursors[ip] = coeff * (c_old + eff_dt*gamma_p * f_d)


def _pwc_compute_fission_rate(self: "TransientSolver") -> None:
    """Compute the point-wise fission rate.
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    self.fission_rate *= 0.0
    for cell in self.mesh.cells:
        volume = cell.volume
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over nodes
        for i in range(view.n_nodes):
            intV_shapeI = view.intV_shapeI[i]

            # Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)
                self.fission_rate[cell.id] += \
                    xs.sigma_f[g]*self.phi[ig] * intV_shapeI/volume




def _pwc_compute_power(self: "TransientSolver") -> float:
    """Compute the fission power in the system.

    Returns
    -------
    float
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    power = 0.0
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over group
        for g in range(self.n_groups):
            sig_f = xs.sigma_f[g]

            # Loop over nodes
            for i in range(view.n_nodes):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)
                power += sig_f * self.phi[ig] * view.intV_shapeI[i]
    return power * self.energy_per_fission
