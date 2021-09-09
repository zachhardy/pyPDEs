"""
All routines of TransientSolver that perform computations
using a finite volume spatial discretization.
"""
import numpy as np

from numpy import ndarray, sqrt
from scipy.sparse import csr_matrix, lil_matrix

from pyPDEs.spatial_discretization import FiniteVolume

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


def _fv_feedback_matrix(self: "TransientSolver") -> csr_matrix:
    """Assemble the feedback matrix.

    Returns
    -------
    csr_matrix (n_cells * n_groups,) * 2
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    T = self.temperature
    T0 = self.initial_temperature

    # Loop over cells
    A = lil_matrix((fv.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Compute feedback coefficient
        f = self.feedback_coeff * (sqrt(T[cell.id]) - sqrt(T0))

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            A[ig, ig] += xs.sigma_t[g] * f * volume
    return A.tocsr()


def _fv_mass_matrix(self: "TransientSolver") -> csr_matrix:
    """Assemble the multigroup mass matrix.

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
            A[ig, ig] += xs.inv_velocity[g] * volume
    return A.tocsr()


def _fv_precursor_substitution_matrix(self: "TransientSolver",
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
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man
    eff_dt = self.effective_time_step(m)

    # Loop over cells
    A = lil_matrix((fv.n_dofs(uk_man),) * 2)
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over precursors
        for p in range(xs.n_precursors):
            lambda_p = xs.precursor_lambda[p]
            gamma_p = xs.precursor_yield[p]

            # Loop over groups
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, uk_man, 0, g)

                chi_d = xs.chi_delayed[g][p]
                coeff = chi_d*lambda_p / (1.0 + eff_dt*lambda_p)
                coeff *= eff_dt * gamma_p

                # Loop over groups
                for gp in range(self.n_groups):
                    igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                    nud_sigf = xs.nu_delayed_sigma_f[gp]
                    A[ig, igp] += coeff * nud_sigf * volume
    return A.tocsr()


def _fv_old_precursor_source(self: "TransientSolver", m: int = 0) -> ndarray:
    """Assemble the delayed terms from the last time step for step `m`.

    Parameters
    ----------
    m : int, default 0
        The step in a multi-step method.

    Returns
    -------
    ndarray (n_cells * n_groups)
    """
    fv: FiniteVolume = self.discretization
    phi_ukm = self.phi_uk_man
    c_ukm = self.precursor_uk_man
    eff_dt = self.effective_time_step(m)

    # Loop over cells
    b = np.zeros(fv.n_dofs(phi_ukm))
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over precursors
        for p in range(xs.n_precursors):
            ip = fv.map_dof(cell, 0, c_ukm, 0, p)
            lambda_p = xs.precursor_lambda[p]
            gamma_p = xs.precursor_yield[p]

            # Get the precursors
            c_old = self.precursors_old[ip]
            if m == 1 and not self.lag_precursors:
                c = self.precursors[ip]
                c_old = (4.0*c - c_old) / 3.0

            # Loop over groups
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, phi_ukm, 0, g)
                chi_d = xs.chi_delayed[g][p]

                # Decay emission coefficient
                coeff = chi_d * lambda_p
                if not self.lag_precursors:
                    coeff /= 1.0 + eff_dt * lambda_p

                b[ig] += coeff * c_old * volume
    return b


def _fv_update_precursors(self: "TransientSolver", m: int = 0) -> None:
    """Update the precursors after a time step.

    Parameters
    ----------
    m : int, default 0
        The step in a multi-step method.
    """
    fv: FiniteVolume = self.discretization
    phi_ukm = self.phi_uk_man
    c_ukm = self.precursor_uk_man
    eff_dt = self.effective_time_step(m)

    # Loop over cells
    for cell in self.mesh.cells:
        xs = self.material_xs[cell.material_id]

        # Compute delayed fission rate
        f_d = 0.0
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, phi_ukm, 0, g)
            nud_sigf = xs.nu_delayed_sigma_f[g]
            f_d += nud_sigf * self.phi[ig]

        # Loop pver precursors
        for p in range(xs.n_precursors):
            ip = fv.map_dof(cell, 0, c_ukm, 0, p)
            lambda_p = xs.precursor_lambda[p]
            gamma_p = xs.precursor_yield[p]

            # Get the precursors
            c_old = self.precursors_old[ip]
            if m == 1:
                c = self.precursors[ip]
                c_old = (4.0*c - c_old) / 3.0

            coeff = 1.0 / (1.0 + eff_dt*lambda_p)
            self.precursors[ip] = coeff * (c_old + eff_dt*gamma_p * f_d)


def _fv_compute_fission_rate(self: "TransientSolver") -> None:
    """Compute the point-wise fission rate.
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    self.fission_rate *= 0.0
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            self.fission_rate[cell.id] += \
                xs.sigma_f[g] * self.phi[ig]


def _fv_compute_power(self: "TransientSolver") -> float:
    """Compute the fission power in the system.

    Returns
    -------
    float
    """
    fv: FiniteVolume = self.discretization
    uk_man = self.phi_uk_man

    # Loop over cells
    power = 0.0
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over group
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            power += xs.sigma_f[g] * self.phi[ig] * volume
    return power * self.energy_per_fission
