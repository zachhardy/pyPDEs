import sys
import numpy as np

from numpy.linalg import norm
from scipy.sparse.linalg import spsolve

from pyPDEs.spatial_discretization import *

from .. import SteadyStateSolver


class KEigenvalueSolver(SteadyStateSolver):
    """Class for solving a k-eigenvalue multigroup diffusion problems.
    """

    def __init__(self) -> None:
        """Class constructor.
        """
        super().__init__()
        self.k_eff: float = 1.0

        # Iterative parameters
        self.tolerance: float = 1.0e-8
        self.max_iterations: int = 500

    def execute(self, verbose: int = 0) -> None:
        """Execute the k-eigenvalue multigroup diffusion solver.

        Parameters
        ----------
        verbose : int, default 0
        """
        uk_man = self.phi_uk_man
        n_dofs = self.discretization.n_dofs(uk_man)
        n_grps = self.n_groups

        # Initialize with unit flux
        self.phi[:] = 1.0
        phi_ell = np.copy(self.phi)

        # Initialize book-keeping parameters
        production = self.compute_fission_production()
        production_ell = production
        k_eff_ell = k_eff_change = 1.0
        phi_change = 1.0

        # Assemble matrices
        F = self.Fp + self.Fd  # Total fission matrix
        A = self.L - self.S  # Diffusion - scattering matrix

        # Start iterative
        nit, converged = 0, False
        for nit in range(self.max_iterations):
            # Set fission source and solve
            b = F @ self.phi / self.k_eff
            self.apply_vector_bcs(b)
            self.phi = spsolve(A, b)

            # Update k-eigenvalue
            production = self.compute_fission_production()
            self.k_eff *= production / production_ell

            # Check convergence
            k_eff_change = abs(self.k_eff - k_eff_ell) / self.k_eff
            phi_change = norm(self.phi - phi_ell)
            k_eff_ell = self.k_eff
            production_ell = production
            phi_ell[:] = self.phi

            # Print iteration summary
            if verbose > 1:
                print(f"\n===== Iteration {nit} =====\n"
                      f"{'k_eff':<15} = {self.k_eff:.6g}\n"
                      f"{'k_eff Change':<15} = {k_eff_change:.3e}\n"
                      f"{'Phi Change':<15} = {phi_change:.3e}")

            # Break, if converged
            if k_eff_change <= self.tolerance and \
                    phi_change <= self.tolerance:
                converged = True
                break

        # Compute precursors
        self.phi /= np.max(self.phi)
        if self.use_precursors:
            self.compute_precursors()
            self.precursors /= self.k_eff

        # Print summary
        if converged:
            msg = "***** k-Eigenvalue Solver Converged! *****"
        else:
            msg = "***** WARNING: k-Eigenvalue Solver NOT Converged *****"
        header = "*" * len(msg)

        if verbose > 0 or not converged:
            print("\n".join(["", header, msg, header]))
            print(f"Final k Effective:\t\t{self.k_eff:.6g}")
            print(f"Final k Effective Change:\t{k_eff_change:3e}")
            print(f"Final Phi Change:\t\t{phi_change:.3e}")
            print(f"# of Iterations:\t\t{nit}")

    def compute_fission_production(self) -> float:
        """Compute the fission production.

        Returns
        -------
        float
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

        # Loop over cells
        production = 0.0
        for cell in self.mesh.cells:
            volume = cell.volume
            xs = self.material_xs[cell.material_id]

            # Loop over groups
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, uk_man, 0, g)
                production += xs.nu_sigma_f[g] * self.phi[ig] * volume
        return production


