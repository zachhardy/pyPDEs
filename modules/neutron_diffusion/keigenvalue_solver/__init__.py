import sys
import numpy as np

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pyPDEs.spatial_discretization import *

from .. import SteadyStateSolver


class KEigenvalueSolver(SteadyStateSolver):
    """k-eigenvalue multigroup diffusion.
    """

    def __init__(self) -> None:
        """Class constructor.
        """
        super().__init__()
        self.k_eff: float = 1.0

        # Iterative parameters
        self.tolerance: float = 1.0e-8
        self.max_iterations: int = 500

    def initialize(self, verbose: int = 0) -> None:
        """Initialize the k-eigenvalue multigroup diffusion solver.

        Parameters
        ----------
        verbose : int, default 0
        """
        SteadyStateSolver.initialize(self)
        self.k_eff = 1.0

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

        # Assemble the matrix
        A = self.assemble_matrix()

        # Start iterative
        nit, converged = 0, False
        for nit in range(self.max_iterations):
            # Set fission source and solve
            b = self.assemble_rhs()
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

    def assemble_matrix(self) -> csr_matrix:
        A = self.L - self.S
        return self.apply_matrix_bcs(A)

    def assemble_rhs(self) -> csr_matrix:
        b = self.Fp @ self.phi / self.k_eff
        if self.use_precursors:
            b += self.Fd @ self.phi / self.k_eff
        return self.apply_vector_bcs(b)

    def compute_fission_production(self) -> float:
        """Compute the fission production.

        Returns
        -------
        float
        """
        production = np.sum(self.Fp @ self.phi)
        if self.use_precursors:
            production += np.sum(self.Fd @ self.phi)
        return production
