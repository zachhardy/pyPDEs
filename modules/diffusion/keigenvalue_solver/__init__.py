import sys

import numpy as np

from numpy.linalg import norm
from scipy.sparse.linalg import spsolve

from pyPDEs.spatial_discretization import (FiniteVolume,
                                           PiecewiseContinuous)

from modules.diffusion import SteadyStateSolver


class KEigenvalueSolver(SteadyStateSolver):
    """Class for solving a multi-group k-eigenvalue problem.
    """

    from ._assemble_fv import _fv_compute_fission_production
    from ._assemble_pwc import _pwc_compute_fission_production

    def __init__(self) -> None:
        super().__init__()
        self.k_eff = 1.0

    def execute(self, verbose: bool = False) -> None:
        """Execute the k-eigenvalue diffusion solver.
        """
        print("\n***** Executing the k-eigenvalue "
              "multi-group diffusion solver.")
        uk_man = self.flux_uk_man
        num_dofs = self.discretization.n_dofs(uk_man)
        n_grps = self.n_groups

        # Initialize with unit flux
        self.phi = np.ones(num_dofs)
        phi_ell = np.copy(self.phi)

        # Initialize book-keeping parameters
        production = self.compute_fission_production()
        production_ell = production
        k_eff_ell = k_eff_change = 1.0
        phi_change = 1.0

        # Start iterating
        nit = 0
        converged = False
        for nit in range(self.max_iterations):

            # Solve group-wise
            if self.use_groupwise_solver:
                # Loop over groups
                for g in range(self.n_groups):
                    # Precompute the fission source
                    self.b *= 0.0
                    self.set_source(False, False, False, True)
                    self.b /= self.k_eff

                    # Add in the scattering + bpundary source
                    self.set_source(False, True, True, False)

                    # Solve group system
                    self.phi[g::n_grps] = spsolve(self.Lg(g), self.bg(g))

            # Solve block system
            else:
                # Precompute fission source
                self.b *= 0.0
                self.set_source(False, False, False, True)
                self.b /= self.k_eff

                # Add in the boundary source
                self.set_source(False, True, False, False)

                # Solve full system
                self.phi = spsolve(self.L - self.S, self.b)

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
            if verbose:
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
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_compute_fission_production()
        else:
            return self._pwc_compute_fission_production()