import numpy as np

from numpy.linalg import norm
from scipy.sparse.linalg import spsolve

from modules.diffusion import SteadyStateSolver


class KEigenvalueSolver(SteadyStateSolver):
    """
    Class for solving a multi-group k-eigenvalue problem.

    Attributes
    ----------
    k_eff : float
    """

    from .assemble_fv import fv_compute_fission_production

    def __init__(self) -> None:
        super().__init__()
        self.k_eff = 1.0

    def execute(self) -> None:
        """
        Execute the k-eigenvalue diffusion solver.
        """
        print("\n***** Executing the multi-group diffusion "
              "k-eigenvalue solver.\n")
        uk_man = self.flux_uk_man
        num_dofs = self.discretization.num_dofs(uk_man)
        n_grps = self.num_groups
        phi_ell = self.phi = np.ones(num_dofs)
        production_ell = self.fv_compute_fission_production(self.phi, )
        k_eff_ell = k_eff_change = 1.0
        phi_change = 1.0

        # ======================================== Start iterating
        nit = 0
        converged = False
        for nit in range(self.max_iterations):

            # =================================== Solve groups-wise
            self.b *= 0.0
            for g in range(self.num_groups):
                # Precompute fission source
                flags = (False, False, True, False)
                self.fv_set_source(g, self.phi, *flags)
                self.b[g::n_grps] /= self.k_eff

                # Add scattering + boundary source
                flags = (False, True, False, True)
                self.fv_set_source(g, self.phi, *flags)
                self.phi[g::n_grps] = spsolve(self.L[g],
                                              self.b[g::n_grps])

            # ============================== Update k-eigenvalue
            production = self.fv_compute_fission_production(self.phi, )
            self.k_eff *= production / production_ell

            # ============================== Check for convergence
            k_eff_change = abs(self.k_eff - k_eff_ell) / self.k_eff
            phi_change = norm(self.phi - phi_ell)
            k_eff_ell = self.k_eff
            production_ell = production
            phi_ell[:] = self.phi
            if k_eff_change <= self.tolerance and \
                    phi_change <= self.tolerance:
                converged = True
                break

        # ======================================== Compute precursors
        self.phi /= np.max(self.phi)
        if self.use_precursors:
            self.fv_compute_precursors()
            self.precursors /= self.k_eff

        # ======================================== Print summary
        if converged:
            msg = "***** k-Eigenvalue Solver Converged!*****"
        else:
            msg = "!!!!! WARNING: k-Eigenvalue Solver NOT Converged !!!!!"
        msg += f"\nFinal k Effective:\t\t{self.k_eff:.6g}"
        msg += f"\nFinal k Effective Change:\t{k_eff_change:3e}"
        msg += f"\nFinal Phi Change:\t\t{phi_change:.3e}"
        msg += f"\n# of Iterations:\t\t{nit}"
        print(msg)
