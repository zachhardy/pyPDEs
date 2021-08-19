import numpy as np

from numpy.linalg import norm
from scipy.sparse.linalg import spsolve

from pyPDEs.spatial_discretization import (FiniteVolume,
                                           PiecewiseContinuous)

from modules.diffusion import SteadyStateSolver


class KEigenvalueSolver(SteadyStateSolver):
    """Class for solving a multi-group k-eigenvalue problem.
    Attributes
    ----------
    mesh : Mesh
        The spatial mesh to solve the problem on.
    discretization : SpatialDiscretization
        The spatial discretization used to solve the problem.
    boundaries : List[Boundary]
        The boundary conditions imposed on the equations.
        There should be a boundary condition for each group
        and boundary. In the list, each boundaries group-wise
        boundary conditions should be listed next to each other.
    material_xs : List[CrossSections]
        The cross sections corresponding to the material IDs
        defined on the cells. There should be as many cross
        sections as unique material IDs on the mesh.
    material_src : List[MultigroupSource]
        The multi-group sources corresponding to the material
        IDs defined on the cells. There should be as many
        multi-group sources as unique material IDs on the mesh.
    use_precursors : bool
        A flag for including delayed neutrons.
    tolerance : float
        The iterative tolerance for the group-wise solver.
    max_iterations : int
        The maximum number of iterations for the group-wise
        solver to take before exiting.
    b : ndarray (n_nodes * n_groups,)
        The right-hand side of the linear system to solve.
    L : List[csr_matrix]
        The group-wise diffusion operators used to solve the
        equations group-wise. There are n_groups matrices stored.
    phi : ndarray (n_nodes * n_groups,)
        The most current scalar flux solution vector.
    flux_uk_man : UnknownManager
        An unknown manager tied to the scalar flux solution vector.
    precurosrs : ndarray (n_nodes * max_precursors_per_material,)
        The delayed neutron precursor concentrations.

        In multi-material problems, this vector stores up to the
        maximum number of precursors that live on any given material.
        This implies that material IDs must be used to map the
        concentration of specific precursor species. This structure
        is used to prevent very sparse vectors in many materials.
    precursor_uk_man : UnknownManager
        An unknown manager tied to the precursor vector.
    k_eff : float
        The most current k-eigenvalue estimate.
    """

    from .assemble_fv import fv_compute_fission_production
    from .assemble_pwc import pwc_compute_fission_production

    def __init__(self) -> None:
        super().__init__()
        self.k_eff = 1.0

    def execute(self, verbose: bool = False) -> None:
        """
        Execute the k-eigenvalue diffusion solver.
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

        # ======================================== Start iterating
        nit = 0
        converged = False
        for nit in range(self.max_iterations):

            # =================================== Solve groups-wise
            self.b *= 0.0
            for g in range(self.n_groups):
                # Precompute fission source
                flags = (False, False, True, False)
                self.set_source(g, self.phi, *flags)
                self.b[g::n_grps] /= self.k_eff

                # Add scattering + boundary source
                flags = (False, True, False, True)
                self.set_source(g, self.phi, *flags)
                self.phi[g::n_grps] = spsolve(self.L[g],
                                              self.b[g::n_grps])

            # ============================== Update k-eigenvalue
            production = self.compute_fission_production()
            self.k_eff *= production / production_ell

            # ============================== Check for convergence
            k_eff_change = abs(self.k_eff - k_eff_ell) / self.k_eff
            phi_change = norm(self.phi - phi_ell)
            k_eff_ell = self.k_eff
            production_ell = production
            phi_ell[:] = self.phi

            if verbose:
                print(f"\n===== Iteration {nit}\n"
                      f"\t{'k_eff':<15} = {self.k_eff:.6g}\n"
                      f"\t{'k_eff Change':<15} = {k_eff_change:.3e}\n"
                      f"\t{'Phi Change':<15} = {phi_change:.3e}")

            if k_eff_change <= self.tolerance and \
                    phi_change <= self.tolerance:
                converged = True
                break

        # ======================================== Compute precursors
        self.phi /= np.max(self.phi)
        if self.use_precursors:
            self.compute_precursors()
            self.precursors /= self.k_eff

        # ======================================== Print summary
        if converged:
            msg = "\n***** k-Eigenvalue Solver Converged!*****"
        else:
            msg = "!!!!! WARNING: k-Eigenvalue Solver NOT Converged !!!!!"
        msg += f"\nFinal k Effective:\t\t{self.k_eff:.6g}"
        msg += f"\nFinal k Effective Change:\t{k_eff_change:3e}"
        msg += f"\nFinal Phi Change:\t\t{phi_change:.3e}"
        msg += f"\n# of Iterations:\t\t{nit}"
        print(msg)
        print("\n***** Done executing k-eigenvalue "
              "multi-group diffusion solver. *****")

    def compute_fission_production(self) -> float:
        """
        Compute the fission production from the most recent
        solution vector.

        Returns
        -------
        float
        """
        if isinstance(self.discretization, FiniteVolume):
            return self.fv_compute_fission_production()
        else:
            return self.pwc_compute_fission_production()