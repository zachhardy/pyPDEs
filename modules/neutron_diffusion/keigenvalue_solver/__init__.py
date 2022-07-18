import sys
import numpy as np

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple

from pyPDEs.spatial_discretization import *

from .. import SteadyStateSolver


class KEigenvalueSolver(SteadyStateSolver):
    """
    k-eigenvalue multigroup diffusion.
    """

    def __init__(self) -> None:
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
        """
        Execute the k-eigenvalue multigroup diffusion solver.

        Parameters
        ----------
        verbose : int, default 0
        """
        if (self.adjoint and not self._adjoint_matrices or
                not self.adjoint and self._adjoint_matrices):
            self._transpose_matrices()

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
                print(f'\n===== Iteration {nit} =====\n'
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
            msg = '***** k-Eigenvalue Solver Converged! *****'
        else:
            msg = '***** WARNING: k-Eigenvalue Solver NOT Converged *****'
        header = '*' * len(msg)

        if verbose > 0 or not converged:
            print('\n'.join(['', header, msg, header]))
            print(f'Final k Effective:\t\t{self.k_eff:.6g}')
            print(f'Final k Effective Change:\t{k_eff_change:3e}')
            print(f'Final Phi Change:\t\t{phi_change:.3e}')
            print(f'# of Iterations:\t\t{nit}')

    def assemble_matrix(self) -> csr_matrix:
        A = self.L - self.S
        return self.apply_matrix_bcs(A)

    def assemble_rhs(self) -> np.ndarray:
        b = self.Fp @ self.phi / self.k_eff
        if self.use_precursors:
            b += self.Fd @ self.phi / self.k_eff
        return self.apply_vector_bcs(b)

    def compute_fission_production(self) -> float:
        """
        Compute the fission production.

        Returns
        -------
        float
        """
        production = np.sum(self.Fp @ self.phi)
        if self.use_precursors:
            production += np.sum(self.Fd @ self.phi)
        return production

    def compute_k_sensitivity(self, variable: str = 'density') -> float:
        """
        Compute the sensitivity of the k-eigenvalue with respect to
        the provided variable.

        Parameters
        ----------
        variable : str, default 'density'
            The variable to modify.

        Returns
        -------
        float
        """
        update, ref = self._get_update_functions(variable)

        # Compute perturbation size
        eps = (1.0 + ref) * np.sqrt(np.finfo(float).eps)

        # Forward perturbation
        update(ref + eps)
        self.execute()
        k_plus = self.k_eff

        # Backward perturbation
        update(ref - eps)
        self.execute()
        k_minus = self.k_eff

        return (k_plus - k_minus) / (2.0 * eps)

    def _get_update_functions(self, variable: str) -> Tuple[callable, float]:
        """
        Get the update function and reference value for a given variable.

        Parameters
        ----------
        variable : str, default 'density'
            The variable to modify.

        Returns
        -------
        callable : A function to modify a solver parameter.
        float : The reference value of the parameter.
        """
        if variable == 'density':
            from pyPDEs.material import CrossSections

            # Get first material's cross sections
            xs = None
            for material_property in self.materials[0].properties:
                if isinstance(material_property, CrossSections):
                    xs = material_property

            # Get the reference density
            ref = xs.density

            # Define the update function
            def update(v: float) -> None:
                xs.density = v
                self.initialize()

        elif variable == 'radius':
            from pyPDEs.mesh import create_1d_mesh
            if self.mesh.dim != 1:
                raise ValueError(
                    'Radius can only be varied in 1D problems.')

            # Get the last vertex coordinate
            ref = self.mesh.vertices[-1].z

            # Define the update function
            def update(v: float) -> None:
                n_cells = self.mesh.n_cells
                coord_sys = self.mesh.coord_sys
                self.mesh = create_1d_mesh([0.0, v], [n_cells],
                                           coord_sys=coord_sys)
                self.discretization = FiniteVolume(self.mesh)
                self.initialize()

        else:
            raise NotImplementedError(
                'The provided variable is not available.')

        return update, ref