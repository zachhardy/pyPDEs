import numpy as np
from numpy.linalg import norm
from numpy import ndarray

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.sparse import csr_matrix, lil_matrix

from matplotlib import pyplot as plt

from typing import List, Union, Callable, Tuple

from pyPDEs.mesh import Mesh
from pyPDEs.spatial_discretization import (SpatialDiscretization,
                                           PiecewiseContinuous)
from pyPDEs.utilities.boundaries import (Boundary, DirichletBoundary,
                                         NeumannBoundary, RobinBoundary)

SolverOutput = Tuple[bool, int, float]

class HeatConductionSolver:
    """
    Class for solving heat conduction problems.
    """

    from ._pwc import (assemble_matrix,
                       assemble_rhs,
                       apply_matrix_bcs,
                       apply_vector_bcs)

    from ._plotting import plot_solution

    def __init__(self) -> None:
        self.mesh: Mesh = None
        self.discretization: SpatialDiscretization = None
        self.boundaries: List[Boundary] = None

        self.k: List[Union[Callable, float]] = None
        self.q: List[float] = None

        self.u: ndarray = None

        self.nonlinear_method: str = 'picard'
        self.nonlinear_tolerance: float = 1.0e-8
        self.nonlinear_max_iterations: int = 1000

        self.linear_tolerance: float = 1.0e-8

    def initialize(self) -> None:
        """
        Initialize the heat conduction solver.
        """
        self._check_inputs()
        sd = self.discretization
        self.u = np.zeros(sd.n_nodes)

    def execute(self, verbose=False) -> None:
        """
        Execute the heat conduction solver.
        """
        print('\n***** Executing the steady-state '
              'heat conduction solver. *****\n')

        # Solve linear problem
        if all([not callable(k) for k in self.k]):
            A = self.assemble_matrix()
            b = self.assemble_rhs()
            self.u = spsolve(A, b)

        # Solve nonlinear problem
        else:
            converged, nit, u_change = None, None, None

            # Picard iterations
            if self.nonlinear_method == 'picard':
                out = self.solve_picard_iterations(verbose)
                converged, nit, u_change = out

            # Newton iterations
            elif 'newton' in self.nonlinear_method:
                out = self.solve_newton_iterations(verbose)
                converged, nit, u_change = out

            # Print summary
            if converged:
                msg = '***** Solver Converged *****'
            else:
                msg = '***** WARNING: Solver NOT Converged *****'
            header = '*' * len(msg)
            print('\n'.join(['', header, msg, header]))
            print(f'Nonlinear Method:\t{self.nonlinear_method}')
            print(f'Final Change:\t\t{u_change:.3e}')
            print(f'# of Iterations:\t{nit}')

    def solve_picard_iterations(self, verbose: bool) -> SolverOutput:
        """
        Solve the problem with Picard iterations

        Parameters
        ----------
        verbose : bool
            Flag for printing iteration information.

        Returns
        -------
        converged : bool
            If True, iterations converged.
        nit : int
            The number of iterations taken.
        u_change : float
            The final difference between iterates.
        """
        u_ell = np.copy(self.u)

        # Start iterating
        u_change, nit, converged = 1.0, 0, False
        for nit in range(self.nonlinear_max_iterations):

            # Solve the system
            A = self.assemble_matrix()
            b = self.assemble_rhs()
            self.u = spsolve(A, b)

            # Check convergence
            u_change = norm(self.u - u_ell)
            u_ell[:] = self.u

            # Iteration summary
            if verbose:
                msg = f'Iteration {nit} --- '
                msg += f'Difference: {u_change:1.4e}'
                print(msg)

            # Break, if converged
            if u_change < self.nonlinear_tolerance:
                converged = True
                break
        return converged, nit, u_change

    def solve_newton_iterations(self, verbose: bool) -> SolverOutput:
        """
        Solve the problem with Newton iterations

        Parameters
        ----------
        verbose : bool
            Flag for printing iteration information.

        Returns
        -------
        converged : bool
            If True, iterations converged.
        nit : int
            The number of iterations taken.
        u_change : float
            The final difference between iterates.
        """
        class GMRESCounter(object):
            def __init__(self) -> None:
                self.nit = 0

            def __call__(self, rk=None) -> None:
                self.nit += 1
        counter = GMRESCounter()

        # Start iterating
        u_change, nit, converged = 1.0, 0, False
        for nit in range(self.nonlinear_max_iterations):
            r = self.residual(self.u)  # compute the residual

            # Direct solve
            if 'direct' in self.nonlinear_method:
                J = self.jacobian(self.u, r)  # define Jacobian
                du = np.linalg.solve(J, -r)

            # GMRES based solve
            else:
                # Determine whether JFNK, or not
                method = self.nonlinear_method
                jfnk = False if 'gmres' in method else True

                # Construct the Jacocian, or its action
                J = self.jacobian(self.u, r, jfnk=jfnk)

                # Solve using GMRES
                counter.nit = 0
                du = gmres(J, -r, x0=self.u, restart=1000,
                           maxiter=10*self.discretization.n_nodes,
                           tol=self.linear_tolerance,
                           callback=counter)[0]

            # Check convergence
            u_change = norm(du)
            self.u += du

            # Print iteration summary
            if verbose:
                msg = f'Iteration {nit:>3} --- '
                msg += f'Difference: {u_change:^.4e}'
                if 'direct' not in self.nonlinear_method:
                    msg += f' --- Linear Iterations: {counter.nit}'
                print(msg)

            # Break, if converged
            if u_change < self.nonlinear_tolerance:
                converged = True
                break
        return converged, nit, u_change

    def residual(self, u: ndarray) -> ndarray:
        """
        Compute the residual associated with a given state.

        Parameters
        ----------
        u : ndarray
            The state to compute a residual with.

        Returns
        -------
        ndarray
        """
        A = self.assemble_matrix()
        b = self.assemble_rhs()
        return A @ u - b

    def jacobian(self, u: ndarray, r: ndarray,
                 jfnk=False) -> Union[ndarray, LinearOperator]:
        """
        Compute the Jacobian matrix associated with a given state.

        This can be computed as an actual matrix by perturbing
        the vector `u` element-wise or by defining the action
        of the Jacobian on a vector.

        Parameters
        ----------
        u : ndarray
            The state to compute the Jacobian from.
        r : ndarray
            The residual to compute the Jacobian from.
        jfnk : bool, default False
            A flag for using the Jacobian-Free-Newton-Krylov
            method, which constructs the action of the
            Jacobian rather than the numerical Jacobian matrix.

        Returns
        -------
        ndarray or LinearOperator
            The prior if jfnk is False, the latter otherwise.
        """
        eps_m = np.finfo(float).eps
        n_nodes = self.discretization.n_nodes

        # Construct the numerical Jacobian
        if not jfnk:
            J = np.zeros((n_nodes,) * 2)

            # Perturb each DoF, compute sensitivity
            for idof in range(n_nodes):
                eps = np.zeros(n_nodes)
                eps[idof] = (1.0 + np.abs(u[idof])) * np.sqrt(eps_m)
                rp = self.residual(u + eps)
                J[:, idof] = (rp - r) / eps[idof]

        # Construct the Jacobian action function
        else:
            def jacobian_action(v):
                if np.all(v == 0):
                    return np.zeros(len(v))
                else:
                    norm_v = norm(v)
                    eps = np.sqrt(eps_m) * np.average(1 + u) / norm_v
                    rp = self.residual(u + eps * v)
                return (rp - r) / eps
            J = LinearOperator((n_nodes,) * 2, matvec=jacobian_action)
        return J

    def _check_inputs(self) -> None:
        """
        Check the inputs of the solver.
        """
        self._check_mesh()
        self._check_discretization()
        self._check_boundaries()
        self._check_materials()

    def _check_mesh(self) -> None:
        if not self.mesh:
            raise AssertionError('No mesh is attached to the solver.')
        elif self.mesh.dim != 1:
            raise NotImplementedError(
                'Only 1D problems have been implemented.')

    def _check_discretization(self) -> None:
        if not self.discretization:
            raise AssertionError(
                'No discretization is attached to the solver.')
        elif self.discretization.type not in ['pwc']:
            raise NotImplementedError(
                'Only piecewise continuous has been implemented.')

    def _check_boundaries(self) -> None:
        if not self.boundaries:
            raise AssertionError(
                'No boundary conditions are attached to the solver.')
        elif len(self.boundaries) != 2:
            raise NotImplementedError(
                'There can only be 2 boundary conditions for 1D problems.')

    def _check_materials(self) -> None:
        mat_ids = [c.material_id for c in self.mesh.cells]
        n_mats = len(np.unique(mat_ids))
        if len(self.k) != n_mats:
            raise AssertionError(
                f'Only {len(self.k)} conductivities provided when there '
                f'are {n_mats} material IDs.')
        if len(self.q) != n_mats:
            raise AssertionError(
                f'Only {len(self.q)} sources provided when there '
                f'are {n_mats} material IDs.')

