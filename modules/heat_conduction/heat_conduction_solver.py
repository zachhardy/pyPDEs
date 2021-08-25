import numpy as np
from numpy.linalg import norm
from numpy import ndarray

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.sparse import csr_matrix

from matplotlib import pyplot as plt

from typing import List, Union, Callable, Tuple

from pyPDEs.mesh import Mesh
from pyPDEs.spatial_discretization import (SpatialDiscretization,
                                           PiecewiseContinuous)
from pyPDEs.utilities.boundaries import (Boundary, DirichletBoundary,
                                         NeumannBoundary, RobinBoundary)

SolverOutput = Tuple[bool, int, float]

class HeatConductionSolver:
    """Class for solving heat conduction problems.
    """

    def __init__(self) -> None:
        self.mesh: Mesh = None
        self.discretization: SpatialDiscretization = None
        self.boundaries: List[Boundary] = None

        self.k: List[Union[Callable, float]] = None
        self.q: List[float] = None

        self.u: ndarray = None

        self.nonlinear_method: str = "PICARD"
        self.nonlinear_tolerance: float = 1.0e-8
        self.nonlinear_max_iterations: int = 1000

        self.linear_tolerance: float = 1.0e-8

    def initialize(self) -> None:
        """Initialize the heat conduction solver.
        """
        self.check_inputs()
        sd = self.discretization
        self.u = np.zeros(sd.n_nodes)

    def execute(self, verbose=False) -> None:
        """Execute the heat conduction solver.
        """
        print("\n***** Executing the steady-state "
              "heat conduction solver. *****\n")

        # Solve linear problem
        if all([not callable(k) for k in self.k]):
            A = self.diffusion_matrix()
            b = self.set_source()
            self.u = spsolve(A, b)

        # Solve nonlinear problem
        else:
            converged, nit, u_change = None, None, None

            # Picard iterations
            if self.nonlinear_method == "PICARD":
                out = self.solve_picard_iterations(verbose)
                converged, nit, u_change = out

            # Newton iterations
            elif "NEWTON" in self.nonlinear_method:
                out = self.solve_newton_iterations(verbose)
                converged, nit, u_change = out

            # Print summary
            if converged:
                msg = "***** Solver Converged *****"
            else:
                msg = "***** WARNING: Solver NOT Converged *****"
            header = "*" * len(msg)
            print("\n".join(["", header, msg, header]))
            print(f"Nonlinear Method:\t{self.nonlinear_method}")
            print(f"Final Change:\t\t{u_change:.3e}")
            print(f"# of Iterations:\t{nit}")

    def solve_picard_iterations(self, verbose: bool) -> SolverOutput:
        """Solve the problem with Picard iterations

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
            A = self.diffusion_matrix()
            b = self.set_source()
            self.u = spsolve(A, b)

            # Check convergence
            u_change = norm(self.u - u_ell)
            u_ell[:] = self.u

            # Iteration summary
            if verbose:
                msg = f"Iteration {nit} --- "
                msg += f"Difference: {u_change:1.4e}"
                print(msg)

            # Break, if converged
            if u_change < self.nonlinear_tolerance:
                converged = True
                break
        return converged, nit, u_change

    def solve_newton_iterations(self, verbose: bool) -> SolverOutput:
        """Solve the problem with Newton iterations

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
            if "DIRECT" in self.nonlinear_method:
                J = self.jacobian(self.u, r)  # define Jacobian
                du = np.linalg.solve(J, -r)

            # GMRES based solve
            else:
                # Determine whether JFNK, or not
                method = self.nonlinear_method
                jfnk = False if "GMRES" in method else True

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
                msg = f"Iteration {nit:>3} --- "
                msg += f"Difference: {u_change:^.4e}"
                if "DIRECT" not in self.nonlinear_method:
                    msg += f" --- Linear Iterations: {counter.nit}"
                print(msg)

            # Break, if converged
            if u_change < self.nonlinear_tolerance:
                converged = True
                break
        return converged, nit, u_change

    def residual(self, u: ndarray) -> ndarray:
        """Compute the residual associated with a given state.

        Parameters
        ----------
        u : ndarray
            The state to compute a residual with.

        Returns
        -------
        ndarray
        """
        A = self.diffusion_matrix()
        b = self.set_source()
        return A @ u - b

    def jacobian(self, u: ndarray, r: ndarray,
                 jfnk=False) -> Union[ndarray, LinearOperator]:
        """Compute the Jacobian matrix associated with a given state.

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

    def diffusion_matrix(self) -> csr_matrix:
        """Assemble the heat conduction matrix.

        Returns
        -------
        csr_matrix
        """
        pwc: PiecewiseContinuous = self.discretization

        # Loop over cells
        rows, cols, data = [], [], []
        for cell in self.mesh.cells:
            view = pwc.fe_views[cell.id]

            # Evaluate the conductivity
            k = self.k[cell.material_id]
            if callable(k):
                u = self.u[view.node_ids]
                uq = view.get_function_values(u)
                k = k(uq)
            else:
                k = np.array([k] * view.n_qpoints)

            # Loop over quadrature
            for qp in range(view.n_qpoints):
                jxw = view.jxw[qp]

                # Loop over test functions
                for i in range(view.n_nodes):
                    ii = pwc.map_dof(cell, i)
                    grad_i = view.grad_shape_values[i][qp]

                    # Loop over trial functions
                    for j in range(view.n_nodes):
                        jj = pwc.map_dof(cell, j)
                        grad_j = view.grad_shape_values[j][qp]

                        value = k[qp] * grad_i.dot(grad_j) * jxw
                        rows.append(ii)
                        cols.append(jj)
                        data.append(value)

            # Loop over faces
            for f_id, face in enumerate(cell.faces):
                if not face.has_neighbor:
                    bndry_id = -1 * (face.neighbor_id + 1)
                    bc = self.boundaries[bndry_id]

                    # Dirichlet boundary
                    if issubclass(type(bc), DirichletBoundary):

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ii = pwc.map_face_dof(cell, f_id, fi)
                            pwc.zero_dirichlet_row(ii, rows, data)
                            rows.append(ii)
                            cols.append(ii)
                            data.append(1.0)

                    # Robin boundary
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ni = view.face_node_mapping[f_id][fi]
                            ii = pwc.map_face_dof(cell, f_id, fi)

                            # Loop over face nodes
                            for fj in range(n_face_nodes):
                                nj = view.face_node_mapping[f_id][fj]
                                jj = pwc.map_face_dof(cell, f_id, fj)

                                intS_mass_ij = \
                                    view.intS_shapeI_shapeJ[f_id][ni][nj]

                                value = bc.a / bc.b * intS_mass_ij
                                rows.append(ii)
                                cols.append(jj)
                                data.append(value)
        return csr_matrix((data, (rows, cols)), shape=(pwc.n_nodes,) * 2)

    def set_source(self) -> ndarray:
        """Assemble the right-hand side.

        Returns
        -------
        ndarray
        """
        pwc: PiecewiseContinuous = self.discretization

        # Loop over cells
        b = np.zeros(pwc.n_nodes)
        for cell in self.mesh.cells:
            view = pwc.fe_views[cell.id]
            q = self.q[cell.material_id]

            # Loop test functions
            for i in range(view.n_nodes):
                ii = pwc.map_dof(cell, i)

                # Loop over quadrature
                for qp in range(view.n_qpoints):
                    b[ii] += q * view.shape_values[i][qp] * view.jxw[qp]

            # Loop over faces
            for f_id, face in enumerate(cell.faces):
                if not face.has_neighbor:
                    bndry_id = -1 * (face.neighbor_id + 1)
                    bc = self.boundaries[bndry_id]

                    # Dirichlet boundary
                    if issubclass(type(bc), DirichletBoundary):
                        bc: DirichletBoundary = bc

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ii = pwc.map_face_dof(cell, f_id, fi)
                            b[ii] = bc.value

                    # Neumann boundary
                    elif issubclass(type(bc), NeumannBoundary):
                        bc: NeumannBoundary = bc

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ni = view.face_node_mapping[f_id][fi]
                            ii = pwc.map_face_dof(cell, f_id, fi)
                            b[ii] += bc.value * view.intS_shapeI[f_id][ni]

                    # Robin boundary
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ni = view.face_node_mapping[f_id][fi]
                            ii = pwc.map_face_dof(cell, f_id, fi)
                            b[ii] += bc.f / bc.b * view.intS_shapeI[f_id][ni]
        return b

    def plot_solution(self, title: str = None) -> None:
        """Plot the currently stored solution.

        Parameters
        ----------
        title : str, default None
            A title for the figure.s
        """
        grid = self.discretization.grid
        if title:
            plt.title(title)
        plt.xlabel("Location")
        plt.ylabel(r"T(r)")
        plt.plot(grid, self.u, '-ob', label='Temperature')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    def check_inputs(self) -> None:
        """Check the inputs of the solver.
        """
        self._check_mesh()
        self._check_discretization()
        self._check_boundaries()
        self._check_materials()

    def _check_mesh(self) -> None:
        if not self.mesh:
            raise AssertionError("No mesh is attached to the solver.")
        elif self.mesh.dim != 1:
            raise NotImplementedError(
                "Only 1D problems have been implemented.")

    def _check_discretization(self) -> None:
        if not self.discretization:
            raise AssertionError(
                "No discretization is attached to the solver.")
        elif self.discretization.type not in ["PWC"]:
            raise NotImplementedError(
                "Only finite volume has been implemented.")

    def _check_boundaries(self) -> None:
        if not self.boundaries:
            raise AssertionError(
                "No boundary conditions are attached to the solver.")
        elif len(self.boundaries) != 2:
            raise NotImplementedError(
                "There can only be 2 * n_groups boundary conditions "
                "for 1D problems.")

    def _check_materials(self) -> None:
        mat_ids = [c.material_id for c in self.mesh.cells]
        n_mats = len(np.unique(mat_ids))
        if len(self.k) != n_mats:
            raise AssertionError(
                f"Only {len(self.k)} conductivities provided when there "
                f"are {n_mats} material IDs.")
        if len(self.q) != n_mats:
            raise AssertionError(
                f"Only {len(self.q)} sources provided when there "
                f"are {n_mats} material IDs.")
