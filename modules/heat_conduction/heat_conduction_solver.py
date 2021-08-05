import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres, LinearOperator
from typing import List, Union, Callable, Tuple

from numpy import ndarray
from scipy.sparse import csr_matrix

from pyPDEs.mesh import Mesh
from pyPDEs.spatial_discretization import (SpatialDiscretization,
                                           PiecewiseContinuous)
from pyPDEs.utilities.boundaries import (Boundary, DirichletBoundary,
                                         NeumannBoundary, RobinBoundary)


class HeatConductionSolver:
    """
    Class for solving heat conduction problems.
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
        """
        Initialize the heat conduction solver.
        """
        self.check_inputs()
        sd = self.discretization

        # ======================================== Initialize system
        self.u = np.zeros(sd.n_nodes)

    def execute(self, verbose=False) -> None:
        """
        Execute the heat conduction solver.
        """
        print("\n***** Executing the steady-state "
              "heat conduction solver. *****\n")

        # Solve linear problem
        if all([not callable(k) for k in self.k]):
            A = self.assemble_matrix()
            b = self.assemble_source()
            self.u = spsolve(A, b)

        # Solve nonlinear problem
        else:
            converged, nit, u_change = None, None, None
            if self.nonlinear_method == "PICARD":
                out = self.solve_picard_iterations(verbose)
                converged, nit, u_change = out
            elif "NEWTON" in self.nonlinear_method:
                out = self.solve_newton_iterations(verbose)
                converged, nit, u_change = out

            # ======================================== Print summary
            if converged:
                msg = "***** Solver Converged *****"
            else:
                msg = "!!!!! WARNING: Solver NOT Converged !!!!!"
            msg += f"\nNonlinear Method:\t{self.nonlinear_method}"
            msg += f"\nFinal Change:\t\t{u_change:.3e}"
            msg += f"\n# of Iterations:\t{nit}"
            print(msg)

        print("\n***** Done executing the steady-state "
              "heat conduction solver. *****\n")

    def solve_picard_iterations(
            self, verbose: bool) -> Tuple[bool, int, float]:
        u_ell = np.copy(self.u)
        u_change, nit, converged = 1.0, 0, False
        for nit in range(self.nonlinear_max_iterations):
            A = self.assemble_matrix()
            b = self.assemble_source()
            self.u = spsolve(A, b)
            u_change = norm(self.u - u_ell)

            if verbose:
                msg = f"Iteration {nit} --- "
                msg += f"Difference: {u_change:1.4e}"
                print(msg)

            u_ell[:] = self.u
            if u_change < self.nonlinear_tolerance:
                converged = True
                break
        return converged, nit, u_change

    def solve_newton_iterations(
            self, verbose: bool) -> Tuple[bool, int, float]:

        class GMRESCounter(object):
            def __init__(self) -> None:
                self.nit = 0

            def __call__(self, rk=None) -> None:
                self.nit += 1
        counter = GMRESCounter()

        u_change, nit, converged = 1.0, 0, False
        for nit in range(self.nonlinear_max_iterations):
            r = self.residual(self.u)

            if "DIRECT" in self.nonlinear_method:
                J = self.jacobian(self.u, r)
                du = np.linalg.solve(J, -r)
            else:
                method = self.nonlinear_method
                jfnk = False if "GMRES" in method else True
                J = self.jacobian(self.u, r, jfnk=jfnk)
                counter.nit = 0
                du = gmres(J, -r, x0=self.u, restart=1000,
                           maxiter=10*self.discretization.n_nodes,
                           tol=self.linear_tolerance,
                           callback=counter)[0]

            u_change = norm(du)
            self.u += du

            if verbose:
                msg = f"Iteration {nit:>3} --- "
                msg += f"Difference: {u_change:^.4e}"
                if "DIRECT" not in self.nonlinear_method:
                    msg += f" --- Linear Iterations: {counter.nit}"
                print(msg)

            if u_change < self.nonlinear_tolerance:
                converged = True
                break
        return converged, nit, u_change

    def residual(self, u: ndarray) -> ndarray:
        A = self.assemble_matrix()
        b = self.assemble_source()
        return A @ u - b

    def jacobian(self, u: ndarray, r: ndarray, jfnk=False) -> ndarray:
        eps_m = np.finfo(float).eps
        n_nodes = self.discretization.n_nodes

        if not jfnk:
            J = np.zeros((n_nodes,) * 2)
            for idof in range(n_nodes):
                eps = np.zeros(n_nodes)
                eps[idof] = (1.0 + np.abs(u[idof])) * np.sqrt(eps_m)
                rp = self.residual(u + eps)
                J[:, idof] = (rp - r) / eps[idof]
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

    def assemble_matrix(self) -> csr_matrix:
        """
        Assemble the heat conduction matrix.
        """
        pwc: PiecewiseContinuous = self.discretization

        # ======================================== Loop over cells
        rows, cols, data = [], [], []
        for cell in self.mesh.cells:
            view = pwc.fe_views[cell.id]
            k = self.k[cell.material_id]
            if callable(k):
                u = self.u[view.node_ids]
                uq = view.get_function_values(u)
                k = k(uq)
            else:
                k = np.array([k] * view.n_qpoints)

            # =================================== Loop over quadrature
            for qp in range(view.n_qpoints):
                jxw = view.jxw[qp]

                # ============================== Loop over test functions
                for i in range(view.n_nodes):
                    ii = pwc.map_dof(cell, i)
                    grad_i = view.grad_shape_values[i, qp]

                    # ========================= Loop over trial functions
                    for j in range(view.n_nodes):
                        jj = pwc.map_dof(cell, j)
                        grad_j = view.grad_shape_values[j, qp]

                        value = grad_i * k[qp] * grad_j * jxw
                        rows.append(ii)
                        cols.append(jj)
                        data.append(value)

            # ======================================== Loop over faces
            for f_id, face in enumerate(cell.faces):
                if not face.has_neighbor:
                    bndry_id = -1 * (face.neighbor_id + 1)
                    bc = self.boundaries[bndry_id]

                    # ============================== Dirichlet boundary
                    if issubclass(type(bc), DirichletBoundary):

                        # ==================== Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ii = pwc.map_face_dof(cell, f_id, fi)
                            pwc.zero_dirichlet_row(ii, rows, data)
                            rows.append(ii)
                            cols.append(ii)
                            data.append(1.0)

                    # ============================== Robin boundary
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc

                        # ==================== Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ni = view.face_node_mapping[f_id][fi]
                            ii = pwc.map_face_dof(cell, f_id, fi)

                            # =============== Loop over face nodes
                            for fj in range(n_face_nodes):
                                nj = view.face_node_mapping[f_id][fj]
                                jj = pwc.map_face_dof(cell, f_id, fj)

                                value = bc.a / bc.b
                                value *= view.intS_shapeI_shapeJ[f_id][ni][nj]
                                rows.append(ii)
                                cols.append(jj)
                                data.append(value)

        return csr_matrix((data, (rows, cols)), shape=(pwc.n_nodes,) * 2)

    def assemble_source(self) -> ndarray:
        """
        Assemble the right-hand side source vector.
        """
        pwc: PiecewiseContinuous = self.discretization

        # ======================================== Loop over cells
        b = np.zeros(pwc.n_nodes)
        for cell in self.mesh.cells:
            view = pwc.fe_views[cell.id]
            q = self.q[cell.material_id]

            # =================================== Loop test functions
            for i in range(view.n_nodes):
                ii = pwc.map_dof(cell, i)

                # ============================== Loop over quadrature
                for qp in range(view.n_qpoints):
                    b[ii] += q * view.shape_values[i, qp] * view.jxw[qp]

            # =================================== Loop over faces
            for f_id, face in enumerate(cell.faces):
                if not face.has_neighbor:
                    bndry_id = -1 * (face.neighbor_id + 1)
                    bc = self.boundaries[bndry_id]

                    # ============================== Dirichlet boundary
                    if issubclass(type(bc), DirichletBoundary):
                        bc: DirichletBoundary = bc

                        # ==================== Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ii = pwc.map_face_dof(cell, f_id, fi)
                            b[ii] = bc.value

                    # ============================== Neumann boundary
                    elif issubclass(type(bc), NeumannBoundary):
                        bc: NeumannBoundary = bc

                        # ==================== Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ni = view.face_node_mapping[f_id][fi]
                            ii = pwc.map_face_dof(cell, f_id, fi)
                            b[ii] += bc.value * view.intS_shapeI[f_id][ni]

                    # ============================== Robin boundary
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc

                        # ==================== Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ni = view.face_node_mapping[f_id][fi]
                            ii = pwc.map_face_dof(cell, f_id, fi)
                            b[ii] += bc.f / bc.b * view.intS_shapeI[f_id][ni]
        return b

    def plot_solution(self, title: str = None) -> None:
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
