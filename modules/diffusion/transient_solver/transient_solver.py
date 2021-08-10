import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
from numpy import ndarray
from scipy.sparse import csr_matrix
from typing import List

from pyPDEs.spatial_discretization import (FiniteVolume,
                                           PiecewiseContinuous,
                                           SpatialDiscretization)

from modules.diffusion.keigenvalue_solver import KEigenvalueSolver


class Outputs:
    def __init__(self):
        self.grid: List[List[float]] = []
        self.time: List[float] = []
        self.power: List[float] = []
        self.flux: List[List[ndarray]] = []

    def store_grid(self, sd: SpatialDiscretization):
        self.grid.clear()
        for point in sd.grid:
            self.grid.append([point.x, point.y, point.z])

    def store_outputs(self, solver: 'TransientSolver',
                      time: float) -> None:
        if time == 0.0:
            self.store_grid(solver.discretization)

        self.time.append(time)

        power = solver.fv_compute_fission_production()
        self.power.append(power)

        n_grps, phi = solver.n_groups, np.copy(solver.phi)
        flux = [phi[g::n_grps] for g in range(n_grps)]
        self.flux.append(flux)

    def write_outputs(self, path: str = ".") -> None:
        if not os.path.isdir(path):
            os.makedirs(path)

        time_path = os.path.join(path, "time.txt")
        np.savetxt(time_path, self.time, fmt="%.6g")

        grid_path = os.path.join(path, "grid.txt")
        np.savetxt(grid_path, self.grid, fmt="%.6g")

        flux_dirpath = os.path.join(path, "flux")
        if not os.path.isdir(flux_dirpath):
            os.makedirs(flux_dirpath)
        os.system(f"rm -r {flux_dirpath}/*")

        for g in range(len(self.flux[0])):
            group_path = os.path.join(flux_dirpath, f"g{g}.txt")
            np.savetxt(group_path, np.array(self.flux)[:, g])

    def reset(self):
        self.grid.clear()
        self.time.clear()
        self.power.clear()
        self.flux.clear()

class TransientSolver(KEigenvalueSolver):
    """
    Transient solver for multi-group diffusion problems.
    """

    from .assemble_fv import fv_assemble_mass_matrix
    from .assemble_fv import fv_set_transient_source
    from .assemble_fv import fv_update_precursors

    from .assemble_pwc import pwc_assemble_mass_matrix
    from .assemble_pwc import pwc_set_transient_source
    from .assemble_pwc import pwc_update_precursors

    def __init__(self) -> None:
        super().__init__()
        self.initial_conditions: list = None

        self.t_final: float = 0.25
        self.dt: float = 5.0e-3
        self.stepping_method: str = "CRANK_NICHOLSON"

        self.output_dir: str = None

        self.lag_precursors: bool = False

        self.phi_old: ndarray = None
        self.precursors_old: ndarray = None

        self.b_old: ndarray = None
        self.M: List[csr_matrix] = None
        self.A: List[List[csr_matrix]] = None

        self.outputs: Outputs = Outputs()

    def initialize(self) -> None:
        """
        Initialize the transient multi-group diffusion solver.
        """
        super(KEigenvalueSolver, self).initialize()

        self.b_old = np.copy(self.b)
        self.phi_old = np.copy(self.phi)
        if self.use_precursors:
            self.precursors_old = np.copy(self.precursors)

        self.M = []
        for g in range(self.n_groups):
            if isinstance(self.discretization, FiniteVolume):
                self.M.append(self.fv_assemble_mass_matrix(g))
            else:
                self.M.append(self.pwc_assemble_mass_matrix(g))

        self.assemble_evolution_matrices()
        self.compute_initial_values()

        self.outputs.reset()
        self.outputs.store_outputs(self, 0.0)

    def execute(self) -> None:
        """
        Execute the transient multi-group diffusion solver.
        """
        print("\n***** Executing the transient "
              "multi-group diffusion solver. *****\n")

        # ======================================== Start time stepping
        time, n_steps = 0.0, 0
        while time < self.t_final - sys.float_info.epsilon:

            # ============================== Force end time
            if time + self.dt > self.t_final:
                self.dt = self.t_final - time
                self.assemble_evolution_matrices()

            # ============================== Solve time step
            self.solve_time_step()
            time += self.dt
            n_steps += 1
            self.outputs.store_outputs(self, time)

            # ============================== Reset vectors
            self.phi_old[:] = self.phi
            if self.use_precursors:
                self.precursors_old[:] = self.precursors

            print(f"*** Time Step: {n_steps}\t "
                  f"Time: [{time - self.dt:.3e}, {time:.3e}] ***")

        print("\n***** Done executing transient "
              "multi-group diffusion solver. *****\n")

    def solve_time_step(self) -> None:
        """
        Solve a full time step.
        """
        # ======================================== First step of time step
        self.solve_system(step=0)

        # ======================================== Compute precursors
        if self.use_precursors:
            if isinstance(self.discretization, FiniteVolume):
                self.fv_update_precursors(step=0)
            else:
                self.pwc_update_precursors(step=0)

        # ======================================== Post-process results
        if self.stepping_method in ["CRANK_NICHOLSON", "TBDF2"]:
            self.phi = 2.0 * self.phi - self.phi_old
            if self.use_precursors:
                self.precursors = 2.0 * self.precursors - self.precursors_old

            # ============================== Second step of time step
            if self.stepping_method == "TBDF2":
                self.solve_system(step=1)

                # ========================= Compute precursors
                if self.use_precursors:
                    if isinstance(self.discretization, FiniteVolume):
                        self.fv_update_precursors(step=1)
                    else:
                        self.pwc_update_precursors(step=1)

    def solve_system(self, step: int = 0) -> None:
        """
        Solve the system for a n'th step of the time step.

        Parameters
        ----------
        step : int, default 0
            The step of the time step.
        """
        n_grps = self.n_groups
        phi_ell = np.copy(self.phi)
        b_old = self.set_old_transient_source(step)

        # ======================================== Start iterating
        converged = False
        for nit in range(self.max_iterations):

            # =================================== Solve group-wise
            self.b[:] = b_old
            for g in range(n_grps):
                if isinstance(self.discretization, FiniteVolume):
                    self.fv_set_transient_source(g, self.phi, step)
                else:
                    self.pwc_set_transient_source(g, self.phi, step)
                self.phi[g::n_grps] = spsolve(self.A[g][step],
                                              self.b[g::n_grps])

            # =================================== Check convergence
            phi_change = norm(self.phi - phi_ell)
            phi_ell[:] = self.phi
            if phi_change <= self.tolerance:
                converged = True
                break

        if not converged:
            print(f"!!!!! WARNING: Solver did not converge. !!!!!")

    def set_old_transient_source(self, step: int = 0) -> ndarray:
        """
        Assemble the previous time step contributions to the right-hand side.

        Parameters
        ----------
        step : int, default 0
            The step of the time step.

        Returns
        -------
        ndarray
        """
        b = np.zeros(self.b.shape)
        n_grps = self.n_groups

        for g in range(self.n_groups):
            phi_old = self.phi_old[g::self.n_groups]

            if self.stepping_method == "BACKWARD_EULER":
                b[g::n_grps] = self.M[g] / self.dt @ phi_old
            elif self.stepping_method == "CRANK_NICHOLSON":
                b[g::n_grps] = 2.0 * self.M[g] / self.dt @ phi_old
            elif self.stepping_method == "TBDF2" and step == 0:
                b[g::n_grps] = 4.0 * self.M[g] / self.dt @ phi_old
            elif self.stepping_method == "TBDF2" and step == 1:
                phi = self.phi[g::self.n_groups]
                b[g::n_grps] = self.M[g] / self.dt @ (4.0 * phi - phi_old)
        return b

    def assemble_evolution_matrices(self) -> csr_matrix:
        """
        Assemble the list of matrices used for evolving the
        solution one time step.

        Returns
        -------
        csr_matrix
        """
        self.A = []
        for g in range(self.n_groups):
            if self.stepping_method == "BACKWARD_EULER":
                matrices = [self.L[g] + self.M[g] / self.dt]
            elif self.stepping_method == "CRANK_NICHOLSON":
                matrices = [self.L[g] + 2.0 * self.M[g] / self.dt]
            elif self.stepping_method == "TBDF2":
                matrices = [self.L[g] + 4.0 * self.M[g] / self.dt,
                            self.L[g] + 3.0 * self.M[g] / self.dt]
            else:
                raise NotImplementedError(
                    f"{self.stepping_method} is not implemented.")

            self.A.append(matrices)

    def effective_dt(self, step: int = 0) -> float:
        """
        Compute the effective time step size for the specified
        step of a time step.

        Parameters
        ----------
        step : int
            The step of the time step.

        Returns
        -------
        float
        """
        if self.stepping_method == "BACKWARD_EULER":
            return self.dt
        elif self.stepping_method == "CRANK_NICHOLSON":
            return self.dt / 2.0
        elif self.stepping_method == "TBDF2" and step == 0:
            return self.dt / 4.0
        elif self.stepping_method == "TBDF2" and step == 1:
            return self.dt / 3.0
        else:
            raise NotImplementedError(
                f"{self.stepping_method} is not implemented.")

    def compute_initial_values(self) -> None:
        """
        Evaluate the initial conditions.
        """
        if not self.initial_conditions:
            super(KEigenvalueSolver, self).execute()
        else:
            n_grps = self.n_groups
            n_prec = self.n_precursors
            num_ics = [n_grps, n_grps + n_prec]
            grid = self.discretization.grid
            grid = np.array([p.z for p in grid])

            # ======================================== Convert to lambdas
            from sympy import lambdify
            from sympy.matrices.dense import MutableDenseMatrix
            if isinstance(self.initial_conditions, MutableDenseMatrix):
                ics = []
                symbols = list(self.initial_conditions.free_symbols)
                for ic in self.initial_conditions:
                    ics.append(lambdify(symbols, ic))
                self.initial_conditions = ics

            # ================================================== Evaluate ics
            if all([len(self.initial_conditions) != n for n in num_ics]):
                raise AssertionError(
                    "Invalid number of initial conditions provided.")

            # ================================================== Flux ics
            for g, ic in enumerate(self.initial_conditions[:n_grps]):
                if callable(ic):
                    self.phi[g::n_grps] = ic(grid)
                elif len(ic) == len(self.phi[g::n_grps]):
                    self.phi[g::n_grps] = ic
                else:
                    raise ValueError(
                        f"Provided initial condition for group {g} "
                        f"does not agree with discretization.")

            # ================================================== Precursor ics
            if self.use_precursors:
                for j, ic in enumerate(self.initial_conditions[n_grps:]):
                    if callable(ic):
                        self.precursors[j::n_prec] = ic(grid)
                    elif len(ic) == len(self.precursors[j::n_prec]):
                        self.precursors[j::n_prec] = ic
                    else:
                        raise ValueError(
                            f"Provided initial condition for family {j} "
                            f"does not agree with discretization.")

            self.phi_old[:] = self.phi
            if self.use_precursors:
                self.precursors_old[:] = self.precursors
