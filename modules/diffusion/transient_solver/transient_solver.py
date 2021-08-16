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

from .. import KEigenvalueSolver
from ..outputs import Outputs


class TransientSolver(KEigenvalueSolver):
    """
    Transient solver for multi-group diffusion problems.
    """

    from .assemble_fv import fv_assemble_mass_matrix
    from .assemble_fv import fv_set_transient_source
    from .assemble_fv import fv_update_precursors
    from .assemble_fv import fv_compute_power

    from .assemble_pwc import pwc_assemble_mass_matrix
    from .assemble_pwc import pwc_set_transient_source
    from .assemble_pwc import pwc_update_precursors
    from .assemble_pwc import pwc_compute_power

    def __init__(self) -> None:
        super().__init__()
        self.initial_conditions: list = None

        self.output_frequency: float = 0.01
        self.next_output_time: float = 0.01
        self.max_dt: float = 0.01

        self.t_final: float = 0.25
        self.dt: float = 5.0e-3
        self.stepping_method: str = "CRANK_NICHOLSON"
        self.adaptivity: bool = False
        self.refine_level: float = 0.01
        self.coarsen_level: float = 0.001

        self.lag_precursors: bool = False

        self.power: float = 1.0
        self.power_old: float = 1.0

        self.energy_per_fission: float = 1.0

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
            self.M.append(self.assemble_mass_matrix(g))

        self.assemble_evolution_matrices()
        self.compute_initial_values()

        self.outputs.reset()
        self.store_outputs(0.0)

    def execute(self, verbose: bool = False) -> None:
        """
        Execute the transient multi-group diffusion solver.
        """
        print("\n***** Executing the transient "
              "multi-group diffusion solver. *****")

        # ======================================== Start time stepping
        time, n_steps, dt0 = 0.0, 0, self.dt
        while time < self.t_final - sys.float_info.epsilon:

            # ============================== Force end time
            if time + self.dt > self.t_final:
                self.dt = self.t_final - time
                self.assemble_evolution_matrices()

            # ============================== Solve time step
            self.solve_time_step()
            time += self.dt
            n_steps += 1
            self.store_outputs(time)

            # ============================== Reset vectors
            self.phi_old[:] = self.phi
            if self.use_precursors:
                self.precursors_old[:] = self.precursors

            if verbose:
                print(f"*** Time Step: {n_steps}\t "
                      f"Time: [{time - self.dt:.3e}, {time:.3e}] ***")

        self.dt = dt0  # reset dt to original
        print("\n***** Done executing transient "
              "multi-group diffusion solver. *****å")

    def solve_time_step(self) -> None:
        """
        Solve a full time step.
        """
        # ======================================== First step of time step
        self.solve_system(step=0)

        # ======================================== Compute precursors
        if self.use_precursors:
            self.update_precursors(step=0)

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
                    self.update_precursors(step=1)

    def solve_system(self, step: int = 0) -> None:
        """
        Solve the system for the n'th step of a time step.

        Parameters
        ----------
        step : int, default 0
            The section of the time step.
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
                self.set_transient_source(g, self.phi, step)
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
        Assemble the previous time step contributions to the
        right-hand side.

        Parameters
        ----------
        step : int, default 0
            The section of the time step.

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

    def assemble_mass_matrix(self, g: int) -> csr_matrix:
        """
        Assemble the mass matrix for time stepping for group `g`

        Parameters
        ----------
        g : int
            The group under consideration.

        Returns
        -------
        csr_matrix
        """
        if isinstance(self.discretization, FiniteVolume):
            return self.fv_assemble_mass_matrix(g)
        else:
            return self.pwc_assemble_mass_matrix(g)

    def set_transient_source(self: 'TransientSolver', g: int,
                             phi: ndarray, step: int = 0) -> None:
        """
        Assemble the right-hand side of the diffusion equation.
        This includes the previous time step contributions and
        material, scattering, fission, and boundary sources for
        group `g`.

        Parameters
        ----------
        g : int
            The group under consideration.
        phi : ndarray
            The vector to compute scattering and fission sources with.
        step : int, default 0
            The section of the time step.
        """
        if isinstance(self.discretization, FiniteVolume):
            self.fv_set_transient_source(g, phi, step)
        else:
            self.pwc_set_transient_source(g, phi, step)

    def update_precursors(self, step: int = 0) -> None:
        """
        Solve a precursor time step.

        Parameters
        ----------
        step : int, default 0
            The section of the time step.
        """
        if isinstance(self.discretization, FiniteVolume):
            self.fv_update_precursors(step)
        else:
            self.pwc_update_precursors(step)

    def compute_power(self) -> float:
        """
        Compute the fission power with the most recent scalar flux solution.

        Returns
        -------
        float
        """
        if isinstance(self.discretization, FiniteVolume):
            return self.fv_compute_power()
        else:
            return self.pwc_compute_power()

    def effective_dt(self, step: int = 0) -> float:
        """
        Compute the effective time step size for the specified
        step of a time step.

        Parameters
        ----------
        step : int
            The section of the time step.

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
        if self.initial_conditions is None:
            super(TransientSolver, self).execute(verbose=False)
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

    def store_outputs(self, time: float) -> None:
        self.outputs.store_outputs(self, time)

    def write_outputs(self, path: str = ".") -> None:
        self.outputs.write_outputs(path)
