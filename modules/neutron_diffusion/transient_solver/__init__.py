import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy import ndarray
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from sympy import MutableDenseMatrix
from typing import List, Union

from pyPDEs.spatial_discretization import *

from .. import KEigenvalueSolver
from ..outputs import Outputs


class TransientSolver(KEigenvalueSolver):
    """Class for solving transinet multigroup diffusion problems.
    """

    from ._fv import (_fv_mass_matrix,
                      _fv_precursor_substitution_matrix,
                      _fv_old_precursor_source,
                      _fv_update_precursors,
                      _fv_compute_power)

    from ._pwc import (_pwc_mass_matrix,
                       _pwc_precursor_substitution_matrix,
                       _pwc_old_precursor_source,
                       _pwc_update_precursors,
                       _pwc_compute_power)

    def __init__(self) -> None:
        """Class constructor.
        """
        super().__init__()
        self.initial_conditions: list = None

        # Time stepping parameters
        self.t_final: float = 0.1
        self.dt: float = 2.0e-3
        self.method: str = "TBDF2"

        # Adaptive time stepping parameters
        self.adaptivity: bool = False
        self.refine_level: float = 0.05
        self.coarsen_level: float = 0.01

        # Physics options
        self.lag_precursors: bool = False

        # Power related parameters
        self.power: float = 1.0
        self.E_fission: float = 1.0

        # Feedback related parameters
        self.use_feedback: bool = False
        self.feedback_coeff: float = 1.0e-3

        # Heat generation parameters
        self.T: float = 300.0  # K
        self.mass_density: float = 19.0  # g/cc
        self.specific_heat: float = 0.12  # J/g-K
        self.domain_volume: float = 0.0  # cc

        # Output options
        self.output_frequency: float = 0.0
        self.outputs: Outputs = Outputs()

        # Precomputed mass matrix storage
        self.M: csr_matrix = None
        self.A: List[csr_matrix] = None

        # Previous time step solutions
        self.phi_old: ndarray = None
        self.precursors_old: ndarray = None

    def initialize(self) -> None:
        """Initialize the solver.
        """
        super(KEigenvalueSolver, self).initialize()

        # Compute the mesh volume
        self.domain_volume = 0.0
        for cell in self.mesh.cells:
            self.domain_volume += cell.volume

        # Set/check output frequency
        if not self.output_frequency:
            self.output_frequency = self.dt
        if self.dt > self.output_frequency:
            self.dt = self.output_frequency

        # Initialize old time solutions
        self.phi_old = np.copy(self.phi)
        if self.use_precursors:
            self.precursors_old = np.copy(self.precursors)

        # Precompute matrices
        self.M: csr_matrix = self.mass_matrix()
        self.A: List[csr_matrix] = self.assemble_transient_matrices()

        # Compute the initial conditions
        self.compute_initial_values()

        # Define energy per fission
        self.E_fission = self.power / self.compute_power()

    def execute(self, verbose: int = 0) -> None:
        """Execute the transient multigroup diffusion solver.

        Parameters
        ----------
        verbose : int, default 0
        """
        power_old = self.power
        T_initial = self.T
        next_output = self.output_frequency
        sigma_t_0 = self.material_xs[0].sigma_t

        # Time stepping loop
        time, n_steps, dt_initial = 0.0, 0, self.dt
        while time < self.t_final:

            # Force coincidence with output time
            if time + self.dt > next_output:
                self.dt = next_output - time
                self.A = self.assemble_transient_matrices()

            # Force coincidence with t_final
            if time + self.dt > self.t_final or \
                    abs(time + self.dt - self.t_final) < 1.0e-12:
                self.dt = self.t_final - time
                self.A = self.assemble_transient_matrices()

            # Solve time step
            self.solve_time_step()
            self.post_process_time_step()
            self.power = self.compute_power()

            # Refinements, if adaptivity is used
            dP = abs(self.power - power_old) / power_old
            while self.adaptivity and dP > self.refine_level:

                # Reset solutions
                self.phi[:] = self.phi_old
                self.precursors[:] = self.precursors_old

                # Half the time step
                self.dt /= 2.0
                self.A = self.assemble_transient_matrices()

                # Take the reduced time step
                self.solve_time_step()
                self.post_process_time_step()
                self.power = self.compute_power()

                # Compute new change in power
                dP = abs(self.power - power_old) / power_old

            # Compute new temperature
            mass = self.mass_density * self.domain_volume
            self.T += self.power * self.dt / (mass * self.specific_heat)

            # Add feedback
            if self.use_feedback:
                sqrt_T = np.sqrt(self.T)
                sqrt_T_initial = np.sqrt(T_initial)
                f = 1.0 + self.feedback_coeff * (sqrt_T - sqrt_T_initial)
                self.material_xs[0].sigma_t[0] = f * sigma_t_0
                self.L = self.diffusion_matrix()
                self.A = self.assemble_transient_matrices()

            # Increment time
            time += self.dt
            n_steps += 1

            # Reinit solutions
            power_old = self.power
            self.phi_old[:] = self.phi
            if self.use_precursors:
                self.precursors_old[:] = self.precursors

            # Output solutions
            if time == next_output:
                self.store_outputs(time)
                next_output += self.output_frequency
                if next_output > self.t_final:
                    next_output = self.t_final

            # Coarsen time steps
            if self.adaptivity:
                if dP < self.coarsen_level:
                    self.dt *= 2.0
                    if self.dt > self.output_frequency:
                        self.dt = self.output_frequency
                    self.A = self.assemble_transient_matrices()
            elif self.dt != dt_initial:
                self.dt = dt_initial
                self.A = self.assemble_transient_matrices()

            # Print time step summary
            if verbose:
                print(f"***** Time Step: {n_steps} *****")
                print(f"Simulation Time:\t{time}")
                print(f"Time Step Size:\t\t{self.dt}")
                print(f"System Power:\t\t{self.power}")
                print(f"Temperature:\t\t{self.T}\n")

        self.dt = dt_initial

    def solve_time_step(self, m: int = 0) -> None:
        """Solve the `m`'th step of a multi-step method.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.
        """
        b = self.assemble_transient_rhs(m)
        self.phi = spsolve(self.A[m], b)
        self.update_precursors(m)

    def post_process_time_step(self) -> None:
        """Post-process the time step results.

        For Backward Euler, nothing is done. For Crank Nicholson,
        this computes the next time step value from the previous and
        half time step values. For TBDF-2, this computes the half time
        step value from the previous and quarter time step values, then
        takes a step of BDF-2.
        """
        # Handle 2nd order methods
        if self.method in ["CRANK_NICHOLSON", "TBDF2"]:
            self.phi = 2.0*self.phi - self.phi_old
            if self.use_precursors:
                self.precursors = 2.0*self.precursors - self.precursors_old

            if self.method == "TBDF2":
                b = self.assemble_transient_rhs(m=1)
                self.phi = spsolve(self.A[1], b)
                self.update_precursors(m=1)

    def assemble_transient_matrices(self) -> List[csr_matrix]:
        """Assemble the multigroup evolution matrix for each step `m`.

        Returns
        -------
        List[csr_matrix (n_cells * n_groups,) * 2]
        """

        if self.method == "BACKWARD_EULER":
            A = [self.L + self.M / self.dt]
        elif self.method == "CRANK_NICHOLSON":
            A = [self.L + 2.0 * self.M / self.dt]
        elif self.method == "TBDF2":
            A = [self.L + 4.0 * self.M / self.dt,
                 self.L + 3.0 * self.M / self.dt]
        else:
            raise NotImplementedError(f"{self.method} is not implemented.")

        for m in range(len(A)):
            D = self.precursor_substitution_matrix(m)
            A[m] -= self.S + self.Fp + D
            A[m] = self.apply_matrix_bcs(A[m])
        return A

    def assemble_transient_rhs(self, m: int = 0) -> ndarray:
        b = self.set_source() + self.set_old_source(m)
        return self.apply_vector_bcs(b)

    def mass_matrix(self) -> csr_matrix:
        """Assemble the multigroup mass matrix.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_mass_matrix()
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_mass_matrix()

    def precursor_substitution_matrix(self, m: int = 0) -> csr_matrix:
        """Assemble the transient precursor substitution matrix for step `m`.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_precursor_substitution_matrix(m)
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_precursor_substitution_matrix(m)

    def set_old_source(self, m: int = 0) -> ndarray:
        """Assemble the right-hand side with terms from last time step.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.

        Returns
        -------
        ndarray (n_cells * n_groups)
        """
        if self.method == "BACKWARD_EULER":
            b = self.M / self.dt @ self.phi_old
        elif self.method == "CRANK_NICHOLSON":
            b = 2.0 * self.M / self.dt @ self.phi_old
        elif self.method == "TBDF2" and m == 0:
            b = 4.0 * self.M / self.dt @ self.phi_old
        elif self.method == "TBDF2" and m == 1:
            b = self.M / self.dt @ (4.0 * self.phi - self.phi_old)

        # Add old time step precursors
        b += self.old_precursor_source(m)
        return b

    def old_precursor_source(self, m: int = 0) -> ndarray:
        """Assemble the delayed terms from the last time step for step `m`.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.

        Returns
        -------
        ndarray (n_cells * n_groups)
        """
        if self.use_precursors:
            if isinstance(self.discretization, FiniteVolume):
                return self._fv_old_precursor_source(m)
            elif isinstance(self.discretization, PiecewiseContinuous):
                return self._pwc_old_precursor_source(m)

    def update_precursors(self, m: int = 0) -> None:
        """Update the precursors after a time step.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.
        """
        if self.use_precursors:
            if isinstance(self.discretization, FiniteVolume):
                self._fv_update_precursors(m)
            elif isinstance(self.discretization, PiecewiseContinuous):
                self._pwc_update_precursors(m)

    def compute_power(self) -> float:
        """Compute the fission power in the system.

        Returns
        -------
        float
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_compute_power()
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_compute_power()

    def effective_time_step(self, m: int = 0) -> float:
        """Compute the effective time step for step `m`.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.

        Returns
        -------
        float
        """
        if self.method == "BACKWARD_EULER":
            return self.dt
        elif self.method == "CRANK_NICHOLSON":
            return self.dt / 2.0
        elif self.method == "TBDF2" and m == 0:
            return self.dt / 4.0
        elif self.method == "TBDF2" and m == 1:
            return self.dt / 3.0
        else:
            raise NotImplementedError(f"{self.method} is not implemented.")

    def compute_initial_values(self) -> None:
        """Evaluate the initial conditions.
        """
        if self.initial_conditions is None:
            KEigenvalueSolver.execute(self, verbose=0)
            msg = f"***** k-Eigenvalue Initialization *****"
            header = "*" * len(msg)
            print("\n".join(["", header, msg, header]))
            print(f"k-Eigenvalue:\t{self.k_eff:.6g}")
        else:
            G, J = self.n_groups, self.n_precursors
            grid = self.discretization.grid
            grid = np.array([p.z for p in grid])

            # Convert to lambdas
            if isinstance(self.initial_conditions, MutableDenseMatrix):
                from sympy import lambdify
                symbols = list(self.initial_conditions.free_symbols)

                ics = []
                for ic in self.initial_conditions:
                    ics += lambdify(symbols, ic)
                self.initial_conditions = ics

            # Check initial conditions
            if len(self.initial_conditions) not in [G, G + J]:
                raise AssertionError(
                    "Invalid number of initial conditions. There must be "
                    "either n_groups or n_groups + n_precursors initial "
                    "conditions.")

            # Initial conditions for scalar flux
            for g in range(G):
                ic = self.initial_conditions[g]
                if callable(ic):
                    self.phi[g::G] = ic(grid)
                elif isinstance(ic, (ndarray, List[float])):
                    if len(ic) == len(self.phi[g::G]):
                        self.phi[g::G] = ic
                    else:
                        raise ValueError(
                            f"Initial condition vector for group {g} is "
                            f"incompatible with the discretization.")

            # Initial conditions for precursors
            if self.use_precursors and len(self.initial_conditions) > G:
                for j in range(J):
                    ic = self.initial_conditions[G + j]
                    if callable(ic):
                        self.precursors[j::J] = ic(grid)
                    elif isinstance(ic, (ndarray, List[float])):
                        if len(ic) == len(self.precursors[j::J]):
                            self.precursors[j::J] = ic
                        else:
                            raise ValueError(
                                f"Initial condition vector for precursor "
                                f"{j} is incompatible with the "
                                f"discretization.")

        # Copy initial conditions to old solutions
        self.phi_old[:] = self.phi
        if self.use_precursors:
            self.precursors_old[:] = self.precursors

    def store_outputs(self, time: float) -> None:
        self.outputs.store_outputs(self, time)

    def write_outputs(self, path: str = ".") -> None:
        self.outputs.write_outputs(path)
