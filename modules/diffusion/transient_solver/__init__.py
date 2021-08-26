import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
from numpy import ndarray
from scipy.sparse import csr_matrix
from typing import List

from pyPDEs.spatial_discretization import *

from .. import KEigenvalueSolver
from ..outputs import Outputs


class TransientSolver(KEigenvalueSolver):
    """Transient solver for multi-group diffusion problems.
    """

    from ._assemble_fv import _fv_mass_matrix
    from ._assemble_fv import _fv_transient_fission_matrix
    from ._assemble_fv import _fv_set_transient_source
    from ._assemble_fv import _fv_update_precursors
    from ._assemble_fv import _fv_compute_power

    from ._assemble_pwc import _pwc_mass_matrix
    from ._assemble_pwc import _pwc_transient_fission_matrix
    from ._assemble_pwc import _pwc_set_transient_source
    from ._assemble_pwc import _pwc_update_precursors
    from ._assemble_pwc import _pwc_compute_power

    def __init__(self) -> None:
        super().__init__()
        self.initial_conditions: list = None

        self.t_final: float = 0.25
        self.dt: float = 5.0e-3
        self.stepping_method: str = "CRANK_NICHOLSON"

        self.output_frequency: float = None

        self.adaptivity: bool = False
        self.refine_level: float = 0.02
        self.coarsen_level: float = 0.005

        self.lag_precursors: bool = False

        self.power: float = 1.0
        self.energy_per_fission: float = 1.0

        self.use_feedback: bool = False
        self.feedback_coeff: float = -1.0e-3

        self.density: float = 19.0  # g / cc
        self.spec_heat: float = 0.12  # J / g-K
        self.volume: float = 0.0  # cc
        self.temperature: float = 293.0  # K

        self.M: csr_matrix = None
        self.A: List[csr_matrix] = None

        self.b_old: ndarray = None
        self.phi_old: ndarray = None
        self.precursors_old: ndarray = None

        self.outputs: Outputs = Outputs()

    def initialize(self) -> None:
        """Initialize the transient multi-group diffusion solver.
        """
        super(KEigenvalueSolver, self).initialize()

        # Compute mesh volume
        self.volume = sum([c.volume for c in self.mesh.cells])

        # Set output frequency, if not set
        if not self.output_frequency:
            self.output_frequency = self.dt

        # Ensure dt <= output frequency
        if self.dt > self.output_frequency:
            self.dt = self.output_frequency

        # Initialize vectors
        self.b_old = np.copy(self.b)
        self.phi_old = np.copy(self.phi)
        if self.use_precursors:
            self.precursors_old = np.copy(self.precursors)

        # Assemble relevant matrices
        self.M = self.mass_matrix()
        self.assemble_evolution_matrices()

        # Compute the initial conditions
        self.compute_initial_values()

        # Compute the effective energy per fission
        self.energy_per_fission = self.power / self.compute_power()

        # Initialize outputs
        self.outputs.reset()
        self.store_outputs(0.0)

    def execute(self, verbose: bool = False) -> None:
        """Execute the transient multi-group diffusion solver.
        """
        print("\n***** Executing the transient "
              "multi-group diffusion solver. *****")
        power_old = self.power
        initial_temperature = self.temperature
        next_output_time = self.output_frequency
        sigma_t0 = self.material_xs[0].sigma_t

        # Start time stepping
        time, n_steps, dt0 = 0.0, 0, self.dt
        while time < self.t_final - sys.float_info.epsilon:

            # Force coincidence with output times
            if time + self.dt > next_output_time:
                self.dt = next_output_time - time
                self.assemble_evolution_matrices()

            # Force coincidence with end time
            if time + self.dt > self.t_final:
                self.dt = self.t_final - time
                self.assemble_evolution_matrices()

            # Solve time step
            self.solve_time_step()
            self.post_process_time_step()
            self.power = self.compute_power()

            # Refinements, if adaptivity is on
            dP = abs(self.power - power_old) / power_old
            while self.adaptivity and dP > self.refine_level:

                # Reset the solutions
                self.phi[:] = self.phi_old
                self.precursors[:] = self.precursors_old

                # Half the time step and assemble matrices
                self.dt /= 2.0
                self.assemble_evolution_matrices()

                # Take the time step again
                self.solve_time_step()
                self.post_process_time_step()

                # Compute the new power, dP
                self.power = self.compute_power()
                dP = abs(self.power - power_old) / power_old

            # Compute new temperature
            c = self.density * self.spec_heat * self.volume
            self.temperature += self.power * self.dt / c

            # Add feedback
            if self.use_feedback:
                dT = self.temperature - initial_temperature
                f = 1.0 - self.feedback_coeff * dT
                self.material_xs[0].sigma_t = f * sigma_t0
                self.L = self.diffusion_matrix()
                self.assemble_evolution_matrices()

            # Increment time
            time += self.dt
            n_steps += 1

            # Reset vectors
            self.phi_old[:] = self.phi
            power_old = self.power
            if self.use_precursors:
                self.precursors_old[:] = self.precursors

            # Output solutions
            if time == next_output_time:
                self.store_outputs(time)
                next_output_time += self.output_frequency
                if next_output_time > self.t_final:
                    next_output_time = self.t_final

            # Coarsen time steps
            if self.adaptivity and dP < self.coarsen_level:
                self.dt *= 2.0
                self.assemble_evolution_matrices()
            if self.adaptivity and self.dt > self.output_frequency:
                self.dt = self.output_frequency
                self.assemble_evolution_matrices()
            if not self.adaptivity:
                self.dt = dt0
                self.assemble_evolution_matrices()

            # Print time step summary
            if verbose:
                print(f"***** Time Step: {n_steps} *****")
                print(f"Simulation Time:\t{time}")
                print(f"Time Step Size:\t\t{self.dt}")
                print(f"System Power:\t\t{self.power}")
                print(f"Temperature:\t\t{self.temperature}\n")

        self.dt = dt0  # reset dt to original

    def solve_time_step(self, step: int = 0) -> None:
        """Solve the system for the n'th step of a time step.

        Parameters
        ----------
        step : int, default 0
            The section of the time step.
        """
        b_old = self.set_old_transient_source(step)

        # Solve the full multi-group system
        if not self.use_groupwise_solver:
            self.b = self.set_old_transient_source(step)
            self.set_transient_source(step)
            # A = self.A[step] - self.S - self.Ft[step]
            self.phi = spsolve(self.A[step], self.b)

        else:
            n_grps = self.n_groups
            phi_ell = np.copy(self.phi)
            b_old = self.set_old_transient_source(step)

            # Start iterating
            converged = False
            for nit in range(self.max_iterations):

                # Solve group-wise
                for g in range(n_grps):
                    self.b[:] = b_old
                    self.set_transient_source(step, *(True,) * 4)
                    self.phi[g::n_grps] = spsolve(self.Ag(g)[step],
                                                  self.b[g::n_grps])

                # Check convergence
                phi_change = norm(self.phi - phi_ell)
                phi_ell[:] = self.phi
                if phi_change <= self.tolerance:
                    converged = True
                    break

            if not converged:
                print(f"!!!!! WARNING: Solver did not converge. "
                      f"Final Change: {phi_change:.3e} !!!!!")

        self.update_precursors(step)

    def post_process_time_step(self) -> None:
        """Post-process the time step results.

        For Backward Euler, nothing is done. For Crank Nicholson,
        this computes the next time step value from the previous and
        half time step values. For TBDF-2, this computes the half time
        step value from the previous and quarter time step values, then
        takes a step of BDF-2.
        """
        # Handle 2nd order methods
        if self.stepping_method in ["CRANK_NICHOLSON", "TBDF2"]:
            self.phi = 2.0 * self.phi - self.phi_old
            if self.use_precursors:
                self.precursors = \
                    2.0 * self.precursors - self.precursors_old

            # Second step of time step
            if self.stepping_method == "TBDF2":
                self.solve_time_step(step=1)

    def set_old_transient_source(self, step: int = 0) -> ndarray:
        """Assemble the previous time step related right-hand side.

        Parameters
        ----------
        step : int, default 0
            The section of the time step.

        Returns
        -------
        ndarray
        """
        n_grps = self.n_groups

        if self.stepping_method == "BACKWARD_EULER":
            b = self.M / self.dt @ self.phi_old
        elif self.stepping_method == "CRANK_NICHOLSON":
            b = 2.0 * self.M / self.dt @ self.phi_old
        elif self.stepping_method == "TBDF2" and step == 0:
             b = 4.0 * self.M / self.dt @ self.phi_old
        else:
            phi, phi_old = self.phi, self.phi_old
            b = self.M / self.dt @ (4.0 * phi - phi_old)
        return b

    def assemble_evolution_matrices(self) -> csr_matrix:
        """Assemble the linear systems for a time step.

        Returns
        -------
        csr_matrix
        """
        # Base evolution matrix
        if self.stepping_method == "BACKWARD_EULER":
            self.A = [self.L + self.M / self.dt]
        elif self.stepping_method == "CRANK_NICHOLSON":
            self.A = [self.L + 2.0 * self.M / self.dt]
        else:
            self.A = [self.L + 4.0 * self.M / self.dt,
                      self.L + 3.0 * self.M / self.dt]

        # Subtract scattering + fission for block solving
        if not self.use_groupwise_solver:
            for i in range(len(self.A)):
                Ft = self.transient_fission_matrix(i)
                self.A[i] -= self.S + Ft

    def mass_matrix(self) -> csr_matrix:
        """Assemble the multi-group mass matrix.

        Returns
        -------
        csr_matrix
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_mass_matrix()
        else:
            return self._pwc_mass_matrix()

    def transient_fission_matrix(self, step: int = 0) -> csr_matrix:
        """Assemble the transient multi-group fission matrix.

        Returns
        -------
        csr_matrix
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_transient_fission_matrix(step)
        else:
            return self._pwc_transient_fission_matrix(step)

    def set_transient_source(
            self: "TransientSolver", step: int = 0,
            apply_material_source: bool = True,
            apply_boundary_source: bool = True,
            apply_scattering_source: bool = False,
            apply_fission_source: bool = False) -> None:
        """Assemble the right-hand side.

        This includes previous time step contributions as well
        as material, scattering, fission, and boundary sources.

        Parameters
        ----------
        step : int, default 0
            The section of the time step.
        apply_material_source : bool, default True
        apply_boundary_source : bool, default True
        apply_scattering_source : bool, default False
        apply_fission_source : bool, default False
        """
        flags = (apply_material_source, apply_boundary_source,
                 apply_scattering_source, apply_fission_source)
        if isinstance(self.discretization, FiniteVolume):
            self._fv_set_transient_source(step, *flags)
        else:
            self._pwc_set_transient_source(step, *flags)

    def update_precursors(self, step: int = 0) -> None:
        """Solve a precursor time step.

        Parameters
        ----------
        step : int, default 0
            The section of the time step.
        """
        if self.use_precursors:
            if isinstance(self.discretization, FiniteVolume):
                self._fv_update_precursors(step)
            else:
                self._pwc_update_precursors(step)

    def compute_power(self) -> float:
        """Compute the fission power.

        Notes
        -----
        This method uses the most current scalar flux solution.

        Returns
        -------
        float
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_compute_power()
        else:
            return self._pwc_compute_power()

    def Mg(self, g: int) -> csr_matrix:
        """Get the `g`'th group's mass matrix.

        Parameters
        ----------
        g : int

        Returns
        -------
        csr_matrix
        """
        if self.flux_uk_man.storage_method == "NODAL":
            return self.M[g::self.n_groups, g::self.n_groups]
        else:
            ni = g * self.discretization.n_nodes
            nf = (g + 1) * self.discretization.n_nodes
            return self.M[ni:nf, ni:nf]

    def Ag(self, g: int) -> List[csr_matrix]:
        """Get the `g`'th group's evolution matrices.

        Parameters
        ----------
        g : int

        Returns
        -------
        List[csr_matrix]
        """
        A = []
        for i in range(len(self.A)):
            if self.flux_uk_man.storage_method == "NODAL":
                Ai = self.A[i][g::self.n_groups, g::self.n_groups]
            else:
                ni = g * self.discretization.n_nodes
                nf = (g + 1) * self.discretization.n_nodes
                Ai = self.A[i][ni:nf, ni:nf]
            A.append(Ai)
        return A

    def effective_dt(self, step: int = 0) -> float:
        """Compute the effective time step.

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
        """Evaluate the initial conditions.
        """
        if self.initial_conditions is None:
            KEigenvalueSolver.execute(self, verbose=False)
            import time
            time.sleep(2.0)
        else:
            n_grps = self.n_groups
            n_prec = self.n_precursors
            num_ics = [n_grps, n_grps + n_prec]
            grid = self.discretization.grid
            grid = np.array([p.z for p in grid])

            # Convert to lambdas
            from sympy import lambdify
            from sympy.matrices.dense import MutableDenseMatrix
            if isinstance(self.initial_conditions, MutableDenseMatrix):
                ics = []
                symbols = list(self.initial_conditions.free_symbols)
                for ic in self.initial_conditions:
                    ics.append(lambdify(symbols, ic))
                self.initial_conditions = ics

            # Check initial conditions
            if all([len(self.initial_conditions) != n for n in num_ics]):
                raise AssertionError(
                    "Invalid number of initial conditions provided.")

            # Flux initial conditions
            for g, ic in enumerate(self.initial_conditions[:n_grps]):
                if callable(ic):
                    self.phi[g::n_grps] = ic(grid)
                elif len(ic) == len(self.phi[g::n_grps]):
                    self.phi[g::n_grps] = ic
                else:
                    raise ValueError(
                        f"Provided initial condition for group {g} "
                        f"does not agree with discretization.")

            # Precursor initial conditions
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

        # Set old solutions with initial conditions
        self.phi_old[:] = self.phi
        if self.use_precursors:
            self.precursors_old[:] = self.precursors

    def store_outputs(self, time: float) -> None:
        self.outputs.store_outputs(self, time)

    def write_outputs(self, path: str = ".") -> None:
        self.outputs.write_outputs(path)
