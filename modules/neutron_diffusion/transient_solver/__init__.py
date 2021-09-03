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
        self.A: List[csr_matrix] = self.evolution_matrices()

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
                self.A = self.evolution_matrices()

            # Force coincidence with t_final
            if time + self.dt > self.t_final or \
                    abs(time + self.dt - self.t_final) < 1.0e-12:
                self.dt = self.t_final - time
                self.A = self.evolution_matrices()

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
                self.A = self.evolution_matrices()

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
                self.A = self.evolution_matrices()

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
                    self.A = self.evolution_matrices()
            elif self.dt != dt_initial:
                self.dt = dt_initial
                self.A = self.evolution_matrices()

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
        b = self.set_source() + self.set_old_source(m)
        self.apply_vector_bcs(b)
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
                b = self.set_source() + self.set_old_source(m=1)
                self.apply_vector_bcs(b)
                self.phi = spsolve(self.A[1], b)
                self.update_precursors(m=1)

    def evolution_matrices(self) -> List[csr_matrix]:
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
            Ft = self.transient_fission_matrix(m)
            A[m] -= self.S + self.Fp + Ft
        return A

    def mass_matrix(self) -> csr_matrix:
        """Assemble the multigroup mass matrix.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

        # Loop over cells
        rows, data = [], []
        for cell in self.mesh.cells:
            volume = cell.volume
            xs = self.material_xs[cell.material_id]

            # Loop over groups
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, uk_man, 0, g)

                # Inverse velocity term
                value = xs.inv_velocity[g] * volume
                rows.append(ig)
                data.append(value)
        return csr_matrix((data, (rows, rows)),
                          shape=(fv.n_dofs(uk_man),) * 2)

    def transient_fission_matrix(self, m: int = 0) -> csr_matrix:
        """Assemble the transient multigroup fission matrix for step `m`.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

        # Construct if using precursors and not lagging
        rows, cols, data = [], [], []
        if self.use_precursors and not self.lag_precursors:
            eff_dt = self.effective_time_step(m)

             # Loop over cells
            for cell in self.mesh.cells:
                volume = cell.volume
                xs = self.material_xs[cell.material_id]

                # Loop over precursors
                for j in range(self.n_precursors):
                    # Loop over groups
                    for g in range(self.n_groups):
                        ig = fv.map_dof(cell, 0, uk_man, 0, g)

                        # Coefficient for delayed fission term
                        coeff = xs.chi_delayed[g][j] * xs.precursor_lambda[j]
                        coeff /= 1.0 + eff_dt * xs.precursor_lambda[j]
                        coeff *= eff_dt * xs.precursor_yield[j]

                        # Loop over groups
                        for gp in range(self.n_groups):
                            igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                            # Delayed fission term
                            value = coeff * xs.nu_delayed_sigma_f[gp] * volume
                            rows.append(ig)
                            cols.append(igp)
                            data.append(value)
        return csr_matrix((data, (rows, cols)),
                          shape=(fv.n_dofs(uk_man),) * 2)

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
        if self.use_precursors:
            fv: FiniteVolume = self.discretization
            phi_uk_man = self.phi_uk_man
            c_uk_man = self.precursor_uk_man
            eff_dt = self.effective_time_step(m)

            # Loop over cells
            for cell in self.mesh.cells:
                volume = cell.volume
                xs = self.material_xs[cell.material_id]

                # Loop over groups
                for g in range(self.n_groups):
                    ig = fv.map_dof(cell, 0, phi_uk_man, 0, g)

                    # Loop over precursors
                    for j in range(xs.n_precursors):
                        ij = fv.map_dof(cell, 0, c_uk_man, 0, j)

                        # Get precursors at this DoF
                        cj = self.precursors[ij]
                        cj_old = self.precursors_old[ij]

                        # Coefficient for precursor term
                        coeff = xs.chi_delayed[g][j] * xs.precursor_lambda[j]
                        if not self.lag_precursors:
                            coeff /= 1.0 + eff_dt * xs.precursor_lambda[j]

                        # Old precursor contributions
                        if m == 0 or self.lag_precursors:
                            b[ig] += coeff * cj_old * volume
                        elif m == 1:
                            tmp = (4.0 * cj - cj_old) / 3.0
                            b[ig] += coeff * tmp * volume
        return b

    def update_precursors(self, m: int = 0) -> None:
        """Update the precursors after a time step.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.
        """
        if self.use_precursors:
            fv: FiniteVolume = self.discretization
            phi_uk_man = self.phi_uk_man
            c_uk_man = self.precursor_uk_man
            eff_dt = self.effective_time_step(m)

            # Loop over cells
            for cell in self.mesh.cells:
                xs = self.material_xs[cell.material_id]

                # Compute delayed fission rate
                f_d = 0.0
                for g in range(self.n_groups):
                    ig = fv.map_dof(cell, 0, phi_uk_man, 0, g)
                    f_d += xs.nu_delayed_sigma_f[g] * self.phi[ig]

                # Loop over precursors
                for j in range(xs.n_precursors):
                    ij = fv.map_dof(cell, 0, c_uk_man, 0, j)

                    # Get the precursors at this DoF
                    cj = self.precursors[ij]
                    cj_old = self.precursors_old[ij]

                    # Coefficient for RHS
                    coeff = (1.0 + eff_dt * xs.precursor_lambda[j])**(-1)

                    # Old precursor contributions
                    if m == 0:
                        self.precursors[ij] = coeff * cj_old
                    else:
                        tmp = (4.0 * cj - cj_old) / 3.0
                        self.precursors[ij] = coeff * tmp

                    # Delayed fission contribution
                    coeff = eff_dt * xs.precursor_yield[j]
                    self.precursors[ij] += coeff * f_d

    def compute_power(self) -> float:
        """Compute the fission power in the system.

        Returns
        -------
        float
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

        # Loop over cells
        power = 0.0
        for cell in self.mesh.cells:
            volume = cell.volume
            xs = self.material_xs[cell.material_id]

            # Loop over groups
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, uk_man, 0, g)
                power += xs.sigma_f[g] * self.phi[ig] * volume
        return power * self.E_fission

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
