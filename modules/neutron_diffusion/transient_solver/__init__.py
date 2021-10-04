import os
import sys
import time

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


class TransientSolver(KEigenvalueSolver):
    """Class for solving transinet multigroup diffusion problems.
    """

    from ._fv import (_fv_mass_matrix,
                      _fv_precursor_substitution_matrix,
                      _fv_old_precursor_source,
                      _fv_update_precursors,
                      _fv_compute_fission_rate)

    from ._pwc import (_pwc_mass_matrix,
                       _pwc_precursor_substitution_matrix,
                       _pwc_old_precursor_source,
                       _pwc_update_precursors,
                       _pwc_compute_fission_rate)

    from ._write_outputs import write_snapshot

    def __init__(self) -> None:
        """Class constructor.
        """
        super().__init__()
        self.initial_conditions: list = None
        self.normalize_fission: bool = True
        self.exact_keff_for_ic: float = None
        self.phi_norm_method: str = "TOTAL"

        # Time stepping parameters
        self.time: float = 0.0
        self.t_start: float = 0.0
        self.t_final: float = 0.1
        self.dt: float = 2.0e-3
        self.method: str = "TBDF2"

        # Nonlinear parameters
        self.is_nonlinear: bool = False
        self.nonlinear_tolerance: float = 1.0e-8
        self.nonlinear_max_iterations: int = 50

        # Adaptive time stepping parameters
        self.adaptivity: bool = False
        self.refine_level: float = 0.05
        self.coarsen_level: float = 0.01

        # Physics options
        self.lag_precursors: bool = False

        # Flag for transient cross sections
        self.has_functional_xs: bool = False

        # Power related parameters
        self.power: float = 1.0  # W
        self.power_old: float = 1.0  # W
        self.initial_power: float = 1.0 # W

        # Physics paramaters
        self.feedback_coeff: float = 0.0  # K^{1/2}
        self.energy_per_fission: float = 3.2e-11  # J / fission
        self.conversion_factor: float = 3.83e-11  # K-cm^3

        # Precomputed mass matrix storage
        self.M: csr_matrix = None

        # Previous time step solutions
        self.phi_old: ndarray = None
        self.precursors_old: ndarray = None

        # Fission density
        self.fission_rate: ndarray = None

        # Temperatures
        self.temperature: ndarray = None
        self.temperature_old: ndarray = None
        self.initial_temperature: float = 300.0  # K

        # Multi-step method vectors
        self.phi_aux: List[ndarray] = None
        self.precursors_aux: List[ndarray] = None
        self.temperature_aux: List[ndarray] = None

        # Output options
        self.write_outputs: bool = False
        self.output_frequency: float = None
        self.output_directory: str = None

    @property
    def power_density(self) -> ndarray:
        return self.energy_per_fission * self.fission_rate

    def initialize(self, verbose: int = 0) -> None:
        """Initialize the solver.
        """
        KEigenvalueSolver.initialize(self)
        KEigenvalueSolver.execute(self, verbose=verbose)
        if verbose > 0:
            time.sleep(2.0)

        # Set transient cross section flag
        for xs in self.material_xs:
            if xs.sigma_a_function is not None:
                self.has_functional_xs = True
                break

        # Set/check output frequency
        self._check_time_step()

        # Initialize power
        self.power = self.initial_power
        self.power_old = self.initial_power

        # Initialize temperature vectors
        T0 = self.initial_temperature
        self.temperature = T0 * np.ones(self.mesh.n_cells)

        # Initialize fission density vector
        self.fission_rate = np.zeros(self.mesh.n_cells)

        # Initalize old vectors
        self.phi_old = np.copy(self.phi)
        self.precursors_old = np.copy(self.precursors)
        self.temperature_old = np.copy(self.temperature)

        # Check output information
        if self.write_outputs:
            if not os.path.isdir(self.output_directory):
                os.makedirs(self.output_directory)
            elif len(os.listdir(self.output_directory)) > 0:
                os.system(f"rm -r {self.output_directory}/*")

        # Evaluate initial conditions
        self.compute_initial_values()
        self.write_snapshot(0)

        # Initialize auxilary vectors
        if self.method == "TBDF2":
            self.phi_aux = [np.copy(self.phi)]
            self.precursors_aux = [np.copy(self.precursors)]
            self.temperature_aux = [np.copy(self.temperature)]

        # Precompute matrices
        self.M = self.mass_matrix()

    def execute(self, verbose: int = 0) -> None:
        """Execute the transient multigroup diffusion solver.

        Parameters
        ----------
        verbose : int, default 0
        """
        next_output = self.output_frequency
        dt_initial = self.dt

        # Time stepping loop
        self.time = self.t_start
        n_steps, n_output = 0, 0
        while self.time < self.t_final:

            # Force coincidence with output time
            if self.time + self.dt > next_output:
                self.dt = next_output - self.time

            # Force coincidence with t_final
            if self.time + self.dt > self.t_final:
                self.dt = self.t_final - self.time

            # Solve time step
            self.solve_time_step(m=0)
            if self.method == "TBDF2":
                self.solve_time_step(m=1)
            self.power = self.compute_power()

            # Refinements, if adaptivity is used
            if self.adaptivity:
                self.refine_time_step()

            # Increment time
            self.time = np.round(self.time + self.dt, 12)
            n_steps += 1

            # Output solutions
            if self.time == next_output:
                n_output += 1
                self.write_snapshot(n_output)
                next_output += self.output_frequency
                next_output = np.round(next_output, 12)
                if next_output > self.t_final:
                    next_output = self.t_final

            # Coarsen time steps
            if self.adaptivity:
                self.coarsen_time_step()
            elif self.dt != dt_initial:
                self.dt = dt_initial

            self.step_solutions()

            # Print time step summary
            if verbose > 0:
                print()
                print(f"***** Time Step: {n_steps} *****")
                print(f"Simulation Time:\t{self.time:.3e} sec")
                print(f"Time Step Size:\t\t{self.dt:.3e} sec")
                print(f"Total Power:\t\t{self.power:.3e} W")
                P_avg = self.average_power_density
                print(f"Average Power Density:\t{P_avg:.3e} W/cm^3")
                T_avg = self.average_temperature
                print(f"Average Temperature:\t{T_avg:.3g} K")

        self.dt = dt_initial

    def refine_time_step(self) -> None:
        """Refine the time step.
        """
        dP = abs(self.power - self.power_old) / self.power_old
        while dP > self.refine_level:
            # Half the time step
            self.dt /= 2.0

            # Take the reduced time step
            self.solve_time_step(m=0)
            if self.method == "TBDF2":
                self.solve_time_step(m=1)
            self.power = self.compute_power()

            # Compute new change in power
            dP = abs(self.power - self.power_old) / self.power_old

    def coarsen_time_step(self):
        """Coarsen the time step.
        """
        dP = abs(self.power - self.power_old) / self.power_old
        if dP < self.coarsen_level:
            self.dt *= 2.0
            if self.dt > self.output_frequency:
                self.dt = self.output_frequency

    def solve_time_step(self, m: int = 0) -> None:
        """Solve the `m`'th step of a multi-step method.
        """
        if not self.is_nonlinear:
            self.update_phi(m)
            self.update_temperature(m)
        else:
            phi_ell = np.copy(self.phi_old)
            T_ell = np.copy(self.temperature_old)
            converged = False
            for nit in range(self.nonlinear_max_iterations):
                self.update_phi(m)
                self.update_temperature(m)
                if m == 0 and self.method == "TBDF2":
                    converged = True
                    break

                change = norm(self.phi - phi_ell)
                change += norm(self.temperature - T_ell)
                phi_ell[:] = self.phi
                T_ell[:] = self.temperature

                if change < self.nonlinear_tolerance:
                    converged = True
                    break

            if not converged:
                print("\n!!! WARNING: Nonlinear iterations "
                      "did not converge !!!\n")

        if self.use_precursors:
            self.update_precursors(m)

        if m == 0 and self.method in ["CN", "TBDF2"]:
            self.phi = 2.0 * self.phi - self.phi_old
            self.temperature = 2.0 * self.temperature - self.temperature_old
            self.compute_fission_rate()

            if self.method == "TBDF2":
                self.phi_aux[0][:] = self.phi
                self.temperature_aux[0][:] = self.temperature
                if self.use_precursors:
                    self.precursors_aux[0][:] = self.precursors

    def update_phi(self, m: int = 0) -> None:
        """Update the scalar flux for the `m`'th step.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.
        """
        A = self.assemble_transient_matrix(m)
        b = self.assemble_transient_rhs(m)
        self.phi = spsolve(A, b)
        self.compute_fission_rate()

    def update_precursors(self, m: int = 0) -> None:
        """Update the precursors for the `m`'th step.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.
        """
        if isinstance(self.discretization, FiniteVolume):
            self._fv_update_precursors(m)
        elif isinstance(self.discretization, PiecewiseContinuous):
            self._pwc_update_precursors(m)

        if m == 0 and self.method in ["CN", "TBDF2"]:
            self.precursors = \
                2.0 * self.precursors - self.precursors_old

    def update_temperature(self, m: int = 0) -> None:
        """Update the temperature for the `m`'th step.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.
        """
        eff_dt = self.effective_time_step(m)
        Ef = self.energy_per_fission

        T_old = self.temperature_old
        if m == 1:
            T = self.temperature_aux[0]
            T_old = (4.0*T - T_old) / 3.0

        # Loop over cells
        for cell in self.mesh.cells:
            Sf = self.fission_rate[cell.id]
            alpha = self.conversion_factor
            self.temperature[cell.id] = \
                T_old[cell.id] + eff_dt * alpha * Sf

    def update_cross_sections(self, t: float) -> None:
        """Update the cell-wise cross sections.

        Parameters
        ----------
        t : float,
            The time to evaluate the cross sections.
        """
        for cell in self.mesh.cells:
            x = [t, self.temperature[cell.id],
                 self.initial_temperature, self.feedback_coeff]
            self.cellwise_xs[cell.id].update(x)

    def step_solutions(self) -> None:
        """Copy the current solutions to the old.
        """
        self.power_old = self.power
        self.phi_old[:] = self.phi
        self.temperature_old[:] = self.temperature
        if self.use_precursors:
            self.precursors_old[:] = self.precursors

    def assemble_transient_matrix(self, m: int = 0) -> List[csr_matrix]:
        """Assemble the multigroup evolution matrix for step `m`.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        if self.has_functional_xs:
            eff_dt = self.effective_time_step(m)
            if m == 1: eff_dt = self.dt
            self.update_cross_sections(self.time + eff_dt)
            self.L = self.diffusion_matrix()

        if self.method == "BE":
            A = self.L + self.M / self.dt
        elif self.method == "CN":
            A = self.L + 2.0 * self.M / self.dt
        elif self.method == "TBDF2" and m == 0:
            A = self.L + 4.0 * self.M / self.dt
        elif self.method == "TBDF2" and m == 1:
            A = self.L + 3.0 * self.M / self.dt
        else:
            raise NotImplementedError(f"{self.method} is not implemented.")

        A -= self.S + self.Fp
        if self.use_precursors and not self.lag_precursors:
            A -= self.precursor_substitution_matrix(m)
        return self.apply_matrix_bcs(A)

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
        if self.method == "BE":
            b = self.M / self.dt @ self.phi_old
        elif self.method == "CN":
            b = 2.0 * self.M / self.dt @ self.phi_old
        elif self.method == "TBDF2" and m == 0:
            b = 4.0 * self.M / self.dt @ self.phi_old
        elif self.method == "TBDF2" and m == 1:
            phi = self.phi_aux[0]
            b = self.M / self.dt @ (4.0 * phi - self.phi_old)

        # Add old time step precursors
        if self.use_precursors:
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
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_old_precursor_source(m)
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_old_precursor_source(m)

    def compute_fission_rate(self) -> None:
        """Compute the fission power averaged over a cell.
        """
        if isinstance(self.discretization, FiniteVolume):
            self._fv_compute_fission_rate()
        elif isinstance(self.discretization, PiecewiseContinuous):
            self._pwc_compute_fission_rate()

    def compute_power(self) -> float:
        """Compute the fission power in the system.

        Returns
        -------
        float
        """
        # Loop over cells
        power = 0.0
        for cell in self.mesh.cells:
            xs = self.material_xs[cell.material_id]
            if xs.is_fissile:
                P = self.power_density[cell.id]
                power += P * cell.volume
        return power

    @property
    def average_power_density(self) -> float:
        """Compute the average power density.

        Parameters
        ----------
        float
        """
        intgrl, volume = 0.0, 0.0
        for cell in self.mesh.cells:
            xs = self.material_xs[cell.material_id]
            if xs.is_fissile:
                P = self.power_density[cell.id]
                intgrl += P * cell.volume
                volume += cell.volume
        return intgrl / volume

    @property
    def average_temperature(self) -> float:
        """Compute the average fuel temperature.

        Returns
        -------
        float
        """
        intgrl, volume = 0.0, 0.0
        for cell in self.mesh.cells:
            xs = self.material_xs[cell.material_id]
            if xs.is_fissile:
                T = self.temperature[cell.id]
                intgrl += T * cell.volume
                volume += cell.volume
        return intgrl / volume

    @property
    def peak_power_density(self) -> float:
        """Get the peak power density.

        Returns
        -------
        float
        """
        return max(self.power_density)

    @property
    def peak_temperature(self) -> float:
        """Get the peak temperature in the system.

        Returns
        -------
        float
        """
        return max(self.temperature)

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
        if self.method == "BE":
            return self.dt
        elif self.method == "CN":
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
        # Evaluate initial condition functions
        if self.initial_conditions is not None:
            self._check_initial_conditions()

            # Get 1D grid
            grid = self.discretization.grid
            grid = np.array([p.z for p in grid])

            # Initial conditions for scalar flux
            for g, ic in enumerate(self.initial_conditions):
                if callable(ic):
                    self.phi[g::self.n_groups] = ic(grid)
                else:
                    self.phi[g::self.n_groups] = ic

            # Compute fission rate and set precursors to zero
            self.compute_fission_rate()
            if self.use_precursors:
                self.precursors[:] = 0.0

        # Initialize with a k-eigenvalue solver
        else:
            # Modify fission cross section
            if self.normalize_fission:
                for xs in self.material_xs:
                    k = self.k_eff
                    if self.exact_keff_for_ic is not None:
                        k = self.exact_keff_for_ic
                    xs.sigma_f /= k

            # Reconstruct pompt/total and delayd matrices
            self.Fp = self.prompt_fission_matrix()
            if self.use_precursors:
                self.Fd = self.delayed_fission_matrix()

            # Normalize phi to initial power conditions
            if self.phi_norm_method is not None:
                self.compute_fission_rate()
                if "TOTAL" in self.phi_norm_method:
                    self.phi *= self.power / self.compute_power()
                elif "AVERAGE" in self.phi_norm_method:
                    P_avg = self.average_power_density
                    self.phi *= self.power / P_avg

            # Compute fission rate and precursors
            self.compute_fission_rate()
            self.power = self.compute_power()
            if self.use_precursors:
                self.compute_precursors()

        self.step_solutions()

    def _check_time_step(self) -> None:
        self.time = self.t_start
        if self.output_frequency is None:
            self.output_frequency = self.dt
        if self.dt > self.output_frequency:
            self.dt = self.output_frequency

    def _check_initial_conditions(self) -> None:
        # Check number of ics
        if len(self.initial_conditions) != self.n_groups:
            raise AssertionError(
                "Invalid number of initial conditions. There must be "
                "as many initial conditions as groups.")

        # Convert to lambdas, if sympy functions
        if isinstance(self.initial_conditions, MutableDenseMatrix):
            from sympy import lambdify
            symbols = list(self.initial_conditions.free_symbols)

            ics = []
            for ic in self.initial_conditions:
                ics += lambdify(symbols, ic)
            self.initial_conditions = ics

        # Check length for vector ics
        if isinstance(self.initial_conditions, list):
            n_phi_dofs = self.discretization.n_dofs(self.phi_uk_man)
            for ic in self.initial_conditions:
                array_like = (ndarray, List[float])
                if not callable(ic) and isinstance(ic, array_like):
                    if len(ic) != n_phi_dofs:
                        raise AssertionError(
                            "Vector initial conditions must agree with "
                            "the number of DoFs associated with the "
                            "attached discretization and phi unknown "
                            "manager.")
