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
    """
    Transinet multigroup diffusion.
    """

    from ._input_checks import (_check_time_step,
                                _check_initial_conditions)

    from ._timestepping import (solve_time_step,
                                refine_time_step,
                                coarsen_time_step,
                                step_solutions)

    from ._updates import (update_phi,
                           update_precursors,
                           update_temperature,
                           update_cross_sections)

    from ._write_outputs import write_snapshot

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

    from ._alphaeigenvalue import solve_alpha_eigenproblem

    def __init__(self) -> None:
        super().__init__()
        self.initial_conditions: list = None
        self.normalize_fission: bool = True
        self.exact_keff_for_ic: float = None
        self.phi_norm_method: str = 'total'

        # Time stepping parameters
        self.time: float = 0.0
        self.t_start: float = 0.0
        self.t_final: float = 0.1
        self.dt: float = 2.0e-3
        self.method: str = 'tbdf2'

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

    @property
    def average_power_density(self) -> float:
        """
        Compute the average power density.

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
        """
        Compute the average fuel temperature.

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
        """
        Get the peak power density.

        Returns
        -------
        float
        """
        return max(self.power_density)

    @property
    def peak_temperature(self) -> float:
        """
        Get the peak temperature in the system.

        Returns
        -------
        float
        """
        return max(self.temperature)

    def initialize(self, verbose: int = 0) -> None:
        """
        Initialize the solver.
        """
        KEigenvalueSolver.initialize(self)
        KEigenvalueSolver.execute(self, verbose=verbose)

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

        # Precompute matrices
        self.M = self.mass_matrix()

        # Compute initial values
        self.compute_initial_values()

    def execute(self, verbose: int = 0) -> None:
        """
        Execute the transient multigroup diffusion solver.

        Parameters
        ----------
        verbose : int, default 0
        """
        if self.adjoint:
            raise NotImplementedError(
                'Adjoint transients have not been implemented.')
        if self._adjoint_matrices:
            self._transpose_matrices()

        # Check output information
        if self.write_outputs:
            if not os.path.isdir(self.output_directory):
                os.makedirs(self.output_directory)
            elif len(os.listdir(self.output_directory)) > 0:
                os.system(f'rm -r {self.output_directory}/*')

        # Store initial conditions
        self.write_snapshot(0)

        # Initialize auxilary vectors
        if self.method == 'tbdf2':
            self.phi_aux = [np.copy(self.phi)]
            self.precursors_aux = [np.copy(self.precursors)]
            self.temperature_aux = [np.copy(self.temperature)]

        # Initialize outputting stuff
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
            if self.method == 'tbdf2':
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
                print(f'***** Time Step: {n_steps} *****')
                print(f'Simulation Time:\t{self.time:.3e} sec')
                print(f'Time Step Size:\t\t{self.dt:.3e} sec')
                print(f'Total Power:\t\t{self.power:.3e} W')
                P_avg = self.average_power_density
                print(f'Average Power Density:\t{P_avg:.3e} W/cm^3')
                T_avg = self.average_temperature
                print(f'Average Temperature:\t{T_avg:.3g} K')

        self.dt = dt_initial

    def assemble_transient_matrix(self, m: int = 0) -> csr_matrix:
        """
        Assemble the multigroup evolution matrix for step `m`.

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
            if m == 1:
                eff_dt = self.dt
            self.update_cross_sections(self.time + eff_dt)
            self.L = self.diffusion_matrix()

        if self.method == 'be':
            A = self.L + self.M / self.dt
        elif self.method == 'cn':
            A = self.L + 2.0 * self.M / self.dt
        elif self.method == 'tbdf2' and m == 0:
            A = self.L + 4.0 * self.M / self.dt
        elif self.method == 'tbdf2' and m == 1:
            A = self.L + 3.0 * self.M / self.dt
        else:
            raise NotImplementedError(f'{self.method} is not implemented.')

        A -= self.S + self.Fp
        if self.use_precursors and not self.lag_precursors:
            A -= self.precursor_substitution_matrix(m)
        return self.apply_matrix_bcs(A)

    def assemble_transient_rhs(self, m: int = 0) -> ndarray:
        b = self.set_source() + self.set_old_source(m)
        return self.apply_vector_bcs(b)

    def mass_matrix(self) -> csr_matrix:
        """
        Assemble the multigroup mass matrix.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_mass_matrix()
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_mass_matrix()

    def precursor_substitution_matrix(self, m: int = 0) -> csr_matrix:
        """
        Assemble the transient precursor substitution matrix for step `m`.

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
        """
        Assemble the right-hand side with terms from last time step.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.

        Returns
        -------
        ndarray (n_cells * n_groups)
        """
        if self.method == 'be':
            b = self.M / self.dt @ self.phi_old
        elif self.method == 'cn':
            b = 2.0 * self.M / self.dt @ self.phi_old
        elif self.method == 'tbdf2' and m == 0:
            b = 4.0 * self.M / self.dt @ self.phi_old
        elif self.method == 'tbdf2' and m == 1:
            phi = self.phi_aux[0]
            b = self.M / self.dt @ (4.0 * phi - self.phi_old)
        else:
            raise NotImplementedError(
                'Invalid method and step provided.')

        # Add old time step precursors
        if self.use_precursors:
            b += self.old_precursor_source(m)
        return b

    def old_precursor_source(self, m: int = 0) -> ndarray:
        """
        Assemble the delayed terms from the last time step for step `m`.

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
        """
        Compute the fission power averaged over a cell.
        """
        if isinstance(self.discretization, FiniteVolume):
            self._fv_compute_fission_rate()
        elif isinstance(self.discretization, PiecewiseContinuous):
            self._pwc_compute_fission_rate()

    def compute_power(self) -> float:
        """
        Compute the fission power in the system.

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

    def effective_time_step(self, m: int = 0) -> float:
        """
        Compute the effective time step for step `m`.

        Parameters
        ----------
        m : int, default 0
            The step in a multi-step method.

        Returns
        -------
        float
        """
        if self.method == 'be':
            return self.dt
        elif self.method == 'cn':
            return self.dt / 2.0
        elif self.method == 'tbdf2' and m == 0:
            return self.dt / 4.0
        elif self.method == 'tbdf2' and m == 1:
            return self.dt / 3.0
        else:
            raise NotImplementedError(f'{self.method} is not implemented.')

    def compute_initial_values(self) -> None:
        """
        Evaluate the initial conditions.
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
                    xs.k_eff = k

                # Reconstruct pompt/total and delayd matrices
                self.Fp = self.prompt_fission_matrix()
                if self.use_precursors:
                    self.Fd = self.delayed_fission_matrix()

        # Normalize phi to initial power conditions
        if self.phi_norm_method is not None:
            self.compute_fission_rate()

            if 'total' in self.phi_norm_method:
                self.phi *= self.power / self.compute_power()
            elif 'average' in self.phi_norm_method:
                P_avg = self.average_power_density
                self.phi *= self.power / P_avg

        # Compute fission rate and precursors
        self.compute_fission_rate()
        self.power = self.compute_power()
        if self.use_precursors and self.initial_conditions is None:
            self.compute_precursors()

        self.step_solutions()
