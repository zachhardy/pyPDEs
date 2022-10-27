import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

from typing import Union

from pyPDEs.math.discretization import FiniteVolume
from pyPDEs.material import Material

from .. import SteadyStateSolver
from .. import KEigenvalueSolver


class TransientSolver(KEigenvalueSolver):
    """
    Implementation of a transient multi-group diffusion solver.
    """

    # Energy released per fission event in J
    energy_per_fission: float = 3.204e-11

    # Conversion factor to convert energy to material temperature (K cm^3)
    conversion_factor: float = 3.83e-11

    from ._assemble_matrix import _assemble_transient_matrices
    from ._assemble_matrix import _assemble_transient_matrix

    from ._assemble_rhs import _assemble_transient_rhs

    from ._solve_timestep import _solve_timestep
    from ._solve_timestep import _solve_timestep_step
    from ._solve_timestep import _refine_timestep
    from ._solve_timestep import _coarsen_timestep

    from ._auxiliary_updates import _update_precursors
    from ._auxiliary_updates import _update_temperature
    from ._auxiliary_updates import _compute_bulk_properties

    from ._write import write_temperature
    from ._write import write_snapshot

    def __init__(
            self,
            discretization: FiniteVolume,
            materials: list[Material],
            boundary_info: list[tuple[str, int]],
            boundary_values: list[list[list[float]]] = None
    ) -> None:
        super().__init__(discretization, materials,
                         boundary_info, boundary_values)

        # ==================== General Options ==================== #

        self.write_outputs: bool = False
        self.output_directory: str = None

        # The frequency outputs are written. If this is None, the solution is
        # written at each time step, otherwise, solutions are written at the
        # specified interval. If adaptive time stepping is on, time steps are
        # modified to ensure coincidence with output times. If adaptive time
        # stepping is off and the output frequency is not an integer multiple
        # of the time step size an error is thrown. If the specified initial
        # time step size is larger than the specified output frequency, the
        # time step size is changed to the output frequency
        self.output_frequency: float = None

        self.lag_precursors: bool = False

        # The normalization method for the initial condition. This is used
        # to map the magnitude of the scalar flux profile to a specified
        # reactor power or average power density.
        self.normalization_method: str = "TOTAL_POWER"

        # A flag for whether to normalize the fission cross-sections
        # to the computed k-eigenvalue.
        self.scale_fission_xs: bool = False

        self.is_nonlinear: bool = False
        self.nonlinear_tolerance: float = 1.0e-8
        self.nonlinear_max_iterations: int = 50

        # ==================== Initial Conditions ==================== #

        self.initial_power: float = 1.0
        self.initial_temperature: float = 300.0

        # The initial conditions. These can either be a dictionary indexed
        # by group number, a numpy.ndarray, or None. If None, a k-eigenvalue
        # initial condition is used.
        self.initial_conditions: dict = None

        # ==================== Time Stepping ==================== #

        self.t_start: float = 0.0
        self.t_end: float = 1.0
        self.dt: float = 0.1

        self.time_stepping_method: str = "CRANK_NICHOLSON"

        # A flag for whether to use adaptive time stepping or not.
        # Currently, adaptive time stepping is based only on the relative
        # power change over a time step. When the relative change in power is
        # greater than the refine threshold, the time step size is halved and
        # the  time step repeated until an acceptable relative power change is
        # achieved or the time step reaches the minimum allowed time step.
        # When the relative change in power is smaller than the coarsen
        # threshold, the time step is doubled. When this occurs the time step
        # is not repeated.
        self.adaptive_time_stepping: bool = False

        self.refine_threshold: float = 0.05
        self.coarsen_threshold: float = 0.01
        self.dt_min: float = 1.0e-6

        # ==================== Problem Information ==================== #

        # A flag for whether the problem has dynamic cross-sections.
        self.has_dynamic_xs: bool = False

        self.time: float = 0.0

        self.power: float = 1.0
        self.power_old: float = 1.0

        self.phi_old: np.ndarray = None
        self.precursors_old: np.ndarray = None

        self.temperature: np.ndarray = None
        self.temperature_old: np.ndarray = None

        self.average_power_density: float = 0.0
        self.peak_power_density: float = 0.0

        self.average_fuel_temperature: float = 300.0
        self.peak_fuel_temperature: float = 300.0

    def initialize(self) -> None:
        """
        Initialize the transient multi-group diffusion solver.
        """
        KEigenvalueSolver.initialize(self)
        KEigenvalueSolver.execute(self)

        # ========================================
        # Physics option checks and sets
        # ========================================

        for xs in self.material_xs:
            if xs.sigma_a_function is not None:
                self.has_dynamic_xs = True
                break

        norm_opts = ["TOTAL_POWER", "AVERAGE_POWER_DENSITY", None]
        if self.normalization_method not in norm_opts:
            msg = "Invalid scalar flux normalization method."
            raise ValueError(msg)

        # ========================================
        # Check time discretization
        # ========================================

        if self.t_start >= self.t_end:
            msg = "The start time must be less than the end time."
            raise AssertionError(msg)
        self.time = self.t_start

        methods = ["BACKWARD_EULER", "CRANK_NICHOLSON", "TBDF2"]
        if self.time_stepping_method not in methods:
            msg = f"Unrecognized time stepping method"
            raise ValueError(msg)

        # ========================================
        # Check output options
        # ========================================

        if self.write_outputs:

            # Set output frequency to time step size if not specified
            if self.output_frequency is None:
                self.output_frequency = self.dt

            # Set time step size to output frequency if larger
            if self.dt > self.output_frequency:
                self.dt = self.output_frequency

            # For non-adaptive time stepping, ensure that the output
            # frequency is an integer multiple of the time step size
            if (not self.adaptive_time_stepping and
                    self.output_frequency % self.dt):
                msg = "The output frequency must be an integer multiple of " \
                      "the time step size when adaptive time stepping is off."
                raise AssertionError(msg)

            print(f"Setting up output directories at {self.output_directory}")
            if not os.path.isdir(self.output_directory):
                os.makedirs(self.output_directory)
            elif len(os.listdir(self.output_directory)) > 0:
                os.system(f"rm -r {self.output_directory}/*")

        # ========================================
        # Initialize auxiliary vectors
        # ========================================

        self.phi_old = np.zeros(self.phi.size)
        if self.use_precursors:
            self.precursors_old = np.zeros(self.precursors.size)

        T_initial = self.initial_temperature
        self.temperature = np.array([T_initial] * self.mesh.n_cells)
        self.temperature_old = np.array([T_initial] * self.mesh.n_cells)

    def execute(self) -> None:
        """
        Execute the transient multi-group diffusion solver.
        """

        eps = 1.0e-10

        msg = "Executing the transient multi-group diffusion solver"
        msg = "\n".join(["", "*" * len(msg), msg, "*" * len(msg), ""])
        print(msg)

        # ========================================
        # Compute initial values
        # ========================================

        self._compute_initial_values()

        # ========================================
        # Setup for outputting
        # ========================================

        output_num = 0
        next_output = self.output_frequency
        if self.write_outputs:
            self.discretization.write_discretization(self.output_directory)
            self.write_snapshot(output_num)
            output_num += 1

        # ========================================
        # Setup matrix
        # ========================================

        self._assemble_transient_matrices(with_scattering=True,
                                          with_fission=True)

        # ========================================
        # Start time stepping
        # ========================================

        step = 0
        dt_initial = self.dt
        self.time = self.t_start
        while self.time < self.t_end - eps:

            # A flag for reconstructing the system matrix.
            reconstruct_matrices = False

            # ========================================
            # Modify time steps to coincide with output times
            # and the end of the simulation
            # ========================================

            if self.write_outputs:
                if self.time + self.dt > next_output + eps:
                    self.dt = next_output - self.time
                    reconstruct_matrices = True

            if self.time + self.dt > self.t_end + eps:
                self.dt = self.t_end - self.time
                reconstruct_matrices = True

            # ========================================
            # Solve the time step
            # ========================================

            self._solve_timestep(reconstruct_matrices)
            self._compute_bulk_properties()

            if self.adaptive_time_stepping:
                self._refine_timestep()

            # ========================================
            # Finalize time step
            # ========================================

            self.time += self.dt
            step += 1

            # Output solutions
            if self.write_outputs:
                if abs(self.time - next_output) < eps:
                    self.write_snapshot(output_num)
                    output_num += 1

                    # Do not pass the end of the simulation
                    next_output += self.output_frequency
                    if (next_output > self.t_end or
                            abs(next_output - self.t_end) < eps):
                        next_output = self.t_end

            # Coarsen time step
            if self.adaptive_time_stepping:
                self._coarsen_timestep()

            # If no adaptive time stepping, reset dt to its original
            # value if modified for outputting. Realistically,  this
            # should not have any significant impact since the output
            # frequency in this case must be an integer multiple of the
            # time step size.
            elif self.dt != dt_initial:
                self.dt = dt_initial
                self._assemble_transient_matrices(with_scattering=True,
                                                  with_fission=True)

            self._step_solutions()

            print(f"***** Time Step {step} *****")
            print(f"Simulation Time         : {self.time:.3g} s")
            print(f"Time Step Size          : {self.dt:.3e} s")
            print(f"Reactor Power           : {self.power:.3e} W")
            print(f"Peak Power Density      : "
                  f"{self.peak_power_density:.3e} W/cc")
            print(f"Average Power Density   : "
                  f"{self.average_power_density:.3e} W/cc")
            print(f"Peak Fuel Temperature   : "
                  f"{self.peak_fuel_temperature:.3f} K")
            print(f"Average Fuel Temperature: "
                  f"{self.average_fuel_temperature:.3f} K")
            print()

        self.dt = dt_initial

    def _step_solutions(self) -> None:
        """
        Reset the solution vectors for next time step.
        """
        self.power_old = self.power
        self.phi_old[:] = self.phi
        self.temperature_old[:] = self.temperature
        if self.use_precursors:
            self.precursors_old[:] = self.precursors

    def _compute_initial_values(self) -> None:
        """
        Compute the initial values.
        """
        print("Computing initial conditions.")

        # ========================================
        # Evaluate initial condition functions
        # ========================================

        if isinstance(self.initial_conditions, dict):

            # Zero the flux moment vector
            self.phi[:] = 0.0

            # Loop over cells
            for cell in self.mesh.cells:

                # Get the nodes on the cell
                nodes = self.discretization.nodes(cell)

                # Loop over the nodes on the cell
                for i in range(len(nodes)):

                    # Go through initial condition dictionary
                    for g, f in self.initial_conditions.items():
                        dof = self.n_groups * len(nodes) * cell.id + g
                        self.phi[dof] = f(nodes[i])

        # ========================================
        # k-eigenvalue initial conditions
        # ========================================

        else:

            # Scale the fission cross-sections for perfectly steady-state
            if self.scale_fission_xs:
                for xs in self.material_xs:
                    xs.sigma_f /= self.k_eff
                    xs.nu_sigma_f /= self.k_eff
                    xs.nu_prompt_sigma_f /= self.k_eff
                    xs.nu_delayed_sigma_f /= self.k_eff

        # ========================================
        # Normalization of initial conditions
        # ========================================

        # Normalize the scalar flux for a reactor power level
        if self.normalization_method is not None:
            msg = "Normalizing the initial condition to the specified "
            if self.normalization_method == "TOTAL_POWER":
                msg += "total power"
            elif self.normalization_method == "AVERAGE_POWER_DENSITY":
                msg += "average power density"
            msg += f" ({self.initial_power:.3e})"
            print(msg)

            self._compute_bulk_properties()

            if self.normalization_method == "TOTAL_POWER":
                self.phi *= self.initial_power / self.power
            elif self.normalization_method == "AVERAGE_POWER_DENSITY":
                self.phi *= self.initial_power / self.average_power_density

        # Initialize bulk properties
        self._compute_bulk_properties()
        if self.use_precursors:
            if self.initial_conditions is None:
                SteadyStateSolver._compute_precursors(self)
            else:
                self.precursors[:] = 0.0

        # Set old vectors
        self._step_solutions()

    def effective_dt(self, step: int = 0) -> float:
        """
        Return the effective time-step size for the discretization method.

        The step parameter is used for designating the step of a multistep
        method.

        Parameters
        ----------
        step : int

        Returns
        -------
        float
        """
        if self.time_stepping_method == "BACKWARD_EULER":
            return self.dt
        elif self.time_stepping_method == "CRANK_NICHOLSON":
            return self.dt / 2.0
        elif self.time_stepping_method == "TBDF2":
            return self.dt / 4.0 if step == 0 else self.dt / 3.0
        else:
            msg = "Unrecognized time stepping method."
            raise NotImplementedError(msg)
