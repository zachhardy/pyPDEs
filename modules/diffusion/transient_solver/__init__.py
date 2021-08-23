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
    """Transient solver for multi-group diffusion problems.

    Attributes
    ----------
    mesh : Mesh
        The spatial mesh to solve the problem on.

    discretization : SpatialDiscretization
        The spatial discretization used to solve the problem.

    boundaries : List[Boundary]
        The boundary conditions imposed on the equations.
        There should be a boundary condition for each group
        and boundary. In the list, each boundaries group-wise
        boundary conditions should be listed next to each other.

    material_xs : List[CrossSections]
        The cross sections corresponding to the material IDs
        defined on the cells. There should be as many cross
        sections as unique material IDs on the mesh.

    material_src : List[MultigroupSource]
        The multi-group sources corresponding to the material
        IDs defined on the cells. There should be as many
        multi-group sources as unique material IDs on the mesh.

    use_precursors : bool
        A flag for including delayed neutrons.
    tolerance : float
        The iterative tolerance for the group-wise solver.
    max_iterations : int
        The maximum number of iterations for the group-wise
        solver to take before exiting.

    b : ndarray (n_nodes * n_groups,)
        The right-hand side of the linear system to solve.
    L : List[csr_matrix]
        The group-wise diffusion operators used to solve the
        equations group-wise. There are n_groups matrices stored.

    phi : ndarray (n_nodes * n_groups,)
        The most current scalar flux solution vector.
    flux_uk_man : UnknownManager
        An unknown manager tied to the scalar flux solution vector.

    precurosrs : ndarray (n_nodes * max_precursors_per_material,)
        The delayed neutron precursor concentrations.

        In multi-material problems, this vector stores up to the
        maximum number of precursors that live on any given material.
        This implies that material IDs must be used to map the
        concentration of specific precursor species. This structure
        is used to prevent very sparse vectors in many materials.
    precursor_uk_man : UnknownManager
        An unknown manager tied to the precursor vector.

    k_eff : float
        The most current k-eigenvalue estimate.

    initial_conditions : list
        An n_groups in length list of initial conditions.
        These may take the form of lambda functions,
        appropriately sized vectors, or a sympy Matrix.

    t_final : float
        The final simulation time.
    dt : float
        The initial, or fixed time step size.
    stepping_method = {"BACKWARD_EULER", "CRANK_NICHOLSON", "TBDF2"}
        The time stepping method to use.

    output_frequency : float
        The intervals in which to write outputs.

    adaptivity : bool
        A flag for using adaptive time stepping.
    refine_level : float
        The relative change in power over a time step at which
        larger changes in power triggers a time step halving.
    coarsen_level : float
        The relative change in power over a time step at which
        smaller changes in power triggers a time step doubling.

    lag_precursors : bool
        A flag for using lagged precursors or an implicit substitution.

    power, power_old : float
        The current and previous time step fission power.
    energy_per_fission : float
        A scaling factor to apply to the computed fission rate to
        match the specified initial power.

    phi_old : ndarray (n_nodes * n_groups,)
        The scalar flux solution last time step.
    precursors_old : ndarray (n_nodes, max_precursors_per_material,)
        The precursor solution last time step.

    b_old : ndarray (n_nodes * n_groups,)
        The right-hand side contributions from last time step. This
        is comprised of portions of the time derivative term containing
        previous solution vectors.
    M : List[csr_matrix]
        The inverse velocity mass matrix group-wise. There are
        n_groups matrices stored for solving group-wise.
    A : List[List[csr_matrix]]
        The evolution operators. The outer list is n_groups in
        length for each group and the inner lists contain all
        matrices necessary for evolving a group through a time
        step. The inner list only has more than one entry if
        TBDF2 time stepping is used.

    outputs : Outputs
        A support class used for storing simulation results as
        the simulation progresses and for writing them at its
        completion.
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

        self.t_final: float = 0.25
        self.dt: float = 5.0e-3
        self.stepping_method: str = "CRANK_NICHOLSON"

        self.output_frequency: float = None

        self.adaptivity: bool = False
        self.refine_level: float = 0.02
        self.coarsen_level: float = 0.005

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
        """Initialize the transient multi-group diffusion solver.
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

        if not self.output_frequency:
            self.output_frequency = self.dt

        if self.dt > self.output_frequency:
            self.dt = self.output_frequency

        self.outputs.reset()
        self.store_outputs(0.0)

        self.energy_per_fission = self.power / self.compute_power()

    def execute(self, verbose: bool = False) -> None:
        """Execute the transient multi-group diffusion solver.
        """
        print("\n***** Executing the transient "
              "multi-group diffusion solver. *****")

        # ======================================== Start time stepping
        time, n_steps, dt0 = 0.0, 0, self.dt
        next_output_time = self.output_frequency
        while time < self.t_final - sys.float_info.epsilon:

            # ==================== Force coincidence with output times
            if time + self.dt > next_output_time:
                self.dt = next_output_time - time
                self.assemble_evolution_matrices()

            # ==================== Force coincidence with end time
            if time + self.dt > self.t_final:
                self.dt = self.t_final - time
                self.assemble_evolution_matrices()

            # ============================== Solve time step
            self.solve_time_step()
            self.post_process_time_step()
            self.power = self.compute_power()

            dP = abs(self.power - self.power_old) / self.power_old
            while self.adaptivity and dP > self.refine_level:
                self.phi[:] = self.phi_old
                self.precursors[:] = self.precursors_old

                self.dt /= 2.0
                self.assemble_evolution_matrices()

                self.solve_time_step()
                self.post_process_time_step()

                self.power = self.compute_power()
                dP = abs(self.power - self.power_old) / self.power_old

            time += self.dt
            n_steps += 1

            # ============================== Reset vectors
            self.phi_old[:] = self.phi
            self.power_old = self.power
            if self.use_precursors:
                self.precursors_old[:] = self.precursors

            # ============================== Output solutions
            if time == next_output_time:
                self.store_outputs(time)
                next_output_time += self.output_frequency
                if next_output_time > self.t_final:
                    next_output_time = self.t_final

            # ============================== Coarsen time steps
            if self.adaptivity and dP < self.coarsen_level:
                self.dt *= 2.0
                self.assemble_evolution_matrices()
            if self.adaptivity and self.dt > self.output_frequency:
                self.dt = self.output_frequency
                self.assemble_evolution_matrices()
            if not self.adaptivity:
                self.dt = dt0
                self.assemble_evolution_matrices()

            # ============================== Print time step summary
            if verbose:
                print(f"***** Time Step: {n_steps} *****")
                print(f"Simulation Time:\t{time}")
                print(f"Time Step Size:\t\t{self.dt}")
                print(f"System Power:\t\t{self.power}\n")

        self.dt = dt0  # reset dt to original

    def solve_time_step(self, step: int = 0) -> None:
        """Solve the system for the n'th step of a time step.

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
            print(f"!!!!! WARNING: Solver did not converge. "
                  f"Final Change: {phi_change:.3e} !!!!!")

        if self.use_precursors:
            self.update_precursors(step)

    def post_process_time_step(self) -> None:
        """Post-process the time step results.

        For Backward Euler, nothing is done. For Crank Nicholson,
        this computes the next time step value from the previous and
        half time step values. For TBDF-2, this computes the half time
        step value from the previous and quarter time step values, then
        takes a step of BDF-2.
        """
        # =================================== Handle 2nd order methods
        if self.stepping_method in ["CRANK_NICHOLSON", "TBDF2"]:
            self.phi = 2.0 * self.phi - self.phi_old
            if self.use_precursors:
                self.precursors = \
                    2.0 * self.precursors - self.precursors_old

            # ============================== Second step of time step
            if self.stepping_method == "TBDF2":
                self.solve_time_step(step=1)

    def set_old_transient_source(self, step: int = 0) -> ndarray:
        """Assemble the previous time step portion of the right-hand side.

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
        """Assemble the linear systems for a time step..

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
        """ Assemble the mass matrix for time stepping for group g

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
        """Assemble the right-hand side of the linear system for group g.

        This includes previous time step contributions as well as material,
        scattering, fission, and boundary sources for group g.

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
        """Solve a precursor time step.

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
        """Compute the fission power.

        Notes
        -----
        This method uses the most current scalar flux solution.

        Returns
        -------
        float
        """
        if isinstance(self.discretization, FiniteVolume):
            return self.fv_compute_power()
        else:
            return self.pwc_compute_power()

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
        """Evaluate the initial conditions."""
        if self.initial_conditions is None:
            KEigenvalueSolver.execute(self, verbose=False)
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
