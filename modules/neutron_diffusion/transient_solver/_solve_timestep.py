from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver

from scipy.sparse.linalg import spsolve


def _solve_timestep(
        self: 'TransientSolver',
        reconstruct_matrices: bool = False
) -> None:
    """
    Solve a time-step.
    """

    # ------------------------------ construct matrices
    if reconstruct_matrices or self.has_dynamic_xs:
        self._assemble_transient_matrices(
            with_scattering=True, with_fission=True
        )

    # ------------------------------ solve the time step
    self._solve_timestep_step(0)
    if self.time_stepping_method == "TBDF2":
        self._solve_timestep_step(1)


def _solve_timestep_step(self: 'TransientSolver', step: int = 0) -> None:
    """
    Solve a time step for the specified step of a multistep method.

    If the reconstruct_matrix flag is set to true, the multi-group
    matrix is reconstructed.
    """

    # ------------------------------ solve the time-step
    self._b[:] = 0.0
    self._assemble_transient_rhs(
        step=step, with_material_src=True, with_boundary_src=True
    )
    self.phi = spsolve(self._A[step], self._b)

    # ------------------------------ update auxiliary unknowns
    self._update_temperature(step)
    if self.use_precursors:
        self._update_precursors(step)

    # ------------------------------ finalize the time-step
    if self.time_stepping_method != "BACKWARD_EULER" and step == 0:

        # -------------------- scalar flux
        self.phi = 2.0 * self.phi - self.phi_old

        # -------------------- temperature
        self.temperature = 2.0 * self.temperature - self.temperature_old

        # -------------------- precursors
        if self.use_precursors:
            self.precursors = 2.0 * self.precursors - self.precursors_old


def _refine_timestep(self: 'TransientSolver') -> None:
    """
    Refine the time step until the relative power change meets the criteria.
    """

    # ------------------------------ compute relative power change
    dP = abs(self.power - self.power_old) / abs(self.power_old)

    # ------------------------------ iterate until criteria is met
    while dP > self.refine_threshold:

        # half the time-step size
        self.dt /= 2.0
        if self.dt < self.dt_min:
            self.dt = self.dt_min

        # ------------------------------ resolve the time-step
        self._solve_timestep(True)
        self._compute_bulk_properties()

        # ------------------------------ exit if dt is min dt
        if self.dt == self.dt_min:
            break

        # ------------------------------ recompute relative power change
        dP = abs(self.power - self.power_old) / abs(self.power_old)


def _coarsen_timestep(self: 'TransientSolver') -> None:
    """
    Coarsen the time step.
    """
    # ------------------------------ compute relative power change
    dP = abs(self.power - self.power_old) / abs(self.power_old)

    # ------------------------------ double the time-step
    if dP < self.coarsen_threshold:
        self.dt *= 2.0

        # do not exceed output frequency
        if self.dt > self.output_frequency:
            self.dt = self.output_frequency

        # rebuild the matrices
        self._assemble_transient_matrices(
            with_scattering=True, with_fission=True
        )
