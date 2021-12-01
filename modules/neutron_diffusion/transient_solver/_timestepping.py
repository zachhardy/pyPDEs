import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


def refine_time_step(self: 'TransientSolver') -> None:
    """
    Refine the time step.
    """
    dP = abs(self.power - self.power_old) / self.power_old
    while dP > self.refine_level:
        # Half the time step
        self.dt /= 2.0

        # Take the reduced time step
        self.solve_time_step(m=0)
        if self.method == 'tbdf2':
            self.solve_time_step(m=1)
        self.power = self.compute_power()

        # Compute new change in power
        dP = abs(self.power - self.power_old) / self.power_old


def coarsen_time_step(self: 'TransientSolver') -> None:
    """
    Coarsen the time step.
    """
    dP = abs(self.power - self.power_old) / self.power_old
    if dP < self.coarsen_level:
        self.dt *= 2.0
        if self.dt > self.output_frequency:
            self.dt = self.output_frequency


def solve_time_step(self: 'TransientSolver', m: int = 0) -> None:
    """
    Solve the `m`'th step of a multi-step method.
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
            if m == 0 and self.method == 'tbdf2':
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
            print('\n!!! WARNING: Nonlinear iterations '
                  'did not converge !!!\n')

    if self.use_precursors:
        self.update_precursors(m)

    if m == 0 and self.method in ['cn', 'tbdf2']:
        self.phi = 2.0 * self.phi - self.phi_old
        self.temperature = 2.0 * self.temperature - self.temperature_old
        self.compute_fission_rate()

        if self.method == 'tbdf2':
            self.phi_aux[0][:] = self.phi
            self.temperature_aux[0][:] = self.temperature
            if self.use_precursors:
                self.precursors_aux[0][:] = self.precursors


def step_solutions(self: 'TransientSolver') -> None:
    """
    Copy the current solutions to the old.
    """
    self.power_old = self.power
    self.phi_old[:] = self.phi
    self.temperature_old[:] = self.temperature
    if self.use_precursors:
        self.precursors_old[:] = self.precursors
