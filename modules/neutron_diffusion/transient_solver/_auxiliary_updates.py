from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver

import numpy as np


def _update_precursors(
        self: 'TransientSolver',
        step: int = 0
) -> None:
    """
    Take a time step for the precursors.

    Parameters
    ----------
    self : TransientSolver
    step : int
    """
    eff_dt = self.effective_dt(step)

    # Loop over cells
    for cell in self.mesh.cells:
        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        # Compute only if fuel with precursors
        if xs.is_fissile and xs.n_precursors > 0:
            uk_map_g = self.n_groups * cell.id
            uk_map_j = self.max_precursors * cell.id

            # Compute the delayed fission rate
            Sf_d = 0.0
            nud_sigf = xs.nu_delayed_sigma_f
            for g in range(self.n_groups):
                Sf_d += nud_sigf[g] * self.phi[uk_map_g + g]

            # Loop over the precursors
            decay = xs.precursor_lambda
            gamma = xs.precursor_yield
            for j in range(xs.n_precursors):
                c_old = self.precursors_old[uk_map_j + j]
                if step == 1:
                    c = self.precursors[uk_map_j + j]
                    c_old = (4.0 * c - c_old) / 3.0

                coeff = 1.0 / (1.0 + eff_dt * decay[j])
                self.precursors[uk_map_j + j] = \
                    coeff * (c_old + eff_dt * gamma[j] * Sf_d)


def _update_temperature(
        self: 'TransientSolver',
        step: int = 0
) -> None:
    """
    Take a time step for the temperatures.

    Parameters
    ----------
    self : TransientSolver
    step : int
    """
    alpha = self.conversion_factor

    # Loop over cells
    eff_dt = self.effective_dt(step)
    for cell in self.mesh.cells:
        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        Sf = 0.0
        if xs.is_fissile:
            for g in range(self.n_groups):
                dof = self.n_groups * cell.id + g
                Sf += xs.sigma_f[g] * self.phi[dof]

        T_old = self.temperature_old[cell.id]
        if step == 1:
            T = self.temperature[cell.id]
            T_old = (4.0 * T - T_old) / 3.0

        self.temperature[cell.id] = T_old + eff_dt * alpha * Sf


def _compute_bulk_properties(self: 'TransientSolver') -> None:
    """
    Compute bulk system properties.

    This updates the reactor power, peak power density, average power
    density, peak fuel temperature, and average fuel temperature.

    Parameters
    ----------
    self : TransientSolver
    """
    Ef = self.energy_per_fission

    n_fuel_cells = 0
    V_fuel = 0.0

    self.power = 0.0
    self.peak_power_density = 0.0
    self.average_power_density = 0.0
    self.peak_fuel_temperature = 0.0
    self.average_fuel_temperature = 0.0
    for cell in self.mesh.cells:
        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        # Compute only if fuel
        if xs.is_fissile:

            # Compute cell power density
            Sf = 0.0
            for g in range(self.n_groups):
                dof = self.n_groups * cell.id + g
                Sf += xs.sigma_f[g] * self.phi[dof]

            n_fuel_cells += 1
            V_fuel += cell.volume
            self.power += Ef * Sf * cell.volume
            self.peak_power_density = max(self.peak_power_density, Ef * Sf)

            T = self.temperature[cell.id]
            self.average_fuel_temperature += T
            self.peak_fuel_temperature = max(self.peak_fuel_temperature, T)

    self.average_power_density = self.power / V_fuel
    self.average_fuel_temperature /= n_fuel_cells
