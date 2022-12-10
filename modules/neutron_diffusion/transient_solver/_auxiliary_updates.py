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
    """
    eff_dt = self.effective_dt(step)

    # ---------------------------------------- loop over cells
    for cell in self.mesh.cells:

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        # compute only if fissile with precursors
        if xs.is_fissile and xs.n_precursors > 0:
            uk_map_g = self.n_groups * cell.id
            uk_map_j = self.max_precursors * cell.id

            # ------------------------------ compute delayed fission rate
            Sf_d = 0.0
            nud_sigf = xs.nu_delayed_sigma_f
            for g in range(self.n_groups):
                Sf_d += nud_sigf[g] * self.phi[uk_map_g + g]

            # ------------------------------ solve precursor time-step
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
    """
    alpha = self.conversion_factor
    eff_dt = self.effective_dt(step)

    # ---------------------------------------- loop over cells
    for cell in self.mesh.cells:

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        # update only needed if fissile
        if xs.is_fissile:

            # ------------------------------ compute fission rate
            Sf = 0.0
            for g in range(self.n_groups):
                dof = self.n_groups * cell.id + g
                Sf += xs.sigma_f[g] * self.phi[dof]

            # ------------------------------ solve time-step
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
    """
    Ef = self.energy_per_fission

    # ---------------------------------------- zero everything
    n_fuel_cells = 0
    V_fuel = 0.0

    self.power = 0.0
    self.peak_power_density = 0.0
    self.average_power_density = 0.0
    self.peak_fuel_temperature = 0.0
    self.average_fuel_temperature = 0.0

    # ---------------------------------------- loop over cells
    for cell in self.mesh.cells:

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        # only consider fuel data
        if xs.is_fissile:

            # ------------------------------ compute fission rate
            Sf = 0.0
            for g in range(self.n_groups):
                dof = self.n_groups * cell.id + g
                Sf += xs.sigma_f[g] * self.phi[dof]

            # ------------------------------ increment fuel counters
            n_fuel_cells += 1
            V_fuel += cell.volume

            # ------------------------------ update power properties
            self.power += Ef * Sf * cell.volume
            self.peak_power_density = max(self.peak_power_density, Ef * Sf)

            # ------------------------------ update temperature properties
            T = self.temperature[cell.id]
            self.average_fuel_temperature += T
            self.peak_fuel_temperature = max(self.peak_fuel_temperature, T)

    # ------------------------------ compute averaged properties
    self.average_power_density = self.power / V_fuel
    self.average_fuel_temperature /= n_fuel_cells
