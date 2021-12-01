from scipy.sparse.linalg import spsolve
from pyPDEs.spatial_discretization import *
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


def update_phi(self: 'TransientSolver', m: int = 0) -> None:
    """
    Update the scalar flux for the `m`'th step.

    Parameters
    ----------
    m : int, default 0
        The step in a multi-step method.
    """
    A = self.assemble_transient_matrix(m)
    b = self.assemble_transient_rhs(m)
    self.phi = spsolve(A, b)
    self.compute_fission_rate()


def update_precursors(self: 'TransientSolver', m: int = 0) -> None:
    """
    Update the precursors for the `m`'th step.

    Parameters
    ----------
    m : int, default 0
        The step in a multi-step method.
    """
    if isinstance(self.discretization, FiniteVolume):
        self._fv_update_precursors(m)
    elif isinstance(self.discretization, PiecewiseContinuous):
        self._pwc_update_precursors(m)

    if m == 0 and self.method in ['cn', 'tbdf2']:
        self.precursors = \
            2.0 * self.precursors - self.precursors_old


def update_temperature(self: 'TransientSolver', m: int = 0) -> None:
    """
    Update the temperature for the `m`'th step.

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


def update_cross_sections(self: 'TransientSolver', t: float) -> None:
    """
    Update the cell-wise cross sections.

    Parameters
    ----------
    t : float,
        The time to evaluate the cross sections.
    """
    for cell in self.mesh.cells:
        x = [t, self.temperature[cell.id],
             self.initial_temperature, self.feedback_coeff]
        self.cellwise_xs[cell.id].update(x)
