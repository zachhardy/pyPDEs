from numpy import ndarray
from typing import TYPE_CHECKING

from pyPDEs.spatial_discretization import FiniteVolume
from pyPDEs.utilities import UnknownManager

if TYPE_CHECKING:
    from .keigenvalue_solver import KEigenvalueSolver


def fv_compute_fission_production(self: 'KEigenvalueSolver') -> float:
    """
    Compute the fission production from the most recent
    solution vector.

    Returns
    -------
    float
    """
    fv: FiniteVolume = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # ======================================== Loop over cells
    production = 0.0
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # =================================== Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            production += xs.nu_sigma_f[g] * self.phi[ig] * volume
    return production
