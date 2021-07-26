from numpy import ndarray
from typing import TYPE_CHECKING

from pyPDEs.spatial_discretization import FiniteVolume
from pyPDEs.utilities import UnknownManager

if TYPE_CHECKING:
    from .keigenvalue_solver import KEigenvalueSolver


def fv_compute_fission_production(self: 'KEigenvalueSolver',
                                  phi: ndarray) -> float:
    """
    Compute the fission production from a given vector for
    finite volume discretizations.

    Parameters
    ----------
    phi : ndarray

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
        for g in range(self.num_groups):
            i = fv.map_dof(cell, 0, uk_man, 0, g)
            production += xs.nu_sigma_f[g] * phi[i] * volume
    return production
