from numpy import ndarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import KEigenvalueSolver

from pyPDEs.spatial_discretization import PiecewiseContinuous
from pyPDEs.utilities import UnknownManager


def _pwc_compute_fission_production(self: "KEigenvalueSolver") -> float:
    """Compute the neutron production rate from fission.

    Notes
    -----
    This routine uses the most up-to-date scalar flux solution.

    Returns
    -------
    float
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    production = 0.0
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over nodes
        for i in range(view.n_nodes):
            intV_shapeI = view.intV_shapeI[i]

            # Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)
                production += xs.nu_sigma_f[g] * \
                              self.phi[ig] * intV_shapeI
    return production
