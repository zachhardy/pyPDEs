from numpy import ndarray
from typing import TYPE_CHECKING

from pyPDEs.spatial_discretization import PiecewiseContinuous
from pyPDEs.utilities import UnknownManager

if TYPE_CHECKING:
    from .keigenvalue_solver import KEigenvalueSolver


def pwc_compute_fission_production(self: 'KEigenvalueSolver',
                                   phi: ndarray) -> float:
    """
    Compute the fission production from a given vector for
    finite volume discretizations.
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # ======================================== Loop over cells
    production = 0.0
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # ============================== Loop over nodes
        for i in range(view.n_nodes):
            intV_shapeI = view.intV_shape_I[i]

            # =================================== Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)
                production += \
                    xs.nu_sigma_f[g] * phi[ig] * intV_shapeI
    return production
