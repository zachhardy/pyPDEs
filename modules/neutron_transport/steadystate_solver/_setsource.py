import numpy as np
from numpy import ndarray

from pyPDEs.spatial_discretization import *
from pyPDEs.material import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def set_source(self: 'SteadyStateSolver', q: ndarray) -> None:
    """
    Compute the source moments on the right-hand side.
    """
    fv: FiniteVolume = self.discretization

    # Loop over cells
    for cell in self.mesh.cells:
        # Get cross section and source IDs
        xs_id = self.matid_to_xs_map[cell.material_id]
        src_id = self.matid_to_src_map[cell.material_id]

        # Get cross section and source objects
        xs: CrossSections = self.material_xs[xs_id]
        src = None
        if src_id >= 0:
            src: IsotropicMultiGroupSource = self.material_src[src_id]

        # Loop over moments
        for m in range(self.n_moments):
            ell = self.harmonic_index_map[m].ell
            dof = fv.map_dof(cell, 0, self.phi_uk_man, m, 0)

            # Loop over groups
            for g in range(self.n_groups):

                # Apply material source
                if src is not None and ell == 0:
                    q[dof + g] += src.values[g]

                # Apply scattering source
                if ell < xs.scattering_order + 1:
                    for gp in range(self.n_groups):
                        sigma_tr = xs.transfer_matrix[ell][gp][g]
                        q[dof + g] = sigma_tr * self.phi_prev[dof + gp]

                # Apply fission source
                if xs.is_fissile and ell == 0:
                    for gp in range(self.n_groups):

                        # Without delayed neutron precursors
                        if not self.use_precursors:
                            q[dof + g] = xs.chi[g] * \
                                         xs.nu_sigma_f[gp] * \
                                         self.phi_prev[dof + gp]

                        # With delayed neutron precursors
                        else:
                            # Prompt fission
                            q[dof + g] = xs.chi_prompt[g] * \
                                         xs.nu_prompt_sigma_f[gp] * \
                                         self.phi_prev[dof + gp]

                            # Delayed fission
                            for j in range(xs.n_precursors):
                                q[dof + g] += xs.chi_delayed[g][j] * \
                                              xs.precursor_yield[j] * \
                                              xs.nu_delayed_sigma_f[gp] * \
                                              self.phi_prev[dof + gp]