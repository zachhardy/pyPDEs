import numpy as np
from numpy import ndarray

from pyPDEs.spatial_discretization import *
from pyPDEs.material import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver

XS = CrossSections
IsoMGSrc = IsotropicMultiGroupSource


def set_source(self: 'SteadyStateSolver', q: ndarray,
               apply_material_source: bool = True,
               apply_scattering_source: bool = True,
               apply_fission_source: bool = True) -> None:
    """
    Compute the source moments on the right-hand side.

    Parameters
    ----------
    q : ndarray
        A vector to add source moments into.
    apply_material_source : bool, default True
    apply_scattering_source : bool, default True
    apply_fission_source : bool, default True
    """
    fv: FiniteVolume = self.discretization

    # Loop over cells
    for cell in self.mesh.cells:
        i = cell.id

        # Get cross section and source IDs
        xs_id = self.matid_to_xs_map[cell.material_id]
        src_id = self.matid_to_src_map[cell.material_id]

        # Get cross section and source objects
        xs: XS = self.material_xs[xs_id]
        src = None
        if src_id >= 0:
            src: IsoMGSrc = self.material_src[src_id]

        # Loop over moments
        for m in range(self.n_moments):
            ell = self.harmonic_index_map[m].ell

            # Loop over groups
            for g in range(self.n_groups):

                # Apply material source
                if apply_material_source:
                    if src is not None and ell == 0:
                        q[m][g][i] += src.values[g]

                # Apply scattering source
                if apply_scattering_source:
                    if ell < xs.scattering_order + 1:
                        for gp in range(self.n_groups):
                            sigma_tr = xs.transfer_matrix[ell][gp][g]
                            q[m][g][i] += sigma_tr * self.phi_prev[m][gp][i]

                # Apply fission source
                if apply_fission_source:
                    if xs.is_fissile and ell == 0:
                        for gp in range(self.n_groups):

                            # Without delayed neutron precursors
                            if not self.use_precursors:
                                q[m][g][i] += xs.chi[g] * \
                                              xs.nu_sigma_f[gp] * \
                                              self.phi_prev[m][gp][i]

                            # With delayed neutron precursors
                            else:
                                # Prompt fission
                                q[m][g][i] += xs.chi_prompt[g] * \
                                              xs.nu_prompt_sigma_f[gp] * \
                                              self.phi_prev[m][gp][i]

                                # Delayed fission
                                for j in range(xs.n_precursors):
                                    q[m][g][i] += \
                                        xs.chi_delayed[g][j] * \
                                        xs.precursor_yield[j] * \
                                        xs.nu_delayed_sigma_f[gp] * \
                                        self.phi_prev[m][gp][i]
