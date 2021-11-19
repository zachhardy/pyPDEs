import numpy as np
from numpy import ndarray

from pyPDEs.material import *
from ..boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def initialize_materials(self: 'SteadyStateSolver') -> None:
    """
    Initialize the materials.
    """
    # Get number of materials and material IDs
    n_materials = len(self.materials)
    material_ids = np.unique(
        [cell.material_id for cell in self.mesh.cells])

    # Clear material xs and sources
    self.material_xs.clear()
    self.material_src.clear()
    self.matid_to_xs_map = [-1 for _ in range(n_materials)]
    self.matid_to_src_map = [-1 for _ in range(n_materials)]

    # Loop over material IDs
    for mat_id in material_ids:
        if mat_id < 0 or mat_id >= n_materials:
            raise ValueError('Invalid material ID encountered.')

        # Get the material for this material ID
        material: Material = self.materials[mat_id]

        # Loop over properties
        found_xs = False
        for prop in material.properties:

            # Get cross sections
            if prop.type == 'xs':
                xs: CrossSections = prop
                self.material_xs.append(xs)
                self.matid_to_xs_map[mat_id] = len(self.material_xs) - 1
                found_xs = True

            # Get sources
            if prop.type == 'isotropic_source':
                src: IsotropicMultiGroupSource = prop
                self.material_src.append(src)
                self.matid_to_src_map[mat_id] = len(self.material_src) - 1

        # Check that cross sections were found
        if not found_xs:
            raise ValueError(
                f'No cross sections found for material {material.name} '
                f'with material ID {mat_id}.')

        # Check scattering order
        xs_id = self.matid_to_xs_map[mat_id]
        if self.material_xs[xs_id] > self.scattering_order:
            import warnings
            warnings.warn(f'Material {material.name} with material ID '
                          f'{mat_id} has a scattering order greater than '
                          f'the specified simulation scattering order. The '
                          f'higher order scattering moments will be ignored.'
                          , RuntimeWarning)

        # Check the source
        src_id = self.matid_to_src_map[mat_id]
        if src_id >= 0:
            src = self.material_src[src_id]
            if self.material_xs[xs_id] != len(src.values):
                raise ValueError(
                    f'Isotropic multigroup source on material '
                    f'{material.name} with material ID {mat_id} must have '
                    f'the same number of entries as the number of groups '
                    f'in this materials cross section set.')

    # Check for group compatibility
    n_groups_ref = self.material_xs[0].n_groups
    for xs in self.material_xs:
        if xs.n_groups != n_groups_ref:
            raise ValueError(
                f'All cross sections must have the same group structure.')

    # Define the number of groups
    self.n_groups = n_groups_ref

    # Set the precursor information
    if self.use_precursors:
        self.n_precursors = self.max_precursors = 0
        for xs in self.material_xs:
            # Increment count
            self.n_precursors += xs.n_precursors

            # Set the max precursors per material
            if xs.n_precursors > self.max_precursors:
                self.max_precursors = xs.n_precursors
    if self.n_precursors == 0:
        self.use_precursors = False
