import numpy as np

from pyPDEs.material import CrossSections, IsotropicMultiGroupSource
from pyPDEs.utilities.boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def _check_mesh(self: 'SteadyStateSolver') -> None:
    """
    Ensure a valid mesh is attached.
    """
    # Is there a mesh?
    if self.mesh is None:
        raise AssertionError(
            'There must be a mesh attached to the solver.')

    # Is it a valid mesh?
    if self.mesh.dim > 2:
        raise NotImplementedError(
            'Only 1D and 2D meshes are implemented.')


def _check_discretization(self: 'SteadyStateSolver') -> None:
    """
    Ensure a valid discretization is attached.
    """
    # Is there a discretization?
    if self.discretization is None:
        raise AssertionError(
            'There must be a discretization attached to the solver.')

    # Is it a supported discretization type?
    if self.discretization.type not in ['fv', 'pwc']:
        raise NotImplementedError(
            'Only finite volume and piecewise continuous spatial '
            'discretizations are implemented.')


def _check_boundaries(self: 'SteadyStateSolver') -> None:
    """
    Ensure valid boundary conditions are attached.
    """
    # Are there boundaries?
    if len(self.boundaries) == 0:
        raise AssertionError(
            'Boundary conditions must be attached to the solver.')

    # Do the boundaries agree with the mesh?
    if self.mesh.type == 'line' and \
            len(self.boundaries) != 2:
        raise NotImplementedError(
            'There must be 2 boundary conditions for 1D problems.')

    elif self.mesh.type == 'ortho_quad' and \
            len(self.boundaries) != 4:
        raise NotImplementedError(
            'There must be 4 boundary conditions for 2D problems.')

    # Do the boundary components agree with n_groups?
    for b, bc in enumerate(self.boundaries):
        error = False

        if hasattr(bc, 'values'):
            if len(bc.values) != self.n_groups:
                error = True
        if hasattr(bc, 'a'):
            vals = [bc.a, bc.b, bc.f]
            if any([len(v) != self.n_groups for v in vals]):
                error = True

        if error:
            raise AssertionError(
                f'Invalid number of components found in boundary {b}.')


def _check_materials(self: 'SteadyStateSolver') -> None:
    """
    Ensure valid material properties are attached.
    """
    # Are there materials?
    if len(self.materials) == 0:
        raise AssertionError(
            'Material must be attached to the solver.')

    # Get number of materials and material IDs
    n_materials = len(self.materials)
    material_ids = \
        np.unique([cell.material_id for cell in self.mesh.cells])

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
        material = self.materials[mat_id]

        # Loop over properties
        found_xs = False
        for prop in material.properties:

            # Get cross sections
            if prop.type == 'xs':
                self.material_xs.append(prop)
                self.matid_to_xs_map[mat_id] = len(self.material_xs) - 1
                found_xs = True

            # Get sources
            if prop.type == 'isotropic':
                self.material_src.append(prop)
                self.matid_to_src_map[mat_id] = len(self.material_src) - 1

        # Check that cross sections were found
        if not found_xs:
            raise ValueError('Each material must have cross sections.')

        # Check sources
        xs_id = self.matid_to_xs_map[mat_id]
        src_id = self.matid_to_src_map[mat_id]

        xs = self.material_xs[xs_id]
        if src_id >= 0:
            src = self.material_src[src_id]
            if xs.n_groups != len(src.values):
                raise ValueError(
                    'Number of isotropic multi-group source values '
                    'does not agree with the number of groups in the '
                    'cross section set.')

    # Check for material compatibility
    n_groups = self.material_xs[0].n_groups
    for xs in self.material_xs:
        if xs.n_groups != n_groups:
            raise ValueError(
                'All cross sections must have the same number '
                'of groups.')

    # Set the number of groups
    self.n_groups = self.material_xs[0].n_groups

    # Set the precursor counts
    if self.use_precursors:
        self.n_precursors = 0
        self.max_precursors = 0
        for xs in self.material_xs:
            # Increment the precursor count
            self.n_precursors += xs.n_precursors

            # Set the max precursor per material
            if xs.n_precursors > self.max_precursors:
                self.max_precursors = xs.n_precursors

    if self.n_precursors == 0:
        self.use_precursors = False
