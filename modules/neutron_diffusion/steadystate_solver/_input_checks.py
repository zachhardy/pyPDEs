import numpy as np

from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def _check_mesh(self: "SteadyStateSolver") -> None:
    """Ensure a valid mesh is attached.
    """
    # Is there a mesh?
    if self.mesh is None:
        raise AssertionError(
            "There must be a mesh attached to the solver.")

    # Is it a valid mesh?
    if self.mesh.dim > 2:
        raise NotImplementedError(
            "Only 1D and 2D meshes are implemented.")


def _check_discretization(self: "SteadyStateSolver") -> None:
    """Ensure a valid discretization is attached.
    """
    # Is there a discretization?
    if self.discretization is None:
        raise AssertionError(
            "There must be a discretization attached to the solver.")

    # Is it a supported discretization type?
    if self.discretization.type not in ["FV", "PWC"]:
        raise NotImplementedError(
            "Only finite volume and piecewise continuous spatial "
            "discretizations are implemented.")


def _check_boundaries(self: "SteadyStateSolver") -> None:
    """Ensure valid boundary conditions are attached.
    """
    # Are there boundaries?
    if len(self.boundaries) == 0:
        raise AssertionError(
            "Boundary conditions must be attached to the solver.")

    # Do the boundaries agree with the mesh?
    if self.mesh.type == "LINE" and \
            len(self.boundaries) != 2:
        raise NotImplementedError(
            "There must be 2 boundary conditions for 1D problems.")

    elif self.mesh.type == "ORTHO_QUAD" and \
            len(self.boundaries) != 4:
        raise NotImplementedError(
            "There must be 4 boundary conditions for 2D problems.")

    # Do the boundary components agree with n_groups?
    for b, bc in enumerate(self.boundaries):
        error = False

        if hasattr(bc, "values"):
            if len(bc.values) != self.n_groups:
                error = True
        if hasattr(bc, "a"):
            vals = [bc.a, bc.b, bc.f]
            if any([len(v) != self.n_groups for v in vals]):
                error = True

        if error:
            raise AssertionError(
                f"Invalid number of components found in boundary {b}.")


def _check_materials(self: "SteadyStateSolver") -> None:
    """Ensure valid material properties are attached.
    """
    # Are there materials?
    if len(self.material_xs) == 0:
        raise AssertionError(
            "Material cross sections must be attached to the solver.")

    # Check each xs set
    for xs in self.material_xs:
        # Is this a CrossSections object?
        if not isinstance(xs, CrossSections):
            raise TypeError(
                "All items in the material_xs list must be of "
                "type CrossSections.")

        # Do the group structures agree?
        if xs.n_groups != self.n_groups:
            raise AssertionError(
                "All cross section sets must have the same number "
                "of groups.")

    # Check the material sources
    for src in self.material_src:
        # Is this a MultiGroupSource object?
        if not isinstance(src, MultiGroupSource):
            raise TypeError(
                "All items in the material_src list must be of "
                "type MultiGroupSource.")

        # Do the group structures agree?
        if len(src.values) != self.n_groups:
            raise AssertionError(
                "All source must have the same number of groups "
                "as the cross section sets.")

    n_srcs_to_add = len(self.material_xs) - len(self.material_src)
    for _ in range(n_srcs_to_add):
        src = MultiGroupSource(np.zeros(self.n_groups))
        self.material_src += [src]