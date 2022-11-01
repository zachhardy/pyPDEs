from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver

from pyPDEs.material import CrossSections
from pyPDEs.material import LightWeightCrossSections
from pyPDEs.material import IsotropicMultiGroupSource

from ..boundaries import Boundary
from ..boundaries import DirichletBoundary
from ..boundaries import NeumannBoundary
from ..boundaries import RobinBoundary


def _initialize_materials(self: 'SteadyStateSolver') -> None:
    """
    Initialize the materials for the problem.

    This routine ensures that the mesh has the same number of
    material IDs as the number of materials stored by the solver,
    pulls out the cross-section and isotropic source properties,
    and performs compatibility checks along the way.

    Parameters
    ----------
    self : SteadyStateSolver
    """
    print("Initializing materials.")

    # ==================================================
    # Check the material IDS
    # ==================================================

    unique_matids = set()
    invalid_cells = []
    for cell in self.mesh.cells:
        unique_matids.add(cell.material_id)
        if cell.material_id == -1:
            invalid_cells.append(cell.id)
    assert (len(invalid_cells) == 0 or
            len(invalid_cells) == len(self.mesh.cells))

    # If all cells material IDs are invalid, set to zero
    if len(invalid_cells) == len(self.mesh.cells):
        unique_matids.clear()
        unique_matids.add(0)
        for cell in self.mesh.cells:
            cell.material_id = 0
    assert len(unique_matids) == len(self.materials)

    # ==================================================
    # Clear current material information
    # ==================================================

    self.material_xs.clear()
    self.material_src.clear()

    n_materials = len(self.materials)
    self.matid_to_xs_map = [-1 for _ in range(n_materials)]
    self.matid_to_src_map = [-1 for _ in range(n_materials)]

    # ==================================================
    # Go through the materials
    # ==================================================

    # For each material, go through its properties to search for
    # cross-sections and multi-group sources. Perform compatibility
    # checks along the way.
    for matid in unique_matids:
        material = self.materials[matid]

        found_xs = False
        for matprop in material.properties:

            # Get cross-section property
            if matprop.type == "XS":
                xs: CrossSections = matprop
                if self.n_groups == 0:
                    self.n_groups = xs.n_groups
                assert xs.n_groups == self.n_groups

                self.material_xs.append(matprop)
                self.matid_to_xs_map[matid] = len(self.material_xs) - 1
                found_xs = True

            # Get multi-group source property
            elif matprop.type == "ISOTROPIC_SOURCE":
                src: IsotropicMultiGroupSource = matprop
                if self.n_groups != 0:
                    assert len(src.values) == self.n_groups

                self.material_src.append(src)
                self.matid_to_src_map[matid] = len(self.material_src) - 1
        assert found_xs

    # ==================================================
    # Define cell-wise cross-sections
    # ==================================================

    for cell in self.mesh.cells:
        xs_id = self.matid_to_xs_map[cell.material_id]
        xs: CrossSections = self.material_xs[xs_id]
        self.cellwise_xs.append(LightWeightCrossSections(xs))

    # ==================================================
    # Define precursor quantities
    # ==================================================

    if self.use_precursors:
        self.max_precursors = 0
        unique_lambdas = set()
        for xs in self.material_xs:
            for j in range(xs.n_precursors):
                unique_lambdas.add(xs.precursor_lambda[j])

            if xs.n_precursors > self.max_precursors:
                self.max_precursors = xs.n_precursors
        self.n_precursors = len(unique_lambdas)

        if self.n_precursors == 0:
            self.use_precursors = False

    print(f"Materials initialized: {len(self.materials)}")


def _initialize_boundaries(self):
    """
    Initialize the boundary conditions.

    This routine takes the boundary condition inputs and creates
    the associated multi-group boundary conditions per boundary.

    Parameters
    ----------
    self : SteadyStateSolver
    """
    print("Initializing boundary conditions.")

    # ==================================================
    # Check the number of boundaries
    # ==================================================

    if self.mesh.dimension == 1 and len(self.boundary_info) != 2:
        raise AssertionError(
            "One-dimensional problems must have two boundary conditions."
        )
    elif self.mesh.dimension == 2 and len(self.boundary_info) != 4:
        raise AssertionError(
            "Two-dimensional problems must have four boundary conditions."
        )

    # ==================================================
    # Check boundary values
    # ==================================================

    if self.boundary_values is not None:
        for bndry_vals in self.boundary_values:
            for group in bndry_vals.keys():
                if group < 0 or group >= self.n_groups:
                    raise ValueError(
                        "Invalid group encountered in boundary condition."
                    )
    # ==================================================
    # Create the multi-group boundaries
    # ==================================================

    for boundary in self.boundary_info:
        btype, bmap = boundary

        # Get the boundary condition type
        valid_btypes = ["DIRICHLET", "ZERO_FLUX",
                        "REFLECTIVE", "VACUUM", "MARSHAK"]
        if btype not in valid_btypes:
            raise ValueError("Invalid boundary type encountered.")

        # Construct a boundary condition for each group
        if btype == "ZERO_FLUX":
            bcs = [DirichletBoundary() for _ in range(self.n_groups)]
        elif btype == "REFLECTIVE":
            bcs = [NeumannBoundary() for _ in range(self.n_groups)]
        elif btype == "VACUUM":
            bcs = [RobinBoundary() for _ in range(self.n_groups)]
        elif btype == "DIRICHLET":
            bcs: list[Boundary] = []
            bndry_vals = self.boundary_values[bmap]
            for g in range(self.n_groups):
                bval = 0.0 if g not in bndry_vals else bndry_vals[g]
                bcs.append(DirichletBoundary(bval))
        elif btype == "MARSHAK":
            bcs: list[Boundary] = []
            bndry_vals = self.boundary_values[bmap]
            for g in range(self.n_groups):
                bval = 0.0 if g not in bndry_vals else bndry_vals[g]
                bcs.append(RobinBoundary(f=bval))
        else:
            raise AssertionError(
                "Invalid boundary condition type specified. Available "
                "boundary conditions are [DIRICHLET, ZERO_FLUX, REFLECTIVE,"
                "VACUUM, MARSHAK]."
            )

        self.boundaries.append(bcs)
    print(f"Boundaries initialized: {len(self.boundaries)}")
