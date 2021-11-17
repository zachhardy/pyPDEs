import numpy as np
from numpy import ndarray

from pyPDEs.mesh import Mesh
from pyPDEs.material import *
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import Boundary
from pyPDEs.utilities.quadratures import ProductQuadrature

from ..boundaries import *
from ._harmonics import HarmonicIndex


class SteadyStateSolver:
    """
    Steady state multigroup neutron transport solver
    """

    from ._harmonics import create_harmonic_indices

    from ._angular_operators import (discrete_to_moment_matrix,
                                     moment_to_discrete_matrix)

    def __init__(self) -> None:
        self.n_groups: int = 0
        self.n_precursors: int = 0
        self.n_moments: int = 0
        self.n_angles: int = 0
        self.max_precursors: int = 0

        self.scattering_order: int = 0
        self.use_precursors: bool = False

        # Domain objects
        self.mesh: Mesh = None
        self.discretization: SpatialDiscretization = None
        self.boundaries: List[Boundary] = []

        # Materials objects
        self.materials: List[Material] = []

        self.material_xs: List[CrossSections] = []
        self.material_src: List[IsotropicMultiGroupSource] = []
        self.matid_to_xs_map: List[int] = []
        self.matid_to_src_map: List[int] = []
        self.cellwise_xs: List[LightWeightCrossSections] = []

        # Quadrature information
        self.quadrature: ProductQuadrature = None
        self.harmonic_index_map: List[HarmonicIndex] = []

        # Flux moment information
        self.phi: ndarray = None
        self.phi_prev: ndarray = None
        self.phi_uk_man: UnknownManager = None

        # Angular flux information
        self.psi: ndarray = None
        self.psi_uk_man: UnknownManager = None

        # Precursor information
        self.precursors: ndarray = None

        # Angular operators
        self.M: ndarray = None
        self.D: ndarray = None

    def initialize(self) -> None:
        """
        Initialize the solver.
        """
        self._check_inputs()

        # Initialize the materials
        self.initialize_materials()

        # Define number of moments and angles
        self.n_angles = self.compute_n_angles()
        self.n_moments = self.compute_n_moments()

        # Initialize the unknown information
        self.initialize_unknowns()

        # Initialize angular operators
        self.D = self.discrete_to_moment_matrix()
        self.M = self.moment_to_discrete_matrix()

    def _check_inputs(self) -> None:
        """
        Check the inputs.
        """
        # Check the mesh
        if not self.mesh:
            raise AssertionError('No mesh is attached to the solver.')
        if self.mesh > 1:
            raise NotImplementedError('Only 1D problems are implementd.')

        # Check the discretization
        if not self.discretization:
            raise AssertionError(
                'No discretization is attached to the solver.')
        if self.discretization.type != 'fv':
            raise NotImplementedError(
                'Only finite volume spatial discretizations are implemented.')

        # Check the boundaries
        if not self.boundaries:
            raise AssertionError(
                'No boundary conditions are attacehd to the solver.')
        if self.mesh.dim == 1 and len(self.boundaries) != 2:
            raise AssertionError(
                '1D problems must have 2 boundary conditions.')
        if self.mesh.dim == 2 and len(self.boundaries) != 4:
            raise AssertionError(
                '2D problems must have 4 boundary conditions.')

        # Check the materials
        if not self.materials:
            raise AssertionError(
                'No materials are attached to the solver.')

        # Check the quadrature
        if not self.quadrature:
            raise AssertionError(
                'No angular quadrature attached to the solver.')
        if len(self.quadrature.abscissae) % 2 != 0:
            raise AssertionError(
                'There must be an even number of angles.')

    def initialize_materials(self) -> None:
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

    def initialize_unknowns(self) -> None:
        """
        Initialize the unknown storage for the problem.
        """
        # Flux moments
        self.phi_uk_man = UnknownManager()
        for m in range(self.n_moments):
            self.phi_uk_man.add_unknown(self.n_groups)
        self.phi = np.zeros(sd.n_dofs(self.phi_uk_man))
        self.phi_prev = np.zeros(sd.n_dofs(self.phi_uk_man))

        # Angular flux
        self.psi_uk_man = UnknownManager()
        for n in range(self.n_angles):
            self.psi_uk_man.add_unknown(self.n_groups)
        self.psi = np.zeros(sd.n_dofs(self.psi_uk_man))

        # Precursors
        if self.use_precursors:
            n_dofs = self.n_precursors * self.mesh.n_cells
            self.precursors = np.zeros(n_dofs)

    def initialize_bondaries(self) -> None:
        """
        Initialize the boundary conditions for the simulation.
        """
        for b, bc in enumerate(self.boundaries):
            if isinstance(bc, VacuumBoundary):
                bc.values = [0.0] * self.n_groups
            elif isinstance(bc, IncidentHomogeneousBoundary):
                if len(bc.values) != self.n_groups:
                    raise ValueError(
                        f'Incident homogeneous boundary conditions must '
                        f'have as many values as the number of groups.')

    def compute_n_angles(self) -> int:
        """
        Compute the number of angles.

        Returns
        -------
        int
        """
        return len(self.quadrature.abscissae)

    def compute_n_moments(self) -> int:
        """
        Compute the number of moments.

        Returns
        -------
        int
        """
        self.create_harmonic_indices()
        return len(self.harmonic_index_map)
