from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def check_inputs(self: 'SteadyStateSolver') -> None:
    """
    Check the inputs.
    """
    # Check the mesh
    if not self.mesh:
        raise AssertionError('No mesh is attached to the solver.')
    if self.mesh.dim > 2:
        raise NotImplementedError('Only 1D and 2D problems are implementd.')

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
