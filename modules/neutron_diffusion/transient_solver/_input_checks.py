from numpy import ndarray
from sympy import MutableDenseMatrix
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


def _check_time_step(self: 'TransientSolver') -> None:
    self.time = self.t_start
    if self.output_frequency is None:
        self.output_frequency = self.dt
    if self.dt > self.output_frequency:
        self.dt = self.output_frequency


def _check_initial_conditions(self: 'TransientSolver') -> None:
    # Check number of ics
    if len(self.initial_conditions) != self.n_groups:
        raise AssertionError(
            'Invalid number of initial conditions. There must be '
            'as many initial conditions as groups.')

    # Convert to lambdas, if sympy functions
    if isinstance(self.initial_conditions, MutableDenseMatrix):
        from sympy import lambdify
        symbols = list(self.initial_conditions.free_symbols)

        ics = []
        for ic in self.initial_conditions:
            ics += lambdify(symbols, ic)
        self.initial_conditions = ics

    # Check length for vector ics
    if isinstance(self.initial_conditions, list):
        n_phi_dofs = self.discretization.n_dofs(self.phi_uk_man)
        for ic in self.initial_conditions:
            array_like = (ndarray, List[float])
            if not callable(ic) and isinstance(ic, array_like):
                if len(ic) != n_phi_dofs:
                    raise AssertionError(
                        'Vector initial conditions must agree with '
                        'the number of DoFs associated with the '
                        'attached discretization and phi unknown '
                        'manager.')