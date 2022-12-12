import numpy as np
from numpy.linalg import norm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def classic_richardson(self: 'SteadyStateSolver',
                       apply_material_source: bool = True,
                       apply_scattering_source: bool = True,
                       apply_fission_source: bool = True,
                       verbose: bool = False) -> None:
    """
    Classic Richardson iterations.

    Parameters
    ----------
    apply_material_source : bool, default True
    apply_scattering_source : bool, default True
    apply_fission_source : bool, default True
    verbose : bool, default False
    """
    # Initial source moments
    init_q_moments = np.copy(self.q_moments)

    # Create flags tuple
    flags = (apply_material_source,
             apply_scattering_source,
             apply_fission_source)

    # Start Richardson iterations
    converged = False
    pw_change_prev = 1.0
    for k in range(self.max_source_iterations):

        # Set source and sweep
        self.q_moments[:] = init_q_moments
        self.set_source(self.q_moments, *flags)
        self.sweep()

        # Compute change in scalar flux
        pw_change = self.compute_piecewise_change()
        self.phi_prev[:] = self.phi

        # Compute spectral radius
        rho = np.sqrt(pw_change / pw_change_prev)
        pw_change_prev = pw_change

        # Check convergence
        if k == 0: rho = 0.0
        if pw_change < self.source_iteration_tolerance * (1.0 - rho):
            converged = True

        # Print iteration summary
        if verbose:
            msg = f'===== Iteration {k}  ' \
                  f'Piecewise change {pw_change:.3e}  ' \
                  f'Spectral radius estimate {rho:.3f}'
            if converged:
                msg += '  CONVERGED\n'
            print(msg)

        # End if converged
        if converged:
            break

    if not converged:
        print(f'!!!!! WARNING: Source iterations not converged.')
