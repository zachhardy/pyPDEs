import numpy as np

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pyPDEs.math.discretization import FiniteVolume
from pyPDEs.material import Material

from modules.neutron_diffusion import SteadyStateSolver


class KEigenvalueSolver(SteadyStateSolver):
    """
    Implementation of a multi-group neutron diffusion k-eigenvalue solver.
    """

    def __init__(
            self,
            discretization: FiniteVolume,
            materials: list[Material],
            boundary_info: list[tuple[str, int]],
            boundary_values: list[dict] = None
    ) -> None:
        super().__init__(discretization, materials,
                         boundary_info, boundary_values)

        self.k_eff: float = 1.0

        self.tolerance: float = 1.0e-8
        self.max_iterations: int = 500

    def execute(self) -> None:
        """Execute the k-eigenvalue multi-group diffusion solver."""

        msg = "Executing the multi-group diffusion k-eigenvalue solver"
        msg = "\n".join(["", "*" * len(msg), msg, "*" * len(msg), ""])
        print(msg)

        # ------------------------------ initialize the system with unit flux
        self.phi[:] = 1.0 / self.phi.size
        self.phi_ell[:] = self.phi

        # ------------------------------ book-keeping
        production = self._compute_fission_production()
        production_ell = production
        k_ell = k_change = phi_change = 1.0

        # ------------------------------ assemble the matrix
        self._assemble_matrix(with_scattering=True,
                              with_fission=False)

        # ------------------------------ start fission source iterations
        nit, converged = 0, False
        for nit in range(self.max_iterations):

            # ------------------------------ compute the fission source
            self._b[:] = 0.0
            self._assemble_rhs(with_fission=True)
            self._b /= self.k_eff

            # ------------------------------ solve the system
            self.phi = spsolve(self._A[0], self._b)
            production = self._compute_fission_production()
            self.k_eff *= production / production_ell

            # ------------------------------ check convergence
            k_change = abs(self.k_eff - k_ell) / self.k_eff
            phi_change = norm(self.phi - self.phi_ell, 1)
            converged = (k_change < self.tolerance and
                         phi_change < self.tolerance)

            # ------------------------------ finalize the iteration
            k_ell = self.k_eff
            production_ell = production
            self.phi_ell = np.copy(self.phi)

            # ------------------------------ print summary
            msg = f"k-iteration  {nit:<4}"
            msg += f"k_eff  {self.k_eff:<10.6g}"
            msg += f"k change  {k_change:<14.6e}"
            msg += f"phi change  {phi_change:<14.6e}"
            if converged:
                msg += "CONVERGED"
            print(msg)

            if converged:
                break

        # ------------------------------ compute precursors
        if self.use_precursors:
            self._compute_precursors()
            self.precursors /= self.k_eff

        # ------------------------------ print simulation summary
        print("\n****** k-Eigenvalue Solver Converged! *****"
              if converged else
              "\n!!*!! WARNING: k-Eigenvalue Solver NOT Converged !!*!!")
        print(f"Final k-Eigenvalue: {self.k_eff:.6g}")
        print(f"Iterations        : {nit}")
        print(f"Final k Change    : {k_change:.6e}")
        print(f"Final phi Change  : {phi_change:.6e}\n")

    def _compute_fission_production(self) -> float:
        """
        Compute the total fission neutron production in the system.
        """
        f = 0.0
        for cell in self.mesh.cells:

            xs_id = self.matid_to_xs_map[cell.material_id]
            xs = self.material_xs[xs_id]

            # only proceed for fissile materials
            if xs.is_fissile:

                uk_map = self.n_groups * cell.id
                for g in range(self.n_groups):
                    f += xs.nu_sigma_f[g] * self.phi[uk_map + g] * cell.volume
        return f
