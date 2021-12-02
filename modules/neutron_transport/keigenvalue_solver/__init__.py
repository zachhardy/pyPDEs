from .. import SteadyStateSolver


class KEigenvalueSolver(SteadyStateSolver):
    def __init__(self) -> None:
        super().__init__()
        self.k_eff: float = 1.0

        self.power_iteration_tolerance: float = 1.0e-8
        self.max_power_iterations: int = 100

    def execute(self, verbose: bool = True) -> None:
        """
        Execute the k-eigenvalue solver.
        """
        self.power_iteration(verbose)

    def power_iteration(self, verbose: bool) -> None:
        """
        Power iteration method for solving k-eigenvalue problems.
        """
        # Unit guess
        self.phi_prev[:] = 1.0

        # Book keeping
        F_prev = 1.0
        k_eff_prev = 1.0
        k_eff_change = 1.0

        # Start power iterations
        nit = 0
        converged = False
        for nit in range(self.max_power_iterations):
            # Reinit source moments
            self.q_moments[:] = 0.0

            # Compute fission source
            flags = (False, False, True)
            self.set_source(self.q_moments, *flags)
            self.q_moments /= self.k_eff

            # Converge the scattering source
            flags = (False, True, False)
            self.classic_richardson(*flags, verbose=False)

            # Compute new k-eigenvalue
            F_new = self.compute_fission_production()
            self.k_eff *= F_new / F_prev
            reactivity = (self.k_eff - 1.0) / self.k_eff * 1.0e5  # pcm

            # Check convergence
            k_eff_change = abs(self.k_eff - k_eff_prev) / self.k_eff
            k_eff_prev = self.k_eff
            F_prev = F_new

            if k_eff_change < self.power_iteration_tolerance:
                converged = True

            if verbose:
                msg = f'===== Iteration {nit}  ' \
                      f'k_eff {self.k_eff:.5f}  ' \
                      f'k_eff change {k_eff_change:.3e}  ' \
                      f'reactivity  {reactivity:.5f} pcm'
                if converged:
                    msg += '  CONVERGED\n'
                print(msg)

            # End, if converged
            if converged:
                break

        if not converged:
            print('!!!!! WARNING: Power iterations did not converge.\n')

        if verbose:
            print()
            print(f'***** FINAL RESULTS *****')
            print(f'Final k-Eigenvalue:\t{self.k_eff:.5f}')
            print(f'Final Change:\t\t{k_eff_change:.3e}')
            print()

    def compute_fission_production(self) -> float:
        """
        Compute the total fission netron production

        Returns
        -------
        float
        """
        # Loop over cells
        production = 0.0
        for cell in self.mesh.cells:
            c = cell.id
            xs_id = self.matid_to_xs_map[cell.material_id]
            xs = self.material_xs[xs_id]

            # Loop over groups
            for g in range(self.n_groups):
                production += xs.nu_sigma_f[g] * \
                              self.phi[0][g][c] * cell.volume
        return production


