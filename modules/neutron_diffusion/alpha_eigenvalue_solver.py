import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.linalg import eig
from scipy.sparse import csr_matrix

from typing import Union

from pyPDEs.math.discretization import FiniteVolume
from pyPDEs.material import Material

from modules.neutron_diffusion import SteadyStateSolver


class AlphaEigenvalueSolver(SteadyStateSolver):
    """
    Implementation of a multi-group neutron diffusion alpha-eigenvalue solver.
    """

    def __init__(
            self,
            discretization: FiniteVolume,
            materials: list[Material],
            boundary_info: list[tuple[str, int]],
            boundary_values: list[dict] = None,
            n_modes: int = -1,
            fit_data: Union[dict, np.ndarray] = None
    ) -> None:
        super().__init__(discretization, materials,
                         boundary_info, boundary_values)

        self.n_modes: int = n_modes
        self.alphas: np.ndarray = None
        self.modes: np.ndarray = None
        self.adjoint_modes: np.ndarray = None
        self.amplitudes: np.ndarray = None

        self._fit_data: Union[dict, np.ndarray] = fit_data

    def execute(self) -> None:
        """
        Execute the multi-group neutron diffusion alpha-eigenvalue solver.
        """
        msg = "Executing the multi-group diffusion alpha-eigenvalue solver"
        msg = "\n".join(["", "*" * len(msg), msg, "*" * len(msg), ""])
        print(msg)

        self._assemble_eigensystem()
        evals, evecs_l, evecs_r = eig(self._A[0].todense(), left=True)

        self.n_modes = len(evals) if self.n_modes == -1 else self.n_modes
        if self.n_modes > len(evals):
            raise ValueError("Invalid number of modes.")

        idx = np.argsort(evals.real)[::-1][:self.n_modes]
        self.alphas = evals[idx]
        self.modes = evecs_r[:, idx]
        self.adjoint_modes = evecs_l[:, idx]

        if self._fit_data is not None:
            self._compute_amplitudes()

    def _assemble_eigensystem(self) -> None:
        """
        Assemble the operator for the alpha-eigensystem.

        Returns
        -------
        csr_matrix
        """
        self._assemble_matrix(with_scattering=True, with_fission=True)
        A = self._A[0].todense()

        # loop over cells
        for cell in self.mesh.cells:
            uk_map = self.n_groups * cell.id

            # get inverse velocity
            xs_id = self.matid_to_xs_map[cell.material_id]
            xs = self.material_xs[xs_id]
            vel = 1.0 / xs.inv_velocity

            # loop over groups
            for g in range(self.n_groups):
                A[uk_map + g] *= -vel[g] / cell.volume
        self._A = [csr_matrix(A)]

    def _compute_amplitudes(self) -> None:
        """
        Fit the alpha-eigenfunctions to the provided data.
        """

        # evaluate callable ICs
        phi = np.zeros(self.phi.shape)
        if isinstance(self._fit_data, dict):

            # loop over cells
            for cell in self.mesh.cells:

                # loop over nodes
                nodes = self.discretization.nodes(cell)
                for i in range(len(nodes)):

                    # loop over initial conditions
                    for g, f in self._fit_data.items():
                        if not callable(f):
                            raise TypeError("Initial condition must be callable.")

                        dof = (cell.id * len(nodes) + i) * self.n_groups + g
                        phi[dof] = f(nodes[i])

        # set vector ICs
        elif isinstance(self._fit_data, np.ndarray):
            if len(self._fit_data) != len(phi):
                raise AssertionError("Invalid initial condition vector.")
            phi[:] = self._fit_data

        else:
            raise TypeError("Unrecognized type for fit data.")

        # compute amplitudes
        evecs_l_star = self.adjoint_modes.conj().transpose()
        A = evecs_l_star @ self.modes
        b = evecs_l_star @ phi.reshape(-1, 1)
        self.amplitudes = np.linalg.solve(A, b).ravel()

        # enforce positive amplitudes
        for m in range(len(self.alphas)):
            if self.amplitudes[m] < 0.0:
                self.amplitudes[m] *= -1.0
                self.modes[:, m] *= -1.0
                self.adjoint_modes[:, m] *= -1.0

        # compute alpha representation to data
        self.phi = self.modes @ self.amplitudes
        diff = np.linalg.norm(phi - self.phi) / np.linalg.norm(phi)
        print(f"Difference in fit:  {diff:.4g}")
        print(f"Number of modes:  {len(self.alphas)}")

        # define dominant mode index
        idx = np.argmax(self.amplitudes.real)
        print(f"Dominant mode:  {idx}, {self.amplitudes[idx]:.3e}\n")

        # plot dominant mode
        plt.figure()
        plt.title(f"Dominant Mode Index: {idx}")
        z = [cell.centroid.z for cell in self.mesh.cells]
        for g in range(self.n_groups):
            b = self.amplitudes[idx]
            mode = self.modes[g::self.n_groups, idx]
            plt.plot(z, b * mode, label=f"Group {g}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # plot error in fit
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        ax[0].set_title("Alpha Expansion")
        ax[1].set_title("Alpha Expansion Error")

        ax[0].set_xlabel("Position (cm)")
        ax[1].set_ylabel("Position (cm)")

        ax[0].set_ylabel("$\phi_{g}(r)$")
        ax[1].set_ylabel("Relative $\ell_2$ Error")

        z = [cell.centroid.z for cell in self.mesh.cells]
        for g in range(self.n_groups):
            color = 'b' if g == 0 else 'r' if g == 1 else 'g'
            ax[0].plot(z, self.phi[g::self.n_groups], f'{color}-',
                       linewidth=1.5, label=f"Group {g}")
            ax[0].plot(z, phi[g::self.n_groups], 'o', ms=4.0, alpha=0.6)


            c = np.linalg.norm(phi[g::self.n_groups])
            if c > 0.0:
                err = self.phi[g::self.n_groups] - phi[g::self.n_groups]
                ax[1].semilogy(z, np.abs(err) / c, label=f"Group {g}")

        for iax in ax:
            iax.grid(True)
            iax.legend()
        plt.tight_layout(w_pad=3.0)
