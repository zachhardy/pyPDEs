import numpy as np
from numpy import ndarray

from ..material_property import MaterialProperty


class CrossSections(MaterialProperty):
    """Neutronics cross sections.
    """

    from ._read_from_dict import read_from_xs_dict
    from ._read_from_file import read_from_xs_file

    def __init__(self):
        super().__init__()
        self.type = "XS"

        self.n_groups: int = 0
        self.n_precursors: int = 0
        self.is_fissile: bool = False
        self.has_precursors: bool = False

        self.sigma_t: ndarray = None
        self.sigma_a: ndarray = None
        self.sigma_r: ndarray = None
        self.sigma_f: ndarray = None
        self.sigma_s: ndarray = None
        self.transfer_matrix: ndarray = None
        self.D: ndarray = None

        self.nu: ndarray = None
        self.nu_prompt: ndarray = None
        self.nu_delayed: ndarray = None

        self.chi: ndarray = None
        self.chi_prompt: ndarray = None
        self.chi_delayed: ndarray = None

        self.velocity: ndarray = None

        self.precursor_lambda: ndarray = None
        self.precursor_yield: ndarray = None

    @property
    def nu_sigma_f(self) -> ndarray:
        """Get total nu times the fission cross sections.

        Returns
        -------
        ndarray (n_groups,)
        """
        return self.nu * self.sigma_f

    @property
    def nu_prompt_sigma_f(self) -> ndarray:
        """Get prompt nu times the fission cross sections.

        Returns
        -------
        ndarray (n_groups,)
        """
        return self.nu_prompt * self.sigma_f

    @property
    def nu_delayed_sigma_f(self) -> ndarray:
        """Get delayed nu times the fission cross sections.

        Returns
        -------
        ndarray (n_groups,)
        """
        return self.nu_delayed * self.sigma_f

    def _validate_xs(self) -> None:
        """Validate the parsed cross sections.
        """
        # Ensure sigma_t was provided
        if self.sigma_t is None:
            raise AssertionError(
                "Total cross sections must be provided.")

        # Ensure transfer matrix was provided
        if self.transfer_matrix is None:
            raise AssertionError(
                "The transfer matrix must be provided.")

        # Compute diffusion coefficient if not provided
        if self.D is None:
            self.D = (3.0 * self.sigma_t) ** (-1)

        # Enforce sigma_t = sigma_a + sigma_s
        if self.sigma_a is None:
            self.sigma_a = self.sigma_t - self.sigma_s
        else:
            self.sigma_t = self.sigma_a + self.sigma_s

        # Compute removal cross sections
        self.sigma_r = self.sigma_t - np.diag(self.transfer_matrix)

        # Check fissile properties
        if self.is_fissile:

            # Enforce nu = nu_prompt + nu_delayed
            has_nu = self.nu is not None
            has_nu_p = self.nu_prompt is not None
            has_nu_d = self.nu_delayed is not None
            if has_nu_p and has_nu_d:
                self.nu = self.nu_prompt + self.nu_delayed
                has_nu = True

            # Ensure appropriate nu values exist
            if self.has_precursors:
                if not has_nu_p or not has_nu_d:
                    raise AssertionError(
                        "Both prompt and delayed nu must be provided "
                        "if the cross sections have precursors.")
            elif not has_nu:
                raise AssertionError(
                    "nu must be provided if the cross sections do "
                    "not have precursors.")

            # Enforce chi = (1.0 - beta) * chi_prompt +
            #               beta * sum_{j=1}^J gamma_j * chi_delayed
            has_chi = self.chi is not None
            has_chi_p = self.chi_prompt is not None
            has_chi_d = self.chi_delayed is not None
            if has_chi_p and has_chi_d:
                beta = self.nu_delayed / self.nu
                self.chi = (1.0 - beta) * self.chi_prompt
                for j in range(self.n_precursors):
                    gamma = self.precursor_yield[j]
                    self.chi += beta * gamma * self.chi_delayed[:, j]
                has_chi = True

            # Ensure appropriate chi values exist
            if self.has_precursors:
                if not has_chi_p or not has_chi_d:
                    raise AssertionError(
                        "Both prompt and delayed chi must be provided "
                        "if the cross sections have precursors.")
            elif not has_chi:
                raise AssertionError(
                    "chi must be provided if the cross sections do "
                    "not have precursors.")
