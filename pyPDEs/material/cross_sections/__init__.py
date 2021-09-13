import numpy as np

from numpy import ndarray
from typing import Callable

from ..material_property import MaterialProperty

XSFunc = Callable[[int, float], float]


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

        self._sigma_t: ndarray = None
        self._sigma_a: ndarray = None
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

        self.sigma_t_function: XSFunc = None
        self.sigma_a_function: XSFunc = None

    def sigma_t(self, g: int, t: float = 0.0) -> float:
        if self.sigma_t_function is not None and \
                self.sigma_a_function is not None:
            raise AssertionError(
                "Only one transient cross section function can "
                "be provided.")

        if self.sigma_t_function is None and \
                self.sigma_a_function is None:
            return self._sigma_t[g]

        elif self.sigma_t_function is not None and \
                self.sigma_a_function is None:
            return self.sigma_t_function(g, t, self._sigma_t)

        elif self.sigma_a_function is not None and \
                self.sigma_t_function is None:
            val = self.sigma_a_function(g, t, self._sigma_a)
            return val + self.sigma_s[g]


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
        if self._sigma_t is None and self._sigma_a is None:
            raise AssertionError(
                "Total or absorption cross sections must be provided.")

        # Ensure transfer matrix was provided
        if self.transfer_matrix is None:
            raise AssertionError(
                "The transfer matrix must be provided.")

        # Compute diffusion coefficient if not provided
        if self.D is None:
            self.D = (3.0 * self._sigma_t) ** (-1)

        # Enforce _sigma_t = sigma_a + sigma_s
        if self._sigma_a is None:
            self._sigma_a = self._sigma_t - self.sigma_s
        else:
            self._sigma_t = self._sigma_a + self.sigma_s

        # Compute removal cross sections
        self.sigma_r = self._sigma_t - np.diag(self.transfer_matrix)

        # Check fissile properties
        if self.is_fissile:

            # Check precursor terms
            if self.has_precursors:
                if self.precursor_lambda is None:
                    raise AssertionError(
                        "Delayed neutron precursor decay constants must be "
                        "supplied for cross sections with precursors.")

                if self.n_precursors > 1:
                    if self.precursor_yield is None:
                        raise AssertionError(
                            "Delayed neutron precursor yields must be "
                            "supplied for cross sections with precursors.")
                else:
                    self.precursor_yield = np.ones(1)

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
            # for multigroup
            if self.n_groups > 1:
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

            # Set all chi's to unit for one group problems
            else:
                self.chi = np.ones(1)
                self.chi_prompt = np.ones(1)
                if self.has_precursors:
                    self.chi_delayed = np.ones((1, self.n_precursors))

        else:
            G, J = self.n_groups, self.n_precursors
            self.sigma_f = np.zeros(G)
            self.nu = np.zeros(G)
            self.nu_prompt = np.zeros(G)
            self.nu_delayed = np.zeros(G)
            self.chi = np.zeros(G)
            self.chi_prompt = np.zeros(G)
            self.chi_delayed = np.zeros((G, J))
            self.precursor_yield = np.zeros(J)
            self.precursor_lambda = np.zeros(J)
            if self.velocity is None:
                self.velocity = np.zeros(G)
