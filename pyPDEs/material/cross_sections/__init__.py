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

        self.inv_velocity: ndarray = None

        self.precursor_lambda: ndarray = None
        self.precursor_yield: ndarray = None

        self.beta: ndarray = None
        self.beta_total: float = 0

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

    def finalize_xs(self) -> None:
        """Compute auxiliary cross sections based upon others.
        """
        self.sigma_s = np.sum(self.transfer_matrix, axis=1)
        if sum(self.D) == 0.0:
            self.D = 1.0 / (3.0 * self.sigma_t)
        if sum(self.sigma_a) == 0.0:
            self.sigma_a = self.sigma_t - self.sigma_s
        self.sigma_r = self.sigma_t - np.diag(self.transfer_matrix)

        nu_p, nu_d = self.nu_prompt, self.nu_delayed
        if sum(nu_p) > 0.0 and sum(nu_d) > 0.0:
            self.nu = nu_p + nu_d

        self.beta = nu_d / self.nu * self.precursor_yield
        self.beta_total = sum(self.beta)

    def reset_groupwise_xs(self) -> None:
        """Reset the general and prompt cross sections.
        """
        self.D = np.zeros(self.n_groups)
        self.sigma_t = np.zeros(self.n_groups)
        self.sigma_f = np.zeros(self.n_groups)
        self.sigma_a = np.zeros(self.n_groups)
        self.sigma_s = np.zeros(self.n_groups)
        self.sigma_r = np.zeros(self.n_groups)
        self.chi = np.zeros(self.n_groups)
        self.chi_prompt = np.zeros(self.n_groups)
        self.nu = np.zeros(self.n_groups)
        self.nu_prompt = np.zeros(self.n_groups)
        self.nu_delayed = np.zeros(self.n_groups)
        self.inv_velocity = np.zeros(self.n_groups)
        self.transfer_matrix = np.zeros((self.n_groups,) * 2)

    def reset_delayed_xs(self) -> None:
        """Reset the cross section terms involving delayed neutrons.
        """
        self.chi_delayed = np.zeros((self.n_groups, self.n_precursors))
        self.precursor_lambda = np.zeros(self.n_precursors)
        self.precursor_yield = np.zeros(self.n_precursors)
        self.beta = np.zeros(self.n_precursors)
        self.beta_total = 0.0
