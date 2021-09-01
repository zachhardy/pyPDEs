import numpy as np
from numpy import ndarray

from ..material_property import MaterialProperty


class CrossSections(MaterialProperty):
    """Neutronics cross sections.

    Attributes
    ----------
    type : str
        The material property type. This is set in the
        constructor of derived classes.
    n_groups : int
        The number of energy group.
    n_precursors : int
        The number of delayed neutron precursors.
    is_fissile : bool
        True if fission cross sections are non-zero.
    has_precursors : bool
        True if n_precursors is greater than zero.
    sigma_t : ndarary (n_groups,)
        The group-wise total cross sections.
    sigma_a : ndarray (n_groups,)
        The group-wise absorption cross sections.

        This is inferred from the total cross section
        and transfer matrix if not provided.
    sigma_r : ndarray (n_groups,)
        The group-wise removal cross sections.

        This is computed by subtracting the out-of-group
        scattering cross sections from the total cross
        sections.
    sigma_f : ndarray (n_groups,)
        The group-wise fission cross sections.
    sigma_s : ndarray (n_groups,)
        The sum of group-wise within-group + out-of-group
        scattering cross sections.
    sigma_tr : ndarray (n_groups, n_groups)
        The group-to-group scattering transfer matrix.

        The rows of the matrix correspond to the initial group
        and the columns to the destination group.
    diffusion_coeff : ndarray (n_groups,)
        The group-wise diffusion coefficients.

        This is computed via the inverse of three times the
        total cross sections.
    nu : ndarray (n_groups,)
        The group-wise total neutrons per fission.
    nu_prompt : ndarray (n_groups,)
        The group-wise prompt neutrons per fission.
    nu_delayed : ndarray (n_groups,)
        The group-wise delayed neutrons per fission
    chi : ndarray (n_groups,)
        The group-wise total fission spectrum.
    chi_prompt : ndarray (n_groups,)
        The group-wise prompt fission spectrum.
    chi_delayed : ndarray (n_groups, n_precursors)
        The group-wise, precursor-wise delayed fission specturm.
    inv_velocity : ndarray (n_groups,)
        The group-wise inverse neutron speed.
    precursor_lambda : ndarray (n_precursors,)
        The precursor-wise decay constants.
    precursor_yield : ndarray (n_precursors,)
        The precursor-wise delayed neutron fraction.
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
        self.sigma_tr: ndarray = None
        self.diffusion_coeff: ndarray = None

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
        self.diffusion_coeff = 1.0 / (3.0 * self.sigma_t)
        self.sigma_s = np.sum(self.sigma_tr, axis=1)
        self.sigma_a = self.sigma_t - self.sigma_s
        self.sigma_r = self.sigma_t - np.diag(self.sigma_tr)

        nu_p, nu_d = self.nu_prompt, self.nu_delayed
        if sum(nu_p) > 0.0 and sum(nu_d) > 0.0:
            self.nu = nu_p + nu_d

        self.beta = nu_d / self.nu * self.precursor_yield
        self.beta_total = sum(self.beta)

    def reset_groupwise_xs(self) -> None:
        """Reset the general and prompt cross sections.
        """
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
        self.sigma_tr = np.zeros((self.n_groups,) * 2)

    def reset_delayed_xs(self) -> None:
        """Reset the cross section terms involving delayed neutrons.
        """
        self.chi_delayed = np.zeros((self.n_groups, self.n_precursors))
        self.precursor_lambda = np.zeros(self.n_precursors)
        self.precursor_yield = np.zeros(self.n_precursors)
        self.beta = np.zeros(self.n_precursors)
        self.beta_total = 0.0
