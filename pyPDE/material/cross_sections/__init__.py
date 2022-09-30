import numpy as np

from ..material_property import MaterialProperty


class CrossSections(MaterialProperty):
    """
    A class for neutronics cross-sections.
    """

    def __init__(self) -> None:
        super().__init__("XS")

        self.n_groups: int = 0
        self.n_precursors: int = 0
        self.n_moments: int = 0  # The number of scattering moments
        self.is_fissile: bool = False

        self.density: float = 1.0  # Atom density in atoms/b-cm

        self.E_bounds: np.ndarray = None  # Energy bin boundaries

        self.sigma_t: np.ndarray = None  # Total cross-section
        self.sigma_a: np.ndarray = None # Absorption cross-section
        self.sigma_s: np.ndarray = None  # Scattering cross-section
        self.sigma_f: np.ndarray = None  # Fission cross-section
        self.sigma_r: np.ndarray = None  # Removal cross-section

        # Moment-wise group-to-group transfer matrices
        self.transfer_matrices: np.ndarray = None

        self.chi: np.ndarray = None  # Total neutrons per fission
        self.chi_prompt: np.ndarray = None  # Prompt neutrons per fission
        self.chi_delayed: np.ndarray = None  # Delayed neutrons per fission

        self.nu: np.ndarray = None  # Total neutrons per fission
        self.nu_prompt: np.ndarray = None  # Prompt neutrons per fission
        self.nu_delayed: np.ndarray = None  # Delayed neutrons per fission

        self.nu_sigma_f: np.ndarray = None
        self.nu_prompt_sigma_f: np.ndarray = None
        self.nu_delayed_sigma_f: np.ndarray = None

        self.beta: np.ndarray = None  # Delayed neutron fraction

        self.precursor_lambda: np.ndarray = None  # Decay constants in 1/s
        self.precursor_yield: np.ndarray = None  # Precursor yielld fractions

        self.inv_velocity: np.ndarray = None  # Inverse speed in s/cm
        self.diffusion_coeff: np.ndarray = None  # Diffusion coefficient (cm)
        self.buckling: np.ndarray = None  # Material buckling

        # A modifier function for the absorption cross-section
        self.sigma_a_function: callable = None

    from ._read import read_xs_file

    def make_pure_scatterer(self) -> None:
        """
        Transform the cross-sections to a pure scattering material.
        """
        if self.n_groups == 0:
            msg = "The cross-sections are uninitialzied."
            raise AssertionError(msg)

        self.sigma_t[:] = self.sigma_s
        self.sigma_a[:] = 0.0
        self.sigma_r[:] = 0.0
        self.sigma_f[:] = 0.0

        self.chi[:] = 0.0
        self.chi_prompt[:] = 0.0

        self.nu[:] = 0.0
        self.nu_prompt[:] = 0.0
        self.nu_delayed[:] = 0.0
        self.beta[:] = 0.0

        self.nu_sigma_f[:] = 0.0
        self.nu_prompt_sigma_f[:] = 0.0
        self.nu_delayed_sigma_f[:] = 0.0

        self.diffusion_coeff[:] = 1.0 / (3.0 * self.sigma_t)

        self.is_fissile = False

    def _compute_macroscopic_cross_sections(self) -> None:
        """
        Compute the macroscopic cross-sections using the current density.
        """
        self.sigma_t *= self.density
        self.sigma_a *= self.density
        self.sigma_f *= self.density

        self.transfer_matrices *= self.density

        if self.diffusion_coeff.sum() == 0.0:
            self.diffusion_coeff = 1.0 / (3.0 * self.sigma_t)

    def _reconcile_cross_sections(self) -> None:
        """
        Enforce physical relationships within the cross-section data.
        """
        # Determine whether total or absorption was provided
        has_sigt = self.sigma_t.sum() > 0.0
        has_siga = self.sigma_a.sum() > 0.0
        if not has_sigt and not has_siga:
            msg = "Total or absorption cross-sections must be specified."
            raise AssertionError(msg)

        # Compute scattering cross-section from transfer matrix
        self.sigma_s = np.sum(self.transfer_matrices[0], axis=0)

        # Enforce total = absorption + scattering
        if has_siga:
            self.sigma_t = self.sigma_a + self.sigma_s
        else:
            self.sigma_a = self.sigma_t - self.sigma_s

        # Compute removal cross-section
        self.sigma_r = self.sigma_t - np.diag(self.transfer_matrices[0])

    def _reconcile_fission_properties(self) -> None:
        """
        Compute fission quantities from the specified inputs.
        """
        has_sigf = self.sigma_f.sum() > 0.0
        has_nusigf = self.nu_sigma_f.sum() > 0.0
        has_nupsigf = self.nu_prompt_sigma_f.sum() > 0.0
        has_nudsigf = self.nu_delayed_sigma_f.sum() > 0.0
        self.is_fissile = has_sigf or has_nusigf or has_nupsigf

        # Clear precursors if not fissile
        if not self.is_fissile and self.n_precursors > 0:
            self.n_precursors = 0
            self.precursor_lambda = np.zeros(0)
            self.precursor_yield = np.zeros(0)
            self.chi_delayed = np.zeros(0)

        # Check fission properties
        if self.is_fissile:
            has_nu = self.nu.sum() > 0.0
            has_nup = self.nu_prompt.sum() > 0.0
            has_nud = self.nu_delayed.sum() > 0.0
            has_beta = self.beta.sum() > 0.0

            has_chi = self.chi.sum() > 0.0
            has_chip = self.chi_prompt.sum() > 0.0
            has_chid = all(self.chi_delayed.sum(axis=0) > 0.0)

            # Prompt + delayed checks
            if self.n_precursors > 0:

                # Check that prompt + delayed spectra provided
                if not has_chip and not has_chid:
                    msg = "Prompt and delayed spectra must be specified."
                    raise AssertionError(msg)

                # ============================== Prompt + delayed
                if has_sigf and has_nup and has_nud:
                    if any(self.sigma_f < 0.0):
                        msg = "Fission cross-sections must be zero or greater."
                        raise ValueError(msg)
                    elif (any(self.nu_prompt <= 0.0) or
                            any(self.nu_delayed <= 0.0)):
                        msg = "Nu prompt/delayed must be positive."
                        raise ValueError(msg)

                    self.nu = self.nu_prompt + self.nu_delayed
                    self.beta = self.nu_delayed / self.nu

                # ============================== Delayed fraction (1)
                elif has_sigf and has_nu and has_beta:
                    if any(self.sigma_f < 0.0):
                        msg = "Fission cross-sections must be zero or greater."
                        raise ValueError(msg)
                    if any(self.nu < 1.0):
                        msg = "Total nu must be one or greater."
                        raise ValueError(msg)
                    if any(self.beta <= 0.0):
                        msg = "Delayed fraction must be greater than zero."
                        raise ValueError(msg)

                    self.nu_prompt = (1.0 - self.beta) * self.nu
                    self.nu_delayed = self.beta * self.nu

                # ============================== Delayed fraction (2)
                elif has_nusigf and has_nu and has_beta:
                    if any(self.nu_sigma_f < 0.0):
                        msg = "Fission cross-sections must be zero or greater."
                        raise ValueError(msg)
                    if any(self.nu < 1.0):
                        msg = "Total nu must be one or greater."
                        raise ValueError(msg)
                    if any(self.beta <= 0.0):
                        msg = "Delayed fraction must be greater than zero."
                        raise ValueError(msg)

                    self.sigma_f = self.nu_sigma_f / self.nu
                    self.nu_prompt = (1.0 - self.beta) * self.nu
                    self.nu_delayed = self.beta * self.nu

                # ============================== No nu
                elif has_nupsigf and has_nudsigf:
                    if (any(self.nu_prompt_sigma_f < 0.0) or
                            any(self.nu_delayed_sigma_f < 0.0)):
                        msg = "Fission cross-sections must be zero or greater."
                        raise ValueError(msg)

                    self.nu[:] = 1.0
                    self.sigma_f = (self.nu_prompt_sigma_f +
                                    self.nu_delayed_sigma_f)

                    self.beta = self.nu_delayed_sigma_f / self.nu_sigma_f
                    self.nu_prompt = 1.0 - self.beta
                    self.nu_delayed = self.beta

                else:
                    msg = "Unrecognized fission property specification."
                    raise AssertionError(msg)

                # ============================== Check the spectra
                if any(self.chi_prompt < 0.0):
                    msg = "Spectra must be zero or greater."
                    raise ValueError(msg)
                for j in range(self.n_precursors):
                    if any(self.chi_delayed[:, j] < 0.0):
                        msg = "Spectra must be zero or greater."
                        raise ValueError(msg)

                # ============================== Normalize the spectra
                self.chi_prompt /= self.chi_prompt.sum()
                for j in range(self.n_precursors):
                    self.chi_delayed[:, j] /= self.chi_delayed[:, j].sum()

                # ============================== Check the precursor properties
                if any(self.precursor_yield < 0.0):
                    msg = "Precursor yields must be zero or greater."
                    raise ValueError(msg)
                if any(self.precursor_lambda <= 0.0):
                    msg = "Precursor decay constants must be positive."
                    raise ValueError(msg)

                # ============================== Normalize the yields
                self.precursor_yield /= self.precursor_yield.sum()

                # ============================== Compute total quantities
                self.nu = self.nu_prompt + self.nu_delayed
                self.chi = (1.0 - self.beta) * self.chi_prompt
                for j in range(self.n_precursors):
                    self.chi += (self.beta * self.precursor_yield[j] *
                                 self.chi_delayed[:, j])

                if any(self.nu < 1.0):
                    msg = "Total nu must be one or greater."
                    raise ValueError(msg)
                if any(self.chi < 0.0):
                    msg = "Spectra must be zero or greater."
                    raise ValueError(msg)
                if abs(self.chi.sum() - 1.0) > 1.0e-12:
                    msg = "Spectra must sum to one."
                    raise ValueError(msg)

                # ============================== Compute nu sigma_f quantities
                self.nu_sigma_f = self.nu * self.sigma_f
                self.nu_prompt_sigma_f = self.nu_prompt * self.sigma_f
                self.nu_delayed_sigma_f = self.nu_delayed * self.sigma_f

            # Total fission checks
            else:
                if not has_nusigf and not has_sigf:
                    msg = "No fission cross-sections provided."
                    raise AssertionError(msg)
                if not has_chi:
                    msg = "No fission spectrum provided."
                    raise AssertionError(msg)

                # ============================== Compute fission cross-section
                if has_nusigf and not has_sigf:
                    if not has_nu:
                        self.nu[:] = 1.0
                        self.sigma_f = self.nu_sigma_f
                    else:
                        self.sigma_f = self.nu_sigma_f / self.nu

                if any(self.sigma_f < 0.0):
                    msg = "Fission cross-sections must be zero or greater."
                    raise ValueError(msg)

                # ============================== Compute nu
                if has_nusigf and has_sigf:
                    self.nu = self.nu_sigma_f / self.sigma_f

                if any(self.nu < 1.0):
                    msg = "Total nu must be one or greater."
                    raise ValueError(msg)

                # ============================== Normalize and check chi
                self.chi /= self.chi.sum()

                if any(self.chi < 0.0):
                    msg = "Spectra must be zero or greater."
                    raise ValueError(msg)

                # ============================== Compute nu sigma_f
                self.nu_sigma_f = self.nu * self.sigma_f

    def _reinit_groupwise_data(self) -> None:
        if self.n_groups == 0:
            raise AssertionError("Cannot reinit with 0 groups.")

        self.E_bounds = np.zeros(self.n_groups + 1)

        self.sigma_t = np.zeros(self.n_groups)
        self.sigma_a = np.zeros(self.n_groups)
        self.sigma_s = np.zeros(self.n_groups)
        self.sigma_r = np.zeros(self.n_groups)
        self.sigma_f = np.zeros(self.n_groups)

        self.chi = np.zeros(self.n_groups)
        self.chi_prompt = np.zeros(self.n_groups)

        self.nu = np.zeros(self.n_groups)
        self.nu_prompt = np.zeros(self.n_groups)
        self.nu_delayed = np.zeros(self.n_groups)
        self.beta = np.zeros(self.n_groups)

        self.nu_sigma_f = np.zeros(self.n_groups)
        self.nu_prompt_sigma_f = np.zeros(self.n_groups)
        self.nu_delayed_sigma_f = np.zeros(self.n_groups)

        self.inv_velocity = np.zeros(self.n_groups)
        self.diffusion_coeff = np.zeros(self.n_groups)
        self.buckling = np.zeros(self.n_groups)

    def _reinit_precursor_data(self) -> None:
        if self.n_groups == 0:
            msg = "Cannot reinit with 0 groups."
            raise AssertionError(msg)

        self.precursor_lambda = np.zeros(self.n_precursors)
        self.precursor_yield = np.zeros(self.n_precursors)
        self.chi_delayed = np.zeros((self.n_groups, self.n_precursors))
