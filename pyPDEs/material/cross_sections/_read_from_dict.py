import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import CrossSections


def read_from_xs_dict(
        self: "CrossSections", xs: dict, density: float = 1.0) -> None:
    """Populate the cross sections with a dictionary.

    Parameters
    ----------
    xs : dict
        The cross sections contained within a dictionary.
    density : float, default 1.0
        A scaling factor for the cross section. This is meant
        to be synonymous with scaling a microscopic cross section
        by an atom density.
    """
    not_found = "must be provided"
    incompat_w_G = "is incompatible with n_groups"
    incompat_w_J = "is incompatible with n_precursors"

    # Get number of groups
    self.n_groups = xs.get("n_groups")
    if not self.n_groups:
        raise KeyError(f"n_groups {not_found}.")

    # Get number of precursors
    self.n_precursors = xs.get("n_precursors")
    if not self.n_precursors:
        self.n_precursors = 0
    self.has_precursors = self.n_precursors > 0

    # Get total cross section
    if "sigma_t" in xs:
        sig_t = np.array(xs.get("sigma_t"))
        if len(sig_t) != self.n_groups:
            raise ValueError( f"sigma_t {incompat_w_G}.")
        self._sigma_t = density * sig_t

    # Get absorption cross section
    if "sigma_a" in xs:
        sig_a = np.array(xs.get("sigma_a"))
        if len(sig_a) != self.n_groups:
            raise ValueError(f"sigma_a {incompat_w_G}.")
        self.sigma_a = density * sig_a

    # Get transfer matrix
    if "transfer_matrix" in xs:
        trnsfr = np.array(xs.get("transfer_matrix"))
        if trnsfr.shape != (self.n_groups,) * 2:
            raise ValueError(
                f"transfer_matrix {incompat_w_G}.")
        self.transfer_matrix = density * trnsfr
        self.sigma_s = np.sum(self.transfer_matrix, axis=1)

    # Get diffusion coefficient or set to default
    if "D" in xs:
        D = np.array(xs.get("D"))
        if len(D) != self.n_groups:
            raise ValueError(f"D {incompat_w_G}.")
        self.D = D

    # Get fission xs
    if "sigma_f" in xs:
        sig_f = np.array(xs.get("sigma_f"))
        if len(sig_f) != self.n_groups:
            raise ValueError(f"sigma_f {incompat_w_G}.")
        self.sigma_f = density * sig_f
        self.is_fissile = sum(sig_f) > 0.0

    # Get nu
    if "nu" in xs:
        nu = np.array(xs.get("nu"))
        if len(nu) != self.n_groups:
            raise ValueError(f"nu {incompat_w_G}.")
        self.nu = nu

    # Get nu prompt
    if "nu_prompt" in xs:
        nu_p = np.array(xs.get("nu_prompt"))
        if len(nu_p) != self.n_groups:
            raise ValueError(f"nu_prompt {incompat_w_G}.")
        self.nu_prompt = nu_p

    # Get nu delayed
    if "nu_delayed" in xs and self.has_precursors:
        nu_d = np.array(xs.get("nu_delayed"))
        if len(nu_d) != self.n_groups:
            raise ValueError(f"nu_delayed {incompat_w_G}.")
        self.nu_delayed = nu_d

    # Get chi total
    if "chi" in xs:
        chi = np.array(xs.get("chi"))
        if len(chi) != self.n_groups:
            raise ValueError(f"chi {incompat_w_G}.")
        self.chi = chi / np.sum(chi)

    # Get chi prompt
    if "chi_prompt" in xs:
        chi_p = np.array(xs.get("chi_prompt"))
        if len(chi_p) != self.n_groups:
            raise ValueError(f"chi_prompt {incompat_w_G}.")
        self.chi_prompt = chi_p / np.sum(chi_p)

    # Get chi delayed
    if "chi_delayed" in xs and self.has_precursors:
        chi_d = xs.get("chi_delayed")
        if len(chi_d) != self.n_groups:
            raise ValueError(f"chi_delayed {incompat_w_G}.")
        for g in range(self.n_groups):
            if len(chi_d[g]) != self.n_precursors:
                raise ValueError(
                    f"chi_delayed group {g} {incompat_w_J}.")
        chi_d = np.array(chi_d)
        for j in range(self.n_precursors):
            chi_dj = chi_d[:, j]
            self.chi_delayed[:, j] = chi_dj / np.sum(chi_dj)

    # Get precursor decay constants
    if "precursor_lambda" in xs and self.has_precursors:
        decay = np.array(xs.get("precursor_lambda"))
        if len(decay) != self.n_precursors:
            raise ValueError(f"precursor_lambda {incompat_w_J}.")
        self.precursor_lambda = decay

    # Get precursor yields
    if "precursor_gamma" in xs and self.has_precursors:
        gamma = np.array(xs.get("precursor_yield"))
        if len(gamma) != self.n_precursors:
            raise ValueError(f"precursor_yield {incompat_w_J}.")
        self.precursor_yield = np.array(gamma) / np.sum(gamma)

    # Velocity term
    if "velocity" in xs:
        v = np.array(xs.get("velocity"))
        if len(v) != self.n_groups:
            raise ValueError(
                f"velocity {incompat_w_G}.")
        self.velocity = v

    self._validate_xs()
