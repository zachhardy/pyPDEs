import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import CrossSections


def read_from_xs_dict(
        self: 'CrossSections', xs: dict, density: float = 1.0) -> None:
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
    self.n_groups = xs.get("n_groups")
    if not self.n_groups:
        raise KeyError("n_groups must be provided.")
    self.reset_groupwise_xs()

    self.n_precursors = xs.get("n_precursors")
    if not self.n_precursors:
        self.n_precursors = 0
    self.has_precursors = self.n_precursors > 0
    self.reset_delayed_xs()

    # Get general xs
    sig_t = xs.get("sigma_t")
    if not sig_t:
        raise KeyError("sigma_t must be provided.")
    if len(sig_t) != self.n_groups:
        raise ValueError(
            "sigma_t is incompatible with num_groups.")
    self.sigma_t = density * np.array(sig_t)

    if "sigma_a" in xs:
        sig_a = xs.get("sigma_a")
        if len(sig_a) != self.n_groups:
            raise ValueError(
                "sigma_a is not compatible with num_groups.")
        self.sigma_a = density * np.array(sig_a)

    if "D" in xs:
        D = xs.get("D")
        if len(D) != self.n_groups:
            raise ValueError(
                "D is not compatible with num_groups.")
        self.D = np.array(D)

    transfer_matrix = xs.get("transfer_matrix")
    if not transfer_matrix:
        raise KeyError(
            "transfer_matrix must be provided")
    if len(transfer_matrix) != len(transfer_matrix[0]) != self.n_groups:
        raise ValueError(
            "transfer_matrix is incompatible with num_groups.")
    self.transfer_matrix = density * np.array(transfer_matrix)

    # Get fission xs
    sig_f = xs.get("sigma_f")
    if sig_f:
        self.is_fissile = True

        if len(sig_f) != self.n_groups:
            raise ValueError(
                "sigma_f is incompatible with num_groups.")
        self.sigma_f = density * np.array(sig_f)

        # Get total fission xs
        chi = xs.get("chi")
        if not chi and not self.has_precursors:
            raise KeyError(
                "chi must be provided if precursors "
                "are not present.")
        if chi:
            if len(chi) != self.n_groups:
                raise ValueError(
                    "chi is incompatible with num_groups.")
            self.chi = np.array(chi) / np.sum(chi)

        nu = xs.get("nu")
        if not nu and not self.has_precursors:
            raise KeyError(
                "nu must be provided if precursors "
                "are not present.")
        if nu:
            if len(nu) != self.n_groups:
                raise ValueError(
                    "nu is incompatible with num_groups.")
            self.nu = np.array(nu)

        # Get prompt fission xs
        chi_p = xs.get("chi_prompt")
        if not chi_p and self.has_precursors:
            raise KeyError(
                "chi_prompt msut be provided if "
                "precursors are present.")
        if chi_p:
            if len(chi) != self.n_groups:
                raise ValueError(
                    "chi_prompt is incompatible with num_groups.")
            self.chi_prompt = np.array(chi_p) / np.sum(chi_p)

        nu_p = xs.get("nu_prompt")
        if not nu_p and self.has_precursors:
            raise KeyError(
                "nu_prompt msut be provided if "
                "precursors are present.")
        if nu_p:
            if len(nu_p) != self.n_groups:
                raise ValueError(
                    "nu_prompt is incompatible with num_groups.")
            self.nu_prompt = np.array(nu_p)

        # Delayed fission xs
        if self.has_precursors:
            p_decay = xs.get("precursor_lambda")
            if not p_decay:
                raise KeyError(
                    "precursor_lambda must be provided for fissile "
                    "cross sections with precursors.")
            if len(p_decay) != self.n_precursors:
                raise ValueError(
                    "precursor_lambda is incompatible "
                    "with num_precursors.")
            self.precursor_lambda = np.array(p_decay)

            p_gamma = xs.get("precursor_yield")
            if not p_gamma:
                raise KeyError("precursor_yield not found.")
            if len(p_decay) != self.n_precursors:
                raise ValueError(
                    "precursor_yield is incompatible "
                    "with num_precursors.")
            self.precursor_yield = np.array(p_gamma)

            nu_d = xs.get("nu_delayed")
            if not nu_d:
                raise KeyError("nu_delayed not found.")
            if len(nu_d) != self.n_groups:
                raise ValueError("nu_delayed is incompatible "
                                 "with num_groups.")
            self.nu_delayed = np.array(nu_d)

            chi_d = xs.get("chi_delayed")
            if not chi_d:
                raise KeyError("chi_delayed not found.")
            if len(chi_d) != self.n_groups:
                raise ValueError(
                    "chi_delayed is incompatible with num_groups.")
            if len(chi_d[0]) != self.n_precursors:
                raise ValueError(
                    "chi_delayed is incompatible with num_precursors.")
            for j in range(self.n_precursors):
                chi_dj = np.array(chi_d[:, j])
                self.chi_delayed[:, j] = chi_dj / np.sum(chi_dj)

    # Inverse velocity term
    inv_vel = xs.get("inv_velocity")
    if inv_vel:
        if len(inv_vel) != self.n_groups:
            raise ValueError(
                "inv_velocity is incompatible with num_groups.")
        self.inv_velocity = np.array(inv_vel)

    # Compute other xs
    self.finalize_xs()
