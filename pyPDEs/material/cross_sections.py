import os.path
import sys

import numpy as np
from numpy import ndarray

from .material import MaterialProperty


class CrossSections(MaterialProperty):
    """
    Class for neutronics cross sections.
    """
    def __init__(self):
        super().__init__("XS")

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

    @property
    def nu_sigma_f(self) -> ndarray:
        return self.nu * self.sigma_f

    @property
    def nu_prompt_sigma_f(self) -> ndarray:
        return self.nu_prompt * self.sigma_f

    @property
    def nu_delayed_sigma_f(self) -> ndarray:
        return self.nu_delayed * self.sigma_f

    def read_from_xs_dict(self, xs: dict, density: float = 1.0) -> None:
        """
        Populate the cross sections with a dictionary.
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

        # ======================================== Get general xs
        sig_t = xs.get("sigma_t")
        if not sig_t:
            raise KeyError("sigma_t must be provided.")
        if len(sig_t) != self.n_groups:
            raise ValueError(
                "sigma_t is incompatible with num_groups.")
        self.sigma_t = density * np.array(sig_t)

        sig_tr = xs.get("sigma_tr")
        if not sig_tr:
            raise KeyError(
                "sigma_tr must be provided")
        if len(sig_tr) != len(sig_tr[0]) != self.n_groups:
            raise ValueError(
                "sigma_tr is incompatible with num_groups.")
        self.sigma_tr = density * np.array(sig_tr)

        # ======================================== Get fission xs
        sig_f = xs.get("sigma_f")
        if sig_f:
            self.is_fissile = True

            if len(sig_f) != self.n_groups:
                raise ValueError(
                    "sigma_f is incompatible with num_groups.")
            self.sigma_f = density * np.array(sig_f)

            # ============================== Get total fission xs
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

            # ============================== Get prompt fission xs
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

            # ======================================== Delayed fission xs
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

        # ======================================== Inverse velocity term
        inv_vel = xs.get("inv_velocity")
        if inv_vel:
            if len(inv_vel) != self.n_groups:
                raise ValueError(
                    "inv_velocity is incompatible with num_groups.")
            self.inv_velocity = np.array(inv_vel)

        # ================================================== Compute other xs
        self.finalize_xs()

    def read_from_xs_file(self, filename: str, density: float = 1.0) -> None:
        """
        Read a ChiTech cross section file.
        """
        def read_1d_xs(key, xs, f, ln):
            words = f[ln + 1].split()
            while words[0] != f"{key}_END":
                ln += 1
                group = int(words[0])
                value = float(words[1])
                xs[group] = value
                words = f[ln + 1].split()
            ln += 1

        def read_transfer_matrix(key, xs, f, ln):
            words = f[ln + 1].split()
            while words[0] != f"{key}_END":
                ln += 1
                if words[0] == "M_GPRIME_G_VAL":
                    if words[1] == "0":
                        gprime = int(words[2])
                        group = int(words[3])
                        value = float(words[4])
                        xs[gprime][group] = value
                words = f[ln + 1].split()
            ln += 1

        def read_chi_delayed(key, xs, f, ln):
            words = f[ln + 1].split()
            while words[0] != f"{key}_END":
                ln += 1
                if words[0] == "G_PRECURSORJ_VAL":
                    group = int(words[1])
                    precursor_num = int(words[2])
                    value = float(words[3])
                    xs[group][precursor_num] = value
                words = f[ln + 1].split()
            ln += 1

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"{filename} could not be found.")

        with open(filename) as file:
            lines = file.readlines()

            # ======================================== Go through file
            line_num = 0
            while line_num < len(lines):
                line = lines[line_num].split()

                # ======================================== Skip empty lines
                while len(line) == 0:
                    line_num += 1
                    line = lines[line_num].split()

                if line[0] == "NUM_GROUPS":
                    self.n_groups = int(line[1])
                    self.reset_groupwise_xs()

                if line[0] == "NUM_PRECURSORS":
                    self.n_precursors = int(line[1])
                    self.has_precursors = self.n_precursors > 0
                    self.reset_delayed_xs()

                if line[0] == "SIGMA_T_BEGIN":
                    read_1d_xs("SIGMA_T", self.sigma_t, lines, line_num)
                    self.sigma_t *= density

                if line[0] == "SIGMA_F_BEGIN":
                    read_1d_xs("SIGMA_F", self.sigma_f, lines, line_num)
                    self.sigma_f *= density
                    if np.sum(self.sigma_f):
                        self.is_fissile = True

                if line[0] == "SIGMA_A_BEGIN":
                    read_1d_xs("SIGMA_A", self.sigma_a, lines, line_num)
                    self.sigma_a *= density

                if line[0] == "SIGMA_S_BEGIN":
                    read_1d_xs("SIGMA_S", self.sigma_s, lines, line_num)
                    self.sigma_s *= density

                if line[0] == "NU_BEGIN":
                    read_1d_xs("NU", self.nu, lines, line_num)

                if line[0] == "NU_PROMPT_BEGIN":
                    read_1d_xs("NU_PROMPT", self.nu_prompt, lines, line_num)

                if line[0] == "NU_DELAYED_BEGIN":
                    read_1d_xs("NU_DELAYED", self.nu_delayed, lines, line_num)

                if line[0] == "CHI_BEGIN":
                    read_1d_xs("CHI", self.chi, lines, line_num)
                    self.chi /= np.sum(self.chi)

                if line[0] == "CHI_PROMPT_BEGIN":
                    read_1d_xs("CHI_PROMPT", self.chi_prompt, lines, line_num)
                    self.chi_prompt /= np.sum(self.chi_prompt)

                if line[0] == "INV_VELOCITY_BEGIN":
                    read_1d_xs("INV_VELOCITY", self.inv_velocity, lines, line_num)

                if line[0] == "TRANSFER_MOMENTS_BEGIN":
                    read_transfer_matrix("TRANSFER_MOMENTS", self.sigma_tr, lines, line_num)
                    self.sigma_tr *= density

                if line[0] == "PRECURSOR_LAMBDA_BEGIN":
                    read_1d_xs("PRECURSOR_LAMBDA", self.precursor_lambda, lines, line_num)

                if line[0] == "PRECURSOR_YIELD_BEGIN":
                    read_1d_xs("PRECURSOR_YIELD", self.precursor_yield, lines, line_num)

                if line[0] == "CHI_DELAYED_BEGIN":
                    read_chi_delayed("CHI_DELAYED", self.chi_delayed, lines, line_num)
                    for j in range(self.n_precursors):
                        self.chi_delayed[:, j] /= np.sum(self.chi_delayed[:, j])

                line_num += 1

        # ================================================== Compute other xs
        self.finalize_xs()

    def finalize_xs(self) -> None:
        """
        Compute auxiliary cross sections based upon others.
        """
        self.diffusion_coeff = 1.0 / (3.0 * self.sigma_t)
        self.sigma_s = np.sum(self.sigma_tr, axis=1)
        self.sigma_a = self.sigma_t - self.sigma_s
        self.sigma_r = self.sigma_t - np.diag(self.sigma_tr)

        nu_p, nu_d = self.nu_prompt, self.nu_delayed
        if sum(nu_p) > 0.0 and sum(nu_d) > 0.0:
            self.nu = nu_p + nu_d

    def reset_groupwise_xs(self) -> None:
        """
        Reset the general and prompt cross sections.
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
        """
        Reset the cross section terms involving delayed neutrons
        """
        self.precursor_lambda = np.zeros(self.n_precursors)
        self.precursor_yield = np.zeros(self.n_precursors)
        self.chi_delayed = np.zeros((self.n_groups, self.n_precursors))
