import os
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import CrossSections


def read_from_xs_file(self: "CrossSections", filename: str,
                      density: float = 1.0) -> None:
    """Populate the cross sections with a ChiTech cross section file.

    Parameters
    ----------
    filename : str
        The path to the ChiTech cross section file.
    density : float, default 1.0
        A scaling factor for the cross section. This is meant
        to be synonymous with scaling a microscopic cross section
        by an atom density.
    """

    not_found = "must be provided"
    incompat_w_G = "is incompatible with n_groups"
    incompat_w_J = "is incompatible with n_precursors"

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

        # Go through file
        line_num = 0
        while line_num < len(lines):
            line = lines[line_num].split()

            # Skip empty lines
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
                self.sigma_t *= density  # scale by density

            if line[0] == "SIGMA_A_BEGIN":
                read_1d_xs("SIGMA_A", self.sigma_a, lines, line_num)
                self.sigma_a *= density  # scale by density

            if line[0] == "DIFFUSION_COEFF_BEGIN":
                read_1d_xs("DIFFUSION_COEFF", self.D, lines, line_num)

            if line[0] == "SIGMA_F_BEGIN":
                read_1d_xs("SIGMA_F", self.sigma_f, lines, line_num)
                self.sigma_f *= density
                if np.sum(self.sigma_f) > 0.0:
                    self.is_fissile = True

            if line[0] == "NU_BEGIN":
                read_1d_xs("NU", self.nu, lines, line_num)

            if line[0] == "NU_PROMPT_BEGIN":
                read_1d_xs("NU_PROMPT", self.nu_prompt, lines, line_num)

            if line[0] == "NU_DELAYED_BEGIN":
                read_1d_xs("NU_DELAYED", self.nu_delayed, lines, line_num)

            if line[0] == "CHI_BEGIN":
                read_1d_xs("CHI", self.chi, lines, line_num)

                # Normalize to unit spectrum
                self.chi /= np.sum(self.chi)

            if line[0] == "CHI_PROMPT_BEGIN":
                read_1d_xs("CHI_PROMPT", self.chi_prompt, lines, line_num)

                # Normalize to unit spectrum
                self.chi_prompt /= np.sum(self.chi_prompt)

            if line[0] == "CHI_DELAYED_BEGIN":
                read_chi_delayed(
                    "CHI_DELAYED", self.chi_delayed, lines, line_num)

                # Normalize to unit spectra
                for j in range(self.n_precursors):
                    chi_dj_sum = np.sum(self.chi_delayed[:, j])
                    self.chi_delayed[:, j] /= chi_dj_sum

            if line[0] == "INV_VELOCITY_BEGIN":
                read_1d_xs(
                    "INV_VELOCITY", self.inv_velocity, lines, line_num)

            if line[0] == "TRANSFER_MOMENTS_BEGIN":
                read_transfer_matrix(
                    "TRANSFER_MOMENTS", self.transfer_matrix, lines, line_num)
                self.transfer_matrix *= density  # scale by density

                # Compute total scattering cross section
                self.sigma_s = np.sum(self.transfer_matrix, axis=1)

            if line[0] == "PRECURSOR_LAMBDA_BEGIN":
                read_1d_xs(
                    "PRECURSOR_LAMBDA", self.precursor_lambda, lines, line_num)


            if line[0] == "PRECURSOR_YIELD_BEGIN":
                read_1d_xs(
                    "PRECURSOR_YIELD", self.precursor_yield, lines, line_num)

                # Normalize to unit yield
                self.precursor_yield /= np.sum(self.precursor_yield)

            line_num += 1

    # Check that mandatory cross sections are provided
    if self.sigma_t is None:
        raise AssertionError(f"sigma_t {not_found}.")

    if self.transfer_matrix is None:
        raise AssertionError(f"transfer_matrix {not_found}.")



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

        # Compute chi from prompt and delayed chi
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


    # Compute other xs
    self.finalize_xs()