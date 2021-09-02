import os
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import CrossSections


def read_from_xs_file(self, filename: str, density: float = 1.0) -> None:
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
                self.sigma_t *= density

            if line[0] == "DIFFUSION_COEFF_BEGIN":
                read_1d_xs("DIFFUSION_COEFF", self.D, lines, line_num)

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
                read_transfer_matrix("TRANSFER_MOMENTS", self.transfer_matrix, lines, line_num)
                self.transfer_matrix *= density

            if line[0] == "PRECURSOR_LAMBDA_BEGIN":
                read_1d_xs("PRECURSOR_LAMBDA", self.precursor_lambda, lines, line_num)

            if line[0] == "PRECURSOR_YIELD_BEGIN":
                read_1d_xs("PRECURSOR_YIELD", self.precursor_yield, lines, line_num)

            if line[0] == "CHI_DELAYED_BEGIN":
                read_chi_delayed("CHI_DELAYED", self.chi_delayed, lines, line_num)
                for j in range(self.n_precursors):
                    self.chi_delayed[:, j] /= np.sum(self.chi_delayed[:, j])

            line_num += 1

    # Compute other xs
    self.finalize_xs()