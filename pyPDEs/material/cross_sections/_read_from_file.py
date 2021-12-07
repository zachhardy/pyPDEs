import os
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import CrossSections


def read_from_xs_file(self: 'CrossSections', filename: str,
                      density: float = 1.0) -> None:
    """
    Populate the cross sections with a ChiTech cross section file.

    Parameters
    ----------
    filename : str
        The path to the ChiTech cross section file.
    density : float, default 1.0
        A scaling factor for the cross section. This is meant
        to be synonymous with scaling a microscopic cross section
        by an atom density.
    """

    not_found = 'must be provided'
    incompat_w_G = 'is incompatible with n_groups'
    incompat_w_J = 'is incompatible with n_precursors'

    def read_1d_xs(key, xs, f, ln):
        # Get first non-empty line
        ln += 1
        words = f[ln].split()
        while len(words) == 0:
            ln += 1
            words = f[ln].split()

        # Get entries until end keyword
        while words[0] != f'{key}_END':
            group = int(words[0])
            value = float(words[1])
            xs[group] = value

            # Get next non-empty line
            ln += 1
            words = f[ln].split()
            while not words:
                ln += 1
                words = f[ln].split()
        ln += 1

    def read_transfer_matrix(key, matrix, f, ln):
        # Get first non-empty line
        ln += 1
        words = f[ln].split()
        while len(words) == 0:
            ln += 1
            words = f[ln].split()

        # Get info until end keyword
        while words[0] != f'{key}_END':
            if words[0] == 'M_GPRIME_G_VAL':
                m = int(words[1])
                gprime = int(words[2])
                group = int(words[3])
                value = float(words[4])
                if m < len(matrix):
                    matrix[m][gprime][group] = value

            # Next non-empty line
            ln += 1
            words = f[ln].split()
            while len(words) == 0:
                ln += 1
                words = f[ln].split()
        ln += 1

    def read_chi_delayed(key, xs, f, ln):
        # Get first non-empty line
        ln += 1
        words = f[ln].split()
        while len(words) == 0:
            ln += 1
            words = f[ln].split()

        while words[0] != f'{key}_END':
            if words[0] == 'G_PRECURSORJ_VAL':
                group = int(words[1])
                precursor_num = int(words[2])
                value = float(words[3])
                xs[group][precursor_num] = value

            # Get next non-empty line
            ln += 1
            words = f[ln].split()
            while len(words) == 0:
                ln += 1
                words = f[ln].split()
        ln += 1

    if not os.path.isfile(filename):
        raise FileNotFoundError(f'{filename} could not be found.')

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

            # Get the number of groups and precursors
            if line[0] == 'NUM_GROUPS':
                self.n_groups = int(line[1])
                self.initialize_groupwise_data()

            if line[0] == 'NUM_MOMENTS':
                assert self.n_groups > 0
                M = int(line[1])
                self.scattering_order = M - 1
                self.transfer_matrix = \
                    np.zeros((M, self.n_groups, self.n_groups))

            if line[0] == 'NUM_PRECURSORS':
                assert self.n_groups > 0
                self.n_precursors = int(line[1])
                if self.n_precursors > 0:
                    self.initialize_precursor_data()
                    if self.n_precursors == 1:
                        self.precursor_yield = np.ones(1)

            if line[0] == 'SIGMA_T_BEGIN':
                read_1d_xs('SIGMA_T', self.sigma_t, lines, line_num)
                self.sigma_t *= density  # scale by density

            if line[0] == 'SIGMA_A_BEGIN':
                read_1d_xs('SIGMA_A', self.sigma_a, lines, line_num)
                self.sigma_a *= density  # scale by density

            if line[0] == 'DIFFUSION_COEFF_BEGIN':
                read_1d_xs('DIFFUSION_COEFF', self.D, lines, line_num)

            if (line[0] == 'BUCKLING_BEGIN'):
                read_1d_xs('BUCKLING', self.B_sq, lines, line_num)

            if line[0] == 'SIGMA_F_BEGIN':
                read_1d_xs('SIGMA_F', self.sigma_f, lines, line_num)
                self.sigma_f *= density

            if line[0] == 'NU_BEGIN':
                read_1d_xs('NU', self.nu, lines, line_num)

            if line[0] == 'NU_PROMPT_BEGIN':
                read_1d_xs('NU_PROMPT', self.nu_prompt, lines, line_num)

            if line[0] == 'NU_DELAYED_BEGIN':
                read_1d_xs('NU_DELAYED', self.nu_delayed, lines, line_num)

            if line[0] == 'CHI_BEGIN':
                read_1d_xs('CHI', self.chi, lines, line_num)

            if line[0] == 'CHI_PROMPT_BEGIN':
                read_1d_xs('CHI_PROMPT', self.chi_prompt, lines, line_num)

            if line[0] == 'CHI_DELAYED_BEGIN':
                read_chi_delayed(
                    'CHI_DELAYED', self.chi_delayed, lines, line_num)

            if line[0] == 'VELOCITY_BEGIN':
                read_1d_xs(
                    'VELOCITY', self.velocity, lines, line_num)

            if line[0] == 'TRANSFER_MOMENTS_BEGIN':
                read_transfer_matrix(
                    'TRANSFER_MOMENTS', self.transfer_matrix, lines, line_num)
                self.transfer_matrix *= density

            if line[0] == 'PRECURSOR_LAMBDA_BEGIN':
                read_1d_xs(
                    'PRECURSOR_LAMBDA', self.precursor_lambda, lines, line_num)

            if line[0] == 'PRECURSOR_YIELD_BEGIN':
                read_1d_xs(
                    'PRECURSOR_YIELD', self.precursor_yield, lines, line_num)

            line_num += 1

    self._validate_xs()