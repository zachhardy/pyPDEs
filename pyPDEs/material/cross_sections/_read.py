import os
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import CrossSections


def read_xs_file(
        self: 'CrossSections',
        xs_path: str,
        density: float = 1.0
) -> None:
    """
    Read a Chi-Tech style cross-section file and scale the relevant
    cross-section data by the specified density.

    Parameters
    ----------
    self : CrossSections
    xs_path : str
    density : float
    """

    # ========================================
    # Utility functions
    # ========================================

    def advance_line(
            f: list[str],
            ln: int
    ) -> tuple[list[str], int]:
        """
        Advance to the next non-empty line.
        """
        ln += 1
        w = f[ln].split()
        while not w:
            ln += 1
            w = f[ln].split()
        return w, ln

    def read_1d_data(
            key: str,
            vector: np.ndarray,
            f: list[str],
            ln: int
    ) -> None:
        """
        Read 1D data.
        """
        w, ln = advance_line(f, ln)
        while w[0] != f"{key}_END":
            group = int(w[0])
            value = float(w[1])
            vector[group] = value

            w, ln = advance_line(f, ln)
        ln += 1

    def read_transfer_matrices(
            key: str,
            matrix: np.ndarray,
            f: list[str],
            ln: int
    ) -> None:
        """
        Read transfer matrix data.
        """
        w, ln = advance_line(f, ln)
        while w[0] != f"{key}_END":
            if w[0] == "M_GPRIME_G_VAL":
                moment = int(w[1])
                gprime = int(w[2])
                group = int(w[3])
                value = float(w[4])
                matrix[moment][group][gprime] = value
            w, ln = advance_line(f, ln)
        ln += 1

    def read_chi_delayed(
            key: str,
            matrix: np.ndarray,
            f: list[str],
            ln: int) -> None:
        """
        Read chi-delayed data.
        """
        w, ln = advance_line(f, ln)
        while w[0] != f"{key}_END":
            if w[0] == "G_PRECURSORJ_VAL":
                group = int(w[1])
                precursor = int(w[2])
                value = float(w[3])
                matrix[group][precursor] = value
            w, ln = advance_line(f, ln)
        ln += 1

    # ========================================
    # Read the file
    # ========================================

    if not os.path.isfile(xs_path):
        msg = f"Invalid path to the cross-section file: {xs_path}."
        raise FileNotFoundError(msg)

    self.density = density

    with open(xs_path) as file:
        lines = file.readlines()

        line_num = 0
        while line_num < len(lines):

            # Get next non-empty line
            words = lines[line_num].split()
            while not words:
                line_num += 1
                words = lines[line_num].split()

            # ========================================
            # Get general information
            # ========================================

            if words[0] == "NUM_GROUPS":
                self.n_groups = int(words[1])
                self._reinit_groupwise_data()

            if words[0] == "NUM_MOMENTS":
                self.n_moments = int(words[1])

                if self.n_groups == 0:
                    msg = "The number of groups must be specified " \
                          "before the number of moments."
                    raise AssertionError(msg)

                shape = (self.n_moments, self.n_groups, self.n_groups)
                self.transfer_matrices = np.zeros(shape)

            if words[0] == "NUM_PRECURSORS":
                self.n_precursors = int(words[1])
                self._reinit_precursor_data()

            # ========================================
            # Parse basic cross-sections
            # ========================================

            if words[0] == "SIGMA_T_BEGIN":
                read_1d_data("SIGMA_T", self.sigma_t, lines, line_num)

            if words[0] == "SIGMA_A_BEGIN":
                read_1d_data("SIGMA_A", self.sigma_a, lines, line_num)

            if words[0] == "DIFFUSION_COEFF_BEGIN":
                read_1d_data("DIFFUSION_COEFF",
                             self.diffusion_coeff, lines, line_num)

            if words[0] == "BUCKLING_BEGIN":
                read_1d_data("BUCKLING", self.buckling, lines, line_num)

            if words[0] == "TRANSFER_MOMENTS_BEGIN":
                read_transfer_matrices(
                    "TRANSFER_MOMENTS",
                    self.transfer_matrices, lines, line_num)

            # ========================================
            # Parse fission cross-sections
            # ========================================

            if words[0] == "SIGMA_F_BEGIN":
                read_1d_data("SIGMA_F", self.sigma_f, lines, line_num)

            if words[0] == "NU_SIGMA_F_BEGIN":
                read_1d_data("NU_SIGMA_F",
                             self.nu_sigma_f, lines, line_num)

            if words[0] == "NU_PROMPT_SIGMA_F_BEGIN":
                read_1d_data("NU_PROMPT_SIGMA_F",
                             self.nu_prompt_sigma_f, lines, line_num)

            if words[0] == "NU_DELAYED_SIGMA_F_BEGIN":
                read_1d_data("NU_DELAYED_SIGMA_F",
                             self.nu_delayed_sigma_f, lines, line_num)

            if words[0] == "CHI_BEGIN":
                read_1d_data("CHI", self.chi, lines, line_num)

            if words[0] == "CHI_PROMPT_BEGIN":
                read_1d_data("CHI_PROMPT",
                             self.chi_prompt, lines, line_num)

            if words[0] == "CHI_DELAYED_BEGIN":
                read_chi_delayed("CHI_DELAYED",
                                 self.chi_delayed, lines, line_num)

            if words[0] == "NU_BEGIN":
                read_1d_data("NU", self.nu, lines, line_num)

            if words[0] == "NU_PROMPT_BEGIN":
                read_1d_data("NU_PROMPT", self.nu_prompt, lines, line_num)

            if words[0] == "NU_DELAYED_BEGIN":
                read_1d_data("NU_DELAYED",
                             self.nu_delayed, lines, line_num)

            if words[0] == "BETA_BEGIN":
                read_1d_data("BETA", self.beta, lines, line_num)

            # ========================================
            # Delayed neutron precursor data
            # ========================================

            if words[0] == "PRECURSOR_LAMBDA_BEGIN":
                read_1d_data("PRECURSOR_LAMBDA",
                             self.precursor_lambda, lines, line_num)

            if words[0] == "PRECURSOR_YIELD_BEGIN":
                read_1d_data("PRECURSOR_YIELD",
                             self.precursor_yield, lines, line_num)

            # ========================================
            # Time-dependent problem data
            # ========================================

            if words[0] == "VELOCITY_BEGIN":
                read_1d_data("VELOCITY",
                             self.inv_velocity, lines, line_num)
                self.inv_velocity = 1.0 / self.inv_velocity

            if (words[0] == "INV_VELOCITY_BEGIN" and
                    self.inv_velocity.sum() == 0.0):
                read_1d_data("INV_VELOCITY",
                             self.inv_velocity, lines, line_num)

            line_num += 1

    self._compute_macroscopic_cross_sections()
    self._reconcile_cross_sections()
    self._reconcile_fission_properties()
