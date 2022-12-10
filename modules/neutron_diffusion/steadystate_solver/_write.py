from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver

import os
import struct
import numpy as np


def write_scalar_flux(
        self: 'SteadyStateSolver',
        directory: str,
        file_prefix: str = "sflux"
) -> None:
    """
    Write the scalar flux to a binary file.

    Parameters
    ----------
    directory : str, The output directory.
    file_prefix : str, The filename.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    assert os.path.isdir(directory)

    filepath = os.path.join(directory, file_prefix)
    if "." in filepath:
        assert filepath.count(".") == 1
        filepath = filepath.split(".")[0]

    n_nodes = self.discretization.n_nodes()
    phi = np.zeros((1, self.n_groups, n_nodes))
    for g in range(self.n_groups):
        phi[0][g] = self.phi[g::self.n_groups]
    np.save(filepath, phi)


def write_precursors(
        self: 'SteadyStateSolver',
        directory: str,
        file_prefix: str = "precursors"
) -> None:
    """
    Write the delayed neutron precursors to a binary file.

    Parameters
    ----------
    directory : str, The output directory.
    file_prefix : str, The filename.
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)
    assert os.path.isdir(directory)

    filepath = os.path.join(directory, file_prefix)
    if "." in filepath:
        assert filepath.count(".") == 1
        filepath = filepath.split(".")[0]

    precursors = np.zeros((self.max_precursors, self.mesh.n_cells))
    for j in range(self.max_precursors):
        precursors[j] = self.precursors[j::self.max_precursors]
    np.save(filepath, precursors)


def write_fission_rate(
        self: 'SteadyStateSolver',
        directory: str,
        file_prefix: str = "fission_rate"
) -> None:
    """
    Write the fission rate to a binary file.

    Parameters
    ----------
    directory : str, The output directory.
    file_prefix : str, The filename.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    assert os.path.isdir(directory)

    filepath = os.path.join(directory, file_prefix)
    if "." in filepath:
        assert filepath.count(".") == 1
        filepath = filepath.split(".")[0]

    fission_rate = np.zeros(self.mesh.n_cells)
    for cell in self.mesh.cells:
        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]
        if xs.is_fissile:
            for g in range(self.n_groups):
                dof = self.n_groups * cell.id + g
                fission_rate[cell.id] += xs.sigma_f[g] * self.phi[dof]
    np.save(filepath, fission_rate)
