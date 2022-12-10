from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver

import os
import struct
import numpy as np


def write_temperature(
        self: 'TransientSolver',
        directory: str,
        file_prefix: str = "temperature"
) -> None:
    """
    Write the temperature to a binary file.

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
    np.save(filepath, self.temperature)


def write_snapshot(self: 'TransientSolver', output_index: int) -> None:
    """
    Write all simulation data to an output file.

    Parameters
    ----------
    output_index : int
    """

    # ------------------------------ create output directory
    directory = str(output_index)
    directory = directory.zfill(4)
    directory = os.path.join(self.output_directory, directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # ------------------------------ write snapshot information
    filepath = os.path.join(directory, "summary.txt")
    with open(filepath, 'a') as file:
        if os.path.getsize(filepath) == 0:
            file.write(f"# {'Time':<18}{'Pwr':<18}"
                       f"{'Peak Pwr Density':<18}"
                       f"{'Avg Pwr Density':<18}"
                       f"{'Peak Fuel Temp':<18}"
                       f"{'Avg Fuel Temp':<18}\n")
        file.write(f"  {self.time:<20.10e}{self.power:<18.10e}"
                   f"{self.peak_power_density:<18.10e}"
                   f"{self.average_power_density:<18.10e}"
                   f"{self.peak_fuel_temperature:<18.10e}"
                   f"{self.average_fuel_temperature:<18.10e}\n")

    # ------------------------------ write simulation data
    self.write_scalar_flux(directory)
    self.write_fission_rate(directory)
    self.write_precursors(directory)
    self.write_temperature(directory)
