import os
import struct
import pickle

import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional
from collections.abc import Iterable

Variables = Union[str, Iterable[str]]


class NeutronicsSimulationReader:
    """
    A utility class for reading neutronics simulation data.
    """

    # Energy released per fission event in J
    energy_per_fission: float = 3.204e-11

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 14

    from ._plot import plot_flux_moment
    from ._plot import plot_power_profile
    from ._plot import plot_temperature_profile

    from ._plot import plot_power
    from ._plot import plot_fuel_temperature

    from ._animate import animate_flux_moment

    def __init__(self, path: str) -> None:
        """
        Create a neutronics simulation readers.

        This constructor sets the path to the output directory
        and specifies which files to read.

        Parameters
        ----------
        path : str, Path to the output binaries.
        """
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a valid directory.")

        self._path: str = os.path.abspath(path)

        self.nodes: np.ndarray = None
        self.centroids: np.ndarray = None
        self.material_ids: np.ndarray = None

        self.times: np.ndarray = None

        self.powers: np.ndarray = None
        self.peak_power_densities: np.ndarray = None
        self.average_power_densities: np.ndarray = None

        self.average_fuel_temperatures: np.ndarray = None
        self.peak_fuel_temperatures: np.ndarray = None

        self.flux_moments: np.ndarray = None
        self.precursors: np.ndarray = None
        self.fission_rates: np.ndarray = None
        self.temperatures: np.ndarray = None

    @property
    def n_snapshots(self) -> int:
        """
        Return the number of time steps.

        Returns
        -------
        int
        """
        return len(self.times)

    @property
    def n_cells(self) -> int:
        """
        Return the number of cells.

        Returns
        -------
        int
        """
        return len(self.centroids)

    @property
    def n_nodes(self) -> int:
        """
        Return the number of nodes.

        Returns
        -------
        int
        """
        return len(self.nodes)

    @property
    def dimension(self) -> int:
        """
        Return the dimension of the spatial domain.

        Returns
        -------
        int
        """
        if all(self.centroids[:, :2].ravel() == 0.0):
            return 1
        elif all(self.centroids[:, 2].ravel() == 0.0):
            return 2
        else:
            return 3

    @property
    def n_materials(self) -> int:
        """
        Return the number of unique materials.

        Returns
        -------
        int
        """
        return len(np.unique(self.material_ids))

    @property
    def n_moments(self) -> int:
        """
        Return the number of flux moments.

        Returns
        -------
        int
        """
        return len(self.flux_moments[0])

    @property
    def n_groups(self) -> int:
        """
        Return the number of energy groups.

        Returns
        -------
        int
        """
        return len(self.flux_moments[0][0])

    @property
    def max_precursors(self) -> int:
        """
        Return the maximum number of precursors per material.

        Returns
        -------
        int
        """
        return len(self.precursors[0])

    def create_simulation_matrix(
            self, variables: Optional[Variables] = None
    ) -> np.ndarray:
        """
        Create a simulation matrix based on the specified variables.

        Parameters
        ----------
        variables : str or Iterable[str], default None
            The variables to include in the matrix.

        Returns
        -------
        numpy.ndarray (varies, n_snapshots)
        """
        if isinstance(variables, str):
            variables = [variables]
        elif variables is None:
            variables = self.default_variables

        X = self.get_variable(variables[0])
        for var in variables[1:]:
            tmp = self.get_variable(var)
            X = np.hstack(X, tmp)
        return X

    def create_simulation_vector(
            self, variables: Optional[Variables] = None
    ) -> np.ndarray:
        """
        Create a simulation vector based on the specified variables.

        Parameters
        ----------
        variables : str or Iterable[str], default None
            The variables to include in the matrix.

        Returns
        -------
        numpy.ndarray (varies * n_snapshots,)
        """
        x = self.create_simulation_matrix(variables)
        return x.reshape(x.size, 1)

    @property
    def default_variables(self) -> list[str]:
        """
        Return the scalar flux moments.

        Returns
        -------
        list[str]
        """
        return ['sflux_m0']

    def get_variable(self, key: str) -> np.ndarray:
        """
        Return the data based on the variable name.

        Parameters
        ----------
        key : str

        Returns
        -------
        numpy.ndarray
        """
        if "sflux" in key:

            if "m" not in key and "g" not in key:
                phi = self.flux_moments

            elif "m" in key and "g" not in key:
                moment = int(key[key.find("m") + 1])
                phi = self.flux_moments[:, moment]

            elif "m" in key and "g" in key:
                moment = int(key[key.find("m") + 1])
                group = int(key[key.find("g") + 1])
                phi = self.flux_moments[:, moment, group]
            else:
                raise ValueError("Invalid sflux specifier.")

            return phi.reshape(self.n_snapshots, -1)

        elif "precursor" in key:
            if "j" not in key:
                Cj = self.precursors
            else:
                precursor = int(key[key.find("j") + 1])
                Cj = self.precursors[:, precursor]
            return Cj.reshape(self.n_snapshots, -1)

        elif key == "power_density":
            Ef = self.energy_per_fission
            return Ef * self.fission_rates

        elif key == "fission_rate":
            return self.fission_rates

        elif key == "temperature":
            return self.temperatures

        else:
            raise KeyError(f"{key} is not a valid variable.")

    def read(
            self,
            parse_flux_moments: bool = True,
            parse_precursors: bool = True,
            parse_fission_rate: bool = True,
            parse_temperature: bool = True
    ) -> 'NeutronicsSimulationReader':
        """
        Read transient simulation data.

        Parameters
        ----------
        parse_flux_moments : bool, Flag for reading multi-group scalar fluxes.
        parse_precursors : bool, Flag for reading precursor concentrations.
        parse_fission_rate : bool, Flag for reading fission rates.
        parse_temperature : bool, Flag for reading temperatures.

        Returns
        -------
        NeutronicsSimulationReader
        """
        self.__init__(self._path)

        # Parse summary file
        tmp = np.loadtxt(f"{self._path}/summary.txt")
        self.times, self.powers = tmp[:, 0], tmp[:, 1]
        self.peak_power_densities = tmp[:, 2]
        self.average_power_densities = tmp[:, 3]
        self.peak_fuel_temperatures = tmp[:, 4]
        self.average_fuel_temperatures = tmp[:, 5]

        # Parse geometry file
        tmp = self.read_geometry_file(f"{self._path}/geom.data")
        self.centroids, self.nodes, self.material_ids = tmp

        # Loop through outputs
        for i, entry in enumerate(sorted(os.listdir(self._path))):
            path = os.path.join(self._path, entry)

            # Go into time step directory
            if os.path.isdir(path):
                for file in os.listdir(path):
                    filepath = os.path.join(path, file)

                    if file == "sflux.npy" and parse_flux_moments:
                        phi = np.load(filepath)
                        if self.flux_moments is None:
                            shape = (self.n_snapshots, *phi.shape)
                            self.flux_moments = np.zeros(shape)
                        self.flux_moments[int(entry)] = phi

                    elif file == "precursors.npy" and parse_precursors:
                        Cj = np.load(filepath)
                        if self.precursors is None:
                            shape = (self.n_snapshots, *Cj.shape)
                            self.precursors = np.zeros(shape)
                        self.precursors[int(entry)] = Cj

                    elif file == "fission_rate.npy" and parse_fission_rate:
                        fission_rate = np.load(filepath)
                        if self.fission_rates is None:
                            shape = (self.n_snapshots, *fission_rate.shape)
                            self.fission_rates = np.zeros(shape)
                        self.fission_rates[int(entry)] = fission_rate

                    elif file == "temperature.npy" and parse_temperature:
                        T = np.load(filepath)
                        if self.temperatures is None:
                            shape = (self.n_snapshots, *T.shape)
                            self.temperatures = np.zeros(shape)
                        self.temperatures[int(entry)] = T
        return self

    def read_geometry_file(self, filename: str) -> None:
        """
        Read the geometry file for a time step.

        Parameters
        ----------
        filename : str

        Returns
        -------
        numpy.ndarray
            The (x, y, z) coordinates of the cell centroids.
        numpy.ndarray
            The (x, y, z) coordinates of the nodes per cell.
        numpy.ndarray
            The material IDs.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Cannot find file {filename}.")

        with open(filename, 'rb') as file:
            file.read(599)

            n_cells = self.read_uint64_t(file)
            n_nodes = self.read_uint64_t(file)

            centroids = np.zeros((n_cells, 3))
            nodes = np.zeros((n_nodes, 3))
            matids = np.zeros(n_cells)

            n = 0
            for c in range(n_cells):
                self.read_uint64_t(file)
                matids[c] = self.read_unsigned_int(file)

                nodes_on_cell = self.read_unsigned_int(file)

                # Parse centroids
                for d in range(3):
                    centroids[c, d] = self.read_double(file)

                # Parse nodes on the cell
                for i in range(nodes_on_cell):
                    for d in range(3):
                        nodes[n, d] = self.read_double(file)
                    n += 1

        return centroids, nodes, matids

    def _interpolate(
            self, times: list[float], data: np.ndarray
    ) -> np.ndarray:
        """
        Return interpolated data for a specified time.

        Parameters
        ----------
        self : NeutronicsSimulationReader
        times : list[float]
        data : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            The values of the specified data at the specified times.
        """
        dt = self.times[1] - self.times[0]
        vals = np.zeros((len(times), *data[0].shape))
        for t, time in enumerate(times):
            i = [int(np.floor(time / dt)), int(np.ceil(time / dt))]
            w = [i[1] - time / dt, time / dt - i[0]]
            if i[0] == i[1]:
                w = [1.0, 0.0]
            vals[t] = w[0] * data[i[0]] + w[1] * data[i[1]]
        return vals

    def save(self, filename: str) -> None:
        """
        Save the simulation reader.

        Parameters
        ----------
        filename : str
            A location to save the file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> 'NeutronicsSimulationReader':
        """
        Load a simulation reader.

        Parameters
        ----------
        filename : str
            The filename where the reader is saved.

        Returns
        -------
        NeutronicsSimulationReader
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def read_double(file) -> float:
        return struct.unpack('d', file.read(8))[0]

    @staticmethod
    def read_uint64_t(file) -> int:
        return struct.unpack('Q', file.read(8))[0]

    @staticmethod
    def read_unsigned_int(file) -> int:
        return struct.unpack('I', file.read(4))[0]

