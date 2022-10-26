import os
import pickle
import numpy as np

from typing import Union

from .simulation_reader import NeutronicsSimulationReader

Simulation = NeutronicsSimulationReader


class NeutronicsDatasetReader:
    """
    A utility class for reading collections of neutronics simulation data.
    """

    def __init__(self, path: str) -> None:
        """
        Create a neutronics dataset readers.

        Parameters
        ----------
        path : str, Path to collection of simulation output data.
        """
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a valid directory.")

        self._path: str = os.path.abspath(path)

        self._simulations: list[Simulation] = []
        self._parameters: np.ndarray = None

        self._parameter_bounds: list[tuple[float, float]] = []
        self._boundary_mask: list[bool] = []

    def __iter__(self) -> list[Simulation]:
        """
        Return the iterator over the simulations.

        Returns
        -------
        list[Simulation]
        """
        return iter(self._simulations)

    def __next__(self) -> Simulation:
        """
        Return the next simulation in the dataset.

        Returns
        -------
        Simulation
        """
        return next(self._simulations)

    def __getitem__(self, item: int) -> Simulation:
        """
        Return the simulation at the specified index.

        Parameters
        ----------
        item : int

        Returns
        -------
        Simulation
        """
        if item < 0 or item >= self.n_simulations:
            raise IndexError(f"{item} is an invalid index.")
        return self._simulations[item]

    @property
    def times(self) -> np.ndarray:
        """
        Return the snapshot times.

        Returns
        -------
        list[float]
        """
        return self._simulations[0].times

    @property
    def n_simulations(self) -> int:
        """
        Return the number of simulations in the dataset.

        Returns
        -------
        int
        """
        return len(self._simulations)

    @property
    def parameters(self) -> np.ndarray:
        """
        Return the parameters.

        Returns
        -------
        numpy.ndarray (n_simulations, n_parameters)
        """
        return self._parameters

    @property
    def n_parameters(self) -> int:
        """
        Return the number of parameters which describe the dataset.

        Returns
        -------
        int
        """
        return self._parameters.shape[1]

    @property
    def boundary_mask(self) -> list[bool]:
        """
        Return a boolean mask to extract simulations which lie on
        the boundary of the parameter space.

        Returns
        -------
        list[bool]
        """
        return self._boundary_mask

    @property
    def interior_mask(self) -> list[bool]:
        """
        Return a boolean mask to extract simulations which lie on
        the interior of the parameter space.

        Returns
        -------
        list[bool]
        """
        return [not mask for mask in self._boundary_mask]

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        """
        Return the bounding values of each parameter.

        Returns
        -------
        list[tuple[float, float]]
        """
        return self._parameter_bounds

    @property
    def n_snapshots(self) -> int:
        """
        Return the number of temporal snapshots.

        Returns
        -------
        int
        """
        return self._simulations[0].n_snapshots

    @property
    def dimension(self) -> int:
        """
        Return the spatial dimension.

        Returns
        -------
        int
        """
        return self._simulations[0].dimension

    @property
    def n_materials(self) -> int:
        """
        Return the number of materials in the problem.

        Returns
        -------
        int
        """
        return self._simulations[0].n_materials

    @property
    def n_cells(self) -> int:
        """
        Return the number of cells.

        Returns
        -------
        int
        """
        return self._simulations[0].n_cells

    @property
    def n_nodes(self) -> int:
        """
        Return the number of nodes

        Returns
        -------
        numpy.ndarray
        """
        return self._simulations[0].n_nodes

    @property
    def n_moments(self) -> int:
        """
        Return the number of flux moments.

        Returns
        -------
        int
        """
        return self._simulations[0].n_moments

    @property
    def n_groups(self) -> int:
        """
        Return the number of energy groups.

        Returns
        -------
        int
        """
        return self._simulations[0].n_groups

    @property
    def max_precursors(self) -> int:
        """
        Return the maximum number of precursors per material.

        Returns
        -------
        int
        """
        return self._simulations[0].max_precursors

    def read(self) -> 'NeutronicsDatasetReader':
        """
        Read in the dataset.

        Returns
        -------
        NeutronicsDatasetReader
        """
        self.__init__(self._path)

        # Loop through simulations
        for i, entry in enumerate(sorted(os.listdir(self._path))):
            path = os.path.join(self._path, entry)
            if entry == "params.txt":
                params = np.loadtxt(path)
                if params.ndim == 1:
                    params = np.atleast_2d(params).T
                self._parameters = params

            elif os.path.isdir(path) and "reference" not in entry:
                reader = Simulation(path).read()
                if self.n_simulations > 1:
                    self._check_compatibility(reader)
                self._simulations.append(reader)

        self._find_parameter_bounds()
        self._define_boundary_mask()
        return self

    def create_3d_matrix(
            self, variables: Union[str, list[str]] = None
    ) -> np.ndarray:
        """
        Create a 3D matrix, or list of 2D simulation matrices.

        Parameters
        ----------
        variables : list[str], default None
            The variables to include in the matrix. When none, only
            the multi-group scalar flux is included.

        Returns
        -------
        numpy.ndarray (n_simumations, n_snapshots, varies)
        """
        X = [self._simulations[0].create_simulation_matrix(variables)]
        for simulation in self._simulations[1:]:
            X.append(simulation.create_simulation_matrix(variables))
        return np.array(X)

    def create_2d_matrix(
            self, variables: Union[str, list[str]] = None
    ) -> np.ndarray:
        """
        Create a 2D matrix whose columns contain full simulation results.

        Parameters
        ----------
        variables : list[str], default None
            The variables to include in the matrix. When none, only
            the multi-group scalar flux is included.

        Returns
        -------
        numpy.ndarray (n_simulations, varies)
        """
        X = self._simulations[0].create_simulation_vector(variables)
        for simulation in self._simulations[1:]:
            tmp = simulation.create_simulation_vector(variables)
            X = np.hstack((X, tmp))
        return X.T

    def unstack_simulation_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Unstack simulation vectors into snapshot matrices.

        Parameters
        ----------
        vector : numpy.ndarray (varies, n_snapshots * varies)
            A set of simulation vectors, where each row is an
            independent simulation.

        Returns
        -------
        numpy.ndarray (varies, n_snapshots, varies)
        """
        vector = np.array(vector)
        if vector.ndim == 1:
            vector = np.atleast_2d(vector)
        return vector.reshape(vector.shape[0], self.n_snapshots, -1)

    def train_test_split(
            self,
            variables: Union[str, list[str]] = None,
            test_size: float = None,
            interior_only: bool = False,
            seed: int = None,
            dimension: int = 2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the training and test set.

        Parameters
        ----------
        variables : list[str], default None
            The variables to include in the data.
        test_size : float, default None
            The fraction of the data to use in the test set. If None,
            the Scikit-Learn default is used.
        interior_only : bool, default True
            Flag for only splitting the interior values. If True, all
            boundary values are included in the training set. If False,
            no preference is given to boundary values in the splitting.
        seed : int, default None
            The seed for the pseudo-random number generator.
        dimension : {2, 3}, default 2
            The dimension of the resulting output data. If 2, individual
            simulations are formatted as vectors. If 3, they are formatted
            as matrices.

        Returns
        -------
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
            The training simulation data, the test simulation data,
            the training parameters, the test parameters.
        """
        from sklearn.model_selection import train_test_split

        if dimension not in [2, 3]:
            raise ValueError("Only 2D and 3D matrices are allowed.")

        if dimension == 2:
            X = self.create_2d_matrix(variables)
        else:
            X = self.create_3d_matrix(variables)
        Y = self._parameters

        if interior_only:
            interior = [not b for b in self._boundary_mask]
            splits = train_test_split(X[interior], Y[interior],
                                      test_size=test_size,
                                      random_state=seed)

            splits[0] = np.vstack((splits[0], X[self._boundary_mask]))
            splits[2] = np.vstack((splits[2], Y[self._boundary_mask]))
        else:
            splits = train_test_split(X, Y,
                                      test_size=test_size,
                                      random_state=seed)
        return splits

    def _find_parameter_bounds(self) -> None:
        """
        Find the bounding values of each parameter.
        """
        bounds = []
        for p in range(self.n_parameters):

            Yp = self._parameters[:, p]
            bounds.append((min(Yp), max(Yp)))
        self._parameter_bounds = bounds

    def _define_boundary_mask(self) -> None:
        """
        Compute a boolean mapping for boundary cases.
        """
        boundary_mask = []
        for i in range(self.n_simulations):
            y = self._parameters[i]

            on_bndry = False
            for p in range(self.n_parameters):
                bounds = self._parameter_bounds[p]
                if y[p] == bounds[0] or y[p] == bounds[1]:
                    on_bndry = True
                    break
            boundary_mask.append(on_bndry)
        self._boundary_mask = boundary_mask

    def _check_compatibility(self, simulation: Simulation) -> None:
        """
        Ensure that the simulations are identical.

        Parameters
        ----------
        simulation : Simulation
        """
        err_msg = "Simulation setup data does not agree."
        if (simulation.n_snapshots != self.n_snapshots or
                simulation.dimension != self.dimension or
                simulation.n_materials != self.n_materials or
                simulation.n_cells != self.n_cells or
                simulation.n_nodes != self.n_nodes or
                simulation.n_moments != self.n_moments or
                simulation.n_groups != self.n_groups or
                simulation.max_precursors != self.max_precursors):
            raise AssertionError(err_msg)

    def save(self, filename: str) -> None:
        """
        Save the data set reader.

        Parameters
        ----------
        filename : str
            A location to save the file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> 'NeutronicsDatasetReader':
        """
        Load a data set reader.

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