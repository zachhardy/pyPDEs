import sys
from typing import List, NewType


class UnknownManager:
    """
    Class which describes a set of unknowns in a problem.

    The unknown manager contains a list of unknowns.
    It is agnostic to the spatial discretization and is
    used as a mapping tool to access a dof on a given node.

    Attributes
    ----------
    unknowns : List[Unknown]
    storage_method : str
        The storage method for unknowns.
        Options are "NODAL" and "BLOCK".

    Parameters
    ----------
    storage_method : str, default "NODAL"
    """
    def __init__(self, storage_method: str = "NODAL") -> None:
        self.unknowns: List[UnknownManager.Unknown] = []
        self.storage_method: str = storage_method

    class Unknown:
        """
        Class describing an individual unknown.

        An unknown is defined by a number of components and
        a starting index within the unknowns list of the
        unknown manager. The starting index is the sum of all
        components of previously entered unknowns.

        Attributes
        ----------
        num_components : int, default 1
        map_begin : int, default 0
        """
        def __init__(self, num_components: int = -1,
                     map_begin: int = 0) -> None:
            self.num_components: int = num_components
            self.map_begin: int = map_begin

        def get_map(self, component: int = 0) -> int:
            """
            Get th index for this component of the unknown.

            Parameters
            ----------
            component : int, default 0

            Returns
            -------
            int
            """
            try:
                if component < 0 or component >= self.num_components:
                    raise ValueError("Invalid component number.")
            except ValueError as err:
                print(f"\n***** ERROR:\t{err.args[0]}\n")
                sys.exit()
            return self.map_begin + component

        def get_map_end(self) -> int:
            """
            Get the last index for this unknown.

            Returns
            -------
            int
            """
            return self.map_begin + self.num_components - 1

    @property
    def num_unknowns(self) -> int:
        """
        Get the total number of unknowns.

        Returns
        -------
        int
        """
        return len(self.unknowns)

    @property
    def total_num_components(self) -> int:
        """
        Get the total number of components.

        Returns
        -------
        int
        """
        return self.unknowns[-1].get_map_end() + 1

    def add_unknown(self, num_components: int = 1) -> None:
        """
        Add an unknown to the unknown manager.

        Parameters
        ----------
        num_components : int, default 1
        """
        map_begin = -1
        if self.unknowns:
            map_begin = self.unknowns[-1].get_map_end()
        unknown = self.Unknown(num_components, map_begin + 1)
        self.unknowns.append(unknown)

    def map_unknown(self, unknown_id: int, component: int = 0) -> int:
        """
        Get the index of a specific component of a specific unknown.

        Parameters
        ----------
        unknown_id : int
        component : int, default 0

        Returns
        -------
        int
        """
        try:
            if unknown_id < 0 or unknown_id > self.num_unknowns:
                raise ValueError("Invalid unknown_id.")
        except ValueError as err:
            print(f"\n***** ERROR:\t{err.args[0]}\n")
            sys.exit()
        return self.unknowns[unknown_id].get_map(component)

    def clear(self):
        self.unknowns.clear()
