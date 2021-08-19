import sys
from typing import List, NewType


class UnknownManager:
    """
    Class which describes a set of unknowns in a problem.

    The unknown manager contains a list of unknowns.
    It is agnostic to the spatial discretization and is
    used as a mapping tool to access a dof on a given node.

    Parameters
    ----------
    storage_method : str, default "NODAL"
        The way DoFs are to be stored within a vector.
        "NODAL" storage orders DoFs by unknown/component and
        then node, while "BLOCK" storage orders DoFs by node
        and then unknown/component.
    """
    def __init__(self, storage_method: str = "NODAL") -> None:
        self.unknowns: List[UnknownManager.Unknown] = []
        self.storage_method: str = storage_method

    class Unknown:
        """
        Class describing an individual unknown.

        An unknown is defined by a number of components and
        a starting index within the unknowns list of the
        UnknownManager.

        Parameters
        ----------
        n_components : int, default 1
            The number of components belonging to this Unknown.
        map_begin : int, default 0
            The cumulative number of components already
            contained within the UnknownManager.
        """
        def __init__(self, n_components: int = 1,
                     map_begin: int = 0) -> None:
            self.num_components: int = n_components
            self.map_begin: int = map_begin

        def get_map(self, component: int = -1) -> int:
            """
            Get the index for this component of the unknown.
            This simply adds the component number to the
            `map_begin` attribute of the Unknown.

            Parameters
            ----------
            component : int, default -1
                The component to get the index for. The
                default returns the last component index.

            Returns
            -------
            int
            """
            if component < 0 or component >= self.num_components:
                raise ValueError("Invalid component number.")
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
    def n_unknowns(self) -> int:
        """
        Get the total number of unknowns.

        Returns
        -------
        int
        """
        return len(self.unknowns)

    @property
    def total_components(self) -> int:
        """
        Get the total number of components.

        Returns
        -------
        int
        """
        return self.unknowns[-1].get_map_end() + 1

    def add_unknown(self, n_components: int = 1) -> None:
        """
        Add an unknown to the unknown manager.

        Parameters
        ----------
        n_components : int, default 1
            The number of components in the Unknown
            being added.
        """
        map_begin = -1
        if self.unknowns:
            map_begin = self.unknowns[-1].get_map_end()
        unknown = self.Unknown(n_components, map_begin + 1)
        self.unknowns.append(unknown)

    def map_unknown(self, unknown_id: int, component: int = 0) -> int:
        """
        Get the index of a specific component of a specific unknown.

        Parameters
        ----------
        unknown_id : int
            The index of the Unknown in the list of Unknowns.
        component : int, default 0
            The component of the specified Unknown.
        """
        if unknown_id < 0 or unknown_id > self.n_unknowns:
            raise ValueError("Invalid unknown_id.")
        return self.unknowns[unknown_id].get_map(component)

    def clear(self):
        self.unknowns.clear()
