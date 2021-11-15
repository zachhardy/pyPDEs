import sys
from typing import List, NewType


class UnknownManager:
    """
    Data structure describing a set of unknowns.

    The unknown manager contains a list of unknowns.
    It is agnostic to the spatial discretization and is
    used as a mapping tool to access a dof on a given node.

    Parameters
    ----------
    storage_method : {'nodal', 'block'}, default 'nodal'
    """

    def __init__(self, storage_method: str = 'nodal') -> None:
        self.unknowns: List[UnknownManager.Unknown] = []
        self.storage_method: str = storage_method

    class Unknown:
        """
        Data structure for describing an unknown.

        An unknown is defined by a number of components and
        a starting index within the unknowns list of the
        UnknownManager.

        Parameters
        ----------
        n_components : int, default 1
            The number of components in the Unknown.
        map_begin : int, default 0
            The starting component index of this unknown within
            the UnknownManager.
        """

        def __init__(self, n_components: int = 1,
                     map_begin: int = 0) -> None:
            self.n_components: int = n_components
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
            if component < 0 or component >= self.n_components:
                raise ValueError('Invalid component number.')
            return self.map_begin + component

        def get_map_end(self) -> int:
            """
            Get the last index for this unknown.

            Returns
            -------
            int
            """
            return self.map_begin + self.n_components - 1

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
        Get the index of a specific unknown and component.

        Parameters
        ----------
        unknown_id : int
            The index of the Unknown in the list of Unknowns.
        component : int, default 0
            The component of the specified Unknown.
        """
        if unknown_id < 0 or unknown_id > self.n_unknowns:
            raise ValueError('Invalid unknown_id.')
        return self.unknowns[unknown_id].get_map(component)

    def clear(self):
        self.unknowns.clear()
