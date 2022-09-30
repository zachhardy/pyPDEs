import os
import struct

import numpy as np

from pyPDE.mesh import Mesh
from pyPDE.mesh import Cell
from pyPDE.mesh import CartesianVector


class SpatialDiscretization:
    """
    Base class for spatial discretizations.
    """

    def __init__(self, mesh: Mesh, discretization_type: str) -> None:
        """
        Parameters
        ----------
        mesh : Mesh
        discretization_type : str
        """
        discretization_types = ["FV"]
        discretization_type = discretization_type.upper()
        if discretization_type not in discretization_types:
            msg = f"Unrecognized discretization type {discretization_type}."
            raise ValueError(msg)

        self.mesh: Mesh = mesh
        self.type: str = discretization_type.upper()

    def n_nodes(self) -> int:
        """
        Return the total number of nodes in the discretization.

        Returns
        -------
        int
        """
        cls_name = self.__class__.__name__
        msg = f"This method has not been implemented in {cls_name}."
        raise NotImplementedError(msg)

    def nodes_per_cell(self, cell: Cell) -> int:
        """
        Return the number of nodes on the specified Cell.

        Parameters
        ----------
        cell : Cell

        Returns
        -------
        int
        """
        cls_name = self.__class__.__name__
        msg = f"This method has not been implemented in {cls_name}."
        raise NotImplementedError(msg)

    def n_dofs(self, n_components: int = 1) -> int:
        """
        Return the total number of degrees of freedom in the discretization.

        Parameters
        ----------
        n_components : int

        Returns
        -------
        int
        """
        cls_name = self.__class__.__name__
        msg = f"This method has not been implemented in {cls_name}."
        raise NotImplementedError(msg)

    def n_dofs_per_cell(self, cell: Cell, n_components: int = 1) -> int:
        """
        Return the number of degrees of freedom on the specified Cell.

        Parameters
        ----------
        cell : Cell
        n_components : int

        Returns
        -------
        int
        """
        cls_name = self.__class__.__name__
        msg = f"This method has not been implemented in {cls_name}."
        raise NotImplementedError(msg)

    def nodes(self, cell: Cell) -> list[CartesianVector]:
        """
        Return the node coordinates on the specified Cell.

        Parameters
        ----------
        cell : Cell

        Returns
        -------
        list[CartesianVector]
        """
        cls_name = self.__class__.__name__
        msg = f"This method has not been implemented in {cls_name}."
        raise NotImplementedError(msg)

    def nodes_as_ndarray(self) -> np.ndarray:
        """
        Return the nodes as a numpy ndarray.

        Returns
        -------
        numpy.ndarray
        """
        nodes = []
        for cell in self.mesh.cells:
            for node in self.nodes(cell):
                nodes.append([node.x, node.y, node.z])
        return np.array(nodes)

    def write_discretization(
            self,
            directory: str,
            file_prefix: str = "geom"
    ) -> None:
        """
        Write the discretization to a binary file.

        Parameters
        ----------
        directory : str, The output directory.
        file_prefix : str, The filename
        """
        if not os.path.isdir(directory):
            os.makedirs(directory)
        assert os.path.isdir(directory)

        filepath = f"{directory}/{file_prefix}"
        if "." in filepath:
            assert filepath.count(".") == 1
            filepath = filepath.split(".")[0]
        filepath += ".data"

        with open(filepath, 'wb') as file:

            size = 600
            header_info = \
                b"pyPDEs Geometry File\n" \
                b"Header Size: " + str(size).encode("utf-8") + b" bytes\n"
            header_info += \
                b"Structure(type-info)\n" \
                b"uint64_t      n_cells\n" \
                b"uint64_t      n_nodes\n" \
                b"Each Cell:\n" \
                b"  uint64_t      cell_id\n" \
                b"  unsigned int  material_id\n" \
                b"  unsigned int  n_nodes\n" \
                b"  Centroid:\n" \
                b"    double        x_position\n" \
                b"    double        y_position\n" \
                b"    double        z_position\n" \
                b"    Each Node:\n" \
                b"      double        x_position\n" \
                b"      double        y_position\n" \
                b"      double        z_position\n"

            header_size = len(header_info)
            header_info += b"-" * (size - 1 - header_size)

            file.write(bytearray(header_info))
            file.write(struct.pack('Q', self.mesh.n_cells))
            file.write(struct.pack('Q', self.n_nodes()))

            for cell in self.mesh.cells:
                file.write(struct.pack('Q', cell.id))
                file.write(struct.pack('I', cell.material_id))

                nodes = self.nodes(cell)
                file.write(struct.pack('I', len(nodes)))

                # Centroid position
                file.write(struct.pack('d', cell.centroid.x))
                file.write(struct.pack('d', cell.centroid.y))
                file.write(struct.pack('d', cell.centroid.z))

                # Node positions
                for node in nodes:
                    file.write(struct.pack('d', node.x))
                    file.write(struct.pack('d', node.y))
                    file.write(struct.pack('d', node.z))
