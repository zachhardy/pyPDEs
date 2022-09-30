from .cartesian_vector import CartesianVector


Point = Normal = CartesianVector


class Cell:
    """
    Implementation of a computational cell.
    """

    def __init__(self, cell_type: str) -> None:
        """
        Parameters
        ----------
        cell_type : str
        """
        cell_type = cell_type.upper()
        cell_types = ["SLAB", "ANNULUS", "SHELL", "QUADRILATERAL"]
        if cell_type not in cell_types:
            msg = f"Unrecognized cell type {cell_type}."
            raise ValueError(msg)

        self.type: str = cell_type.upper()

        self.id: int = -1
        self.material_id: int = -1
        self.vertex_ids: list[int] = []

        self.volume: float = 0.0
        self.centroid: Point = Point()

        self.faces: list[Face] = []

    @property
    def n_faces(self) -> int:
        """
        Return the number of faces on the cell.

        Returns
        -------
        int
        """
        return len(self.faces)

    @property
    def n_vertices(self) -> int:
        """
        Return the number of vertices on the cell.

        Returns
        -------
        int
        """
        return len(self.vertex_ids)

    def __str__(self) -> str:
        """
        Return the contents of the cell as a string.

        Returns
        -------
        str
        """
        s = f"***** Cell {self.id} *****\n" \
            f"type: {self.type}\n" \
            f"material_id: {self.material_id}\n" \
            f"centroid: {str(self.centroid)}\n" \
            f"volume: {self.volume}\n" \
            f"n_vertices: {len(self.vertex_ids)}\n" \
            f"vertex_ids: ["
        for v in self.vertex_ids:
            s += f"{v} "
        s += "]\n"

        s += f"n_faces: {len(self.faces)}\n"
        for f, face in enumerate(self.faces):
            s += f"----- Face {f} -----\n{face}"
        return s

    def __eq__(self, other: 'Cell') -> bool:
        """
        Test the equality of two cells.

        Parameters
        ----------
        other : Cell

        Returns
        -------
        bool
        """
        if len(self.faces) != len(other.faces):
            return False

        for (f0, f1) in zip(self.faces, other.faces):
            if not f0 == f1:
                return False
        return True

    def __ne__(self, other: 'Cell') -> bool:
        """
        Test the equality of two cells.

        Parameters
        ----------
        other : Cell

        Returns
        -------
        bool
        """
        return not self == other


class Face:
    """
    Implementation of a computational face.
    """

    def __init__(self) -> None:
        self.vertex_ids: list[int] = []

        self.neighbor_id: int = -1
        self.has_neighbor: bool = False

        self.area: float = 0.0
        self.normal: Normal = Normal()
        self.centroid: Point = Point()

    @property
    def n_vertices(self) -> int:
        """
        Return the number of vertices on the face.

        Returns
        -------
        int
        """
        return len(self.vertex_ids)

    def __str__(self) -> str:
        """
        Return the contents of the face as a string.

        Returns
        -------
        str
        """
        s = f"n_vertex_ids: {len(self.vertex_ids)}\n" \
            f"vertex_ids: ["
        for v in self.vertex_ids:
            s += f"{v} "
        s += "]\n"

        s += f"has_neighbor: {self.has_neighbor}\n" \
             f"neighbor_id: {self.neighbor_id}" \
             f"normal: {self.normal}\n" \
             f"centroid: {self.centroid}\n" \
             f"area: {self.area}\n"
        return s

    def __eq__(self, other: 'Face') -> bool:
        """
        Test the equality of two faces.

        Parameters
        ----------
        other : Face

        Returns
        -------
        bool
        """
        assert type(other) == type(self)
        return set(self.vertex_ids) == set(other.vertex_ids)

    def __ne__(self, other: 'Face') -> bool:
        """
        Test the inequality of two faces.

        Parameters
        ----------
        other : Face

        Returns
        -------
        bool
        """
        return not self == other
