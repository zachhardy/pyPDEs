from typing import List, Set
from copy import deepcopy

__all__ = ['DirectedGraph', 'VertexAccessor', 'GraphVertex']


class DirectedGraph:
    """
    Implementation of a directed graph.
    """
    def __init__(self) -> None:
        self.vertices: VertexAccessor = None
        self.valid_flags: List[bool] = []

    def add_vertex(self, v_id: int = -1) -> None:
        """
        Add a vertex to the directed graph.

        Parameters
        ----------
        v : int, default -1
            The ID to assign to the vertex.
        """
        if self.vertices is None:
            self.vertices = VertexAccessor()
        self.vertices.add_vertex(v_id)

    def remove_vertex(self, v: int) -> None:
        """
        Remove vertex `v` from the directed graph.

        Parameters
        ----------
        v : int
        """
        if self.vertices is None:
            raise AssertionError(
                f'VertexAccessor has not been initialized.')
        self.vertices.remove_vertex(v)

    def add_edge(self, start: int, end: int) -> None:
        """
        Add an edge to the directed graph.

        Parameters
        ----------
        start : int
            The vertex index the edge comes from.
        end : int
            The vertex index the edge goes to.
        """
        self.vertices[start].downstream_edge.add(end)
        self.vertices[end].upstream_edge.add(start)

    def remove_vertex(self, v: int) -> None:
        """
        Remove a vertex from the directed graph.

        Parameters
        ----------
        v : int
            The vertex index to remove.
        """
        if v < 0 or v >= len(self.vertices):
            raise ValueError(f'Invalid vertex index {v}.')

        vertex = self.vertices[v]

        # Get adjacent vertices
        adj_vertices: List[int] = []
        for u in vertex.upstream_edge:
            adj_vertices.append(u)
        for u in vertex.downstream_edge:
            adj_vertices.append(u)

        # Remove vertex from adjecent vertices
        for u in adj_vertices:
            self.vertices[u].upstream_edge.remove(v)
            self.vertices[u].downstream_edge.remove(v)

        # Change valid flag
        self.valid_flags[v] = False

    def remove_edge(self, start: int, end: int) -> None:
        """
        Remove an edge from the directed graph.

        Parameters
        ----------
        start : int
            The vertex index the edge comes from.
        end : int
            The vertex index the edge goes to.
        """
        self.vertices[start].downstream_edge.remove(end)
        self.vertices[end].upstream_edge.remove(start)

    def create_topological_sort(self) -> List[int]:
        """
        Generate a topological sorting from Kahn's algorithm.

        Returns
        -------
        List[int]
        """
        L: List[int] = []
        S: List[GraphVertex] = []

        # Copy the vertices
        vertices: VertexAccessor = deepcopy(self.vertices)

        # Identify vertices with no incoming edge
        for vertex in vertices:
            vertex: GraphVertex = vertices
            if len(vertex.upstream_edge) == 0:
                S.append(vertex)

        if len(S) == 0:
            raise AssertionError(
                'There must be vertices with no incoming edge.')

        # Repeatedly remove vertices
        while len(S) > 0:
            node_n: GraphVertex = S.pop()
            L.append(node_n.id)

            nodes_m = node_n.downstream_edge
            for m in nodes_m:
                node_m: GraphVertex = vertices[m]

                node_n.downstream_edge.remove(m)
                node_m.upstream_edge.remove(n)

                if len(node_m.upstream_edge) == 0:
                    S.append(node_m)
        return L


class VertexAccessor:
    """
    Vertex accessor class.
    """
    def __init__(self) -> None:
        self.vertices: List[GraphVertex] = []
        self.vertex_valid_flags: List[bool] = []

    def __getitem__(self, v: int) -> None:
        """
        Get vertex `v`.

        Parameters
        ----------
        v : int
            The vertex index to get.
        """
        if not self.vertex_valid_flags[v]:
            raise ValueError(f'Invalid vertex index {v}.')
        return self.vertices[v]

    def add_vertex(self, v_id: int = -1) -> None:
        """
        Add a vertex to the directed graph.

        Parameters
        ----------
        v_id : int, default -1
            An ID to attach to the vertex.
        """
        v_id = v_id if v_id >= 0 else len(self.vertices)
        self.vertices.append(GraphVertex(v_id))
        self.vertex_valid_flags.append(True)

    def remove_vertex(self, v: int) -> None:
        """
        Remove a vertex from the directed graph.

        Parameters
        ----------
        v : int
            The vertex to remove.
        """
        if v < 0 or v >= len(self.vertices):
            raise ValueError(f'Invalid vertex ID {v}.')

        vertex = self.vertices[v]

        # Get adjacent vertices
        adj_verts: List[int] = []
        for u in vertex.upstream_edge:
            adj_verts.append(u)
        for u in vertex.downstream_edge:
            adj_verts.append(u)

        # Remove vertex fro up/downstream of neighbors
        for u in adj_verts:
            self.vertices[u].upstream_edge.remove(v)
            self.vertices[u].downstream_edge.remove(v)

        # Change flag
        self.vertex_valid_flags[v] = False


class GraphVertex:
    """
    Implementation of a graph vertex.

    Parameters
    ----------
    v : int, default -1
    """
    def __init__(self, v: int = -1) -> None:
        self.id: int = v
        self.upstream_edge: Set[int] = set()
        self.downstream_edge: Set[int] = set()
