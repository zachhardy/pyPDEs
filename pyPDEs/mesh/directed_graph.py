from copy import deepcopy


class DirectedGraph:
    """
    Implementation of a directed graph.
    """
    def __init__(self) -> None:
        self.vertices: list[GraphVertex] = []
        self.vertex_valid_flags: list[bool] = []

    def add_vertex(self, v_id: int = -1) -> None:
        """
        Add a vertex to the directed graph.

        Parameters
        ----------
        v : int, default -1
            The ID to assign to the vertex.
        """
        v_id = v_id if v_id >= 0 else len(self.vertices)
        self.vertices.append(GraphVertex(v_id))
        self.vertex_valid_flags.append(True)

    def remove_vertex(self, v: int) -> None:
        """
        Remove vertex `v` from the directed graph.

        Parameters
        ----------
        v : int
        """
        if v < 0 or v >= len(self.vertices):
            raise ValueError(f'Invalid vertex index {v}.')

        # Get adjacent vertices
        adj_verts = []
        for u in self.vertices[v].upstream_edge:
            adj_verts.append(u)
        for u in self.vertices[v].downstream_edge:
            adj_verts.append(u)

        # Remove vertex from up/downstreams
        for u in adj_verts:
            self.vertices[u].upstream_edge.remove(v)
            self.vertices[u].downstream_edge.remove(v)

        # Change valid flag
        self.vertex_valid_flags[v] = False

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
        if not self.vertex_valid_flags[start]:
            raise ValueError(f'Invalid upstream vertex index {start}.')
        if not self.vertex_valid_flags[end]:
            raise ValueError(f'Invalid downstream vertex index {end}.')
        self.vertices[start].downstream_edge.add(end)
        self.vertices[end].upstream_edge.add(start)

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
        if not self.vertex_valid_flags[start]:
            raise ValueError(f'Invalid upstream vertex index {start}.')
        if not self.vertex_valid_flags[end]:
            raise ValueError(f'Invalid downstream vertex index {end}.')
        self.vertices[start].downstream_edge.remove(end)
        self.vertices[end].upstream_edge.remove(start)

    def create_topological_sort(self) -> list[int]:
        """
        Generate a topological sorting from Kahn's algorithm.

        Returns
        -------
        list[int]
        """
        L: list[int] = []
        S: list[GraphVertex] = []

        # Identify count of upstream edges, add vertices
        # with no upstream edges to queue
        for v in range(len(self.vertices)):
            node: GraphVertex = self.vertices[v]
            if len(node.upstream_edge) == 0:
                S.append(node)

        # Repeatedly remove vertices
        while len(S) > 0:
            node_n: GraphVertex = S.pop()
            L.append(node_n.id)

            ds_num = 0
            while node_n.downstream_edge:
                m = node_n.downstream_edge.pop()
                node_m: GraphVertex = self.vertices[m]

                # Remove upstream edge
                node_m.upstream_edge.remove(node_n.id)

                # Add no upstream node to queue
                if len(node_m.upstream_edge) == 0:
                    S.append(node_m)
        return L


class GraphVertex:
    """
    Implementation of a graph vertex.

    Parameters
    ----------
    v_id : int, default -1
    """
    def __init__(self, v_id: int = -1) -> None:
        self.id: int = v_id
        self.upstream_edge: set[int] = set()
        self.downstream_edge: set[int] = set()
