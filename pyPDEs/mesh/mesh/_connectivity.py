from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import  Mesh


def establish_connectivity(self: 'Mesh') -> None:
    """
    Establish the cell/face connectivity.

    Notes
    -----
    This routine is quite slow and should only be used when
    there are no predefined rules to determine mesh connectivity.
    Primarily, this will be used with unstructured meshes.
    """
    # Vertex-cell mapping
    vc_map = [set()] * len(self.vertices)
    for cell in self.cells:
        for vid in cell.vertex_ids:
            vc_map[vid].add(cell.id)

    # Loop over cells
    cells_to_search = set()
    for cell in self.cells:

        # Get neighbor cells
        cells_to_search.clear()
        for vid in cell.vertex_ids:
            for cid in vc_map[vid]:
                if cid != cell.id:
                    cells_to_search.add(cid)

        # Loop over faces
        for face in cell.faces:
            if face.has_neighbor:
                continue

            this_vids = set(face.vertex_ids)

            # Loop over neighbors
            nbr_found = False
            for nbr_cell_id in cells_to_search:
                nbr_cell: Cell = self.cells[nbr_cell_id]

                # Loop over neighbor faces
                for nbr_face in nbr_cell.faces:
                    nbr_vids = set(nbr_face.vertex_ids)

                    if this_vids == nbr_vids:
                        face.neighbor_id = nbr_cell.id
                        nbr_face.neighbor_id = cell.id

                        face.has_neighbor = True
                        nbr_face.has_neighbor = True

                        nbr_found = True

                    # Break loop over neighbor faces
                    if nbr_found:
                        break

                # Break loop over neighbor cells
                if nbr_found:
                    break

        if any([f.has_neighbor for f in cell.faces]):
            self.boundary_cell_ids.append(cell.id)
