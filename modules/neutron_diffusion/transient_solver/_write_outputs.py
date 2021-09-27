import os
import struct

from pyPDEs.spatial_discretization import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


def write_snapshot(self: "TransientSolver",
                   output_num: int) -> None:
    """Write the system state to a binary file.

    Parameters
    ----------
    output_num : int
        The output index.
        This is used to name the snapshot file.
    """
    file_base = str(output_num).zfill(4)
    file_path = \
        os.path.join(self.output_directory, file_base + ".data")

    header_info = \
        b"pyPDEs neutron_diffusion TransientSolver: Snapshot file\n" \
        b"Header size: 1500 bytes\n" \
        b"Structure(type-info):\n" \
        b"uint64_t      time_step_index\n" \
        b"double        time\n" \
        b"double        system_power\n" \
        b"uint64_t      num_global_cells\n" \
        b"uint64_t      num_global_nodes\n" \
        b"uint64_t      num_groups\n" \
        b"uint64_t      num_precursors\n" \
        b"uint64_t      max_precursors\n" \
        b"unsigned int  num_data_blocks\n" \
        b"Each cell:\n" \
        b"   uint64_t    cell_global_id\n" \
        b"   uint64_t    cell_material_id\n" \
        b"   uint64_t    num_nodes\n" \
        b"   double      centroid_x_position\n" \
        b"   double      centroid_y_position\n" \
        b"   double      centroid_z_poisition\n" \
        b"   Each node:\n" \
        b"       double  x_position\n" \
        b"       double  y_position\n" \
        b"       double  z_position\n" \
        b"Flux Moment Records:\n" \
        b"   unsigned int    record_type\n" \
        b"   uint64_t        num_records\n" \
        b"   Each Record:\n" \
        b"   uint64_t        cell_global_id\n" \
        b"   unsigned int    node_num\n" \
        b"   unsigned int    moment_num\n" \
        b"   unsigned int    group_num\n" \
        b"   double          flux_moment_value\n" \
        b"Precursor Records:\n" \
        b"   unsigned int    record_type\n" \
        b"   uint64_t        num_records\n" \
        b"       uint64_t        cell_global_id\n" \
        b"       uint64_t        cell_material_id\n" \
        b"       unsigned int    precursor_num\n" \
        b"       double          precursor_value\n" \
        b"Temperature Records:\n" \
        b"   unsigned int    record_type\n" \
        b"   uint64_t        num_records\n" \
        b"   Each Record:\n" \
        b"       uint64_t    cell_global_id\n" \
        b"       double      temperature_value\n" \
        b"Power Density Records:\n" \
        b"   unsigned int    record_type\n" \
        b"   uint64_t        num_records\n" \
        b"   Each Record:\n" \
        b"       uint64_t    cell_global_id\n" \
        b"       double      power_density_value\n"

    header_size = len(header_info)
    header_info += b"-" * (1499 - header_size)

    with open(file_path, "wb") as f:
        f.write(bytearray(header_info))

        f.write(struct.pack("Q", output_num))
        f.write(struct.pack("d", self.time))
        f.write(struct.pack("d", self.power))
        f.write(struct.pack("Q", self.mesh.n_cells))
        f.write(struct.pack("Q", self.discretization.n_nodes))
        f.write(struct.pack("Q", 1))
        f.write(struct.pack("Q", self.n_groups))
        f.write(struct.pack("Q", self.n_precursors))
        f.write(struct.pack("Q", self.max_precursors))
        f.write(struct.pack("I", 4))

        # Write grid information
        for cell in self.mesh.cells:
            f.write(struct.pack("Q", cell.id))
            f.write(struct.pack("Q", cell.material_id))

            # Write FV grid information
            if self.discretization.type == "FV":
                f.write(struct.pack("Q", 1))

                # Write the centroid
                f.write(struct.pack("d", cell.centroid.x))
                f.write(struct.pack("d", cell.centroid.y))
                f.write(struct.pack("d", cell.centroid.z))

                # Write the node (which is the centroid)
                f.write(struct.pack("d", cell.centroid.x))
                f.write(struct.pack("d", cell.centroid.y))
                f.write(struct.pack("d", cell.centroid.z))

            # Write PWC grid information
            elif self.discretization.type == "PWC":
                pwc: PiecewiseContinuous = self.discretization
                view = pwc.fe_views[cell.id]
                f.write(struct.pack("Q", view.n_nodes))
                for n in range(view.n_nodes):
                    f.write(struct.pack("d", view.nodes[n].x))
                    f.write(struct.pack("d", view.nodes[n].y))
                    f.write(struct.pack("d", view.nodes[n].z))

            else:
                raise NotImplementedError(
                    "Only FV and PWC num_nodes can be written.")

        # Write scalar flux data
        f.write(struct.pack("I", 0))

        n_dofs = self.discretization.n_dofs(self.phi_uk_man)
        f.write(struct.pack("Q", n_dofs))

        for cell in self.mesh.cells:
            cell_id = cell.id

            # Write FV scalar flux information
            if self.discretization.type == "FV":
                fv: FiniteVolume = self.discretization

                # Loop over groups
                for g in range(self.n_groups):
                    dof_map = fv.map_dof(cell, 0, self.phi_uk_man, 0, g)

                    assert dof_map < len(self.phi)
                    value = self.phi[dof_map]

                    f.write(struct.pack("Q", cell_id))
                    f.write(struct.pack("I", 0))
                    f.write(struct.pack("I", 0))
                    f.write(struct.pack("I", g))
                    f.write(struct.pack("d", value))

            # Write PWC scalar flux information
            elif self.discretization.type == "PWC":
                pwc: PiecewiseContinuous = self.discretization
                view = pwc.fe_views[cell_id]

                # Loop over nodes
                for n in range(view.n_nodes):
                    # Loop over groups
                    for g in range(self.n_groups):
                        dof_map = pwc.map_dof(cell, n, self.phi_uk_man, 0, g)

                        assert dof_map < len(self.phi)
                        value = self.phi[dof_map]

                        f.write(struct.pack("Q", cell_id))
                        f.write(struct.pack("I", n))
                        f.write(struct.pack("I", 0))
                        f.write(struct.pack("I", g))
                        f.write(struct.pack("d", value))

        # Write precursor data
        f.write(struct.pack("I", 1))

        n_dofs = self.mesh.n_cells * self.max_precursors
        f.write(struct.pack("Q", n_dofs))

        for cell in self.mesh.cells:
            cell_id = cell.id
            mat_id = cell.material_id

            for j in range(self.max_precursors):
                dof_map = cell_id * self.max_precursors + j

                assert dof_map < len(self.precursors)
                value = self.precursors[dof_map]

                f.write(struct.pack("Q", cell_id))
                f.write(struct.pack("Q", mat_id))
                f.write(struct.pack("I", j))
                f.write(struct.pack("d", value))

        # Write temperature data
        f.write(struct.pack("I", 2))
        f.write(struct.pack("Q", self.mesh.n_cells))

        for cell in self.mesh.cells:
            f.write(struct.pack("Q", cell.id))
            f.write(struct.pack("d", self.temperature[cell.id]))

        # Write power density data
        f.write(struct.pack("I", 3))
        f.write(struct.pack("Q", self.mesh.n_cells))

        power_density = self.energy_per_fission * self.fission_density
        print(min(power_density), max(power_density))
        for cell in self.mesh.cells:
            f.write(struct.pack("Q", cell.id))
            f.write(struct.pack("d", power_density[cell.id]))
