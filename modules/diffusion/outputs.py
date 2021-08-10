import os
import numpy as np

from pyPDEs.spatial_discretization import SpatialDiscretization

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


class Outputs:
    def __init__(self):
        self.grid: List[List[float]] = []
        self.time: List[float] = []
        self.power: List[float] = []
        self.flux: List[List[ndarray]] = []

    def store_grid(self, sd: SpatialDiscretization):
        self.grid.clear()
        for point in sd.grid:
            self.grid.append([point.x, point.y, point.z])

    def store_outputs(self, solver: 'TransientSolver',
                      time: float) -> None:
        if time == 0.0:
            self.store_grid(solver.discretization)

        self.time.append(time)

        power = solver.fv_compute_fission_production()
        self.power.append(power)

        n_grps, phi = solver.n_groups, np.copy(solver.phi)
        flux = [phi[g::n_grps] for g in range(n_grps)]
        self.flux.append(flux)

    def write_outputs(self, path: str = ".") -> None:
        if not os.path.isdir(path):
            os.makedirs(path)

        time_path = os.path.join(path, "time.txt")
        np.savetxt(time_path, self.time, fmt="%.6g")

        grid_path = os.path.join(path, "grid.txt")
        np.savetxt(grid_path, self.grid, fmt="%.6g")

        flux_dirpath = os.path.join(path, "flux")
        if not os.path.isdir(flux_dirpath):
            os.makedirs(flux_dirpath)
        os.system(f"rm -r {flux_dirpath}/*")

        for g in range(len(self.flux[0])):
            group_path = os.path.join(flux_dirpath, f"g{g}.txt")
            np.savetxt(group_path, np.array(self.flux)[:, g])

    def reset(self):
        self.grid.clear()
        self.time.clear()
        self.power.clear()
        self.flux.clear()