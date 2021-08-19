import os
import numpy as np
from numpy import ndarray

from pyPDEs.spatial_discretization import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


class Outputs:
    def __init__(self):
        self.grid: List[List[float]] = []
        self.time: List[float] = []
        self.power: List[float] = []
        self.flux: List[List[ndarray]] = []
        self.precursors: List[List[ndarray]] = []

    def store_grid(self, sd: SpatialDiscretization):
        self.grid.clear()
        for point in sd.grid:
            self.grid.append([point.x, point.y, point.z])

    def store_outputs(self, solver: 'TransientSolver',
                      time: float) -> None:
        if time == 0.0:
            self.store_grid(solver.discretization)

        self.time.append(time)

        self.power.append(solver.power)

        n_grps, phi = solver.n_groups, np.copy(solver.phi)
        flux = [phi[g::n_grps] for g in range(n_grps)]
        self.flux.append(flux)

        if solver.use_precursors:
            n_dnps = solver.n_precursors
            precursors = np.copy(solver.precursors)
            precursors = [precursors[j::n_dnps] for j in range(n_dnps)]
            self.precursors.append(precursors)

    def write_outputs(self, path: str = ".") -> None:
        if not os.path.isdir(path):
            os.makedirs(path)
        if len(os.listdir(path)) > 0:
            os.system(f"rm -r {path}/*")

        time_path = os.path.join(path, "times.txt")
        np.savetxt(time_path, self.time, fmt="%.6g")

        grid_path = os.path.join(path, "grid.txt")
        np.savetxt(grid_path, self.grid, fmt="%.6g")

        power_path = os.path.join(path, "power.txt")
        np.savetxt(power_path, self.power, fmt="%.6e")

        flux_dirpath = os.path.join(path, "flux")
        if not os.path.isdir(flux_dirpath):
            os.makedirs(flux_dirpath)
        if len(os.listdir(flux_dirpath)) > 0:
            os.system(f"rm -r {flux_dirpath}/*")

        for g in range(len(self.flux[0])):
            filepath = os.path.join(flux_dirpath, f"g{g}.txt")
            np.savetxt(filepath, np.array(self.flux)[:, g])

        if len(self.precursors) > 0:
            precursor_dirpath = os.path.join(path, "precursors")
            if not os.path.isdir(precursor_dirpath):
                os.makedirs(precursor_dirpath)
            if len(os.listdir(precursor_dirpath)) > 0:
                os.system(f"rm - r {precursor_dirpath}/*")

            for j in range(len(self.precursors[0])):
                filepath = os.path.join(precursor_dirpath, f"j{j}.txt")
                np.savetxt(filepath, np.array(self.precursors)[:, j])

    def reset(self):
        self.grid.clear()
        self.time.clear()
        self.power.clear()
        self.flux.clear()
        self.precursors.clear()