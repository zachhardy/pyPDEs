import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray

from simulation_reader import SimulationReader

script_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_path, "outputs")

sim = SimulationReader(path)
sim.read_simulation_data()

sim.plot_power(mode=1, log_scale=True)
sim.plot_temperatures(mode=0)
plt.show()

from rom.dmd import DMD
X = sim.create_simulation_matrix(variables="power_density")

dmd = DMD(svd_rank=10, opt=True)
dmd.fit(X[:137], sim.times[:137])

dmd1 = DMD(svd_rank=-1)
dmd1.fit(X[136:151], sim.times[136:151])

dmd2 = DMD(svd_rank=40)
dmd2.fit(X[150:], sim.times[150:])

plt.show()

