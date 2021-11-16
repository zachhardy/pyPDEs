import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray

from simulation_reader import SimulationReader

script_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_path, 'outputs')

sim = SimulationReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, 0, [0.0, 1.44])
sim.plot_power(mode=1, log_scale=True)
sim.plot_temperatures(mode=0)
plt.show()
