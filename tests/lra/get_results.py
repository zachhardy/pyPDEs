import os
import sys
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
sim.plot_power_densities(times=[0.0, 1.43, 2.0])
sim.plot_temperature_profiles(times=[0.0, 1.43, 2.0])
plt.show()
