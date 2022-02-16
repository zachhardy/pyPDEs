import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from studies.utils import *


########################################
# Parse the results
########################################
problem, study = int(sys.argv[1]), int(sys.argv[2])
dataset = get_data('infinite_slab', problem, study)
n_params = dataset.n_parameters

simulations = [dataset.simulations[0],
               dataset.simulations[-1]]

for simulation in simulations:
    simulation.plot_power_densities([0.0, 2.0])
    simulation.plot_power()
plt.show()
