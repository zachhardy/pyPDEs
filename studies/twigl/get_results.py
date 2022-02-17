import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from studies.utils import *

########################################
# Parse the results
########################################
study = int(sys.argv[1])
dataset = get_data('twigl', study)
n_params = dataset.n_parameters

simulations = [dataset.simulations[0],
               dataset.simulations[-1]]

for simulation in simulations:
    simulation.plot_flux_moments(0, [0, 1], [0.0, 0.5], 'group')
    simulation.plot_power()
plt.show()
