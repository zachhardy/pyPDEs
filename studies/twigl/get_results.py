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
    simulation.plot_power()
plt.show()
