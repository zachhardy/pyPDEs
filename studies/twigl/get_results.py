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

for s, simulation in enumerate(simulations):
    simulation.plot_power()
    ylabel = "Power (W)" if s == 0 else ""
    plt.gca().set_xlabel(f"Time (s)", fontsize=12)
    plt.gca().set_ylabel(ylabel, fontsize=12)
plt.show()
