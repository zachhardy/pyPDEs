import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from studies.utils import *

########################################
# Parse the results
########################################
study = int(sys.argv[1])
dataset = get_data('lra', study)
n_params = dataset.n_parameters
times = dataset.times

plt.figure()
for s, simulation in enumerate(dataset.simulations):
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")

    P_peak = simulation.peak_power_densities
    P_avg = simulation.average_power_densities

    plt.plot(times, P_peak, '-*', ms=2.5, label=f"Simulation {s}")
plt.grid(True)
plt.legend()
plt.show()