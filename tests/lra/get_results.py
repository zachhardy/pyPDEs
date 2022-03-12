import os
import sys
import time
import warnings

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from readers import NeutronicsSimulationReader

warnings.filterwarnings('ignore')

script_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_path, 'outputs')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, [0, 1], [0.0], grouping='time')
plt.gcf().suptitle("")
for i, ax in enumerate(plt.gcf().get_axes()[::2]):
    ylabel = "Y (cm)" if i == 0 else ""
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Group {i}", fontsize=12)
    ax.tick_params(labelsize=12)

sim.plot_power(mode=1, log_scale=False)
plt.gca().legend(fontsize=12)
plt.gca().tick_params(labelsize=12)

sim.plot_temperatures()
plt.gca().legend(fontsize=12)
plt.gca().tick_params(labelsize=12)

sim.plot_flux_moments(0, [0], [0.0, 1.44], grouping='group')
plt.gcf().suptitle("")
for i, ax in enumerate(plt.gcf().get_axes()[::2]):
    ylabel = "Y (cm)" if i == 0 else ""
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=12)

plt.show()
