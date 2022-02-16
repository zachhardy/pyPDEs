import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from studies.utils import *


def plot_minmax_power():
    plt.figure()
    plt.xlabel("Time ($\mu$s)", fontsize=12)
    plt.ylabel("Power (W)", fontsize=12)

    styles = ['-b*', '-ro']
    for i, s in enumerate([0, -1]):
        powers = dataset.simulations[s].powers
        r_b = dataset.parameters[s][0]
        plt.plot(dataset.times, powers, styles[i],
                 label=f"$r_b$ = {r_b:.4f} cm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


study = int(sys.argv[1])
dataset = get_data('three_group_sphere', study)

parameter_names = dataset.path.split('/')[-1].split('_')
parameters = dataset.parameters
if parameters.ndim == 1:
    parameters = parameters.reshape(-1, 1)

plot_minmax_power()
