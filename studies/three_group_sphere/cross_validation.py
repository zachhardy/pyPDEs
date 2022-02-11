import sys
import time
import warnings

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold

from studies.utils import *
from pyROMs.pod import POD

warnings.filterwarnings('ignore')

########################################
# Parse the data
########################################
study = int(sys.argv[1])
dataset = get_data('three_group_sphere', study)

var = 'power_density'
n_splits = 5 if study == 0 else 2
n_repeats = 500 // n_splits
interior_only = True
tau = 1.0e-8
interp = 'rbf_gaussian' if study == 0 else 'rbf_cubic'
eps = 1.0 if study == 0 else 200.0
if interior_only and study == 3:
    n_splits = 4
    n_repeats = 100
    interp = 'rbf'

X = dataset.create_dataset_matrix(var)
Y = dataset.parameters

# KFold cross validation
count = 0
cv = {'mean': [], 'std': [],
      'max': [], 'min': [],
      'construction': [], 'query': []}

cross_validator = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
if interior_only:
    interior = dataset.interior_map
    iterator = cross_validator.split(X[interior], Y[interior])
else:
    iterator = cross_validator.split(X, Y)

for train, test in iterator:
    if interior_only:
        boundary = dataset.boundary_map
        X_train, Y_train = X[interior][train], Y[interior][train]
        X_test, Y_test = X[interior][test], Y[interior][test]
        X_train = np.vstack((X_train, X[boundary]))
        Y_train = np.vstack((Y_train, Y[boundary]))
    else:
        X_train, Y_train = X[train], Y[train]
        X_test, Y_test = X[test], Y[test]

    # Construct POD model, predict test data
    start_time = time.time()
    svd_rank = 1.0 - tau
    pod = POD(svd_rank=svd_rank)
    pod.fit(X_train, Y_train, interp, epsilon=eps)
    contruction_time = time.time() - start_time
    cv['construction'].append(contruction_time)

    # Predict results
    X_pred = []
    predict_time = 0.0
    for y_test in Y_test:
        start_time = time.time()
        X_pred.append(pod.predict(y_test))
        query_time = time.time() - start_time
    cv['query'].append(query_time)

    # Format datasets
    X_pred = dataset.unstack_simulation_vector(X_pred)
    X_test = dataset.unstack_simulation_vector(X_test)

    # Compute simulation errors
    errors = np.zeros(len(X_test))
    for i in range(len(X_test)):
        errors[i] = norm(X_test[i] - X_pred[i]) / norm(X_test[i])

    cv['mean'].append(np.mean(errors))
    cv['min'].append(np.min(errors))
    cv['max'].append(np.max(errors))

import seaborn as sb
from typing import List
from matplotlib.pyplot import Figure, Axes

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig: Figure = fig
axs: List[Axes] = axs.ravel()

for i, ax in enumerate(axs):
    data = cv['mean'] if i == 0 else cv['max']
    title = "Mean Error" if i == 0 else "Maximum Error"
    ylabel = "Probability" if i == 0 else ""
    sb.histplot(data, bins=10, stat='probability',
                kde=True, log_scale=True, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Relative $L^2$ Error", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True)
plt.tight_layout()
plt.show()

msg = "\\begin{tabular}{|c|c|c|c|}" \
      "\n\t\hline" \
      "\n\t\\textbf{Quantity} & \\textbf{Value} \\\\ \hline"
msg += f"\n\t \hline Mean of Set Means & {np.mean(cv['mean']):.3e}"
msg += f"\n\t \hline Maximum of Set Means & {np.max(cv['mean']):.3e}"
msg += f"\n\t \hline Minimum of Set Means & {np.min(cv['mean']):.3e}"
msg += f"\n\t \hline Maximum of Set Maximums & {np.max(cv['max']):.3e}"
msg += f"\n\t \hline Minimum of Set Minimums & {np.min(cv['min']):.3e}"
msg += f"\n\t \hline \end{{tabular}}"
print(msg)

print()
print(f"Number of POD Modes:\t\t{pod.n_modes}")
print(f"Number of Snapshots:\t\t{pod.n_snapshots}")
print(f"Number of Validations:\t\t{len(X_test)}")
print(f"Average Construction Time:\t{np.mean(cv['construction']):.3e} s")
print(f"Average Query Time:\t\t{np.mean(cv['query']):.3e} s")
