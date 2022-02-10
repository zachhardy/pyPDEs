import sys
import time
import warnings

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold

from studies.utils import *
from pyROMs.pod import POD
from pyROMs.dmd import DMD

warnings.filterwarnings('ignore')

only_interior = False

########################################
# Parse the data
########################################
case, study = int(sys.argv[1]), int(sys.argv[2])
dataset = get_data('three_group_sphere', case, study)

X = dataset.create_dataset_matrix('power_density')
Y = dataset.parameters

# KFold cross validation
count = 0
cv = {'mean': [], 'std': [],
      'max': [], 'min': [],
      'construction': [], 'query': []}

cross_validator = RepeatedKFold(n_splits=10, n_repeats=50)
if only_interior:
    interior = dataset.interior_map
    iterator = cross_validator.split(X[interior], Y[interior])
else:
    iterator = cross_validator.split(X, Y)

for train, test in iterator:
    if only_interior:
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
    svd_rank = 1.0 - 1.0e-10
    pod = POD(svd_rank=svd_rank)
    pod.fit(X_train, Y_train, 'rbf_gaussian', epsilon=100.0)
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

# Print aggregated POD results
msg = f"===== Summary of {len(cv['mean'])} CV Sets ====="
header = '=' * len(msg)
print('\n'.join(['', header, msg, header]))
print(f"Number of Snapshots:\t\t{pod.n_snapshots}")
print(f"Number of Validations:\t\t{len(X_test)}")
print(f"Number of POD Modes:\t\t{pod.n_modes}")
print()
print(f"Mean of Mean Prediction Error:\t\t{np.mean(cv['mean']):.3e}")
print(f"Max of Mean Prediction Error:\t\t{np.max(cv['mean']):.3e}")
print(f"Min of Mean Prediction Error:\t\t{np.min(cv['mean']):.3e}")
print()
print(f"Mean of Max Prediction Error:\t\t{np.mean(cv['max']):.3e}")
print(f"Max of Max Prediction Error:\t\t{np.max(cv['max']):.3e}")
print(f"Min of Max Prediction Error:\t\t{np.min(cv['max']):.3e}")
print()
print(f"Mean of Min Prediction Error:\t\t{np.mean(cv['min']):.3e}")
print(f"Max of Min Prediction Error:\t\t{np.max(cv['min']):.3e}")
print(f"Min of Min Prediction Error:\t\t{np.min(cv['min']):.3e}")
print()
print(f"Construction Time:\t{np.mean(cv['construction']):.3e} s")
print(f"Average Query Time:\t{np.mean(cv['query']):.3e} s")

import seaborn as sb

plt.figure()
plt.subplot(121)
plt.title("Mean Error")
sb.histplot(cv['mean'], bins=20, stat='probability',
            kde=True, log_scale=True)
plt.subplot(122)
plt.title("Maximum Error")
sb.histplot(cv['max'], bins=20, stat='probability',
            kde=True, log_scale=True)
plt.tight_layout()
plt.show()
